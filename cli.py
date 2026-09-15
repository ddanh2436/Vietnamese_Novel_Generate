"""Author Console — giao diện dòng lệnh (§16.1).

    python cli.py write --chapter 1
    python cli.py write --chapter 2 --llm gemini
    python cli.py status
    python cli.py show --chapter 1

Mặc định backend là `fake`: chạy `cli.py` mà vô tình đốt hạn ngạch vì quên đặt
biến môi trường là chuyện không nên xảy ra. Muốn gọi model thật thì phải nói
ra — `--llm gemini` hoặc `NOVEL_LLM=gemini`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from novel_engine.audit.timeline_rules import check_continuity
from novel_engine.canon.models import Assertion, Entity, Relation, StateDelta
from novel_engine.canon.timeline import ContinuityFrame
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.reconcile.classify import classify_delta
from novel_engine.reconcile.commit import reconcile
from novel_engine.llm.env import load_dotenv

DB_PATH = "novel_storage.db"
OUT_CHAPTERS = Path("output/chapters")
OUT_REPORTS = Path("output/reports")


def _llm(name: str):
    from novel_engine.llm.gemini import build_llm
    return build_llm(name)


def _render_markdown(eng, chapter: int, scene_outputs: list[dict]) -> str:
    parts = [f"# Chương {chapter} — {eng.planner.title(chapter)}", ""]
    for i, s in enumerate(scene_outputs):
        parts += [f"## Cảnh {i}", "", s["prose"].strip(), ""]
    return "\n".join(parts)


# ═══════════════════════════ write ═══════════════════════════

def cmd_write(args) -> int:
    eng = build_engines(_llm(args.llm), db_path=args.db)
    try:
        # Idempotency (§16.1 CP-2): chạy lại một chương đã có là thao tác PHÁ
        # HUỶ — nó xoá frame, digest và delta của bản cũ. Phải nói ra rõ ràng.
        if eng.store.has_chapter(args.chapter):
            if not args.force:
                print(f"Chương {args.chapter} đã có trong {args.db}.\n"
                      f"  • Xem lại:   python cli.py show --chapter {args.chapter}\n"
                      f"  • Viết lại:  python cli.py write --chapter "
                      f"{args.chapter} --force   (XOÁ bản cũ)",
                      file=sys.stderr)
                return 2
            print(f"[--force] xoá bản cũ của chương {args.chapter}…")
            eng.store.clear_chapter(args.chapter)
            # Canon trong bộ nhớ vẫn chứa sự thật của bản cũ — fold lại.
            eng.rebuild_canon()

        truoc = eng.store.get_deltas(args.chapter - 1) if args.chapter > 1 else []
        if truoc and not truoc[0].committed:
            print(f"⚠ chương {args.chapter - 1} chưa ghi canon — chương này sẽ "
                  f"không thấy các sự thật của nó. Chạy `commit --chapter "
                  f"{args.chapter - 1}` trước nếu muốn.", file=sys.stderr)
        print(f"Viết chương {args.chapter} · backend={args.llm} · db={args.db}")
        t0 = time.time()
        out = run_chapter(eng, args.chapter)
        dt = time.time() - t0

        if out.get("escalated"):
            print(f"\n⚠ ESCALATION sau {dt:.0f}s: {out.get('escalation_reason')}",
                  file=sys.stderr)
            if args.traceback and out.get("traceback"):
                print(out["traceback"], file=sys.stderr)
            # Escalate ở bước CUỐI (extract) nghĩa là toàn bộ văn xuôi đã được
            # viết và đã trả tiền. Thoát mà không lưu là vứt đi phần đắt nhất.
            if out.get("scene_outputs"):
                OUT_CHAPTERS.mkdir(parents=True, exist_ok=True)
                draft = OUT_CHAPTERS / f"ch{args.chapter:03d}.escalated.md"
                draft.write_text(_render_markdown(eng, args.chapter,
                                                  out["scene_outputs"]),
                                 encoding="utf-8")
                print(f"  văn xuôi đã viết được giữ ở {draft}", file=sys.stderr)
            print(f"  viết lại: python cli.py write --chapter {args.chapter} --force",
                  file=sys.stderr)
            return 1

        OUT_CHAPTERS.mkdir(parents=True, exist_ok=True)
        md_path = OUT_CHAPTERS / f"ch{args.chapter:03d}.md"
        md = _render_markdown(eng, args.chapter, out["scene_outputs"])
        md_path.write_text(md, encoding="utf-8")

        # Ghi delta vào LOG. §3.1: log chỉ GHI THÊM theo thứ tự chương — đó là
        # lịch sử sáng tác, bất biến, dùng để rollback. Ghi vào GRAPH là việc
        # của `reconcile_node` (GĐ3), sau khi tác giả duyệt ở CP-2. Hai thao
        # tác khác nhau: lưu bằng chứng ≠ chấp nhận bằng chứng.
        delta = StateDelta.model_validate(out["delta"]) if out.get("delta") \
            else StateDelta().stamp(args.chapter)
        eng.store.append_delta(delta)

        frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
        # Luật liên tục chạy trên DÒNG ĐỜI nhân vật, XUYÊN CHƯƠNG (§3.6.1) —
        # không phải trong phạm vi một chương. Cảnh hồi ức của Chương 2 neo
        # vào `CH001_S01`; đưa cho nó mỗi frame của Chương 2 thì nó không tra
        # được mốc neo, và `no_teleport` cũng không thấy nhân vật đi từ đâu
        # tới. Lấy tất cả từ store — bản vừa viết đã nằm trong đó.
        findings = check_continuity(eng.store.get_frames(), eng.graph)
        coerced = [(s["scene_id"], c) for s in out["scene_outputs"]
                   for c in s.get("coerced", [])]

        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        report = {
            "chapter": args.chapter, "backend": args.llm,
            "seconds": round(dt, 1), "words": len(md.split()),
            "scenes": len(out["scene_outputs"]),
            "epoch_span": [frames[0].time.epoch_tick, frames[-1].time.end_tick],
            "continuity_findings": findings,
            "coerced": [{"scene_id": s, "note": c} for s, c in coerced],
            "time_drift": out.get("time_drift", []),
            "extraction": out.get("extraction_report", {}),
            "unresolved": out.get("unresolved", []),
        }
        (OUT_REPORTS / f"ch{args.chapter:03d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n✓ {dt:.0f}s · {len(md.split())} từ · {len(frames)} cảnh")
        print(f"  epoch {frames[0].time.epoch_tick} → {frames[-1].time.end_tick}")
        print(f"  {md_path}")
        if coerced:
            print(f"  ⚠ hàng rào canon đã chặn {len(coerced)} trường hợp:")
            for sid, c in coerced[:5]:
                print(f"      {sid} | {c}")
        ex = out.get("extraction_report", {})
        if ex:
            print(f"  trích xuất: {ex.get('assertions_kept', 0)} mệnh đề giữ · "
                  f"{ex.get('new_entities', 0)} thực thể mới · "
                  f"{len(ex.get('rejected_spans', []))} span bị loại")
            for x in ex.get("rejected_spans", [])[:3]:
                print(f"      [{x['reason']}] {str(x.get('span',''))[:62]}")
            if ex.get("promised_plants"):
                print(f"      manh mối: {len(ex.get('fulfilled_plants', []))}"
                      f"/{len(ex['promised_plants'])} cài được")
        for d in out.get("time_drift", []):
            print(f"  ⏱ {d}")
        if findings:
            print(f"  ⚠ {len(findings)} vi phạm liên tục:")
            for f in findings[:5]:
                print(f"      [{f['severity']}] {f['message']}")
        else:
            print("  ✓ liên tục: sạch")
        if out.get("unresolved"):
            print(f"  … {len(out['unresolved'])} chỉ mục còn treo")
        print(f"  → duyệt:     python cli.py review --chapter {args.chapter}")
        print(f"  → ghi canon: python cli.py commit --chapter {args.chapter}")
        return 0
    finally:
        eng.store.close()


# ═══════════════════════════ status ═══════════════════════════

def cmd_status(args) -> int:
    eng = build_engines(_llm("fake"), db_path=args.db)
    try:
        frames = eng.store.get_frames()
        if not frames:
            print(f"{args.db}: chưa có chương nào.")
            return 0
        by_ch: dict[int, list] = {}
        for f in frames:
            ch = int(f.scene_id.split("_")[0].removeprefix("CH"))
            by_ch.setdefault(ch, []).append(f)
        print(f"{args.db} — {len(by_ch)} chương, {len(frames)} cảnh\n")
        print(f"  {'ch':>3}  {'cảnh':>4}  {'epoch':>15}  {'đọc':>7}  {'canon':>6}  tiêu đề")
        for ch in sorted(by_ch):
            fs = sorted(by_ch[ch], key=lambda f: f.time.narrative_order)
            present = [f for f in fs if f.time.mode == "present"]
            span = (f"{min(f.time.epoch_tick for f in present)}"
                    f"→{max(f.time.end_tick for f in present)}"
                    if present else "—")
            orders = f"{fs[0].time.narrative_order}–{fs[-1].time.narrative_order}"
            special = [f for f in fs if f.time.mode != "present"]
            title = eng.planner.title(ch) if ch in eng.planner.chapters else ""
            ds = eng.store.get_deltas(ch)
            canon = "đã ghi" if ds and ds[0].committed else ("chờ" if ds else "—")
            print(f"  {ch:>3}  {len(fs):>4}  {span:>15}  {orders:>7}  {canon:>6}  {title}")
            for f in special:
                print(f"         └ {f.scene_id} [{f.time.mode}] tick "
                      f"{f.time.epoch_tick} ← neo {f.time.anchor_scene}")
        # Kiểm tra liên tục TOÀN CỤC — trên dòng đời nhân vật, xuyên chương.
        findings = check_continuity(frames, eng.graph)
        print(f"\n  liên tục toàn cục: "
              f"{'sạch' if not findings else str(len(findings)) + ' vi phạm'}")
        for f in findings[:8]:
            print(f"    [{f['severity']}] {f['message']}")
        return 0
    finally:
        eng.store.close()


def _summary_of(cls: dict, retr: dict, quarantine: list) -> dict:
    return {
        "enrichment": sum(1 for v in cls.values() if v == "enrichment"),
        "improvement": sum(1 for v in cls.values() if v == "improvement"),
        "contradictions": (sum(1 for v in cls.values() if v == "contradiction")
                           + sum(1 for v in retr.values() if v == "contradiction")),
        "quarantined": len(quarantine),
        "irony_seeds": 0,
    }


def _describe(item) -> str:
    if isinstance(item, Entity):
        return f"{item.id} ({item.kind}) “{item.name}”"
    if isinstance(item, Relation):
        return f"{item.src} -[{item.type}]-> {item.dst}"
    if isinstance(item, Assertion):
        who = f"  ← {item.holder}" if item.holder else ""
        return f"{item.subject}.{item.predicate} = {str(item.object)[:60]}{who}"
    return str(item)


def cmd_classify(args) -> int:
    """CP-2 (§16.1) — xem TRƯỚC những gì sẽ vào canon, nhóm theo NƠI ĐẾN.

    Không ghi gì, không gọi LLM. Delta đã ghi thì hiển thị quyết định ĐÃ LƯU:
    phân loại lại trên canon đã chứa chính delta đó sẽ thấy mọi thứ "đã có".

    Mã thoát: 0 sạch · 2 chưa có delta · 3 có mâu thuẫn canon.
    """
    eng = build_engines(_llm("fake"), db_path=args.db)
    try:
        deltas = eng.store.get_deltas(args.chapter)
        if not deltas:
            print(f"chưa có delta cho chương {args.chapter} — chạy "
                  f"`write --chapter {args.chapter}` trước", file=sys.stderr)
            return 2
        delta = deltas[0]
        if delta.committed:
            out = {"classification": delta.classification,
                   "retractions": delta.retraction_classification,
                   "quarantine": delta.quarantine, "irony_seeds": [],
                   "findings": [],
                   "summary": _summary_of(delta.classification,
                                          delta.retraction_classification,
                                          delta.quarantine)}
        else:
            out = classify_delta(delta, eng.graph, eng.planner,
                                 eng.store.get_frames(args.chapter))

        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        rp = OUT_REPORTS / f"ch{args.chapter:03d}_classify.json"
        rp.write_text(json.dumps({"delta_id": delta.delta_id,
                                  "committed": delta.committed, **out},
                                 ensure_ascii=False, indent=2), encoding="utf-8")

        sm = out["summary"]
        trang_thai = "ĐÃ GHI canon" if delta.committed else "chờ duyệt"
        print(f"Phân loại {delta.delta_id} ({trang_thai}): enrichment "
              f"{sm['enrichment']} · improvement {sm['improvement']} · mâu thuẫn "
              f"{sm['contradictions']} · cách ly {sm['quarantined']}")

        if delta.extraction_issues:
            dem: dict[str, int] = {}
            for x in delta.extraction_issues:
                k = f"{x.get('stage')}:{x.get('reason')}"
                dem[k] = dem.get(k, 0) + 1
            print(f"  ⚠ TRÍCH XUẤT · {len(delta.extraction_issues)} vấn đề trước "
                  f"phân loại: " + ", ".join(f"{k}×{v}" for k, v in sorted(dem.items())))
            for x in delta.extraction_issues:
                if x.get("reason") == "empty_extraction":
                    print(f"      [major] {x['message']}")
        groups: dict[str, list] = {"entity": [], "relation": [], "truth": [],
                                   "belief": [], "claim": [], "contradiction": []}
        for k, v in out["classification"].items():
            try:
                item = delta.item(k)
            except KeyError:
                continue
            if v == "contradiction":
                grp = "contradiction"
            elif isinstance(item, Assertion) and item.epistemic == "believed_by":
                grp = "belief"
            elif isinstance(item, Assertion) and item.epistemic == "claimed_by":
                grp = "claim"
            elif isinstance(item, Assertion):
                grp = "truth"
            elif isinstance(item, Relation):
                grp = "relation"
            else:
                grp = "entity"
            groups[grp].append((v, item))
        titles = [
            ("entity", "→ WORLD GRAPH · thực thể"),
            ("relation", "→ WORLD GRAPH · quan hệ"),
            ("truth", "→ WORLD GRAPH · sự thật khách quan"),
            ("belief", "→ HỒ SƠ NHÂN VẬT · niềm tin"),
            ("claim", "→ LỜI KHAI · ghi là điều đã nói, KHÔNG thành niềm tin của người nói"),
            ("contradiction", "✗ MÂU THUẪN · chặn ghi cả chương"),
        ]
        for grp, title in titles:
            if groups[grp]:
                print(f"  {title} ({len(groups[grp])})")
                for v, item in groups[grp]:
                    print(f"      [{v:11}] {_describe(item)}")
        if out["retractions"]:
            print(f"  → ĐÓNG QUAN HỆ ({len(out['retractions'])})")
            for k, v in out["retractions"].items():
                print(f"      [{v:11}] {k}")
        by_reason: dict[str, list[dict]] = {}
        for q in out["quarantine"]:
            by_reason.setdefault(q["reason"], []).append(q)
        for reason, items in sorted(by_reason.items()):
            print(f"  ⏸ CÁCH LY · {reason} ({len(items)}) — không ghi")
            for q in items[:4]:
                print(f"      [{q['severity']}] {q['message'][:96]}")
        print(f"  {rp}")
        if not delta.committed:
            print(f"  bước tiếp: python cli.py commit --chapter {args.chapter}")
        return 3 if sm["contradictions"] else 0
    finally:
        eng.store.close()


def cmd_commit(args) -> int:
    """Ghi delta đã duyệt vào canon (§11).

    Mã thoát: 0 đã ghi · 2 chưa có delta / đã ghi trước đó · 3 bị chặn.
    """
    eng = build_engines(_llm("fake"), db_path=args.db)
    try:
        deltas = eng.store.get_deltas(args.chapter)
        if not deltas:
            print(f"chưa có delta cho chương {args.chapter}", file=sys.stderr)
            return 2
        delta = deltas[0]
        if delta.committed:
            print(f"{delta.delta_id} đã được ghi canon trước đó", file=sys.stderr)
            return 2
        res = reconcile(delta, eng)
        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        rp = OUT_REPORTS / f"ch{args.chapter:03d}_commit.json"
        rp.write_text(json.dumps({"delta_id": delta.delta_id, **res},
                                 ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
        if res["status"] == "escalated":
            print(f"⚠ KHÔNG GHI {delta.delta_id} — {res['escalation_reason']}",
                  file=sys.stderr)
            return 3
        a = res["applied"]
        print(f"✓ Ghi canon {delta.delta_id}: {a['entities']} thực thể · "
              f"{a['relations']} quan hệ · {a['truths']} sự thật · "
              f"{a['beliefs']} niềm tin · {a['claims']} lời khai · "
              f"{a['retracted']} quan hệ đóng · {a['clues_touched']} manh mối")
        if any(x.get("reason") == "empty_extraction"
               for x in delta.extraction_issues):
            print(f"  ⚠ trích xuất RỖNG — canon không nhận thêm sự thật nào từ "
                  f"chương này. Xem `raw` trong output/reports/"
                  f"ch{args.chapter:03d}.json; viết lại bằng "
                  f"`write --chapter {args.chapter} --force` nếu cần.")
        for p in res["plan_patches"]:
            print(f"  ⚑ {p['patch_id']}: đụng tới {', '.join(p['invalidated_beats'])}"
                  f" — tác giả phải quyết định (CP-4)")
            for r in p["suggested_rewrites"][:4]:
                print(f"      {r['item']}: {r['reason']}")
        for seed in res["classify"].get("irony_seeds", []):
            print(f"  ◆ irony: {seed['holder']} nói/tin '{seed['claim']}' — sai với canon")
        print(f"  {rp}")
        return 0
    finally:
        eng.store.close()


def cmd_show(args) -> int:
    p = OUT_CHAPTERS / f"ch{args.chapter:03d}.md"
    if not p.exists():
        print(f"chưa có {p}", file=sys.stderr)
        return 2
    print(p.read_text(encoding="utf-8"))
    return 0


def main(argv=None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="cli.py", description="Novel Engine v2.5")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="viết một chương")
    w.add_argument("--chapter", type=int, required=True)
    w.add_argument("--llm", default=None,
                   help="fake (mặc định) | gemini")
    w.add_argument("--db", default=DB_PATH)
    w.add_argument("--force", action="store_true",
                   help="XOÁ bản cũ rồi viết lại chương đã có")
    w.add_argument("--traceback", action="store_true")
    w.set_defaults(func=cmd_write)

    s = sub.add_parser("status", help="trạng thái canon store")
    s.add_argument("--db", default=DB_PATH)
    s.set_defaults(func=cmd_status)

    sh = sub.add_parser("show", help="in một chương đã viết")
    sh.add_argument("--chapter", type=int, required=True)
    sh.set_defaults(func=cmd_show)

    c = sub.add_parser("classify", help="phân loại delta đã lưu (§11)")
    c.add_argument("--chapter", type=int, required=True)
    c.add_argument("--db", default=DB_PATH)
    c.set_defaults(func=cmd_classify)

    rv = sub.add_parser("review", help="CP-2: xem trước những gì sẽ vào canon")
    rv.add_argument("--chapter", type=int, required=True)
    rv.add_argument("--db", default=DB_PATH)
    rv.set_defaults(func=cmd_classify)

    cm = sub.add_parser("commit", help="ghi delta đã duyệt vào canon (§11)")
    cm.add_argument("--chapter", type=int, required=True)
    cm.add_argument("--db", default=DB_PATH)
    cm.set_defaults(func=cmd_commit)

    args = ap.parse_args(argv)
    if getattr(args, "llm", None) is None:
        import os
        args.llm = os.environ.get("NOVEL_LLM", "fake")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
