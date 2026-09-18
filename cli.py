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
import re
import sys
import time
from pathlib import Path

from novel_engine.audit.timeline_rules import check_continuity
from novel_engine.canon.models import Assertion, Entity, Relation, StateDelta
from novel_engine.canon.timeline import ContinuityFrame
from novel_engine.graph.build import run_chapter
from novel_engine.eval.debt import story_debt
from novel_engine.eval.harness import evaluate
from novel_engine.eval.regression import run_regression
from novel_engine.graph.checkpoint import (
    CHECKPOINT_DB, chapter_thread, drop_thread, open_checkpointer, thread_status,
)
from novel_engine.graph.engines import build_engines
from novel_engine.reconcile.classify import classify_delta, downstream_chapters
from novel_engine.reconcile.commit import reconcile
from novel_engine.audit.deterministic import (
    audit_contract, audit_stats, deterministic_audit,
)
from novel_engine.llm.env import load_dotenv
from novel_engine.render import render_chapter_markdown as _render_markdown

DB_PATH = "novel_storage.db"
OUT_CHAPTERS = Path("output/chapters")
OUT_REPORTS = Path("output/reports")


def _llm(name: str, writer_model: str = ""):
    from novel_engine.llm.gemini import build_llm
    kw = {"role_models": {"writer": writer_model}} if writer_model else {}
    return build_llm(name, **kw)


def _render_blocked(b: dict) -> str:
    lines = [f"## Cảnh {b['scene_index']} — BỊ CHẶN sau {b['revisions']} lần viết lại",
             "", "> Lỗi còn lại:"]
    lines += [f"> - [{f.get('severity')}] {f.get('check')}: {f.get('message', '')}"
              for f in b.get("findings", [])
              if f.get("severity") in ("blocker", "major")]
    lines += ["", (b.get("prose") or "").strip(), ""]
    return "\n".join(lines)


def _audit_summary(out: dict) -> dict:
    scenes = [{"scene_id": s["scene_id"], **s.get("audit", {})}
              for s in out.get("scene_outputs", [])]
    return {
        "revisions": sum(s.get("revisions", 0) for s in scenes),
        "polish_called": sum(1 for s in scenes if s.get("polish", {}).get("called")),
        "polish_accepted": sum(1 for s in scenes
                               if s.get("polish", {}).get("accepted")),
        "residual_major": sum(1 for s in scenes for f in s.get("residual", [])
                              if f["severity"] == "major"),
        "hygiene": out.get("hygiene_notes", []),
        "scenes": scenes,
        "log": out.get("audit_log", []),
    }


# ═══════════════════════════ write ═══════════════════════════

def cmd_write(args) -> int:
    eng = build_engines(_llm(args.llm, getattr(args, "writer_model", "")),
                        db_path=args.db)
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
        if args.no_checkpoint:
            out = run_chapter(eng, args.chapter)
        else:
            with open_checkpointer(args.checkpoint_db) as cp:
                tt = thread_status(cp, args.chapter)
                if args.resume and not tt["exists"]:
                    print(f"không có checkpoint cho chương {args.chapter} — bỏ "
                          f"`--resume` để viết từ đầu", file=sys.stderr)
                    return 2
                if args.resume and tt["done"]:
                    print(f"checkpoint chương {args.chapter} đã ở trạng thái XONG; "
                          f"viết lại bằng `--force`", file=sys.stderr)
                    return 2
                if args.resume:
                    print(f"  [--resume] tiếp tục từ {', '.join(tt['next']) or 'đầu'} "
                          f"· {tt['scenes_done']} cảnh đã xong")
                elif tt["exists"]:
                    # Chạy lại cùng thread mà KHÔNG xoá: LangGraph chạy lại từ
                    # đầu và CỘNG DỒN lên state cũ — chương ra 12 cảnh, frame
                    # trùng lặp, không một lỗi nào được ném.
                    drop_thread(cp, args.chapter)
                    print(f"  bỏ checkpoint cũ của chương {args.chapter}")
                out = run_chapter(eng, args.chapter, checkpointer=cp,
                                  thread_id=chapter_thread(args.chapter),
                                  resume=args.resume)
        dt = time.time() - t0

        if out.get("escalated"):
            print(f"\n⚠ ESCALATION sau {dt:.0f}s: {out.get('escalation_reason')}",
                  file=sys.stderr)
            if args.traceback and out.get("traceback"):
                print(out["traceback"], file=sys.stderr)
            # Escalate ở bước CUỐI (extract) nghĩa là toàn bộ văn xuôi đã được
            # viết và đã trả tiền. Thoát mà không lưu là vứt đi phần đắt nhất.
            # Escalate trong vòng kiểm toán (§9.3) thì bản nháp BỊ CHẶN cũng được
            # giữ, kèm lỗi — đó chính là thứ tác giả cần đọc để quyết định.
            blocked = out.get("escalated_scene")
            if out.get("scene_outputs") or blocked:
                OUT_CHAPTERS.mkdir(parents=True, exist_ok=True)
                draft = OUT_CHAPTERS / f"ch{args.chapter:03d}.escalated.md"
                md = _render_markdown(eng, args.chapter, out.get("scene_outputs", []))
                if blocked:
                    md += _render_blocked(blocked)
                draft.write_text(md, encoding="utf-8")
                print(f"  văn xuôi đã viết được giữ ở {draft}", file=sys.stderr)
            if blocked:
                for f in blocked["findings"]:
                    if f.get("severity") in ("blocker", "major"):
                        print(f"      [{f['severity']}] {f.get('check')}: "
                              f"{f.get('message', '')[:110]}", file=sys.stderr)
            OUT_REPORTS.mkdir(parents=True, exist_ok=True)
            (OUT_REPORTS / f"ch{args.chapter:03d}.escalated.json").write_text(
                json.dumps({"chapter": args.chapter, "backend": args.llm,
                            "reason": out.get("escalation_reason"),
                            "rolled_back": bool(out.get("rolled_back")),
                            "scenes_done": [s["scene_id"]
                                            for s in out.get("scene_outputs", [])],
                            "escalated_scene": blocked,
                            "audit_log": out.get("audit_log", [])},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            if out.get("rolled_back"):
                print("  canon store đã được trả về trạng thái trước chương này",
                      file=sys.stderr)
            force = " --force" if eng.store.has_chapter(args.chapter) else ""
            print(f"  viết lại: python cli.py write --chapter {args.chapter}{force}",
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
            "audit": _audit_summary(out),
            "foreshadow": out.get("foreshadow_report", {}),
        }
        (OUT_REPORTS / f"ch{args.chapter:03d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n✓ {dt:.0f}s · {len(md.split())} từ · {len(frames)} cảnh")
        print(f"  epoch {frames[0].time.epoch_tick} → {frames[-1].time.end_tick}")
        print(f"  {md_path}")
        au = report["audit"]
        print(f"  kiểm toán: {au['revisions']} lần viết lại · polish nhận "
              f"{au['polish_accepted']}/{au['polish_called']} · "
              f"{au['residual_major']} major còn lại")
        for h in au.get("hygiene", []):
            print(f"      🧹 {h['scene_id']}: {'; '.join(h['notes'])[:100]}")
        for s in au["scenes"]:
            pr = s.get("polish", {})
            if pr.get("called") and not pr.get("accepted"):
                print(f"      {s['scene_id']} polish bị từ chối: {pr.get('reason', '')[:90]}")
        if coerced:
            print(f"  ⚠ hàng rào canon đã chặn {len(coerced)} trường hợp:")
            for sid, c in coerced[:5]:
                print(f"      {sid} | {c}")
        fs = out.get("foreshadow_report") or {}
        if fs:
            modes: dict[str, int] = {}
            for d in fs.get("directives", []):
                modes[d["mode"]] = modes.get(d["mode"], 0) + 1
            print(f"  manh mối: {len(fs.get('directives', []))} chỉ thị"
                  + "".join(f" · {m}×{n}" for m, n in sorted(modes.items()))
                  + f" · {len(fs.get('unplaced', []))} không xếp được"
                  + f" · {len(fs.get('escalations', []))} cần tác giả quyết")
            for x in fs.get("unplaced", [])[:3]:
                print(f"      [không xếp] {x['clue_id']}: {x['reason']}")
            for x in fs.get("blocked", [])[:3]:
                print(f"      [chờ tiền đề] {x['clue_id']}: {x['reason']} "
                      f"{x.get('waiting_on', [])}")
            for x in fs.get("escalations", [])[:3]:
                print(f"      [CP-4] {x['clue_id']} quá hạn {x['overdue_by']} chương — "
                      f"{x['question']}")
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
        book = getattr(eng.graph, "relationships", None)
        if book is not None and book.states:
            print("\n  quan hệ:")
            for k, st in sorted(book.states.items()):
                print(f"    {k:<26} {st.stage.value:<13} {st.chapters_in_stage} chương · "
                      f"gắn kết {st.intimacy:.0f} · va chạm {st.friction:.0f} · "
                      f"xung đột lợi ích {st.stake_conflict:.0f} · sẹo {len(st.scars)}")
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


_DICH_DEN = {
    "entity": "world graph · thực thể",
    "relation": "world graph · quan hệ",
    "truth": "world graph · sự thật khách quan",
    "belief": "hồ sơ nhân vật · niềm tin",
    "claim": "lời khai — KHÔNG thành niềm tin của người nói",
}


def _dest(item) -> str:
    if isinstance(item, Entity):
        return _DICH_DEN["entity"]
    if isinstance(item, Relation):
        return _DICH_DEN["relation"]
    if item.epistemic == "believed_by":
        return _DICH_DEN["belief"]
    if item.epistemic == "claimed_by":
        return _DICH_DEN["claim"]
    return _DICH_DEN["truth"]


def _diff_lines(delta, out: dict, eng, chapter: int) -> list[str]:
    """CP-2 (§16.1): "diff ngắn, không phải JSON thô".

    Mỗi dòng trả lời ba câu: ghi gì, xếp hạng gì, và đi đâu. Hạng `improvement`
    kèm luôn hai lựa chọn của tác giả (CP-4) và danh sách chương bị đụng — quyết
    định mà không biết cái giá thì không phải quyết định.
    """
    dau = {"enrichment": "+", "improvement": "!", "contradiction": "✗"}
    lines = [f"  Chương {chapter} đề xuất ghi vào canon:"]
    for k, v in out["classification"].items():
        try:
            item = delta.item(k)
        except KeyError:
            continue
        lines.append(f"  {dau.get(v, '?')} {_describe(item)}   [{v}] → {_dest(item)}")
        if v == "improvement":
            ds = downstream_chapters(item, eng.planner, chapter)
            if ds:
                lines.append(f"      ⚠ đụng kế hoạch chương "
                             f"{', '.join(str(x) for x in ds)}")
            lines.append("      [G] Giữ chi tiết mới, sửa kế hoạch hạ nguồn"
                         "    [B] Bỏ chi tiết, viết lại cảnh")
        elif v == "contradiction":
            lines.append("      ✗ chặn ghi CẢ chương — sửa văn xuôi hoặc bible rồi "
                         "`write --force`")
    for cid, st in (delta.clue_transitions or {}).items():
        c = eng.graph.clues.get(cid)
        lines.append(f"  ~ Manh mối {cid}: → {getattr(st, 'value', st)}"
                     f"   [code suy ra từ bằng chứng]"
                     + (f" · hạn trả bài ch{c.payoff_deadline}" if c else ""))
    n_pe = sum(1 for e in delta.plant_evidence if e.verified)
    if n_pe:
        lines.append(f"  ◇ {n_pe} bằng chứng manh mối đã lên trang")
    if delta.relationship_events:
        kinds: dict[str, int] = {}
        for ev in delta.relationship_events:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        lines.append("  ♦ Quan hệ: "
                     + ", ".join(f"{k}×{n}" for k, n in sorted(kinds.items()))
                     + "   [chỉ số do code tính]")
    if len(lines) == 1:
        lines.append("  (không có mục nào — xem phần trích xuất ở trên)")
    return lines


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
                                 eng.store.get_frames(args.chapter),
                                 vocabulary=eng.vocabulary)

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
        for line in _diff_lines(delta, out, eng, args.chapter):
            print(line)
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
        if a.get("relationship_events") or a.get("relationship_transitions"):
            print(f"  quan hệ: {a.get('relationship_events', 0)} sự kiện · "
                  f"{len(a.get('relationship_skipped', []))} bị bỏ")
        nw = res.get("news") or {}
        if nw.get("records"):
            print(f"  tin tức: {nw['records']} bản ghi tri thức · "
                  f"{nw['rejected']} lần từ chối đính chính")
        for t in a.get("relationship_transitions", []):
            print(f"  ♦ {t['pair']}: {t['from']} → {t['to']} — {t['reason'][:90]}")
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


def _split_chapter_md(md: str) -> list[tuple[int, str]]:
    parts = re.split(r"^## Cảnh (\d+)\s*$", md, flags=re.M)
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def cmd_audit(args) -> int:
    """Kiểm toán tất định (§10.3) trên chương ĐÃ VIẾT. Không gọi LLM.

    In kèm số đo thô (σ, tỉ lệ câu ngắn, mệnh đề phụ, kênh giác quan, thoại gán
    được) — §10.3.3 nói ngưỡng phải hiệu chỉnh trên văn xuôi thật.

    Mã thoát: 0 không có blocker · 2 chưa có chương · 3 có blocker.
    """
    path = OUT_CHAPTERS / f"ch{args.chapter:03d}.md"
    if not path.exists():
        print(f"chưa có {path}", file=sys.stderr)
        return 2
    eng = build_engines(_llm("fake"), db_path=args.db)
    try:
        scenes = _split_chapter_md(path.read_text(encoding="utf-8"))
        report = {"chapter": args.chapter, "scenes": []}
        dem: dict[str, int] = {}
        blocker = False
        print(f"Kiểm toán tất định chương {args.chapter} · {len(scenes)} cảnh")
        truoc: list[dict] = []
        for si, prose in scenes:
            c = audit_contract(eng, args.chapter, si)
            findings = deterministic_audit(prose, c, args.chapter, eng,
                                           previous_scenes=truoc)
            st = audit_stats(prose, c, previous_scenes=truoc)
            truoc.append({"scene_id": c["scene_id"], "prose": prose,
                          "location": eng.planner.location_id(args.chapter, si),
                          "cast": [x["id"] for x in c["active_characters"]]})
            report["scenes"].append({"scene_id": c["scene_id"], "stats": st,
                                     "findings": findings})
            r = st["rhythm"] or {}
            gan = sum(st["dialogue_attributed"].values())
            print(f"  {c['scene_id']} · {st['words']} từ · σ={r.get('sd', '-')} · "
                  f"câu ngắn {r.get('short_ratio', '-')} · mệnh đề phụ "
                  f"{r.get('sub_ratio', '-')} · {len(st['sensory'])} kênh · thoại gán "
                  f"{gan}/{gan + st['dialogue_unattributed']}")
            for x in findings:
                dem[x["check"]] = dem.get(x["check"], 0) + 1
                blocker = blocker or x["severity"] == "blocker"
                print(f"      [{x['severity']}] {x['check']}: {x['message'][:110]}")
        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        rp = OUT_REPORTS / f"ch{args.chapter:03d}_audit.json"
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print("  tổng: " + (", ".join(f"{k}×{v}" for k, v in sorted(dem.items()))
                            or "sạch"))
        print(f"  {rp}")
        return 3 if blocker else 0
    finally:
        eng.store.close()


def cmd_eval(args) -> int:
    """Chấm bộ chỉ số M1–M15 trên các chương đã ghi (§13).

    Mã thoát: 0 không có báo động · 3 có báo động đỏ.
    """
    eng = build_engines(_llm("fake"), db_path=args.db)
    try:
        den = args.den or max((int(f.scene_id[2:5]) for f in eng.store.get_frames()),
                              default=args.tu)
        so = list(range(args.tu, den + 1))
        judge = _llm(args.judge) if args.judge else None
        rep = evaluate(eng, so, judge=judge)
        print(f"Chỉ số M1–M15 · chương {args.tu}–{den} · "
              f"judge={args.judge or 'không'}")
        for name, res in rep["metrics"].items():
            v = res.get("value")
            gia_tri = "chưa đo được" if v is None else f"{v}"
            co = any(a["metric"] == name for a in rep["alerts"])
            print(f"  {'⚠' if co else ' '} {name:<26} {gia_tri:>14}  "
                  f"(mục tiêu {res['target']})"
                  + (f"  — {res.get('reason')}" if v is None else ""))
        for a in rep["alerts"]:
            print(f"  ⚠ {a['metric']}: {a['value']} ngoài mục tiêu "
                  f"{a['target']} — {a['likely_cause']}")
        if args.baseline:
            base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            reg = run_regression(rep, base)
            rep["regression"] = reg
            print(f"  hồi quy so với {args.baseline}: {reg['verdict']}"
                  + (f" — tụt: {', '.join(reg['regressed'])}" if reg["regressed"] else ""))
        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        rp = OUT_REPORTS / (args.out or f"eval_ch{args.tu:03d}_{den:03d}.json")
        rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
        print(f"  {rp}")
        return 3 if rep["alerts"] else 0
    finally:
        eng.store.close()


def cmd_report_debt(args) -> int:
    """Nợ tự sự toàn cục (§6.4, §16.1) — thứ đã hứa với độc giả mà chưa trả.

    Mã thoát: 0 trong ngưỡng · 3 vượt ngưỡng, nên trả nợ trước khi mở tuyến mới.
    """
    eng = build_engines(_llm("fake"), db_path=args.db)
    try:
        den = args.den or max((int(f.scene_id[2:5]) for f in eng.store.get_frames()),
                              default=0)
        d = story_debt(eng, last_chapter=den)
        c = d["clues"]
        print(f"Nợ tự sự tới chương {den} · debt_load {d['debt_load']}"
              f"/{d['block_threshold']}"
              + ("  ⚠ VƯỢT NGƯỠNG" if d["blocked"] else ""))
        for m in c["overdue"]:
            print(f"  ⚠ manh mối QUÁ HẠN {m.get('clue', m.get('npc'))} "
                  f"— trễ {m['overdue_by']} chương")
        for m in c["late_to_plant"]:
            print(f"  ⚠ chưa cài kịp {m['clue']} — hạn cài ch{m['plant_by']}, "
                  f"trễ {m['late_by']} chương")
        for m in c["at_risk"]:
            print(f"  · sắp đến hạn {m['clue']} — còn {m['slack']} chương")
        for m in c["forgotten"]:
            print(f"  · độc giả đã quên {m['clue']} — salience {m['salience']}, "
                  f"im {m['silent_for']} chương")
        for r in d["relationships"]:
            print(f"  · quan hệ {r['pair']}: {r['reason']}")
        for n in d["news"]:
            print(f"  · tin {n['news_id']} {n['reason']}")
        for p in d["plan_patches"]:
            print(f"  ⚑ CP-4 {p.get('patch_id')}: chờ tác giả quyết")
        if d["unresolved_total"]:
            print(f"  · {d['unresolved_total']} chỉ mục treo; gần nhất:")
            for u in d["unresolved"][-4:]:
                print(f"      ch{u['chapter']}: {u['thread'][:90]}")
        OUT_REPORTS.mkdir(parents=True, exist_ok=True)
        rp = OUT_REPORTS / "debt.json"
        rp.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
        print(f"  {rp}")
        return 3 if d["blocked"] else 0
    finally:
        eng.store.close()


def cmd_show(args) -> int:
    p = OUT_CHAPTERS / f"ch{args.chapter:03d}.md"
    if not p.exists():
        print(f"chưa có {p}", file=sys.stderr)
        return 2
    print(p.read_text(encoding="utf-8"))
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from novel_engine.web import create_app

    uvicorn.run(create_app(default_db=args.db), host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="cli.py", description="Novel Engine v2.5")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="viết một chương")
    w.add_argument("--chapter", type=int, required=True)
    w.add_argument("--llm", default=None,
                   help="fake (mặc định) | gemini")
    w.add_argument("--writer-model", dest="writer_model", default="",
                   help="model riêng cho vai Writer, vd gemini-3.8-flash — "
                        "6/26 lượt mỗi chương, nên chậm hơn nhưng không gấp 13 lần")
    w.add_argument("--db", default=DB_PATH)
    w.add_argument("--force", action="store_true",
                   help="XOÁ bản cũ rồi viết lại chương đã có")
    w.add_argument("--traceback", action="store_true")
    w.add_argument("--resume", action="store_true",
                   help="tiếp tục chương đang dở từ checkpoint")
    w.add_argument("--checkpoint-db", dest="checkpoint_db", default=CHECKPOINT_DB)
    w.add_argument("--no-checkpoint", dest="no_checkpoint", action="store_true")
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

    au = sub.add_parser("audit", help="kiểm toán tất định chương đã viết (§10.3)")
    au.add_argument("--chapter", type=int, required=True)
    au.add_argument("--db", default=DB_PATH)
    au.set_defaults(func=cmd_audit)

    ev = sub.add_parser("eval", help="chấm bộ chỉ số M1–M15 (§13)")
    ev.add_argument("--from", dest="tu", type=int, default=1)
    ev.add_argument("--to", dest="den", type=int)
    ev.add_argument("--judge", default="", help="fake|gemini — bật M6, M7")
    ev.add_argument("--baseline", default="", help="file eval JSON để so hồi quy")
    ev.add_argument("--out", default="")
    ev.add_argument("--db", default=DB_PATH)
    ev.set_defaults(func=cmd_eval)

    rd = sub.add_parser("report-debt", help="nợ tự sự toàn cục (§6.4)")
    rd.add_argument("--to", dest="den", type=int)
    rd.add_argument("--db", default=DB_PATH)
    rd.set_defaults(func=cmd_report_debt)

    sv = sub.add_parser("serve", help="chạy giao diện web Xưởng viết")
    sv.add_argument("--db", default=DB_PATH)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    if getattr(args, "llm", None) is None:
        import os
        args.llm = os.environ.get("NOVEL_LLM", "fake")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
