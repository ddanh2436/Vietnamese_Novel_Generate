"""Phân luồng ghi canon (§11) — Ngày 8.

    sự thật khách quan, thực thể, quan hệ  → World Graph
    believed_by                            → hồ sơ nhân vật  (+ graph.beliefs)
    claimed_by                             → graph.beliefs   (KHÔNG thành niềm tin)
    quan hệ thu hồi                        → close_relation
    cách ly / mâu thuẫn                    → không ghi gì

═══ CANON LÀ FOLD, KHÔNG PHẢI BẢNG ════════════════════════════════════════

CLI chạy đa tiến trình, và graph được dựng lại từ bible ở MỖI tiến trình. Nếu
"ghi canon" chỉ là sửa graph trong bộ nhớ, mọi thứ đã ghi biến mất khi tiến
trình thoát. §3.1 đã nói đúng cách: canon = fold các delta đã ghi. Nên:

- `reconcile()` phân loại, kiểm tra, rồi gọi `apply_delta()` và đánh dấu
  `committed` trong log.
- `replay_committed()` chạy lúc dựng engine, gọi CHÍNH `apply_delta()` đó trên
  mọi delta đã ghi.

Một hàm cho cả hai đường (NT-11). Hai bản sao của luật ghi sẽ khiến trạng thái
"lúc ghi" và "lúc phát lại" lệch nhau, và lệch loại đó không bao giờ tự lộ.

Hệ quả có lợi: rollback (§3.1, "bỏ chương 18, viết lại") không cần code riêng —
xoá delta khỏi log rồi dựng lại canon.

Fold theo `epoch_tick` của chương, không theo số chương (§3.6.4).
"""
from __future__ import annotations

from novel_engine.canon.flashback import flashback_admissible
from novel_engine.canon.models import (
    Assertion, Entity, Relation, StateDelta, item_key,
)
from novel_engine.canon.timeline import ContinuityFrame
from novel_engine.character.beliefs import update_belief
from novel_engine.reconcile.classify import (
    classify_delta, is_objective, retract_key,
)
from novel_engine.reconcile.replan import replan_downstream
from novel_engine.canon.models import ClueStatus
from novel_engine.foreshadow.scheduler import decay
from novel_engine.world.news import chapter_end_tick, settle_news

DEFAULT_PLANT_INTENSITY = 0.6


def _as_frames(frames) -> list[ContinuityFrame]:
    return [f if isinstance(f, ContinuityFrame)
            else ContinuityFrame.model_validate(f) for f in (frames or [])]


def _scene_index(scene_id: str) -> int | None:
    try:
        return int(scene_id.rsplit("_S", 1)[1])
    except (IndexError, ValueError):
        return None


def scene_ticks(frames: list[ContinuityFrame],
                chapter: int) -> tuple[dict[int, int], int]:
    """`{chỉ số cảnh: epoch_tick}` và mốc cơ sở của chương.

    Mốc cơ sở là cảnh `present` sớm nhất — không phải cảnh sớm nhất, vì một
    chương có hồi ức sẽ có cảnh ở tick 2.480 giữa các cảnh ở tick 20.000.
    """
    prefix = f"CH{chapter:03d}_"
    own = [f for f in frames if f.scene_id.startswith(prefix)]
    ticks: dict[int, int] = {}
    for f in own:
        i = _scene_index(f.scene_id)
        if i is not None:
            ticks[i] = f.time.epoch_tick
    present = [f.time.epoch_tick for f in own if f.time.mode == "present"]
    base = min(present) if present else (min(ticks.values()) if ticks else 0)
    return ticks, base


# ═══════════════════════ HÀM GHI DUY NHẤT ═══════════════════════

def apply_delta(delta: StateDelta, graph, chars: dict, frames) -> dict:
    """Áp các mục ĐÃ PHÂN LOẠI của một delta vào graph và hồ sơ nhân vật.

    Không phân loại lại: quyết định đã lưu trong delta là quyết định. Nhờ vậy
    phát lại ra đúng trạng thái lúc ghi, kể cả khi canon lúc phát lại đã khác.
    """
    ch = delta.chapter
    ticks, base = scene_ticks(_as_frames(frames), ch)
    prov = f"extracted_ch{ch}"
    applied = {"entities": 0, "relations": 0, "truths": 0, "beliefs": 0,
               "claims": 0, "retracted": 0, "clues_touched": 0,
               "clue_transitions": 0, "understood_by": 0}

    # Thứ tự trong `classification` là thực thể → quan hệ → mệnh đề (thứ tự
    # `classify_delta` duyệt), nên đầu mút luôn tồn tại trước khi quan hệ ghi.
    for key, verdict in delta.classification.items():
        if verdict not in ("enrichment", "improvement"):
            continue
        item = delta.item(key)

        if isinstance(item, Entity):
            graph.upsert_entity(item.model_copy(update={
                "first_appearance": item.first_appearance or ch,
                "attributes": {**item.attributes, "provenance": prov}}))
            applied["entities"] += 1

        elif isinstance(item, Relation):
            graph.upsert_relation(item.model_copy(update={
                "provenance": prov, "since_chapter": ch, "since_tick": base}))
            applied["relations"] += 1

        elif isinstance(item, Assertion):
            # Mốc của CHÍNH cảnh chứa mệnh đề — mệnh đề của cảnh hồi ức ghi ở
            # tick hồi ức, không ở tick hiện tại (§3.6.4).
            tick = ticks.get(item.scene, base)
            if is_objective(item):
                graph.commit_truth(item.subject, item.predicate, item.object,
                                   tick=tick)
                applied["truths"] += 1
                continue
            graph.commit_belief(holder=item.holder, subject=item.subject,
                                predicate=item.predicate, object=item.object,
                                kind=item.epistemic, since_tick=tick)
            if item.epistemic == "believed_by":
                prof = chars.get(item.holder)
                if prof is not None:
                    is_true = (not graph.conflicts_with_truth(item)
                               if graph.has_truth(item.subject, item.predicate)
                               else None)
                    update_belief(
                        prof,
                        proposition=f"{item.subject} {item.predicate} {item.object}",
                        confidence=item.confidence,
                        source=f"believed_by_ch{ch}", is_actually_true=is_true)
                applied["beliefs"] += 1
            else:
                applied["claims"] += 1

    # F7: đóng quan hệ bị thu hồi.
    for r in delta.retracted_relations:
        if delta.retraction_classification.get(retract_key(r)) == "enrichment":
            graph.close_relation(r.src, r.dst, r.type, until_chapter=ch)
            applied["retracted"] += 1

    # E4: không chỗ nào cập nhật `last_touched_chapter`/`salience` thì `decay()`
    # trả 0.0 mãi mãi và hệ thống ép cài lại cùng một manh mối ở mọi chương.
    touched: dict[str, float] = {}
    for pe in delta.plant_evidence:
        clue = graph.clues.get(pe.clue_id)
        if not pe.verified or clue is None:
            continue
        inten = delta.plant_intensity.get(pe.clue_id, DEFAULT_PLANT_INTENSITY)
        touched[pe.clue_id] = max(touched.get(pe.clue_id, 0.0), inten)
        if pe.surface_form:
            graph.record_surface_form(pe.clue_id, pe.surface_form)
        applied["understood_by"] += _ghi_nhan_hieu(clue, pe, chars)
    for cid, inten in sorted(touched.items()):
        clue = graph.clues[cid]
        # Nhắc thoáng qua không khôi phục trí nhớ độc giả bằng một cảnh nhấn
        # mạnh — nhưng cũng KHÔNG làm độc giả quên bớt. Bản trước GHI ĐÈ
        # `salience = 0.45 + 0.55·cường độ`: một manh mối đang ở 0.79 được nhắc
        # thoáng (0.2) thì tụt về 0.56. Ở đây cộng dồn bão hoà trên mức ĐÃ PHAI.
        # Nhiều bằng chứng cùng manh mối trong một chương chỉ chạm MỘT lần.
        cur = decay(clue, ch)
        boost = 0.45 + 0.55 * inten
        clue.salience = round(min(1.0, 1 - (1 - cur) * (1 - boost)), 4)
        clue.last_touched_chapter = ch
        applied["clues_touched"] += 1

    for cid, status in delta.clue_transitions.items():
        clue = graph.clues.get(cid)
        if clue is None:
            continue
        clue.status = status
        if status.value == "planted" and clue.planted_in_chapter is None:
            clue.planted_in_chapter = ch
        applied["clue_transitions"] += 1

    # Quan hệ nhân vật (§7.2–7.3): chỉ SỰ KIỆN đã xác minh. `relationship_updates`
    # — trạng thái do LLM viết, kể cả `stage` — không bao giờ được đọc.
    book = getattr(graph, "relationships", None)
    if book is not None:
        rel = book.settle(ch, delta.relationship_events, _as_frames(frames), chars)
        applied["relationship_events"] = rel["applied"]
        applied["relationship_skipped"] = rel["skipped"]
        applied["relationship_transitions"] = rel["transitions"]

    return applied


def _ghi_nhan_hieu(clue, pe, chars: dict) -> int:
    """`concluded_by` → `understood_by_characters`.

    Extractor khai ai đã RÚT RA KẾT LUẬN từ một manh mối, mục ấy được parse vào
    `PlantEvidence`… rồi không ai đọc. `understood_by_characters` — thứ quyết
    định nhân vật nào được phép nhắc tới manh mối ở chương sau (§5.4.1, và
    `known_by` ở networkx_graph) — không bao giờ được cập nhật. Cả một vòng đời
    tri thức nằm chết trong một trường được điền đủ.

    Chỉ ghi khi span ĐÃ XÁC MINH (NT-5): ai đó hiểu ra điều gì là chuyện phải
    có trên trang giấy, không phải chuyện model tuyên bố.

    Mã nhân vật được chuẩn hoá hoa-thường trước khi đối chiếu: lượt viết lại
    Arc 1 trả về "CHAR_KAELen", và vì không ai đọc trường này nên chẳng ai kêu.
    """
    if not chars:
        return 0
    tra = {c.upper(): c for c in chars}
    them = 0
    for cid in pe.concluded_by:
        that = tra.get(str(cid).strip().upper())
        if that and that not in clue.understood_by_characters:
            clue.understood_by_characters.append(that)
            them += 1
    return them


def derive_clue_transitions(delta: StateDelta, clues: dict) -> tuple[dict, list[dict]]:
    """Trạng thái manh mối do CODE suy ra, không do LLM khai (NT-13).

    `plan_coverage` gác transition bằng bằng chứng xác minh — nhưng GIÁ TRỊ vẫn
    là của model. Manh mối được giao để NHẮC LẠI, có bằng chứng thật, model khai
    `paid_off` → canon ghi đã trả bài một chi tiết chưa ai ghép nối. Bằng chứng
    chứng minh chi tiết đã lên trang; mode của chỉ thị cho biết nó lên trang để
    làm gì. Hai thứ đó đủ để suy ra trạng thái:

        nháp            + bằng chứng           → planted
        đang sống       + mode payoff          → paid_off
        planted         + bằng chứng khác      → reinforced

    Trả bài chỉ đi qua chỉ thị `payoff`. Bằng chứng tình cờ (Writer cài một manh
    mối không được giao) tính là cài/nhắc, không bao giờ là trả bài.
    """
    out: dict[str, ClueStatus] = {}
    for pe in delta.plant_evidence:
        c = clues.get(pe.clue_id)
        if not pe.verified or c is None or pe.clue_id in out:
            continue
        if c.status in (ClueStatus.PAID_OFF, ClueStatus.RETIRED):
            continue
        mode = delta.plant_modes.get(pe.clue_id)
        if c.status == ClueStatus.DRAFTED:
            new = ClueStatus.PLANTED
        elif mode == "payoff":
            new = ClueStatus.PAID_OFF
        elif c.status == ClueStatus.PLANTED:
            new = ClueStatus.REINFORCED
        else:
            new = c.status
        if new != c.status:
            out[pe.clue_id] = new
    notes = [{"clue_id": cid, "proposed": st.value,
              "derived": out[cid].value if cid in out else None}
             for cid, st in delta.clue_transitions.items() if out.get(cid) != st]
    return out, notes


def flashback_findings(delta: StateDelta, frames, graph) -> list[dict]:
    """Forward-Reachability Audit (§3.6.4), CHỈ trên mục thuộc cảnh hồi ức.

    §11 chạy `flashback_admissible(delta, frame)` với TOÀN BỘ delta cho mỗi
    frame hồi ức. Nghĩa là "Kaelen sờ vết sẹo mới" ở cảnh HIỆN TẠI của Chương 2
    bị kiểm như thể xảy ra ở tick 2.480 — và dựng blocker nhân quả giả.

    Quan hệ bị thu hồi không mang chỉ số cảnh nên không quy được về cảnh hồi
    ức; chúng được ghi chú thay vì bị chặn.
    """
    frames = _as_frames(frames)
    prefix = f"CH{delta.chapter:03d}_"
    ghi = {k for k, v in delta.classification.items()
           if v in ("enrichment", "improvement")}
    out: list[dict] = []
    co_hoi_uc = False
    for f in frames:
        if not f.scene_id.startswith(prefix) or f.time.mode not in ("flashback", "vision"):
            continue
        co_hoi_uc = True
        si = _scene_index(f.scene_id)
        sub = StateDelta(assertions=[
            a for a in delta.assertions
            if a.scene == si and is_objective(a) and item_key(a) in ghi]).stamp(delta.chapter)
        for x in flashback_admissible(sub, f, graph):
            out.append({**x, "scene_id": f.scene_id})
    if co_hoi_uc and delta.retracted_relations:
        out.append({"severity": "note", "check": "flashback_retraction_unattributed",
                    "message": ("chương có cảnh hồi ức và có quan hệ bị thu hồi; "
                                "không xác định được việc thu hồi thuộc cảnh nào")})
    return out


def reconcile(delta: StateDelta, eng, frames=None, contracts=None) -> dict:
    """Phân loại → kiểm hồi ức → chặn mâu thuẫn → PlanPatch → ghi → đánh dấu.

    Chặn là chặn CẢ delta: không ghi một phần. Ghi một phần một chương có mâu
    thuẫn tạo ra canon mà không phiên bản văn xuôi nào khớp.
    Delta vẫn được LƯU kèm quyết định khi bị chặn, để CP-2 thấy vì sao.
    """
    result = {"status": "", "escalation_reason": "", "classify": {},
              "flashback": [], "plan_patches": [], "applied": {}}
    if delta.committed:
        result["status"] = "already_committed"
        return result

    frames = _as_frames(frames if frames is not None
                        else eng.store.get_frames(delta.chapter))
    out = classify_delta(delta, eng.graph, eng.planner, frames,
                         vocabulary=getattr(eng, "vocabulary", None))
    delta.retraction_classification = out["retractions"]
    delta.quarantine = out["quarantine"]
    result["classify"] = out

    for c in contracts or []:
        for d in c.get("plant_directives", []):
            if d.get("clue_id"):
                delta.plant_intensity[d["clue_id"]] = float(
                    d.get("intensity", DEFAULT_PLANT_INTENSITY))
                if d.get("mode"):
                    delta.plant_modes[d["clue_id"]] = d["mode"]

    derived, overridden = derive_clue_transitions(delta, eng.graph.clues)
    delta.clue_transitions = derived
    result["clue_transitions_overridden"] = overridden

    fb = flashback_findings(delta, frames, eng.graph)
    result["flashback"] = fb
    blockers = [x for x in fb if x["severity"] == "blocker"]
    if blockers:
        eng.store.append_delta(delta)
        result["status"] = "escalated"
        result["escalation_reason"] = ("nghịch lý nhân quả hồi ức: "
                                       + "; ".join(x["message"] for x in blockers))
        return result

    if out["summary"]["contradictions"]:
        keys = ([k for k, v in out["classification"].items() if v == "contradiction"]
                + [k for k, v in out["retractions"].items() if v == "contradiction"])
        eng.store.append_delta(delta)
        result["status"] = "escalated"
        result["escalation_reason"] = f"mâu thuẫn canon: {keys}"
        return result

    # NT-14: `improvement` vẫn được GHI. Văn xuôi đã viết ra là sự thật; từ chối
    # ghi chỉ tạo ra phân ly canon–văn bản. PlanPatch đưa quyết định cho tác giả.
    improvements = [k for k, v in out["classification"].items() if v == "improvement"]
    if improvements:
        patch = replan_downstream(delta.chapter,
                                  [(k, delta.item(k)) for k in improvements],
                                  eng.planner)
        if patch is not None:
            eng.store.put_plan_patch(delta.chapter, patch.model_dump())
            result["plan_patches"] = [patch.model_dump()]

    result["applied"] = apply_delta(delta, eng.graph, eng.chars, frames)
    # Tin tức: dựng lại tri thức tới cuối chương này, trên vị trí THẬT của nhân vật.
    result["news"] = settle_news(eng.graph, eng.chars, eng.store.get_frames(),
                                 upto_tick=chapter_end_tick(frames))
    delta.committed = True
    eng.store.append_delta(delta)
    result["status"] = "committed"
    return result


def replay_committed(graph, chars: dict, store) -> int:
    """Fold mọi delta đã ghi vào graph vừa nạp từ bible. Trả số delta đã áp."""
    deltas = store.get_deltas(committed_only=True)
    deltas.sort(key=lambda d: (store.tick_of_chapter(d.chapter), d.chapter))
    all_frames = store.get_frames()
    for d in deltas:
        own = store.get_frames(d.chapter)
        apply_delta(d, graph, chars, own)
        settle_news(graph, chars, all_frames, upto_tick=chapter_end_tick(own))
    return len(deltas)
