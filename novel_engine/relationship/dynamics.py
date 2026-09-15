"""Cập nhật chỉ số và cơ chế thoái lui (§7.3).

LLM KHÔNG sinh con số. Extractor khai SỰ KIỆN (`RelationshipEvent`: loại, ai,
câu trích nguyên văn); bảng `EFFECTS` dưới đây là nơi DUY NHẤT biến sự kiện
thành điểm (NT-1, NT-11). `apply_scene_effects` của §7.3 nhận `d_friction`,
`d_stake` thô từ Extractor — model khai `d_stake: 60` là cặp nhân vật nhảy từ
bộc lộ tổn thương sang thử thách trong một chương, guard vẫn "đúng".

═══ BỐN CHỖ KHÁC MÃ §7.3 ═══════════════════════════════════════════════════

1. CATHARSIS CHỈ TỚI ĐƯỢC QUA PHẢN BỘI. Guard TRIAL→CATHARSIS đòi `scars`, và
   trong §7.3 sẹo CHỈ sinh từ `betrayal` — thứ cũng đẩy cặp vào RUPTURE. Một hy
   sinh thật không để lại sẹo, trái với chính `UNBLOCK_HINT[TRIAL]` ("cần một hy
   sinh có thật, để lại hậu quả không đảo ngược"). Mọi quan hệ muốn đi hết vòng
   đều phải phản bội nhau trước. Ở đây `sacrifice` để lại sẹo mà không đổ vỡ.

2. CHƯƠNG PHẢN BỘI ĐƯỢC ĐẾM LÀ MỘT CHƯƠNG ĐỔ VỠ. `betrayal` đặt
   `chapters_in_stage = 0` rồi cuối hàm cộng 1 vì `last_counted_chapter` chưa
   đổi. Trong khi `apply_relationship_advancement` đặt `last_counted_chapter`
   khi tiến giai đoạn — hai lối vào giai đoạn, hai quy tắc đếm. Ở đây mọi lối
   vào giai đoạn đi qua `_enter`.

3. SEVERED KHÔNG BAO GIỜ XẢY RA NẾU HAI NGƯỜI KHÔNG GẶP NHAU. `chapters_in_stage`
   chỉ tăng khi có sự kiện, nên một cặp đổ vỡ rồi tránh mặt nhau mãi mãi ở
   RUPTURE. Tiến lên cần TƯƠNG TÁC (đếm chương có tương tác); chấm dứt thì cần
   THỜI GIAN (đếm chương trôi qua từ lúc đổ vỡ). Hai đồng hồ, hai ý nghĩa.

4. `intimacy_asym += d_asym` không chặn biên; pydantic không validate khi gán,
   nên giá trị vượt ±50 lọt vào canon không một lỗi.

Thêm: tối đa MỘT lần chuyển giai đoạn mỗi chương. Guard kiểm từng bước riêng lẻ
thì một chương đủ sự kiện đi thẳng STRANGERS → FRICTION → … trong một lần chốt.
"""
from __future__ import annotations

from novel_engine.canon.models import RelationshipEvent, RelationStage
from novel_engine.relationship.machine import can_advance, next_stage
from novel_engine.relationship.models import RelationshipState

R = RelationStage

# Gắn kết tăng chậm — và chỉ tăng thật qua HÀNH ĐỘNG có giá, không qua lời nói.
EFFECTS: dict[str, dict] = {
    "acted_against_own_interest_for_other": {"intimacy": 9, "asym": 3},
    "shared_ordeal":         {"intimacy": 4, "ordeal": True},
    "verbal_affection_only": {"intimacy": 1},
    "sacrifice":             {"intimacy": 12, "asym": 5, "scar": True},
    "betrayal":              {"intimacy": -28, "scar": True, "rupture": True},
    "value_clash":           {"friction": 8},
    "reconciled_method":     {"friction": -8},     # hoà giải được bằng đối thoại
    "interests_collide":     {"stake": 12},        # chỉ đổi khi hoàn cảnh đổi
    "interests_align":       {"stake": -12},
}
MAX_FRICTION_PER_CHAPTER = 16
MAX_STAKE_PER_CHAPTER = 20
SCAR_CAP_STEP = 7
SEVER_AFTER_ELAPSED = 6


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _enter(st: RelationshipState, stage: RelationStage, chapter: int, reason: str) -> dict:
    """Lối vào giai đoạn DUY NHẤT."""
    t = {"pair": st.pair_key, "chapter": chapter, "from": st.stage.value,
         "to": stage.value, "reason": reason}
    st.stage = stage
    st.stage_entered_chapter = chapter
    st.chapters_in_stage = 0
    st.last_counted_chapter = chapter     # chương chuyển KHÔNG tính cho giai đoạn mới
    st.history.append({"stage": stage.value, "chapter": chapter,
                       "intimacy": round(st.intimacy, 1), "scars": len(st.scars),
                       "reason": reason})
    return t


def apply_event(st: RelationshipState, ev: RelationshipEvent, chapter: int,
                acc: dict) -> dict | None:
    """Áp MỘT sự kiện. Không đếm chương, không xét guard. Trả transition nếu có."""
    eff = EFFECTS[ev.kind]
    d_int = eff.get("intimacy", 0)
    if st.stage == R.SEVERED and d_int > 0:
        d_int = 0                          # đã chấm dứt: cử chỉ không hàn gắn được
    st.intimacy = clamp(st.intimacy + d_int)
    acc["friction"] += eff.get("friction", 0)
    acc["stake"] += eff.get("stake", 0)
    if eff.get("asym"):
        sign = 1 if ev.actor == st.a else (-1 if ev.actor == st.b else 0)
        st.intimacy_asym = clamp(st.intimacy_asym + sign * eff["asym"], -50.0, 50.0)
    note = f"ch{chapter}: {ev.span[:80]}"
    if eff.get("ordeal") and note not in st.shared_ordeals:
        st.shared_ordeals.append(note)
    if eff.get("scar"):
        st.scars.append(f"ch{chapter} {ev.kind}: {ev.span[:80]}")
    if st.scars:
        # SẸO KHÔNG LÀNH: mỗi vết sẹo đặt trần vĩnh viễn cho intimacy.
        st.intimacy = min(st.intimacy, 100 - SCAR_CAP_STEP * len(st.scars))
    if eff.get("rupture") and st.stage not in (R.RUPTURE, R.SEVERED):
        return _enter(st, R.RUPTURE, chapter, f"phản bội: {ev.span[:80]}")
    return None


def settle_chapter(st: RelationshipState, events: list[RelationshipEvent], chapter: int,
                   *, interacted: bool) -> list[dict]:
    """PHÁN QUYẾT cuối chương cho một cặp. Idempotent theo chương."""
    if st.last_settled_chapter is not None and st.last_settled_chapter >= chapter:
        return []
    transitions: list[dict] = []
    acc = {"friction": 0.0, "stake": 0.0}
    for ev in events:
        t = apply_event(st, ev, chapter, acc)
        if t:
            transitions.append(t)
    st.friction = clamp(st.friction + clamp(acc["friction"], -MAX_FRICTION_PER_CHAPTER,
                                            MAX_FRICTION_PER_CHAPTER))
    st.stake_conflict = clamp(st.stake_conflict + clamp(acc["stake"], -MAX_STAKE_PER_CHAPTER,
                                                        MAX_STAKE_PER_CHAPTER))

    active = interacted or bool(events)
    # Đếm MỘT lần mỗi chương (C7), độc lập với số cảnh và số sự kiện.
    if active and st.last_counted_chapter != chapter:
        st.chapters_in_stage += 1
        st.last_counted_chapter = chapter

    if not transitions:                    # tối đa một lần chuyển mỗi chương
        ok, reason = can_advance(st, chapter) if active else (False, "")
        nxt = next_stage(st)
        if ok and nxt is not None:
            transitions.append(_enter(st, nxt, chapter, reason))
        elif (st.stage == R.RUPTURE
              and chapter - st.stage_entered_chapter >= SEVER_AFTER_ELAPSED):
            transitions.append(_enter(st, R.SEVERED, chapter,
                                      f"{chapter - st.stage_entered_chapter} chương "
                                      f"đổ vỡ không hàn gắn"))
    st.last_settled_chapter = chapter
    return transitions
