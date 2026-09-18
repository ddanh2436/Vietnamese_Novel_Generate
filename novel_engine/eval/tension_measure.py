"""Đo độ căng THỰC của một chương — mảnh còn thiếu của vòng phản hồi (§8.1, §13).

`director_node` gọi `store.measured_tension(ch - 1)` để lái đường cong căng thẳng,
và `M8_tension_mae` (§13) so số đo đó với đường cong mục tiêu. Nhưng KHÔNG CHỖ NÀO
trong tài liệu ghi số đo ấy vào store: `put_chapter_summary` không có người gọi,
nên `measured_tension` trả 0.5 mãi mãi. Hai hệ quả im lặng:

- Vòng phản hồi căng thẳng HỞ: Director luôn tin chương trước ở mức trung tính.
- M8 đo một hằng số, nên nó luôn "đạt" hoặc luôn "trượt" tuỳ đường cong.

Cùng lúc đó, bảng `chapter_summaries` rỗng làm tầng L2 của bộ nhớ (§4.1 — "các
chương gần đây") không bao giờ có nội dung. Cả hai đóng lại ở đây, bằng code.

ĐÂY LÀ PROXY, KHÔNG PHẢI CHÂN LÝ. Độ căng cảm nhận được không đo trực tiếp được;
ba tín hiệu dưới đây đếm được, tất định, và không cần LLM:

1. `change_rate` — mật độ thay đổi trạng thái mỗi cảnh (mệnh đề, thực thể, sự
   kiện quan hệ, chuyển trạng thái manh mối). Cảnh không đổi gì là cảnh phẳng.
2. `staccato`   — tỉ lệ câu trần thuật rất ngắn (§10.3.1). Văn dồn thì câu ngắn.
3. `open_loops` — số chỉ mục còn treo cuối chương (`SceneClose.unresolved`).

Trọng số cố ý thô: một proxy giả vờ tinh vi là một proxy dễ bị tin nhầm.
"""
from __future__ import annotations

import re

from novel_engine.audit.rhythm import THRESH, narrative_sentences, word_count

W_CHANGE, W_STACCATO, W_LOOPS = 0.45, 0.25, 0.30
CHANGE_PER_SCENE_FULL = 3.0      # ≥3 thay đổi/cảnh coi là kịch tính tối đa
STACCATO_FULL = 0.25             # ≥25% câu ngắn coi là nhịp dồn tối đa
LOOPS_FULL = 6.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def measure_tension(scenes: list[dict], delta, unresolved) -> float:
    if not scenes:
        return 0.5
    changes = (len(delta.assertions) + len(delta.new_entities)
               + len(delta.new_relations) + len(delta.relationship_events)
               + len(delta.clue_transitions))
    change = _clamp01(changes / len(scenes) / CHANGE_PER_SCENE_FULL)

    prose = "\n\n".join(s.get("prose", "") for s in scenes)
    sents = narrative_sentences(prose)
    short_max = THRESH["vi"]["short_max"]
    staccato = _clamp01(
        (sum(1 for s in sents if word_count(s) <= short_max) / len(sents))
        / STACCATO_FULL) if sents else 0.0

    loops = _clamp01(len(list(unresolved or [])) / LOOPS_FULL)
    return round(W_CHANGE * change + W_STACCATO * staccato + W_LOOPS * loops, 3)


def chapter_summary(scenes: list[dict], max_chars: int = 700) -> str:
    """Tóm tắt L2 (§4.1) — câu đầu của mỗi scene digest, ghép lại.

    Không gọi LLM: digest đã là bản tóm tắt do model viết ở ranh giới cảnh; tóm
    tắt lại bằng một lượt gọi nữa là trả tiền hai lần cho cùng một việc.
    """
    cau = []
    for s in scenes:
        d = (s.get("digest") or "").strip()
        if d:
            cau.append(re.split(r"(?<=[.!?…])\s+", d)[0].strip())
    out = " ".join(cau)
    return out if len(out) <= max_chars else out[:max_chars - 1].rstrip() + "…"
