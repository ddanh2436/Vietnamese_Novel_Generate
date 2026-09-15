"""Cập nhật niềm tin của nhân vật (§11, F6).

CHỈ `believed_by` đi vào đây. `claimed_by` thì KHÔNG.

§11 gọi `eng.chars.update_belief(char_id=item.holder, ...)` cho mọi mệnh đề phi
khách quan, kể cả lời khai. Hệ quả: Serena nói dối "Hạm Đội Số 3 đã giải tán"
thì hệ thống ghi rằng SERENA TIN hạm đội đã giải tán. Từ đó `deliberate()` (§5.2)
cho cô hành động như thể lời nói dối của chính mình là thật — người nói dối bị
biến thành người bị lừa.

Lời khai là một SỰ KIỆN (ai đã nói gì, lúc nào), ghi ở `graph.beliefs` với
`kind="claimed_by"`. Người NGHE có tin hay không là chuyện của News Dispatcher
và `accept_correction` (§5.6.6), không phải của người nói.
"""
from __future__ import annotations

import unicodedata

from novel_engine.character.models import BeliefState, CharacterProfile


def _key(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").lower().split())


def update_belief(profile: CharacterProfile, proposition: str, confidence: float,
                  source: str, is_actually_true: bool | None = None) -> BeliefState:
    """Thêm hoặc cập nhật một niềm tin. Cùng mệnh đề (so theo NFC, không phân
    biệt hoa thường) thì cập nhật tại chỗ — không chồng bản sao mỗi chương."""
    conf = max(0.0, min(1.0, float(confidence)))
    for b in profile.beliefs:
        if _key(b.proposition) == _key(proposition):
            b.confidence = conf
            b.source = source
            if is_actually_true is not None:
                b.is_actually_true = is_actually_true
            return b
    b = BeliefState(proposition=proposition, confidence=conf, source=source,
                    is_actually_true=is_actually_true)
    profile.beliefs.append(b)
    return b
