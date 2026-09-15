"""Interface của tầng LLM.

Toàn bộ `graph/nodes.py` chỉ nói chuyện qua Protocol này. Đổi nhà cung cấp —
Gemini free hôm nay, Anthropic cho bản chốt, model local về sau — là đổi một
dòng config, không đụng một chữ nào trong node.

NT-1: LLM chỉ DIỄN ĐẠT, không QUYẾT ĐỊNH. Mọi logic (cấp phát thời gian, chọn
manh mối, guard quan hệ, chấm điểm nhịp văn) là code thuần. Vì vậy `FakeLLM`
chạy được cả pipeline: những gì nó trả về là văn xuôi giả, nhưng mọi quyết
định của hệ thống vẫn là quyết định thật.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    def invoke(self, prompt: str, *, role: str = "") -> str:
        """Gọi model, trả về CHUỖI thô.

        `role` cho adapter biết đây là lượt gọi loại gì ("writer", "director",
        "scene_digest", "auditor", "extractor") — dùng để chọn nhiệt độ, chọn
        model rẻ hơn cho việc có cấu trúc (§14.2 đòn 4), và để `FakeLLM` biết
        phải trả về hình dạng gì.
        """
        ...
