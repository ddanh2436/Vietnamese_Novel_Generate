"""Đừng để một ngoại lệ giết cả chương (§9.5, lớp 4).

Trong một lần chạy 40 chương, SẼ có node ném lỗi vì lý do nào đó. Lỗi phải
biến thành escalation, không thành stack trace. Với checkpointer đã bật
(§9.4), một node hỏng chỉ mất công đoạn đó chứ không mất cả chương — sửa rồi
chạy tiếp từ checkpoint.
"""
from __future__ import annotations

import functools
import traceback

from langchain_core.runnables import RunnableConfig


def safe_node(fn):
    @functools.wraps(fn)
    def wrapper(state, config: RunnableConfig):
        try:
            return fn(state, config)
        except Exception as e:      # noqa: BLE001 — đây đúng là chỗ bắt hết
            return {
                "escalated": True,
                "escalation_reason": f"{fn.__name__} lỗi: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:],
            }
    return wrapper
