"""Checkpoint LangGraph (§9.4) — dừng giữa chương rồi chạy tiếp.

Một chương mất 2–3 phút và khoảng 30 lượt gọi LLM. Mất mạng ở cảnh 5 mà phải
viết lại từ cảnh 0 là trả tiền hai lần cho bốn cảnh đã đúng.

═══ CẠM BẪY: THREAD CŨ CÒN SỐNG ═══════════════════════════════════════════

`interrupt_before` và `SqliteSaver` ở §9.4 chỉ nói cách BẬT checkpoint, không
nói chuyện gì xảy ra khi chạy lại. Đo trên chính LangGraph 1.2:

- `invoke(state_ban_dau, cfg)` trên một thread ĐÃ CÓ checkpoint = chạy lại từ
  START **cộng dồn lên state cũ**. Các kênh reducer (`scene_outputs`, `frames`)
  giữ nguyên phần cũ rồi nối thêm lượt mới: chương ra 8 hay 12 cảnh, `frames`
  trùng lặp, và KHÔNG một lỗi nào được ném.
- `invoke(None, cfg)` = đi tiếp từ chỗ dừng. Đây là cách DUY NHẤT để resume,
  và là lý do `run_chapter` có tham số `resume`.

Vì vậy trước mỗi lượt viết mới, thread cũ phải bị XOÁ tường minh; chỉ `--resume`
mới đi tiếp thread đang dở.
"""
from __future__ import annotations

from contextlib import contextmanager

CHECKPOINT_DB = "novel_checkpoints.db"


def chapter_thread(chapter: int) -> str:
    return f"ch{chapter:03d}"


@contextmanager
def open_checkpointer(path: str = CHECKPOINT_DB):
    """`SqliteSaver.from_conn_string` là context manager — giữ kết nối mở suốt
    lượt chạy, đóng sau đó."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(path) as cp:
        yield cp


def thread_status(checkpointer, chapter: int, *, auto_commit: bool = False) -> dict:
    """`{exists, done, next, scene_index}` của thread một chương."""
    from novel_engine.graph.build import build_chapter_graph
    g = build_chapter_graph(checkpointer, auto_commit=auto_commit)
    snap = g.get_state({"configurable": {"thread_id": chapter_thread(chapter)}})
    exists = snap.created_at is not None
    return {"exists": exists, "done": exists and not snap.next,
            "next": list(snap.next),
            "scene_index": (snap.values or {}).get("scene_index"),
            "scenes_done": len((snap.values or {}).get("scene_outputs", []))}


def drop_thread(checkpointer, chapter: int) -> None:
    checkpointer.delete_thread(chapter_thread(chapter))
