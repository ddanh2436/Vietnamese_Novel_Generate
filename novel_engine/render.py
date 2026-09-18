"""Định dạng văn xuôi đã viết thành markdown chương.

Tách khỏi `api.py`: đây là một lựa chọn TRÌNH BÀY (định dạng `## Cảnh N` mà
`api.doc_chuong` biết đọc lại), không phải một phần hợp đồng dữ liệu của giao
diện. Cả `cli.py` lẫn `novel_engine/web.py` dùng chung một hàm này để hai nơi
không lặng lẽ trôi ra hai định dạng khác nhau.
"""
from __future__ import annotations


def render_chapter_markdown(eng, chapter: int, scene_outputs: list[dict]) -> str:
    parts = [f"# Chương {chapter} — {eng.planner.title(chapter)}", ""]
    for i, s in enumerate(scene_outputs):
        parts += [f"## Cảnh {i}", "", s["prose"].strip(), ""]
    return "\n".join(parts)
