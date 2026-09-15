"""Bounded Critique Loop và định tuyến (§9.3).

NT-9: router là hàm ĐIỀU KIỆN THUẦN. Nó đọc state và trả về tên nhánh — nó
không được sửa gì. Mọi biến đếm cập nhật trong node.
"""
from __future__ import annotations

from novel_engine.graph.state import ChapterState

MAX_REVISIONS = 2


def after_audit(state: ChapterState) -> str:
    """Ba quy tắc:

    1. BLOCKER được viết lại tối đa 2 lần rồi escalate. Nếu model không sửa
       được sau 2 lần, lần thứ 3 cũng sẽ không sửa được — lặp thêm chỉ đốt tiền.
    2. MAJOR chỉ được viết lại 1 lần. Lệch tính cách nhẹ tốt hơn một đoạn văn
       đã bị mài mòn qua bốn vòng revision — văn qua nhiều vòng thường trở nên
       an toàn và nhạt.
    3. MINOR không bao giờ gây viết lại. Chúng đi thẳng vào Polish.
    """
    sev = state.get("max_severity", "note")
    n = state.get("revision_count", 0)

    if sev == "blocker":
        return "escalate" if n >= MAX_REVISIONS else "revise"
    if sev == "major":
        return "polish" if n >= 1 else "revise"
    return "polish"


def after_scene_boundary(state: ChapterState) -> str:
    """`scene_index` ĐÃ được tăng trong node, nên so sánh không cộng thêm 1."""
    if state.get("escalated"):
        return "escalate"
    if state["scene_index"] < len(state["contracts"]):
        return "next_scene"
    return "done"


def guard(state: ChapterState) -> str:
    """Cạnh kiểm cờ `escalated` sau mỗi node (§9.5). `safe_node` biến ngoại lệ
    thành cờ thay vì stack trace; không có cạnh này thì cờ được dựng lên rồi
    đồ thị vẫn chạy tiếp với state dở dang."""
    return "escalate" if state.get("escalated") else "ok"
