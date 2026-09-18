"""State của LangGraph (§9.1).

NT-9: router của LangGraph là hàm điều kiện THUẦN — nó KHÔNG đổi được state.
Mọi biến điều khiển vòng lặp (`scene_index`, `revision_count`) chỉ được cập
nhật trong một NODE. Ở đây node đó là `scene_boundary_node`, và nó là nơi DUY
NHẤT được đụng vào chúng. Thiếu quy tắc này là C1: đồ thị viết lại Cảnh 0 vô
hạn cho tới khi cạn quota.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ChapterState(TypedDict, total=False):
    # ── đầu vào ──
    chapter: int
    total_chapters: int
    outline_beat: str

    # ── do Director sinh ──
    contracts: list[dict]          # SceneContract cho từng cảnh
    beats: list[dict]

    # ── do Writer sinh ──
    scene_index: int
    current_draft: str
    # Một mục MỖI CẢNH: {scene_id, prose, digest}. Thay cho `drafts` là
    # list[str] trần — Extractor cần biết span thuộc cảnh nào (C2).
    scene_outputs: Annotated[list[dict], operator.add]
    frames: Annotated[list[dict], operator.add]      # ContinuityFrame (§10.1)
    # Ghi nhận mỗi lần thời lượng THỰC lệch dự kiến và kế hoạch phải nhường
    # (NT-14). Tín hiệu cho Director chương sau: beat nào đang bị ước lượng sai.
    time_drift: Annotated[list[str], operator.add]

    # ── do Auditor / Polish sinh (§9.3) ──
    findings: list[dict]           # của bản nháp HIỆN TẠI; reset ở ranh giới cảnh
    max_severity: str              # mức ĐỊNH TUYẾN — đã loại chỉ số nhịp (§10.3.2)
    revision_count: int
    # Một mục MỖI VÒNG kiểm toán, cộng dồn cả chương: dữ liệu cho eval (M-*)
    # và cho câu hỏi "Writer có sửa được sau phản hồi không".
    audit_log: Annotated[list[dict], operator.add]
    polish_report: dict            # nhận/từ chối bản Polish và lý do
    # Ký tự rác code đã dọn ở Writer — ghi ra để tác giả thấy (NT-13).
    hygiene_notes: Annotated[list[dict], operator.add]
    # Bản nháp bị chặn sau MAX_REVISIONS, kèm lỗi — thứ tác giả cần đọc.
    escalated_scene: dict
    # LangGraph BỎ QUA mọi khoá không khai trong TypedDict — node trả về khoá lạ
    # thì nó biến mất không báo lỗi. Cờ này cho CLI biết store đã được dọn.
    rolled_back: bool

    # ── kết quả ──
    polished: str                  # CHỈ cảnh hiện tại — đừng dùng cho cả chương
    delta: dict
    extraction_report: dict
    clue_escalations: list[dict]   # F5: quyết định CP-4, không chặn sản xuất
    # Lịch manh mối của chương: chỉ thị, manh mối không xếp được, bị chặn, và
    # báo cáo nợ (§6.4). Khai ở đây vì LangGraph bỏ khoá không khai.
    foreshadow_report: dict
    unresolved: Annotated[list[str], operator.add]   # chỉ mục treo, từ SceneClose
    irony_seeds: list[dict]
    reconcile_report: dict         # chỉ có khi đồ thị chạy với auto_commit
    escalated: bool
    escalation_reason: str
    traceback: str
