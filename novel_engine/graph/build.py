"""Lắp đồ thị (§9.4).

    START → director → writer → auditor ─┬→ writer   (revise: blocker ≤2, major ≤1)
                                         ├→ polish → scene_boundary ─┬→ writer (cảnh tiếp)
                                         │                           └→ extract ─→ END
                                         └→ escalate → END              (│ auto_commit:
                                                                         └→ reconcile → END)

Mọi node có cạnh `guard` tới `escalate`: `safe_node` biến ngoại lệ thành cờ,
và thiếu cạnh kiểm thì đồ thị chạy tiếp với state dở dang.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from novel_engine.graph.nodes import (
    auditor_node, director_node, extractor_node, polish_node, reconcile_node,
    scene_boundary_node, writer_node,
)
from novel_engine.graph.routing import (
    MAX_REVISIONS, after_audit, after_scene_boundary, guard,
)
from novel_engine.graph.state import ChapterState


def escalate_node(state: ChapterState, config: RunnableConfig) -> dict:
    """Điểm dừng CÓ KIỂM SOÁT. NT-17: escalation không đồng nghĩa với hỏng — nó
    nghĩa là cần một quyết định của con người. "Thoát an toàn" gồm hai việc:

    1. GIỮ bằng chứng. Khi dừng vì blocker còn nguyên sau MAX_REVISIONS, bản
       nháp bị chặn và danh sách lỗi đi ra ngoài trong `escalated_scene` — không
       có nó, tác giả chỉ thấy "escalate" mà không biết cảnh nào, vì sao.
    2. KHÔNG để lại canon dở dang. `scene_boundary_node` đã ghi digest và frame
       của các cảnh trước vào store. Để lại chúng thì `last_epoch_tick` của
       chương sau đọc mốc thời gian của nửa chương bị bỏ, và luật liên tục chạy
       trên những cảnh không bao giờ được xuất bản. Văn xuôi của các cảnh đó
       vẫn nằm trong `scene_outputs` để CLI lưu ra file.

       Ngoại lệ: đã có `delta` (escalate ở extract/reconcile) thì GIỮ — frame là
       thứ `review` cần để phân loại delta đang chờ tác giả duyệt.
    """
    out: dict = {"escalated": True}
    reason = state.get("escalation_reason")
    idx = state.get("scene_index", 0)
    contracts = state.get("contracts") or []

    if not reason and state.get("max_severity") == "blocker" and idx < len(contracts):
        c = contracts[idx]
        blockers = [f for f in state.get("findings", []) if f.get("severity") == "blocker"]
        n = state.get("revision_count", 0)
        reason = (f"{c['scene_id']}: còn {len(blockers)} blocker sau {n} lần viết lại "
                  f"(tối đa {MAX_REVISIONS}) — "
                  + "; ".join(f"[{f.get('check')}] {f.get('message', '')[:120]}"
                              for f in blockers[:3]))
        out["escalated_scene"] = {
            "scene_id": c["scene_id"], "scene_index": idx, "revisions": n,
            "prose": state.get("current_draft", ""),
            "findings": state.get("findings", []),
        }
    out["escalation_reason"] = reason or "không rõ nguyên nhân"

    eng = (config or {}).get("configurable", {}).get("engines")
    if eng is not None and not state.get("delta") and state.get("chapter"):
        eng.store.discard_scene_records(state["chapter"])
        out["rolled_back"] = True
    return out


def build_chapter_graph(checkpointer=None, *, auto_commit: bool = False):
    """`auto_commit=False` (mặc định): đồ thị dừng sau `extract`, delta chờ tác
    giả duyệt ở CP-2. `auto_commit=True`: ghi canon ngay — dùng cho test và cho
    lượt chạy không người giám sát."""
    g = StateGraph(ChapterState)

    g.add_node("director", director_node)
    g.add_node("writer", writer_node)
    g.add_node("auditor", auditor_node)
    g.add_node("polish", polish_node)
    g.add_node("scene_boundary", scene_boundary_node)
    g.add_node("extract", extractor_node)
    g.add_node("escalate", escalate_node)

    g.add_edge(START, "director")
    g.add_conditional_edges("director", guard,
                            {"ok": "writer", "escalate": "escalate"})
    g.add_conditional_edges("writer", guard,
                            {"ok": "auditor", "escalate": "escalate"})
    # `after_audit` tự kiểm cờ `escalated` trước — nó thay chỗ `guard` ở đây.
    g.add_conditional_edges("auditor", after_audit,
                            {"revise": "writer", "polish": "polish",
                             "escalate": "escalate"})
    g.add_conditional_edges("polish", guard,
                            {"ok": "scene_boundary", "escalate": "escalate"})
    g.add_conditional_edges(
        "scene_boundary", after_scene_boundary,
        {"next_scene": "writer", "done": "extract", "escalate": "escalate"})
    if auto_commit:
        g.add_node("reconcile", reconcile_node)
        g.add_conditional_edges("extract", guard,
                                {"ok": "reconcile", "escalate": "escalate"})
        g.add_conditional_edges("reconcile", guard,
                                {"ok": END, "escalate": "escalate"})
    else:
        g.add_conditional_edges("extract", guard,
                                {"ok": END, "escalate": "escalate"})
    g.add_edge("escalate", END)

    return g.compile(checkpointer=checkpointer)


def recursion_limit_for(n_scenes: int) -> int:
    """Trần số bước cho TRƯỜNG HỢP XẤU NHẤT, tính từ MAX_REVISIONS.

    Mỗi cảnh tối đa (MAX_REVISIONS+1) × (writer + auditor) + polish +
    scene_boundary. Để LangGraph chạm trần thì lỗi đọc như vòng lặp vô hạn chứ
    không như một giới hạn cấu hình — nên trần phải suy ra, không đoán.
    """
    per_scene = 2 * (MAX_REVISIONS + 1) + 2
    return n_scenes * per_scene + 10       # director, extract, reconcile, escalate


def run_chapter(engines, chapter: int, *, checkpointer=None,
                auto_commit: bool = False,
                thread_id: str | None = None) -> dict:
    """Chạy một chương. Trả về state cuối."""
    graph = build_chapter_graph(checkpointer, auto_commit=auto_commit)
    n_scenes = engines.planner.n_scenes(chapter)
    cfg = {
        "configurable": {"engines": engines,
                         "thread_id": thread_id or f"ch{chapter:03d}"},
        "recursion_limit": recursion_limit_for(n_scenes),
    }
    return graph.invoke({
        "chapter": chapter,
        "total_chapters": engines.total_chapters,
        "outline_beat": engines.planner.outline_beat(chapter),
        "scene_outputs": [], "frames": [], "unresolved": [], "time_drift": [],
        "audit_log": [],
    }, cfg)
