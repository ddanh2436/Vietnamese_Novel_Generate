"""Lắp đồ thị (§9.4).

GĐ1 — ba node:

    START → director → writer → scene_boundary ─┬→ writer   (cảnh tiếp)
                                                ├→ escalate
                                                └→ extract → END

GĐ2 chèn auditor + polish giữa writer và scene_boundary; GĐ3 chèn reconcile
giữa extract và END. Chỗ nối đó đã có sẵn trong `routing.py`.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from novel_engine.graph.nodes import (
    director_node, extractor_node, reconcile_node, scene_boundary_node,
    writer_node,
)
from novel_engine.graph.routing import after_scene_boundary, guard
from novel_engine.graph.state import ChapterState


def escalate_node(state: ChapterState, config: RunnableConfig) -> dict:
    """Điểm dừng có kiểm soát. NT-17: escalation KHÔNG đồng nghĩa với hỏng —
    nó nghĩa là cần một quyết định của con người."""
    return {"escalated": True,
            "escalation_reason": state.get("escalation_reason",
                                           "không rõ nguyên nhân")}


def build_chapter_graph(checkpointer=None, *, auto_commit: bool = False):
    """`auto_commit=False` (mặc định): đồ thị dừng sau `extract`, delta chờ tác
    giả duyệt ở CP-2. `auto_commit=True`: ghi canon ngay — dùng cho test và cho
    lượt chạy không người giám sát."""
    g = StateGraph(ChapterState)

    g.add_node("director", director_node)
    g.add_node("writer", writer_node)
    g.add_node("scene_boundary", scene_boundary_node)
    g.add_node("extract", extractor_node)
    g.add_node("escalate", escalate_node)

    g.add_edge(START, "director")

    # Cạnh kiểm cờ sau mỗi node: `safe_node` biến ngoại lệ thành cờ
    # `escalated`, và không có cạnh này thì cờ dựng lên rồi đồ thị vẫn chạy
    # tiếp với state dở dang — đúng kiểu lỗi mà §9.5 lớp 4 sinh ra để chặn.
    g.add_conditional_edges("director", guard,
                            {"ok": "writer", "escalate": "escalate"})
    g.add_conditional_edges("writer", guard,
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


def run_chapter(engines, chapter: int, *, checkpointer=None,
                auto_commit: bool = False,
                thread_id: str | None = None) -> dict:
    """Chạy một chương. Trả về state cuối.

    `recursion_limit` phải nới: 6 cảnh × 2 node + director + các cạnh điều
    kiện vượt mặc định 25 của LangGraph, và khi chạm trần thì lỗi đọc như một
    vòng lặp vô hạn chứ không như một giới hạn cấu hình.
    """
    graph = build_chapter_graph(checkpointer, auto_commit=auto_commit)
    n_scenes = engines.planner.n_scenes(chapter)
    cfg = {
        "configurable": {"engines": engines,
                         "thread_id": thread_id or f"ch{chapter:03d}"},
        "recursion_limit": max(50, n_scenes * 6 + 20),
    }
    return graph.invoke({
        "chapter": chapter,
        "total_chapters": engines.total_chapters,
        "outline_beat": engines.planner.outline_beat(chapter),
        "scene_outputs": [], "frames": [], "unresolved": [], "time_drift": [],
    }, cfg)
