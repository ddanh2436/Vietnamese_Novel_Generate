"""PlanPatch — khi một chi tiết mới đụng tới kế hoạch các chương sau (§11).

Nguyên tắc của §11: hệ thống KHÔNG BAO GIỜ tự ý xoá một chi tiết hay, và cũng
KHÔNG BAO GIỜ tự ý sửa kế hoạch đã chốt. Khi hai điều đó va nhau, con người
quyết định (CP-4).

Vì vậy hàm này chỉ LẬP HỒ SƠ. Nó không sửa `bible/outline.yaml` — dàn ý là đầu
vào của tác giả, và agent không được ghi vào `bible/`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from novel_engine.canon.models import Assertion, Entity, Relation
from novel_engine.reconcile.classify import downstream_chapters

OPTIONS = ["giữ chi tiết mới và sửa dàn ý chương sau",
           "gỡ chi tiết mới khỏi canon (viết lại chương nguồn)"]


class PlanPatch(BaseModel):
    patch_id: str
    source_chapter: int
    from_chapter: int
    invalidated_beats: list[str] = Field(default_factory=list)
    invalidated_clues: list[str] = Field(default_factory=list)
    suggested_rewrites: list[dict] = Field(default_factory=list)
    author_decision_required: bool = True


def _reason(item, n: int) -> str:
    if isinstance(item, Entity):
        return f"dàn ý chương {n} nhắc tới '{item.name}'"
    if isinstance(item, Relation):
        return (f"quan hệ mới {item.src} -[{item.type}]-> {item.dst}, cả hai "
                f"cùng xuất hiện ở chương {n}")
    if isinstance(item, Assertion):
        return (f"sự thật vĩnh viễn {item.subject}.{item.predicate} về một nhân "
                f"vật còn xuất hiện ở chương {n}")
    return f"đụng tới chương {n}"


def replan_downstream(source_chapter: int, items: list[tuple[str, object]],
                      planner) -> PlanPatch | None:
    """Lập PlanPatch cho các mục `improvement`. Trả `None` khi không chương sau
    nào bị đụng tới."""
    beats: set[str] = set()
    clues: set[str] = set()
    rewrites: list[dict] = []
    for key, item in items:
        for n in downstream_chapters(item, planner, source_chapter):
            beats.add(f"CH{n:03d}")
            rewrites.append({"chapter": n, "item": key,
                             "reason": _reason(item, n), "options": OPTIONS})
            cid = getattr(item, "id", "") or ""
            if cid.startswith("CLUE_"):
                clues.add(cid)
    if not rewrites:
        return None
    return PlanPatch(
        patch_id=f"patch_ch{source_chapter:03d}",
        source_chapter=source_chapter, from_chapter=source_chapter + 1,
        invalidated_beats=sorted(beats), invalidated_clues=sorted(clues),
        suggested_rewrites=rewrites)
