"""Interface của tầng lưu trữ tuần tự (§3.1, §4.2, §8.2.1, §9.2, §11).

`store` được gọi rải rác khắp kiến trúc v2.5 — `store.last_epoch_tick`,
`store.recent_scene_digests`, `store.measured_tension`, `store.put_frame` —
nhưng chưa bao giờ được định nghĩa ở một chỗ. Protocol này quy chúng về một
hợp đồng duy nhất.

PHÂN VAI với `GraphPort`, vì hai thứ này rất dễ lẫn:

    GraphPort  →  world state HIỆN TẠI, truy vấn theo quan hệ (materialized view)
    StorePort  →  lịch sử TUẦN TỰ: delta log, digest, frame, con trỏ thời gian

Nói cách khác: graph trả lời "ai đang ở đâu", store trả lời "chuyện gì đã được
viết ra, theo thứ tự nào". Fold delta log (store) sinh ra graph — không phải
ngược lại (§3.1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from novel_engine.canon.models import StateDelta
    from novel_engine.canon.timeline import ContinuityFrame


@runtime_checkable
class StorePort(Protocol):
    # ── 1. Quản lý con trỏ thời gian (§8.2.1) ──
    def last_epoch_tick(self, chapter: int) -> int: ...
    def last_narrative_order(self, chapter: int) -> int: ...
    def epoch_tick_of(self, scene_id: str) -> int: ...
    def tick_of_chapter(self, chapter: int) -> int: ...

    # ── 2. Memory Hierarchy L1, L2, L3 (§4.1, §4.2) ──
    def put_scene_digest(self, chapter: int, scene_idx: int,
                         digest: str, scene_id: str = "") -> None: ...
    def recent_scene_digests(self, chapter: int, scene_idx: int,
                             k: int = 2) -> list[str]: ...
    def put_chapter_summary(self, chapter: int, summary: str,
                            measured_tension: float = 0.5) -> None: ...
    def chapter_summaries(self, from_ch: int, to_ch: int) -> list[str]: ...
    def arc_summaries(self, before_chapter: int) -> list[str]: ...
    def measured_tension(self, chapter: int) -> float: ...

    # ── 3. Trạng thái vật lý & Event Sourcing (§9.2, §10.1, §11) ──
    def put_frame(self, frame: "ContinuityFrame") -> None: ...
    def get_frames(self, chapter: int | None = None) -> list["ContinuityFrame"]: ...
    def append_delta(self, delta: "StateDelta") -> None: ...
    def get_deltas(self, chapter: int | None = None, *,
                   committed_only: bool = False) -> list["StateDelta"]: ...

    # ── 4. PlanPatch — quyết định CP-4 của tác giả (§11) ──
    def put_plan_patch(self, chapter: int, patch: dict) -> None: ...
    def get_plan_patches(self, chapter: int | None = None) -> list[dict]: ...
