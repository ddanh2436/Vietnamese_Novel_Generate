"""Interface của tầng truy cập đồ thị (§3.7).

Bọc tầng này sau một Protocol để hoán đổi `NetworkXGraph` ↔ `Neo4jGraph` mà
không đụng tới phần còn lại. Với một tiểu thuyết 60 chương / ~150 thực thể,
SQLite + NetworkX cho kết quả tương đương Neo4j với 0 chi phí vận hành.

BỔ SUNG SO VỚI §3.7 — ba method được GỌI ở chỗ khác nhưng không có trong danh
sách Protocol của tài liệu. Thiếu chúng thì Protocol không mô tả đúng cái mà
hệ thống thật sự cần, và lỗi chỉ lộ ra khi chạy:

- `travel_ticks`    — `_no_teleport_epoch` (§3.6.1) gọi `graph.travel_ticks(a, b)`
- `routes_from`     — `propagate` (§5.6.3) gọi `graph.routes_from(loc)`
- `pov_can_observe` — `director_node` (§9.2) gọi để lọc plant directive
- `entity_index`    — `extractor_node` (§9.2) truyền vào EXTRACT_EMERGENT_TMPL
- `conflicts_with_truth` — `classify_delta` (§11) gọi để gieo irony seed
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from novel_engine.canon.flashback import AttributeWindow
    from novel_engine.canon.models import Assertion, Entity, Relation


@runtime_checkable
class GraphPort(Protocol):
    # ── Ghi ──────────────────────────────────────────────────────────────
    def upsert_entity(self, e: "Entity") -> None: ...
    def upsert_relation(self, r: "Relation") -> None: ...
    def close_relation(self, src: str, dst: str, rel_type: str,
                       until_chapter: int) -> None: ...             # F7
    def commit_belief(self, holder: str, subject: str, predicate: str,
                      object, kind: str, since_tick: int) -> None: ...  # F6

    # ── Đọc: POV firewall ────────────────────────────────────────────────
    def known_by(self, pov_id: str, epoch_tick: int) -> dict: ...   # NT-6

    # ── Đọc: Forward-Reachability Audit của flashback (§3.6.4) ───────────
    # Thiếu ba method này thì `flashback_admissible` không chạy được (F8).
    def exists(self, entity_id: str) -> bool: ...
    def alive_after(self, entity_id: str, tick: int) -> bool: ...
    def attribute_window(self, entity_id: str,
                         attribute: str) -> "AttributeWindow": ...

    # ── Đọc: canon & kế hoạch ────────────────────────────────────────────
    def is_locked(self, src: str, dst: str, rel_type: str) -> bool: ...
    def faction_tensions(self, location_id: str, chapter: int) -> list[dict]: ...
    def due_clues(self, chapter: int, lookahead: int = 3) -> list[dict]: ...
    def record_surface_form(self, clue_id: str, form: str) -> None: ...
    def used_surface_forms(self, clue_id: str) -> list[str]: ...
    def conflicts_with_truth(self, a: "Assertion") -> bool: ...
    def entity_index(self) -> list[dict]: ...
    def pov_can_observe(self, pov_id: str, clue_id: str, chapter: int) -> bool: ...

    # ── Đọc: Unified Route Graph (§3.6.1 + §5.6.1) ───────────────────────
    # Một đồ thị địa lý DUY NHẤT, hai phép chiếu. Xem `networkx_graph` để
    # biết vì sao không tách làm hai bảng khoảng cách.
    def travel_ticks(self, a: str, b: str, *, mode: str = "foot",
                     hostile_to: tuple[str, ...] = ()) -> int | None: ...
    def routes_from(self, location_id: str, *,
                    channel: str | None = None) -> list[dict]: ...
    def location_ids(self) -> list[str]: ...

    # ── Đọc: phục vụ classify_delta (§11) ────────────────────────────────
    # `is_locked` khoá QUAN HỆ; `entity_locked` khoá DANH TÍNH thực thể.
    def entity_locked(self, entity_id: str) -> bool: ...
    def active_relation(self, src: str, dst: str, rel_type: str) -> bool: ...
    def conflicting_relations(self, src: str, dst: str,
                              rel_type: str) -> list[dict]: ...
    def names_of(self, entity_id: str) -> list[str]: ...
