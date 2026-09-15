"""Triển khai `GraphPort` bằng NetworkX — bản chạy cục bộ, 0 chi phí vận hành.

═══ UNIFIED ROUTE GRAPH ═══════════════════════════════════════════════════

Địa lý của thế giới là MỘT. Một đèo núi hiểm trở hay một trạm kiểm soát bị
phong toả phải tác động đồng thời lên cả việc di chuyển của nhân vật lẫn tốc
độ truyền tin. Giữ hai bảng khoảng cách rời nhau (một cho `travel_ticks`
§3.6.1, một cho `latency_ticks` §5.6.1) vi phạm Single Source of Truth, và
chúng sẽ lệch nhau ở chương thứ ba — theo đúng cơ chế của NT-8.

Vì vậy: MỘT đồ thị tuyến đường, HAI phép chiếu có trọng số khác nhau.

    cạnh ROUTE(u, v) = {
        base_ticks:       int,        # 1 tick = 1 giờ đi đường
        terrain:          str,        # plains | mountain_pass | sea | cathedral_ward
        blocked_by:       list[str],  # phe/sự kiện chặn đường
        allowed_channels: list[str],  # kênh tin đi qua được
    }

    ├─ Di chuyển  (travel_ticks):  cost = base_ticks × speed_multiplier
    │                              bỏ cạnh nếu blocked_by ∩ phe thù địch
    └─ Truyền tin (latency_ticks): cost = base_ticks × channel.latency_multiplier
                                   bỏ cạnh nếu kênh ∉ allowed_channels
                                   hoặc blocked_by ∩ news.suppressed_by

`routes_from` trả `latency_ticks` = `base_ticks` THÔ, chưa nhân hệ số kênh, vì
`propagate` (§5.6.3) tự nhân `ch.latency_multiplier`. Nhân hai lần là một lỗi
âm thầm điển hình.

═══ HAI ĐỒ THỊ, KHÔNG PHẢI MỘT ════════════════════════════════════════════

`self.g` giữ thực thể + quan hệ ngữ nghĩa (MultiDiGraph — nhiều loại cạnh
giữa cùng cặp node). `self.routes` giữ địa lý (DiGraph — giữa hai địa điểm chỉ
có một tuyến, nhưng có hướng, vì một chốt kiểm soát có thể chặn chiều vào mà
không chặn chiều ra).

Đây KHÔNG mâu thuẫn với "unified": thống nhất nằm ở chỗ chỉ có MỘT nguồn dữ
liệu địa lý mà cả hai cơ chế cùng chiếu từ đó — không phải ở chỗ nhét quan hệ
chính trị và đường đi vào chung một cấu trúc.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Iterable

import networkx as nx

from novel_engine.canon.flashback import AttributeWindow
from novel_engine.canon.models import (
    Assertion, Clue, ClueStatus, Entity, Relation,
)

# Hệ số tốc độ theo phương tiện. Số nhỏ = đi nhanh.
SPEED_MULTIPLIERS: dict[str, float] = {
    "foot": 1.0,        # đi bộ
    "mecha": 0.6,       # thiết kỵ
    "armored": 0.4,     # xe bọc thép
}

# Quan hệ loại trừ lẫn nhau — dùng cho Q4 (§3.5) khi dò mâu thuẫn.
MUTUALLY_EXCLUSIVE: dict[str, set[str]] = {
    "BETRAYED": {"PROTECTS"},
    "PROTECTS": {"BETRAYED"},
    "MEMBER_OF": {"HAS_FEUD_WITH"},
}

_CLOSED_CLUE_STATUSES = {ClueStatus.PAID_OFF, ClueStatus.RETIRED}


class NetworkXGraph:
    """Triển khai `GraphPort` (§3.7). Bắt đầu ở đây; chuyển sang Neo4j khi
    thật sự chạm trần (>2.000 node / >8.000 cạnh)."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()          # thực thể + quan hệ ngữ nghĩa
        self.routes = nx.DiGraph()          # UNIFIED ROUTE GRAPH — địa lý
        self.clues: dict[str, Clue] = {}
        # F6: niềm tin KHÔNG được nằm chung với sự thật khách quan.
        self.beliefs: list[dict] = []
        # Sự thật khách quan đã chốt: (subject, predicate) → object
        self._truth: dict[tuple[str, str], Any] = {}
        # Cửa sổ khởi phát của thuộc tính vĩnh viễn (§3.6.4)
        self._windows: dict[tuple[str, str], AttributeWindow] = {}
        self._deaths: dict[str, int] = {}   # entity_id → tick chết

    # ══════════════════════════════ GHI ══════════════════════════════

    def upsert_entity(self, e: Entity) -> None:
        self.g.add_node(e.id, **e.model_dump())
        if e.kind == "location" and e.id not in self.routes:
            self.routes.add_node(e.id)

    def upsert_relation(self, r: Relation) -> None:
        """Khoá cạnh theo `type` để (src, dst, type) là duy nhất — upsert thật
        sự, không chồng thêm bản sao mỗi lần chương sau nhắc lại quan hệ cũ."""
        self.g.add_edge(r.src, r.dst, key=r.type, **r.model_dump())

    def close_relation(self, src: str, dst: str, rel_type: str,
                       until_chapter: int) -> None:
        """F7: không có hàm này thì quan hệ cũ không bao giờ đóng — nhân vật
        vẫn `MEMBER_OF` một phe họ đã phản bội từ mười chương trước."""
        if self.g.has_edge(src, dst, key=rel_type):
            self.g.edges[src, dst, rel_type]["until_chapter"] = until_chapter

    def commit_belief(self, holder: str, subject: str, predicate: str,
                      object: Any, kind: str, since_tick: int) -> None:
        """F6: mệnh đề phi khách quan đi vào HỒ SƠ NGƯỜI TIN, không vào graph.

        Đây là chỗ v2.3 + v2.4 hợp lại thành lỗi: v2.3 xếp lời nói dối vào
        `enrichment` (đúng), v2.4 cho commit mọi `enrichment` vào graph (đúng
        trong phạm vi của nó) — hợp lại thì lời nói dối thành sự thật lịch sử,
        đúng cái mà `epistemic` sinh ra để chặn (NT-15).
        """
        self.beliefs.append({
            "holder": holder, "subject": subject, "predicate": predicate,
            "object": object, "kind": kind, "since_tick": since_tick,
        })

    def commit_truth(self, subject: str, predicate: str, object: Any,
                     tick: int) -> None:
        """Ghi một sự thật KHÁCH QUAN. Tách khỏi `commit_belief` một cách có
        chủ ý: hai đường vào khác nhau thì không thể lẫn."""
        self._truth[(subject, predicate)] = object
        if predicate == "is_dead" and object:
            self._deaths[subject] = tick
        self.record_attestation(subject, predicate, tick, present=bool(object))

    def record_attestation(self, entity_id: str, attribute: str, tick: int, *,
                           present: bool) -> None:
        """Ghi nhận bằng chứng thuộc tính CÓ / CHƯA CÓ tại một tick (§3.6.4)."""
        w = self._windows.setdefault(
            (entity_id, attribute),
            AttributeWindow(entity=entity_id, attribute=attribute))
        (w.attested_present if present else w.attested_absent).append(tick)

    def add_clue(self, c: Clue) -> None:
        self.clues[c.clue_id] = c
        self.upsert_entity(Entity(id=c.clue_id, kind="clue", name=c.clue_id,
                                  attributes={"target": c.macro_event_target}))

    # ══════════════════ UNIFIED ROUTE GRAPH — nạp ══════════════════

    def add_route(self, u: str, v: str, *, base_ticks: int, terrain: str,
                  blocked_by: Iterable[str] = (),
                  allowed_channels: Iterable[str] = (),
                  one_way: bool = False) -> None:
        """Thêm một tuyến đường. Mặc định HAI CHIỀU: địa lý đối xứng là trường
        hợp thường gặp, còn chặn một chiều (chốt kiểm soát chặn lối vào nhưng
        không chặn lối ra) thì khai `one_way` rồi thêm chiều kia riêng."""
        attrs = {
            "base_ticks": int(base_ticks),
            "terrain": terrain,
            "blocked_by": list(blocked_by),
            "allowed_channels": list(allowed_channels),
        }
        self.routes.add_edge(u, v, **attrs)
        if not one_way:
            self.routes.add_edge(v, u, **dict(attrs))

    # ══════════════ PHÉP CHIẾU 1: DI CHUYỂN NHÂN VẬT ══════════════

    def travel_ticks(self, a: str, b: str, *, mode: str = "foot",
                     hostile_to: tuple[str, ...] = ()) -> int | None:
        """Thời gian đi từ `a` tới `b`, tính trên ROUTE graph.

        KHÔNG phải khoảng cách euclid: núi, chốt kiểm soát và biển làm hỏng
        mọi tính toán theo bán kính (§3.6.1).

        Trả `None` khi không tồn tại tuyến đường — `_no_teleport_epoch` dựa
        vào đúng giá trị này để phân biệt "đi quá nhanh" với "không có đường".
        """
        if a == b:
            return 0
        if a not in self.routes or b not in self.routes:
            return None
        speed = SPEED_MULTIPLIERS.get(mode, 1.0)
        hostile = set(hostile_to)

        # Dijkstra thuần, cost thực; làm tròn LÊN đúng một lần ở cuối. Ceil
        # từng cạnh sẽ cộng dồn sai số và làm đường dài nhiều chặng đắt giả tạo.
        dist: dict[str, float] = {a: 0.0}
        pq: list[tuple[float, str]] = [(0.0, a)]
        seen: set[str] = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == b:
                return math.ceil(d)
            for v in self.routes.successors(u):
                e = self.routes.edges[u, v]
                if hostile and hostile.intersection(e["blocked_by"]):
                    continue          # đường bị phe thù địch phong toả
                nd = d + e["base_ticks"] * speed
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return None

    # ══════════════ PHÉP CHIẾU 2: LAN TRUYỀN TIN TỨC ══════════════

    def routes_from(self, location_id: str, *,
                    channel: str | None = None) -> list[dict]:
        """Các cạnh đi ra từ một địa điểm, dưới dạng `propagate` (§5.6.3) cần.

        `latency_ticks` trả về là `base_ticks` THÔ. `propagate` tự nhân
        `ch.latency_multiplier`; nhân sẵn ở đây là nhân hai lần.
        """
        if location_id not in self.routes:
            return []
        out = []
        for v in self.routes.successors(location_id):
            e = self.routes.edges[location_id, v]
            if channel is not None and channel not in e["allowed_channels"]:
                continue              # kênh này không đi qua được địa hình đó
            out.append({
                "to": v,
                "latency_ticks": e["base_ticks"],
                "blocked_by": list(e["blocked_by"]),
                "terrain": e["terrain"],
                "allowed_channels": list(e["allowed_channels"]),
            })
        return out

    # ══════════════════════════ POV FIREWALL ══════════════════════════

    def known_by(self, pov_id: str, epoch_tick: int) -> dict:
        """Q2′ (§3.6.2) — nhân vật POV thực sự biết gì TẠI MỐC EPOCH NÀY.

        NT-6: lọc theo `since_tick`, KHÔNG theo `since_chapter`. Một cảnh hồi
        ức ở chương 30 kể sự kiện tại tick 400 trong khi truyện đang ở tick
        9.000 — lọc theo chương thì nhân vật trong hồi ức "biết" mọi thứ học
        được suốt 8.600 tick sau đó, và model sẽ không phát hiện ra vì trong
        nội bộ cảnh mọi thứ đọc vẫn hợp lý.

        Cố ý KHÔNG lọc theo `until_chapter`: trộn trục chương vào một truy vấn
        khoá theo epoch chính là NT-8. Tri thức một khi có thì không mất —
        quên là một sự kiện riêng, cần cơ chế riêng, không phải tác dụng phụ
        của việc đóng cạnh.
        """
        known, suspected = [], []
        if pov_id in self.g:
            for _, dst, key, d in self.g.out_edges(pov_id, keys=True, data=True):
                if d.get("since_tick", 0) > epoch_tick:
                    continue
                name = self.g.nodes[dst].get("name", dst) if dst in self.g else dst
                if key == "KNOWS_ABOUT":
                    known.append({"id": dst, "name": name, "certainty": "known"})
                elif key == "SUSPECTS":
                    suspected.append({"id": dst, "name": name,
                                      "certainty": "suspected",
                                      "conf": d.get("weight", 0.5)})

        clues_held = [
            {"id": c.clue_id,
             "form": c.surface_forms[0] if c.surface_forms else c.clue_id}
            for c in self.clues.values()
            if pov_id in c.understood_by_characters
        ]
        return {"known": known, "suspected": suspected, "clues_held": clues_held}

    def pov_can_observe(self, pov_id: str, clue_id: str, chapter: int) -> bool:
        """Cảnh này có được phép mang manh mối đó không (§9.2).

        Ba điều kiện, tất cả đều tất định:
        1. manh mối còn sống (chưa trả bài, chưa bị bỏ);
        2. mọi `prerequisites` đã được cài — không thể hiểu "con dấu bị ăn mòn
           bởi axit công nghiệp" trước khi độc giả biết "giáo hội cấm công
           nghiệp hoá" (§3.3);
        3. POV chưa tự mình HIỂU nó — cài lại cho người đã hiểu rồi là thừa.
        """
        c = self.clues.get(clue_id)
        if c is None or c.status in _CLOSED_CLUE_STATUSES:
            return False
        for pre in c.prerequisites:
            p = self.clues.get(pre)
            if p is None or p.status == ClueStatus.DRAFTED:
                return False
        return pov_id not in c.understood_by_characters

    # ══════════════════ FLASHBACK AUDIT (§3.6.4, F8) ══════════════════

    def exists(self, entity_id: str) -> bool:
        return entity_id in self.g

    def alive_after(self, entity_id: str, tick: int) -> bool:
        """Còn sống SAU `tick` không? Chưa có bản ghi chết = còn sống."""
        d = self._deaths.get(entity_id)
        return d is None or d > tick

    def attribute_window(self, entity_id: str, attribute: str) -> AttributeWindow:
        """Trả BẢN SAO. `flashback_admissible` append vào `attested_present`
        để thử giả thuyết — trả bản gốc thì mỗi lần kiểm tra lại làm bẩn canon
        bằng chính cái tick mà nó đang nghi ngờ."""
        w = self._windows.get((entity_id, attribute))
        if w is None:
            return AttributeWindow(entity=entity_id, attribute=attribute)
        return w.model_copy(deep=True)

    # ══════════════════════ CANON & KẾ HOẠCH ══════════════════════

    def is_locked(self, src: str, dst: str, rel_type: str) -> bool:
        """QUAN HỆ này có bị tác giả chốt không — agent cấm sửa.

        Chỉ khoá theo `provenance="author"` của chính cạnh đó.

        Bản Ngày 2 khoá thêm khi một đầu mút là thực thể `canon_locked`. Nghe an
        toàn, nhưng mọi nhân vật trong bible đều `canon_locked`, nên mọi quan hệ
        giữa hai nhân vật chính đều bị coi là khoá: Serena không bao giờ phản
        bội được Kaelen, và mỗi lần thử lại thành một escalation. Khoá DANH
        TÍNH của thực thể (`entity_locked`) và khoá QUAN HỆ giữa chúng là hai
        việc khác nhau — gộp làm một là NT-8 ở tầng quyền hạn.
        """
        if not self.g.has_edge(src, dst, key=rel_type):
            return False
        return self.g.edges[src, dst, rel_type].get("provenance") == "author"

    def entity_locked(self, entity_id: str) -> bool:
        """DANH TÍNH thực thể có bị tác giả chốt không (tên, loại, thuộc tính)."""
        return entity_id in self.g and bool(
            self.g.nodes[entity_id].get("canon_locked"))

    def active_relation(self, src: str, dst: str, rel_type: str) -> bool:
        """Quan hệ tồn tại VÀ còn hiệu lực (chưa đóng bằng `until_chapter`)."""
        if not self.g.has_edge(src, dst, key=rel_type):
            return False
        return self.g.edges[src, dst, rel_type].get("until_chapter") is None

    def names_of(self, entity_id: str) -> list[str]:
        """Mã, tên và bí danh — mọi cách văn xuôi có thể gọi thực thể này."""
        if entity_id not in self.g:
            return []
        d = self.g.nodes[entity_id]
        return [entity_id, d.get("name") or "", *d.get("aliases", [])]

    def conflicting_relations(self, src: str, dst: str,
                              rel_type: str) -> list[dict]:
        """Q4 (§3.5) — quan hệ mới có phá quan hệ cũ còn hiệu lực không."""
        excl = MUTUALLY_EXCLUSIVE.get(rel_type, set())
        out = []
        for _, d2, key, data in self.g.out_edges(src, keys=True, data=True):
            if d2 == dst and key in excl and data.get("until_chapter") is None:
                out.append({"existing": key, "since": data.get("since_chapter"),
                            "provenance": data.get("provenance")})
        return out

    def has_truth(self, subject: str, predicate: str) -> bool:
        return (subject, predicate) in self._truth

    def truth_of(self, subject: str, predicate: str, default: Any = None) -> Any:
        """Sự thật KHÁCH QUAN đã ghi cho (chủ thể, vị từ). Lời khai và niềm tin
        không bao giờ nằm ở đây — chúng ở `self.beliefs` (F6)."""
        return self._truth.get((subject, predicate), default)

    def conflicts_with_truth(self, a: Assertion) -> bool:
        """Điều nhân vật TIN/NÓI có lệch với sự thật khách quan không (§11).

        Chênh lệch này KHÔNG phải lỗi — nó là nguyên liệu của mỉa mai kịch
        tính (M12). Ghi nhận, đừng chặn.
        """
        key = (a.subject, a.predicate)
        return key in self._truth and self._truth[key] != a.object

    def faction_tensions(self, location_id: str, chapter: int) -> list[dict]:
        """Q1 (§3.5) — bối cảnh chính trị quanh một địa điểm, sâu 2 tầng."""
        if location_id not in self.g:
            return []
        factions: set[str] = set()
        frontier = {location_id}
        for _hop in range(2):
            nxt: set[str] = set()
            for node in frontier:
                for src, _dst, key in self.g.in_edges(node, keys=True):
                    if key in ("CONTROLS", "LOCATED_IN"):
                        nxt.add(src)
                        if self.g.nodes[src].get("kind") == "faction":
                            factions.add(src)
            frontier = nxt

        out = []
        for f in sorted(factions):
            feuds = []
            for _s, other, key, d in self.g.out_edges(f, keys=True, data=True):
                if key != "HAS_FEUD_WITH":
                    continue
                if d.get("since_chapter", 0) > chapter:
                    continue
                until = d.get("until_chapter")
                if until is not None and until <= chapter:
                    continue
                feuds.append({"with": self.g.nodes[other].get("name", other),
                              "intensity": d.get("weight", 1.0)})
            out.append({"faction": self.g.nodes[f].get("name", f),
                        "faction_id": f, "feuds": feuds[:3]})
        return out[:5]

    def due_clues(self, chapter: int, lookahead: int = 3) -> list[dict]:
        """Q3 (§3.5) — manh mối đến hạn, sắp theo slack tăng dần."""
        live = {ClueStatus.PLANTED, ClueStatus.REINFORCED,
                ClueStatus.PARTIALLY_READ}
        out = [
            {"clue_id": c.clue_id, "payoff_deadline": c.payoff_deadline,
             "salience": c.salience, "slack": c.payoff_deadline - chapter}
            for c in self.clues.values()
            if c.status in live
            and c.payoff_deadline <= chapter + lookahead
            and c.payoff_threshold <= chapter
        ]
        out.sort(key=lambda d: (d["slack"], -d["salience"]))
        return out

    def entity_index(self) -> list[dict]:
        """Danh mục thực thể đã biết, đưa vào EXTRACT_EMERGENT_TMPL (§9.2) để
        Extractor không tạo trùng một thực thể đã có dưới tên khác."""
        return [{"id": n, "name": d.get("name", n), "kind": d.get("kind"),
                 "aliases": d.get("aliases", [])}
                for n, d in sorted(self.g.nodes(data=True))
                if d.get("kind")]

    def location_ids(self) -> list[str]:
        """Tập ĐÓNG mã địa điểm hợp lệ. Đưa vào prompt digest và dùng làm hàng
        rào ở `_coerce_continuity` — model rất hay bịa mã nghe hợp lý cho những
        góc nhỏ mà văn xuôi nhắc tới."""
        return sorted(n for n, d in self.g.nodes(data=True)
                      if d.get("kind") == "location")
