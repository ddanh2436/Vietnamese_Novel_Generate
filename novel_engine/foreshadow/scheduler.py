"""Foreshadowing Engine — lập lịch manh mối (§6.2, §6.3, §6.3.1).

Greedy có điểm số, chạy MỘT LẦN mỗi chương ở Director (F4). Bài toán và bảy
ràng buộc ở §6.1; ở đây là cách giữ cả bảy mà không khoá cứng.

═══ NĂM CHỖ KHÁC MÃ §6.3 / §6.3.1 ═══════════════════════════════════════════

1. QUÁ HẠN VƯỢT QUA RÀNG BUỘC THỨ TỰ. `_score` trả `10 + |slack|·2` cho manh
   mối quá hạn TRƯỚC khi kiểm `prerequisites`. `CLUE_THIRD_SIGNATURE` (cần con
   dấu ăn mòn VÀ nhật ký thiếu) quá hạn thì được lên lịch — rồi nấc 2 ép trả
   bài — trong khi độc giả chưa từng thấy con dấu. Vi phạm ràng buộc 1.

   Gốc bệnh sâu hơn: hạn chót của manh mối TIỀN ĐỀ không biết gì về manh mối
   PHỤ THUỘC vào nó. `CLUE_MISSING_LOGS` có deadline 6 nên áp lực thấp, nhưng
   `CLUE_THIRD_SIGNATURE` phải trả bài trước chương 5, tức phải được cài trước
   chương 5, tức nhật ký phải được cài trước nữa. `plant_by()` truyền hạn chót
   ngược qua DAG; manh mối nháp chấm điểm theo hạn CÀI đó, không theo hạn trả.

2. CHỈ MỘT POV CHO CẢ CHƯƠNG. §9.2 gọi `schedule(..., pov_for(ch, 0))` — POV
   cảnh 0 — rồi `build_beats` `zip` manh mối vào beat yên tĩnh, rồi Director lọc
   lại theo POV của từng cảnh. Manh mối rơi vào cảnh có POV khác thì bị lọc bỏ
   không ai ghi nhận: đã tiêu ngân sách chú ý, không được cài, không có báo cáo.
   `zip` còn cắt cụt khi manh mối nhiều hơn beat yên tĩnh (cảnh `escalate` chỉ
   có 2). Ở đây Scheduler XẾP CẢNH luôn, kiểm POV và vật mang CỦA CẢNH ĐÓ, và
   mọi manh mối đủ điểm mà không xếp được đi vào `unplaced` — không im lặng.

3. NẤC 2 ÉP THOẠI VÀO CẢNH KHÔNG AI NÓI. `carrier = ... or "dialogue"` bất kể
   cảnh có mấy người. Ngày 10 cho thấy chính xác điều gì xảy ra với ràng buộc
   không thể thoả: model phá nó. Cảnh một mình thì "trả bài qua thư từ, bản
   tin" (§6.3.1) — vật mang `object`.

4. NHẮC LẠI MỌI CHƯƠNG. Mọi manh mối đang sống có `deadline_pressure > 0`, nên
   greedy nhắc lại tất cả cho tới khi cạn ngân sách — salience không bao giờ
   phai, và `subtlety_target` mất nghĩa. §6.2 nói re-plant là BẮT BUỘC khi
   salience tụt dưới ngưỡng; ở đây nhắc lại chỉ khi sắp quên (`REINFORCE_AT`)
   hoặc đã vào cửa sổ trả bài.

5. `used_surface_forms` KHÔNG TỒN TẠI, và `unused[0] if unused else
   surface_forms[-1]` lặp mãi dạng cuối khi đã dùng hết — đúng điều §3.3 sinh
   `surface_forms` ra để tránh. Đồ thị ghi dạng đã dùng lúc commit; hết dạng
   mới thì lấy dạng dùng LÂU NHẤT. `surface_forms` rỗng thì `IndexError` — ở
   đây thành `blocked`.

Ngoài ra: nấc 2 ép một manh mối CHƯA TỪNG CÀI sang `payoff` — trả bài một chi
tiết độc giả chưa thấy là gian lận. Manh mối nháp bị ép thì cài mạnh, không trả.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from novel_engine.canon.models import Clue, ClueStatus

DECAY_LAMBDA = 0.18          # nửa đời ≈ 3,85 chương
SALIENCE_FLOOR = 0.25        # dưới mức này độc giả đã quên
REINFORCE_AT = 0.40          # nhắc lại TRƯỚC khi chạm sàn — lịch có thể trượt
MAX_PLANTS_PER_CH = 3
ATTENTION_BUDGET = 1.8       # tổng "trọng lượng chú ý" một chương chịu được
URGENT_SLACK = 3             # còn ≤3 chương tới deadline = khẩn cấp
FORCE_RESOLVE_AFTER = 3      # quá hạn ≥3 chương: bỏ qua ràng buộc mềm (nấc 2)
ABANDON_AFTER = 8            # quá hạn ≥8 chương: tác giả quyết (nấc 3)
MIN_PLANT_LEAD = 1           # cài và trả bài không cùng một chương

CARRIER_ORDER = ("object", "setting", "behavior", "dialogue")
MAX_PER_SCENE = 1
MAX_PER_SCENE_FORCING = 2

_DRAFTED = ClueStatus.DRAFTED
_CLOSED = {ClueStatus.PAID_OFF, ClueStatus.RETIRED}
_LIVE = {ClueStatus.PLANTED, ClueStatus.REINFORCED, ClueStatus.PARTIALLY_READ}
_PREREQ_READY = _LIVE | {ClueStatus.PAID_OFF}


def decay(clue: Clue, chapter: int) -> float:
    """salience(c, n) = s₀ · e^(−λ·(n − t_last)) — §6.2."""
    if clue.last_touched_chapter is None:
        return 0.0
    gap = max(0, chapter - clue.last_touched_chapter)
    return clue.salience * math.exp(-DECAY_LAMBDA * gap)


@dataclass
class PlantDirective:
    """Chỉ thị đưa cho Writer. KHÔNG chứa `description` của manh mối (§3.3)."""
    clue_id: str
    surface_form: str
    intensity: float            # 0..1 — 0.2 = thoáng qua, 0.9 = nhấn mạnh
    carrier: str                # object | setting | behavior | dialogue
    mode: str                   # plant | reinforce | payoff
    instruction: str
    weight: float = 0.6         # trọng lượng chú ý tiêu tốn
    scene_index: int = -1
    forced: bool = False        # đi qua nấc 2

    def to_contract(self) -> dict:
        return asdict(self)


@dataclass
class SceneSlot:
    """Một cảnh nhìn từ phía Scheduler: ai kể, có vật mang gì, có yên tĩnh không."""
    index: int
    pov: str
    affordances: dict[str, list[str]]
    quiet: bool = False


@dataclass
class ScheduleResult:
    directives: list[PlantDirective] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)   # nấc 3 → tác giả
    unplaced: list[dict] = field(default_factory=list)      # đủ điểm, không xếp được
    blocked: list[dict] = field(default_factory=list)       # chưa đủ điều kiện

    def for_scene(self, index: int) -> list[dict]:
        return [d.to_contract() for d in self.directives if d.scene_index == index]

    def report(self) -> dict:
        return {"directives": [d.to_contract() for d in self.directives],
                "unplaced": self.unplaced, "blocked": self.blocked,
                "escalations": self.escalations}


class ForeshadowScheduler:
    def __init__(self, graph, clues: dict[str, Clue] | None = None):
        self.g = graph
        self.clues = graph.clues if clues is None else clues

    # ── hạn cài, truyền ngược qua DAG ───────────────────────────────
    def plant_by(self, c: Clue, _stack: tuple[str, ...] = ()) -> int:
        """Chương MUỘN NHẤT manh mối phải được cài để mọi manh mối phụ thuộc
        (còn nháp) vẫn kịp cài và trả bài đúng hạn."""
        limit = c.payoff_deadline - MIN_PLANT_LEAD
        if c.clue_id in _stack:              # DAG hỏng — không đệ quy vô hạn
            return limit
        for d in self.clues.values():
            if c.clue_id in d.prerequisites and d.status == _DRAFTED:
                limit = min(limit, self.plant_by(d, _stack + (c.clue_id,)) - 1)
        return limit

    def _prereq_problem(self, c: Clue) -> str | None:
        for p in c.prerequisites:
            pc = self.clues.get(p)
            if pc is None:
                return "prerequisite_missing"
            if pc.status == ClueStatus.RETIRED:
                return "prerequisite_retired"
            if pc.status not in _PREREQ_READY:
                return "prerequisite_not_planted"
        return None

    # ── điểm ưu tiên ────────────────────────────────────────────────
    def _score(self, c: Clue, chapter: int) -> float:
        if c.status in _CLOSED or self._prereq_problem(c):
            return -1.0
        if c.status == _DRAFTED:
            slack = self.plant_by(c) - chapter
            if slack < 0:
                return 10.0 + abs(slack) * 2.0       # trễ cài — cao nhất
            virgin = 1.4 if slack <= URGENT_SLACK * 2 else 0.0
            return 1.0 / max(slack, 1) ** 0.7 + virgin

        slack = c.payoff_deadline - chapter
        if slack < 0:
            # DEADLOCK §6.3: quá hạn là ưu tiên CAO NHẤT, không phải loại trừ.
            return 10.0 + abs(slack) * 2.0
        cur = decay(c, chapter)
        forget = (SALIENCE_FLOOR - cur) * 3.2 if cur < SALIENCE_FLOOR else 0.0
        due = chapter >= c.payoff_threshold and slack <= URGENT_SLACK
        if not due and cur >= REINFORCE_AT:
            return -1.0                               # độc giả còn nhớ
        return 1.0 / max(slack, 1) ** 0.7 + forget

    @staticmethod
    def _mode(c: Clue, chapter: int) -> str:
        if c.status == _DRAFTED:
            return "plant"
        if chapter >= c.payoff_threshold and c.payoff_deadline - chapter <= URGENT_SLACK:
            return "payoff"
        return "reinforce"

    def _pick_form(self, c: Clue) -> str | None:
        if not c.surface_forms:
            return None
        used = list(self.g.used_surface_forms(c.clue_id)) if self.g is not None else []
        for f in c.surface_forms:
            if f not in used:
                return f
        last_use = {f: i for i, f in enumerate(used)}
        return min(c.surface_forms, key=lambda f: last_use.get(f, -1))

    @staticmethod
    def _intensity(c: Clue, mode: str, slack: int) -> float:
        if mode == "payoff":
            return 0.95
        base = 1.0 - c.subtlety_target          # subtlety cao → intensity thấp
        ramp = max(0.0, (8 - slack) / 8) * 0.35  # càng gần deadline càng rõ
        return min(0.9, base + ramp)

    # ── xếp cảnh ────────────────────────────────────────────────────
    def _place(self, c: Clue, chapter: int, mode: str, slots: list[SceneSlot],
               load: dict[int, int], forcing: bool):
        allowed = [k for k in CARRIER_ORDER if k in (c.carriers or CARRIER_ORDER)]
        cap = MAX_PER_SCENE_FORCING if forcing else MAX_PER_SCENE
        cands = []
        for s in slots:
            if load[s.index] >= cap:
                continue
            car = next((k for k in allowed if s.affordances.get(k)), None)
            if car is None:
                continue                                   # ràng buộc 5
            if not self.g.pov_can_observe(s.pov, c.clue_id, chapter):
                continue                                   # ràng buộc 6
            cands.append((s, car))
        if not cands and forcing:
            # Nấc 2: bỏ ràng buộc vật mang và POV. Thoại cần người nói; cảnh
            # một mình thì qua thư từ, bản tin — một vật.
            cands = [(s, "dialogue" if s.affordances.get("dialogue") else "object")
                     for s in slots if load[s.index] < cap]
        if not cands:
            return None
        if mode == "payoff":       # trả bài ở nhịp dồn, về cuối chương
            return min(cands, key=lambda x: (x[0].quiet, load[x[0].index], -x[0].index))
        # cài/nhắc ở beat YÊN TĨNH (§8.3), sớm, cảnh còn trống
        return min(cands, key=lambda x: (not x[0].quiet, load[x[0].index], x[0].index))

    @staticmethod
    def _instruction(mode: str, intensity: float, carrier: str) -> str:
        if mode == "payoff":
            return ("TRẢ BÀI: để nhân vật tự ghép nối, KHÔNG giải thích lại chuỗi "
                    "manh mối cho độc giả. Độc giả phải hiểu trước nhân vật khoảng "
                    "nửa nhịp.")
        if intensity < 0.35:
            return (f"Cài qua {carrier}. Đặt ở VỊ TRÍ KHÔNG NHẤN — giữa một đoạn "
                    f"đang nói chuyện khác. Không có câu nào bình luận về nó. Không "
                    f"có nhân vật nào phản ứng.")
        if intensity < 0.65:
            return (f"Cài qua {carrier}. Một nhân vật để ý nhưng hiểu SAI ý nghĩa, "
                    f"hoặc bị cắt ngang trước khi kịp hỏi.")
        return (f"Cài qua {carrier} và cho nhân vật phản ứng rõ. Được phép dừng "
                f"lại một nhịp, nhưng KHÔNG được kết luận.")

    def _retire_impact(self, c: Clue, chapter: int) -> dict:
        return {
            "orphaned_clues": sorted(x.clue_id for x in self.clues.values()
                                     if c.clue_id in x.prerequisites),
            "macro_event_at_risk": c.macro_event_target,
            "planted_in_chapter": c.planted_in_chapter,
            # §6.3.1 ghi `chapters_since_planted: c.planted_in_chapter` — đó là
            # SỐ CHƯƠNG đã cài, không phải số chương đã trôi qua.
            "chapters_since_planted": (chapter - c.planted_in_chapter
                                       if c.planted_in_chapter is not None else None),
        }

    # ── API chính ───────────────────────────────────────────────────
    def schedule(self, chapter: int, slots: list[SceneSlot]) -> ScheduleResult:
        res = ScheduleResult()
        ranked = sorted(((self._score(c, chapter), c) for c in self.clues.values()),
                        key=lambda x: (-x[0], x[1].clue_id))
        load = {s.index: 0 for s in slots}
        spent = 0.0

        for score, c in ranked:
            if c.status in _CLOSED:
                continue
            overdue = max(0, chapter - c.payoff_deadline)

            # Nấc 3 — TRƯỚC mọi bộ lọc, kể cả prerequisite: manh mối kẹt vì
            # tiền đề 8 chương vẫn là quyết định của tác giả, không phải im lặng.
            if overdue >= ABANDON_AFTER:
                res.escalations.append({
                    "clue_id": c.clue_id, "overdue_by": overdue,
                    "question": "Gia hạn payoff_deadline, hay RETIRE manh mối này?",
                    "retire_cost": self._retire_impact(c, chapter)})
                continue

            problem = self._prereq_problem(c)
            if problem:
                if overdue or self.plant_by(c) <= chapter + URGENT_SLACK:
                    res.blocked.append({
                        "clue_id": c.clue_id, "reason": problem, "overdue_by": overdue,
                        "waiting_on": [p for p in c.prerequisites
                                       if p not in self.clues
                                       or self.clues[p].status not in _PREREQ_READY]})
                continue
            if score < 0:
                continue          # `continue`, KHÔNG `break` (§6.3.1)

            if len(res.directives) >= MAX_PLANTS_PER_CH:
                res.unplaced.append({"clue_id": c.clue_id, "reason": "max_plants_per_chapter",
                                     "score": round(score, 2)})
                continue

            forcing = overdue >= FORCE_RESOLVE_AFTER
            mode = self._mode(c, chapter)
            if forcing and c.status != _DRAFTED:
                mode = "payoff"

            form = self._pick_form(c)
            if form is None:
                res.blocked.append({"clue_id": c.clue_id, "reason": "no_surface_forms",
                                    "overdue_by": overdue, "waiting_on": []})
                continue

            inten = 0.95 if forcing else self._intensity(c, mode, c.payoff_deadline - chapter)
            w = round(0.35 + inten * 0.7, 2)
            # Nấc 1 — quá hạn được vượt ngân sách chú ý.
            if spent + w > ATTENTION_BUDGET and not forcing:
                res.unplaced.append({"clue_id": c.clue_id, "reason": "attention_budget",
                                     "score": round(score, 2)})
                continue

            placed = self._place(c, chapter, mode, slots, load, forcing)
            if placed is None:
                res.unplaced.append({"clue_id": c.clue_id,
                                     "reason": "no_scene_with_carrier_and_pov",
                                     "score": round(score, 2)})
                continue
            slot, carrier = placed
            res.directives.append(PlantDirective(
                clue_id=c.clue_id, surface_form=form, intensity=round(inten, 2),
                carrier=carrier, mode=mode, weight=w, scene_index=slot.index,
                forced=forcing,
                instruction=self._instruction(mode, inten, carrier)))
            load[slot.index] += 1
            spent += w
        return res
