"""Sổ quan hệ — trạng thái mọi cặp nhân vật, nằm trong canon.

Không có kho riêng: `RelationshipState` là một phần của WorldState, nên nó được
dựng như mọi thứ khác — nạp từ bible, rồi fold các delta đã ghi (§3.1). Rollback
một chương và viết lại cho ra đúng quan hệ như chưa từng viết bản cũ.

`StateDelta.relationship_updates` (§3.4) chứa NGUYÊN TRẠNG THÁI do LLM viết,
kể cả `stage`. Áp nó là để model đặt `stage: catharsis` và vượt qua mọi guard
của §7.2. Sổ này KHÔNG BAO GIỜ đọc trường đó; nó chỉ nhận `relationship_events`.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import yaml

from novel_engine.canon.models import RelationshipEvent, RelationStage
from novel_engine.relationship.dynamics import settle_chapter
from novel_engine.relationship.machine import advance_or_hold
from novel_engine.relationship.models import RelationshipState


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


class RelationshipBook:
    def __init__(self, states: dict[str, RelationshipState] | None = None):
        self.states: dict[str, RelationshipState] = dict(states or {})

    @classmethod
    def from_bible(cls, bible_dir: Path | str) -> "RelationshipBook":
        p = Path(bible_dir) / "relationships.yaml"
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        book = cls()
        for r in raw.get("relationships", []):
            a, b = sorted((r["a"], r["b"]))
            st = RelationshipState.model_validate({**r, "a": a, "b": b})
            book.states[st.pair_key] = st
        return book

    def get(self, a: str, b: str) -> RelationshipState:
        k = pair_key(a, b)
        if k not in self.states:
            x, y = sorted((a, b))
            self.states[k] = RelationshipState(a=x, b=y)
        return self.states[k]

    def snapshot(self) -> dict[str, dict]:
        return {k: st.model_dump(mode="json") for k, st in sorted(self.states.items())}

    # ── GHI: gọi từ apply_delta, cả lúc commit lẫn lúc replay ──────────
    def settle(self, chapter: int, events: list[RelationshipEvent], frames,
               known_chars) -> dict:
        prefix = f"CH{chapter:03d}_"
        own = [f for f in frames if f.scene_id.startswith(prefix)]
        mode_of = {f.scene_id: f.time.mode for f in own}

        # Tương tác = cùng cảnh HIỆN TẠI, cùng địa điểm. Cảnh hồi ức là quá khứ —
        # bible đã kể nó; đếm lại là tính một nghịch cảnh hai lần (NT-6).
        interacted: set[str] = set()
        for f in own:
            if f.time.mode != "present":
                continue
            by_loc: dict[str, list[str]] = defaultdict(list)
            for cid, loc in (f.locations or {}).items():
                by_loc[loc].append(cid)
            for group in by_loc.values():
                for a, b in combinations(sorted(set(group)), 2):
                    interacted.add(pair_key(a, b))

        by_pair: dict[str, list[RelationshipEvent]] = defaultdict(list)
        skipped: list[dict] = []
        for ev in events:
            reason = None
            if not ev.verified:
                reason = "unverified_span"
            elif ev.a == ev.b or ev.a not in known_chars or ev.b not in known_chars:
                reason = "unknown_character"
            elif mode_of.get(ev.scene_id, "present") != "present":
                reason = "non_present_scene"
            if reason:
                skipped.append({"reason": reason, "pair": f"{ev.a}|{ev.b}",
                                "kind": ev.kind, "scene_id": ev.scene_id})
            else:
                by_pair[pair_key(ev.a, ev.b)].append(ev)

        rupture = {k for k, st in self.states.items() if st.stage == RelationStage.RUPTURE}
        transitions: list[dict] = []
        for k in sorted(set(by_pair) | interacted | rupture):
            a, b = k.split("|")
            transitions += settle_chapter(self.get(a, b), by_pair.get(k, []), chapter,
                                          interacted=k in interacted)
        return {"applied": sum(len(v) for v in by_pair.values()),
                "skipped": skipped, "transitions": transitions}

    # ── ĐỌC: chỉ thị cho Director ──────────────────────────────────────
    def directives_for_chapter(self, chapter: int, planner, chars: dict,
                               scene_modes: list[str]) -> dict[int, list[dict]]:
        """Mỗi cặp nhận chỉ thị ở MỘT cảnh — cảnh HIỆN TẠI cuối cùng cả hai có mặt.

        §9.2 gắn `rel_dirs` của MỌI cặp vào MỌI contract, kể cả cảnh cặp đó vắng
        mặt. "Cảnh phải chứa một bất đồng về phương pháp" giao cho cảnh Kaelen
        một mình là ràng buộc không thể thoả (Ngày 10), và giao cho sáu cảnh là
        sáu lần dựng cùng một bước ngoặt.
        """
        last: dict[str, tuple[int, str, str]] = {}
        for si, mode in enumerate(scene_modes):
            if mode != "present":
                continue
            present = sorted({c for c in planner.present_characters(chapter, si) if c in chars})
            for a, b in combinations(present, 2):
                last[pair_key(a, b)] = (si, a, b)
        out: dict[int, list[dict]] = defaultdict(list)
        for k, (si, a, b) in sorted(last.items()):
            d = advance_or_hold(self.get(a, b), chapter)
            out[si].append({"pair": k, "a": a, "b": b,
                            "names": [chars[a].name, chars[b].name], **d})
        return out
