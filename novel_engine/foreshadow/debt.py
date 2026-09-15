"""Phát hiện nợ tự sự (§6.4) — chạy độc lập với Writer.

Hai chỗ khác §6.4:

- Manh mối quá hạn mang `severity: "major"`, không phải `"blocker"`. NT-17:
  nợ manh mối là quyết định KẾ HOẠCH (CP-4), không phải lỗi chặn (CP-3). Báo cáo
  này đi vào prompt Director và Author Console; nó không dừng chương nào.
- Thêm `late_to_plant`: manh mối còn nháp đã qua hạn CÀI (`plant_by`, truyền
  ngược qua DAG). §6.4 chỉ đếm theo hạn TRẢ, nên tiền đề của một manh mối sắp
  đến hạn không bao giờ hiện là nợ — cho tới khi đã quá muộn để cứu cả chuỗi.
"""
from __future__ import annotations

from novel_engine.canon.models import Clue, ClueStatus
from novel_engine.foreshadow.scheduler import (
    SALIENCE_FLOOR, URGENT_SLACK, ForeshadowScheduler, decay,
)


def narrative_debt_report(clues: dict[str, Clue], registry_obligations: list,
                          chapter: int) -> dict:
    sched = ForeshadowScheduler(None, clues)
    overdue, at_risk, forgotten, late = [], [], [], []

    for c in sorted(clues.values(), key=lambda x: x.clue_id):
        if c.status in (ClueStatus.PAID_OFF, ClueStatus.RETIRED):
            continue
        slack = c.payoff_deadline - chapter
        if slack < 0:
            overdue.append({"clue": c.clue_id, "overdue_by": -slack, "severity": "major"})
        elif slack <= URGENT_SLACK:
            at_risk.append({"clue": c.clue_id, "slack": slack})
        if c.status == ClueStatus.DRAFTED:
            pb = sched.plant_by(c)
            if pb < chapter:
                late.append({"clue": c.clue_id, "plant_by": pb, "late_by": chapter - pb})
        elif decay(c, chapter) < SALIENCE_FLOOR:
            forgotten.append({"clue": c.clue_id,
                              "salience": round(decay(c, chapter), 3),
                              "silent_for": chapter - (c.last_touched_chapter or 0)})

    for ob in registry_obligations:
        if ob["must_resurface_by"] < chapter and not ob.get("fulfilled"):
            overdue.append({"npc": ob["npc_id"], "kind": ob["kind"],
                            "overdue_by": chapter - ob["must_resurface_by"],
                            "severity": "major"})

    return {"overdue": overdue, "at_risk": at_risk, "forgotten": forgotten,
            "late_to_plant": late,
            "debt_load": len(overdue) * 2 + len(at_risk) + len(late)}
