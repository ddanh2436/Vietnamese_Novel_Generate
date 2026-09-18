"""Nợ tự sự TOÀN CỤC cho Author Console (`report-debt`, §6.4 + §16.1).

`narrative_debt_report` (§6.4) chỉ đếm manh mối. Nhưng thứ làm một tiểu thuyết
dài kỳ "phình ra rồi không đóng lại được" không chỉ là manh mối:

- quan hệ kẹt một giai đoạn quá lâu, hoặc đổ vỡ rồi không ai hàn gắn;
- tin tức tác giả đã khai mà chưa tới tai ai (§5.6);
- chỉ mục treo cuối mỗi cảnh (`SceneClose.unresolved`) không ai nhặt lại;
- PlanPatch đang chờ tác giả quyết (CP-4).

Tất cả đều là NỢ: thứ đã hứa với độc giả mà chưa trả. Gom về một chỗ, vì tác giả
chỉ mở một màn hình.
"""
from __future__ import annotations

import json
from pathlib import Path

from novel_engine.foreshadow.debt import narrative_debt_report

STUCK_AFTER = 5          # chương ở yên một giai đoạn thì coi là kẹt
DEBT_BLOCK = 8           # §6.4: vượt ngưỡng thì đừng mở cốt truyện phụ mới
UNRESOLVED_KEEP = 12


def story_debt(eng, *, last_chapter: int,
               reports_dir: Path | str = Path("output/reports")) -> dict:
    clues = narrative_debt_report(getattr(eng.graph, "clues", {}) or {}, [], last_chapter)

    rel = []
    for key, st in sorted(getattr(eng.graph.relationships, "states", {}).items()):
        if st.stage.value == "rupture":
            ly_do = "đổ vỡ chưa hàn gắn"
        elif st.chapters_in_stage >= STUCK_AFTER:
            ly_do = f"kẹt ở '{st.stage.value}' ≥{STUCK_AFTER} chương"
        else:
            continue
        rel.append({"pair": key, "stage": st.stage.value,
                    "chapters_in_stage": st.chapters_in_stage, "reason": ly_do})

    heard = {nid for (_c, nid) in (getattr(eng.graph, "news_knowledge", {}) or {})}
    news = [{"news_id": n.news_id, "origin_tick": n.origin_tick,
             "reason": "chưa tới tai ai"}
            for n in sorted(getattr(eng.graph.news, "items", {}).values(),
                            key=lambda x: x.news_id)
            if n.news_id not in heard and n.origin_tick <= _last_tick(eng)]

    reports_dir = Path(reports_dir)
    unresolved = []
    for n in range(1, last_chapter + 1):
        rp = reports_dir / f"ch{n:03d}.json"
        if not rp.exists():
            continue
        data = json.loads(rp.read_text(encoding="utf-8"))
        unresolved += [{"chapter": n, "thread": u} for u in data.get("unresolved", [])]

    patches = [p for p in eng.store.get_plan_patches()
               if p.get("author_decision_required", True)]

    load = clues["debt_load"] + len(rel) + len(patches)
    return {"last_chapter": last_chapter, "clues": clues, "relationships": rel,
            "news": news, "unresolved": unresolved[-UNRESOLVED_KEEP:],
            "unresolved_total": len(unresolved), "plan_patches": patches,
            "debt_load": load, "blocked": load > DEBT_BLOCK,
            "block_threshold": DEBT_BLOCK}


def _last_tick(eng) -> int:
    return max((f.time.end_tick for f in eng.store.get_frames()
                if f.time.mode == "present"), default=0)
