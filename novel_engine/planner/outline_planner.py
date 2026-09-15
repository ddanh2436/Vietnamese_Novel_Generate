"""Planner đọc `bible/outline.yaml`.

§9.2 gọi `eng.planner.pov_for(ch, si)`, `.present_characters(...)`,
`.location(...)`, `.goal_of(...)`, `.hidden_action(...)`, `.secrets_of(...)`,
`.time_overrides(ch)` — nhưng tài liệu không định nghĩa `planner` ở đâu cả.
Đây là bản tối giản: đọc dàn ý của tác giả, không suy diễn, không gọi LLM.

NT-1: mọi thứ ở đây là QUYẾT ĐỊNH — ai kể, ở đâu, ai có mặt. Quyết định là
việc của tác giả và của code, không phải của model.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from novel_engine.canon.bible import DEFAULT_BIBLE


class OutlinePlanner:
    def __init__(self, outline_path: Path | str | None = None,
                 scenes_per_chapter: int = 6) -> None:
        p = Path(outline_path) if outline_path else DEFAULT_BIBLE / "outline.yaml"
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        self.chapters: dict[int, dict] = {
            c["chapter"]: c for c in raw.get("chapters", [])}
        self.scenes_per_chapter = scenes_per_chapter

    # ── truy cập thô ──────────────────────────────────────────────

    def chapter(self, ch: int) -> dict:
        if ch not in self.chapters:
            raise KeyError(
                f"chương {ch} không có trong outline.yaml — dàn ý là đầu vào "
                f"của TÁC GIẢ, hệ thống không tự bịa ra chương mới")
        return self.chapters[ch]

    def scene(self, ch: int, si: int) -> dict:
        scenes = self.chapter(ch).get("scenes", [])
        if not scenes:
            raise KeyError(f"chương {ch} không có cảnh nào trong outline")
        # Lặp vòng khi số beat vượt số cảnh đã soạn: thà lặp còn hơn ném giữa
        # chương, và số cảnh thật do `build_beats(n=…)` quyết định.
        return scenes[si % len(scenes)]

    def n_scenes(self, ch: int) -> int:
        return len(self.chapter(ch).get("scenes", [])) or self.scenes_per_chapter

    def outline_beat(self, ch: int) -> str:
        return (self.chapter(ch).get("outline_beat") or "").strip()

    def title(self, ch: int) -> str:
        return self.chapter(ch).get("title", f"Chương {ch}")

    # ── §9.2 gọi những hàm này ────────────────────────────────────

    def pov_for(self, ch: int, si: int) -> str:
        return self.scene(ch, si)["pov"]

    def present_characters(self, ch: int, si: int) -> list[str]:
        s = self.scene(ch, si)
        present = list(s.get("present") or [])
        pov = s["pov"]
        if pov not in present:       # POV luôn có mặt trong cảnh của chính mình
            present.insert(0, pov)
        return present

    def location_id(self, ch: int, si: int) -> str:
        return self.scene(ch, si)["location"]

    # `location` và `location_id` cùng trả mã LOC_* một cách CÓ CHỦ Ý.
    # §9.2 dùng hai tên cho hai chỗ; để chúng trả hai KHÔNG GIAN ĐỊNH DANH
    # khác nhau (tên hiển thị vs mã) là tái tạo đúng NT-8 — tên đi vào
    # ContinuityFrame rồi `travel_ticks` tra bằng mã và không khớp mãi mãi.
    location = location_id

    def goal_of(self, cid: str, ch: int, si: int) -> str:
        return (self.scene(ch, si).get("goals") or {}).get(cid, "")

    def hidden_action(self, cid: str, ch: int, si: int) -> str | None:
        return (self.scene(ch, si).get("hidden_actions") or {}).get(cid)

    def secrets_of(self, cid: str, ch: int) -> list[str]:
        """Điều nhân vật TUYỆT ĐỐI không nói ra trong chương này."""
        return list((self.chapter(ch).get("secrets") or {}).get(cid, []))

    def time_overrides(self, ch: int) -> dict[int, dict]:
        """Khoá YAML là chuỗi hay số tuỳ cách viết — chuẩn hoá về int, vì
        `allocate_scene_times` tra bằng chỉ số cảnh kiểu int."""
        raw = self.chapter(ch).get("time_overrides") or {}
        return {int(k): v for k, v in raw.items()}

    def action_options(self, cid: str, ch: int, si: int) -> list[dict]:
        """Tập hành động cho `deliberate()` (§5.2). Chưa cài ở GĐ1 — trả rỗng
        để Director chạy được, và `deliberation` sẽ là dict rỗng."""
        return []

    def scene_affordances(self, ch: int, si: int) -> dict[str, list[str]]:
        """Vật mang manh mối khả dụng trong cảnh (§6.1 ràng buộc 5).

        §9.2 gọi `planner.chapter_affordances(ch)` — MỘT bộ cho cả chương — nhưng
        vật mang là của CẢNH: Chương 2 có năm cảnh một mình, và "lời thoại" chỉ
        tồn tại ở cảnh có người thứ hai. Outline khai `affordances` thì dùng
        nguyên; không thì suy từ ai có mặt và ở đâu.
        """
        s = self.scene(ch, si)
        if s.get("affordances"):
            return {k: list(v or []) for k, v in s["affordances"].items()}
        others = [c for c in self.present_characters(ch, si) if c != s["pov"]]
        return {"setting": [s["location"]],
                "object": list(s.get("props") or [s["location"]]),
                "behavior": others,
                "dialogue": others}
