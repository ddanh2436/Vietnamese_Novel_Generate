"""Trình nạp Bible — từ YAML của tác giả vào `NetworkXGraph`.

`bible/` là NGUỒN SỰ THẬT GỐC (§15.1). Agent không được ghi vào đây; mọi thứ
agent sinh ra đi qua `StateDelta` và chỉ vào canon sau khi tác giả duyệt ở
CP-2 (§16.1). Vì vậy mọi thực thể nạp từ bible đều mang `provenance="author"`,
và `is_locked` dựa vào đúng dấu hiệu đó.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from novel_engine.canon.models import Clue, Entity, Relation
from novel_engine.canon.networkx_graph import NetworkXGraph
from novel_engine.character.models import CharacterProfile

DEFAULT_BIBLE = Path(__file__).resolve().parents[2] / "bible"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_characters(bible_dir: Path) -> dict[str, CharacterProfile]:
    """Nạp hồ sơ nhân vật. Sắp theo tên file để thứ tự nạp tất định — bộ hồi
    quy §13.2 cần cùng seed cho ra cùng kết quả."""
    out: dict[str, CharacterProfile] = {}
    for f in sorted((bible_dir / "characters").glob("*.yaml")):
        p = CharacterProfile.model_validate(_read(f))
        out[p.id] = p
    return out


def load_bible(bible_dir: Path | str = DEFAULT_BIBLE
               ) -> tuple[NetworkXGraph, dict[str, CharacterProfile], dict]:
    """Dựng graph từ bible. Trả `(graph, characters, meta)`.

    THỨ TỰ NẠP QUAN TRỌNG: thực thể trước, quan hệ sau. `upsert_relation` của
    NetworkX tự tạo node cho đầu mút chưa tồn tại — node trần, không có `kind`
    hay `name`. Nạp ngược thứ tự thì `faction_tensions` lọc `kind == "faction"`
    trên node trần và im lặng trả rỗng.
    """
    bible_dir = Path(bible_dir)
    g = NetworkXGraph()

    world = _read(bible_dir / "world.yaml")
    meta = world.get("meta", {})

    # 1. Thực thể — phe phái, địa điểm
    for f in world.get("factions", []):
        g.upsert_entity(Entity(kind="faction", **f))
    for loc in world.get("locations", []):
        g.upsert_entity(Entity(kind="location", **loc))

    # 2. Nhân vật
    chars = load_characters(bible_dir)
    for p in chars.values():
        g.upsert_entity(Entity(
            id=p.id, kind="character", name=p.name, canon_locked=True,
            first_appearance=1,
            attributes={"role_tier": p.role_tier, "moral_line": p.moral_line}))

    # 3. Manh mối — trước quan hệ, vì bible có cạnh KNOWS_ABOUT trỏ tới clue
    for c in _read(bible_dir / "clues.yaml").get("clues", []):
        g.add_clue(Clue(**c))

    # 4. UNIFIED ROUTE GRAPH — địa lý, một nguồn cho cả di chuyển lẫn truyền tin
    for r in world.get("routes", []):
        g.add_route(r["from"], r["to"],
                    base_ticks=r["base_ticks"], terrain=r["terrain"],
                    blocked_by=r.get("blocked_by", []),
                    allowed_channels=r.get("allowed_channels", []),
                    one_way=r.get("one_way", False))

    # 5. Quan hệ — sau cùng, khi mọi đầu mút đã là node có thuộc tính đầy đủ
    for rel in world.get("relations", []):
        g.upsert_relation(Relation(provenance="author", **rel))

    # 6. Quan hệ nhân vật (§7.1) — ĐIỂM XUẤT PHÁT. Mọi thay đổi sau đó là sự
    # kiện trích từ văn xuôi, fold từ delta; file bible không bao giờ bị sửa.
    from novel_engine.relationship.book import RelationshipBook
    g.relationships = RelationshipBook.from_bible(bible_dir)

    # 7. Tin tức (§5.6) — tin do tác giả khai; ai nghe, bản nào, lúc nào thì
    # hệ thống tính khi chốt chương.
    from novel_engine.world.news import NewsDispatcher
    g.news = NewsDispatcher.from_bible(bible_dir)

    return g, chars, meta
