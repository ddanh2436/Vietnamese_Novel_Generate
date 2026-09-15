"""Bộ từ vựng vị từ — tập đóng do tác giả khai (`bible/predicates.yaml`).

Quyết định Ngày 8. Lượt Gemini thật ở Chương 2 lách luật "snake_case không
dấu" bằng cách BỎ DẤU các động từ kể sự kiện (`CHAR_SERENA.dung_cach = "6 met"`,
`buoc_vao`, `nhan_dien`). Chặn một cách thì model lách cách khác. Chỉ tập đóng
mới bền: không có trong danh sách thì không vào canon.

Hai tầng, cùng một danh sách (NT-11):
- prompt Extractor nhận danh sách qua `render_for_prompt()`;
- `classify_delta` kiểm lại bằng `check_assertion()` — prompt là lời khuyên,
  code là hàng rào.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from novel_engine.canon.bible import DEFAULT_BIBLE

ObjectType = Literal["entity", "text", "number", "bool", "any"]


def _key(name: str) -> str:
    s = unicodedata.normalize("NFC", name or "").strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


class PredicateSpec(BaseModel):
    name: str
    description: str = ""
    subject_kinds: list[str] = Field(default_factory=lambda: ["any"])
    object_type: ObjectType = "any"
    object_kinds: list[str] = Field(default_factory=lambda: ["any"])
    permanent: bool = False
    aliases: list[str] = Field(default_factory=list)


class Vocabulary(BaseModel):
    predicates: dict[str, PredicateSpec]
    alias_map: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_specs(cls, specs: list[PredicateSpec]) -> "Vocabulary":
        preds: dict[str, PredicateSpec] = {}
        aliases: dict[str, str] = {}
        for sp in specs:
            k = _key(sp.name)
            if k in preds:
                raise ValueError(f"vị từ khai trùng: {sp.name}")
            preds[k] = sp.model_copy(update={"name": k})
        for sp in preds.values():
            for al in sp.aliases:
                ak = _key(al)
                if ak in preds or aliases.get(ak, sp.name) != sp.name:
                    raise ValueError(f"bí danh '{al}' trùng với một vị từ khác")
                aliases[ak] = sp.name
        return cls(predicates=preds, alias_map=aliases)

    def resolve(self, name: str) -> str | None:
        """Tên chuẩn của một vị từ (qua bí danh nếu cần), hoặc `None`."""
        k = _key(name)
        if k in self.predicates:
            return k
        return self.alias_map.get(k)

    def get(self, name: str) -> PredicateSpec | None:
        k = self.resolve(name)
        return self.predicates.get(k) if k else None

    @property
    def permanent(self) -> set[str]:
        return {n for n, sp in self.predicates.items() if sp.permanent}

    def render_for_prompt(self) -> str:
        lines = []
        for sp in self.predicates.values():
            val = sp.object_type
            if sp.object_type == "entity" and "any" not in sp.object_kinds:
                val = f"mã thực thể ({', '.join(sp.object_kinds)})"
            lines.append(f"- {sp.name} — chủ thể: {', '.join(sp.subject_kinds)}; "
                         f"giá trị: {val}. {sp.description}")
        return "\n".join(lines)


def load_vocabulary(bible_dir: Path | str | None = None) -> Vocabulary | None:
    """`None` khi bible không có `predicates.yaml` — khi đó phân loại quay về
    luật snake_case cũ, để một bible cũ vẫn chạy được."""
    p = (Path(bible_dir) if bible_dir else DEFAULT_BIBLE) / "predicates.yaml"
    if not p.exists():
        return None
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Vocabulary.from_specs(
        [PredicateSpec.model_validate(x) for x in raw.get("predicates", [])])


def normalize_predicates(delta, vocab: Vocabulary) -> list[dict]:
    """Quy bí danh về tên chuẩn. Chạy TRƯỚC khi sinh khoá phân loại: khoá của
    Assertion chứa `predicate`, nên đổi tên sau đó thì `delta.item(k)` hỏng."""
    notes: list[dict] = []
    for a in delta.assertions:
        canon = vocab.resolve(a.predicate)
        if canon and canon != a.predicate:
            notes.append({"source": "vocab", "field": "assertions",
                          "action": "coerced",
                          "reason": f"predicate '{a.predicate}' → '{canon}'",
                          "item": a.span[:120]})
            a.predicate = canon
    return notes


def check_assertion(a, vocab: Vocabulary, index: dict,
                    accepted_new: dict) -> tuple[str, str] | None:
    """Kiểm một mệnh đề khách quan theo từ vựng. Trả `(lý do, thông điệp)` khi
    vi phạm, `None` khi hợp lệ."""
    sp = vocab.get(a.predicate)
    if sp is None:
        return ("predicate_not_in_vocabulary",
                f"vị từ '{a.predicate}' không có trong bible/predicates.yaml — "
                f"lời kể sự kiện, hoặc tác giả cần khai thêm vị từ")

    def kind_of(eid) -> str | None:
        if not isinstance(eid, str):
            return None
        if eid in accepted_new:
            return accepted_new[eid].kind
        return index.get(eid, {}).get("kind")

    sk = kind_of(a.subject)
    if "any" not in sp.subject_kinds and sk not in sp.subject_kinds:
        return ("subject_kind_not_allowed",
                f"'{sp.name}' chỉ nhận chủ thể {sp.subject_kinds}, "
                f"{a.subject} là {sk}")

    obj, t = a.object, sp.object_type
    sai = ("object_type_mismatch",
           f"'{sp.name}' cần giá trị {t}, nhận {obj!r}")
    if t == "bool" and not isinstance(obj, bool):
        return sai
    if t == "number" and (isinstance(obj, bool) or not isinstance(obj, (int, float))):
        return sai
    if t == "text" and not isinstance(obj, str):
        return sai
    if t == "entity":
        ok = kind_of(obj)
        if ok is None:
            return ("unknown_object",
                    f"'{sp.name}' cần mã thực thể có trong canon, nhận {obj!r}")
        if "any" not in sp.object_kinds and ok not in sp.object_kinds:
            return ("object_type_mismatch",
                    f"'{sp.name}' chỉ nhận giá trị loại {sp.object_kinds}, "
                    f"{obj} là {ok}")
    return None
