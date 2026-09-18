"""Parse `StateDelta` KHOAN DUNG — từng mục một (§9.5, lớp 4 ở mức mục).

Lượt Gemini thật ở Chương 2: sau 90 giây viết sáu cảnh, Extractor trả về một
thực thể `kind: "document"`. `Entity.kind` không có giá trị đó, nên
`StateDelta.model_validate_json` ném `ValidationError` cho CẢ delta, lượt tự sửa
cũng không cứu được, và cả chương escalate. Một mục sai giết mọi mục đúng.

§9.5 lớp 4 nói lỗi phải thành escalation chứ không thành stack trace. Ở mức
cả chương thì đúng; ở mức từng mục thì còn làm tốt hơn được: validate RIÊNG
từng phần tử, cứu cái cứu được, loại cái hỏng, và GHI LẠI đã loại gì. Không
mục nào biến mất im lặng.

Hai loại xử lý, cả hai đều được báo cáo:

- `coerced` — đồng nghĩa hiển nhiên của một `kind` hợp lệ (`document` →
  `object`). Ánh xạ là một bảng CỐ ĐỊNH, không đoán: thứ không có trong bảng
  thì bị loại chứ không bị ép.
- `dropped` — mục vẫn không hợp lệ sau ánh xạ.

JSON hỏng hẳn (không parse được thành object) thì không có mục nào để cứu từng
cái; khi đó quay về `parse_model` với lượt tự sửa của §9.5.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from novel_engine.canon.models import (
    Assertion, ClueStatus, Entity, PlantEvidence, Relation, RelationshipEvent,
    StateDelta,
)
from novel_engine.llm.json_io import parse_model, strip_fences
from novel_engine.relationship.models import RelationshipState

KIND_SYNONYMS: dict[str, str] = {
    "document": "object", "doc": "object", "paper": "object", "papers": "object",
    "item": "object", "artifact": "object", "tool": "object", "weapon": "object",
    "vehicle": "object", "record": "object", "letter": "object",
    "person": "character", "people": "character", "npc": "character",
    "place": "location", "area": "location", "room": "location",
    "building": "location",
    "organization": "faction", "organisation": "faction", "group": "faction",
    "org": "faction",
    "rule": "doctrine", "law": "doctrine", "policy": "doctrine",
    "custom": "doctrine",
    "incident": "event",
}

_LIST_FIELDS = {
    "new_entities": Entity,
    "new_relations": Relation,
    "retracted_relations": Relation,
    "assertions": Assertion,
    "plant_evidence": PlantEvidence,
    "relationship_updates": RelationshipState,
    "relationship_events": RelationshipEvent,
}

# Đồng nghĩa HIỂN NHIÊN của loại sự kiện quan hệ. Bảng cố định như KIND_SYNONYMS:
# thứ không có trong bảng thì bị loại, không bị đoán.
REL_KIND_SYNONYMS: dict[str, str] = {
    "sacrificed": "sacrifice", "self_sacrifice": "sacrifice",
    "betrayed": "betrayal", "treachery": "betrayal",
    "ordeal": "shared_ordeal", "shared_hardship": "shared_ordeal",
    "disagreement": "value_clash", "argument": "value_clash", "conflict": "value_clash",
    "reconciliation": "reconciled_method",
    "affection": "verbal_affection_only",
    "protected_other": "acted_against_own_interest_for_other",
}


def _brief(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:200]
    except (TypeError, ValueError):
        return str(obj)[:200]


MAX_COERCE = 4


def _validate_khoan_dung(model, it):
    """Validate một mục, tự sửa những sai kiểu HIỂN NHIÊN rồi thử lại.

    Lượt Gemini Arc 1: Extractor trả `"concluded_by": "CHAR_KAELEN"` — một chuỗi
    ở chỗ schema đòi danh sách. Cả mục `plant_evidence` bị loại, nên manh mối ĐÃ
    được cài và ĐÃ được trích xuất vẫn bị tính là trượt: 0/2 ở cả năm chương,
    M11 = 0.0, và nợ manh mối vượt ngưỡng. Một trường sai kiểu giết một cơ chế.

    Chỉ sửa MỘT loại sai: giá trị đơn ở chỗ cần danh sách (`list_type`). Không
    đoán gì thêm — mọi lần sửa đều được ghi lại như `coerced`.
    """
    sua = []
    for _ in range(MAX_COERCE):
        try:
            return model.model_validate(it), sua
        except ValidationError as e:
            loi = e.errors()
            hong = [x for x in loi
                    if x.get("type") == "list_type" and len(x.get("loc", ())) == 1
                    and isinstance(it.get(x["loc"][0]), (str, int, float, bool))]
            if not hong:
                return None, loi
            it = dict(it)
            for x in hong:
                f = x["loc"][0]
                sua.append((f, _brief(it[f])))
                it[f] = [it[f]]
    return None, [{"loc": ("?",), "msg": "vẫn không hợp lệ sau khi ép kiểu"}]


def parse_delta_lenient(raw: str, repair_llm=None,
                        source: str = "") -> tuple[StateDelta, list[dict]]:
    """Trả `(delta, issues)`. `issues` gồm mọi mục bị ép kiểu hoặc bị loại."""
    text = strip_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return parse_model(raw, StateDelta, repair_llm=repair_llm), []
    if not isinstance(data, dict):
        raise ValueError(f"không parse được StateDelta: gốc JSON là "
                         f"{type(data).__name__}, không phải object")

    issues: list[dict] = []
    clean: dict[str, list] = {}
    for field, model in _LIST_FIELDS.items():
        items = data.get(field) or []
        if not isinstance(items, list):
            issues.append({"source": source, "field": field, "action": "dropped",
                           "reason": "not_a_list", "item": _brief(items)})
            clean[field] = []
            continue
        ok = []
        for i, it in enumerate(items):
            if isinstance(it, dict):
                if field == "new_entities":
                    k = str(it.get("kind", "")).strip().lower()
                    if k in KIND_SYNONYMS:
                        issues.append({"source": source, "field": field,
                                       "index": i, "action": "coerced",
                                       "reason": f"kind '{k}' → '{KIND_SYNONYMS[k]}'",
                                       "item": _brief(it)})
                        it = {**it, "kind": KIND_SYNONYMS[k]}
                elif field == "relationship_events":
                    k = str(it.get("kind", "")).strip().lower()
                    if k in REL_KIND_SYNONYMS:
                        issues.append({"source": source, "field": field,
                                       "index": i, "action": "coerced",
                                       "reason": f"kind '{k}' → '{REL_KIND_SYNONYMS[k]}'",
                                       "item": _brief(it)})
                        it = {**it, "kind": REL_KIND_SYNONYMS[k]}
                    it = {k2: v for k2, v in it.items() if k2 != "verified"}
                elif field == "assertions":
                    # NT-13: `chapter`/`scene` là metadata của hệ thống — model
                    # hay bỏ trống, `stamp()` sẽ điền `chapter` sau.
                    it = {"chapter": 0, "scene": 0, **it}
            obj, sua = _validate_khoan_dung(model, it)
            if obj is not None:
                for f, v in sua:
                    issues.append({"source": source, "field": field, "index": i,
                                   "action": "coerced",
                                   "reason": f"'{f}': {v} → danh sách một phần tử",
                                   "item": _brief(it)})
                ok.append(obj)
            else:
                err = sua[0]
                issues.append({
                    "source": source, "field": field, "index": i,
                    "action": "dropped", "reason": "invalid_item",
                    "error": f"{'.'.join(str(x) for x in err.get('loc', ()))}: "
                             f"{err.get('msg', '')}"[:200],
                    "item": _brief(it)})
        clean[field] = ok

    transitions: dict[str, ClueStatus] = {}
    ct = data.get("clue_transitions") or {}
    if isinstance(ct, dict):
        for cid, st in ct.items():
            try:
                transitions[cid] = ClueStatus(st)
            except ValueError:
                issues.append({"source": source, "field": "clue_transitions",
                               "action": "dropped", "reason": "invalid_status",
                               "item": _brief({cid: st})})

    return StateDelta(**clean, clue_transitions=transitions), issues
