"""Hồi quy từ lượt Gemini thật ở Chương 2: một thực thể `kind: "document"` làm
đổ CẢ delta sau 90 giây viết, và CLI thoát mà không lưu văn xuôi."""
from __future__ import annotations

import json

import pytest

from novel_engine.canon.models import ClueStatus
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.reconcile.lenient import KIND_SYNONYMS, parse_delta_lenient
from novel_engine.reconcile.verify import merge_extractions

GEMINI_CH2 = json.dumps({
    "new_entities": [
        {"id": "OBJ_LOCATION_TAG", "kind": "object", "name": "Thẻ định vị"},
        {"id": "DOC_SHIP_MANIFEST", "kind": "document", "name": "Bản kê hàng tàu quặng"},
        {"id": "X_BAD", "kind": "spaceship", "name": "??"},
    ],
    "assertions": [
        {"subject": "OBJ_LOCATION_TAG", "predicate": "is_active", "object": False,
         "span": "Thẻ định vị đã dừng phát tín hiệu", "confidence": 0.8},
    ],
    "clue_transitions": {"CLUE_SEAL_CORROSION": "planted", "CLUE_X": "exploded"},
}, ensure_ascii=False)


def test_kind_document_duoc_quy_ve_object_khong_do_ca_delta():
    d, issues = parse_delta_lenient(GEMINI_CH2, source="luot_2")
    kinds = {e.id: e.kind for e in d.new_entities}
    assert kinds == {"OBJ_LOCATION_TAG": "object", "DOC_SHIP_MANIFEST": "object"}
    act = {(i["action"], i.get("index")) for i in issues if i["field"] == "new_entities"}
    assert ("coerced", 1) in act and ("dropped", 2) in act


def test_muc_hong_bi_loai_RIENG_muc_dung_con_nguyen():
    d, issues = parse_delta_lenient(GEMINI_CH2)
    assert len(d.assertions) == 1                 # thiếu chapter/scene vẫn cứu được
    assert d.clue_transitions == {"CLUE_SEAL_CORROSION": ClueStatus.PLANTED}
    assert any(i["reason"] == "invalid_status" for i in issues)


def test_kind_la_khong_co_trong_bang_thi_bi_loai_khong_bi_doan():
    """Ánh xạ là bảng CỐ ĐỊNH. `spaceship` không có trong bảng → loại, không ép."""
    assert "spaceship" not in KIND_SYNONYMS
    _d, issues = parse_delta_lenient(GEMINI_CH2)
    bad = next(i for i in issues if i.get("index") == 2 and i["field"] == "new_entities")
    assert bad["action"] == "dropped" and "kind" in bad["error"]


def test_merge_extractions_bao_cao_muc_bi_loai():
    dropped: list[dict] = []
    m = merge_extractions('{"assertions": []}', GEMINI_CH2, chapter=2, dropped=dropped)
    assert m.chapter == 2 and len(m.new_entities) == 2
    assert {i["source"] for i in dropped} == {"luot_2"}


def test_json_hong_han_van_nem_valueerror():
    with pytest.raises(ValueError, match="StateDelta"):
        parse_delta_lenient("hoàn toàn không phải json")


def test_goc_json_khong_phai_object_thi_nem():
    with pytest.raises(ValueError, match="không phải object"):
        parse_delta_lenient("[1, 2, 3]")


def test_prompt_noi_ro_doc_la_hoc_thuyet_va_liet_ke_kind():
    """`DOC_` trong quy ước tiền tố cũ bị model đọc thành *document*."""
    from novel_engine.prompts import EXTRACT_EMERGENT_TMPL
    assert "không có kind `document`" in EXTRACT_EMERGENT_TMPL
    assert "KHÔNG phải giấy tờ" in EXTRACT_EMERGENT_TMPL


class _LuotHaiKieuGemini(FakeLLM):
    def invoke(self, prompt: str, *, role: str = "") -> str:
        if role == "extractor_emergent":
            self.calls.append({"role": role, "prompt": prompt})
            return GEMINI_CH2
        return super().invoke(prompt, role=role)


def test_chuong_khong_escalate_vi_mot_thuc_the_sai_kind():
    e = build_engines(_LuotHaiKieuGemini(), db_path=":memory:")
    try:
        st = run_chapter(e, 1)
        assert not st.get("escalated"), st.get("escalation_reason")
        rep = st["extraction_report"]
        assert any(i["action"] == "dropped" for i in rep["dropped_items"])
        assert any(i["action"] == "coerced" for i in rep["dropped_items"])
        ids = {x["id"] for x in st["delta"]["new_entities"]}
        assert "DOC_SHIP_MANIFEST" in ids and "X_BAD" not in ids
    finally:
        e.store.close()


def test_cli_giu_van_xuoi_khi_escalate(tmp_path, monkeypatch):
    """Escalate ở bước cuối = văn xuôi đã viết và đã trả tiền. Không được vứt."""
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_LLM", "fake")
    monkeypatch.setattr(cli, "run_chapter", lambda eng, ch: {
        "escalated": True, "escalation_reason": "extractor_node lỗi: giả lập",
        "scene_outputs": [{"scene_id": "CH001_S00",
                           "prose": "Văn xuôi đắt tiền không được mất."}]})
    assert cli.main(["write", "--chapter", "1"]) == 1
    p = tmp_path / "output" / "chapters" / "ch001.escalated.md"
    assert "Văn xuôi đắt tiền" in p.read_text(encoding="utf-8")
