"""Hồi quy từ lượt Gemini thật ở Chương 2 (bản prompt mới):

- 4.692 từ văn xuôi → 0 mệnh đề, 0 thực thể; delta rỗng được GHI vào canon
  không một lời cảnh báo.
- Lượt 1 bịa `plant_evidence` với clue_id là mã cảnh (`CH002_S00`) cho cả sáu
  cảnh, vì phần kế hoạch BỎ dòng `plant_directives` khi cảnh không có manh mối.
- Phản hồi thô của model không được lưu, nên không chẩn đoán được vì sao rỗng.
"""
from __future__ import annotations

import json

from novel_engine.canon.models import StateDelta
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.graph.nodes import _contracts_brief_for_extract
from novel_engine.llm.fake import FakeLLM

PHANTOM_LUOT_1 = json.dumps({"plant_evidence": [
    {"clue_id": f"CH001_S0{i}", "scene_id": f"CH001_S0{i}",
     "span": "Kaelen cúi xuống nhét thiết bị vào khe hở", "carrier_used": "object",
     "concluded_by": "CHAR_KAELEN"}            # sai kiểu: phải là list
    for i in range(6)]}, ensure_ascii=False)


class _GeminiRong(FakeLLM):
    """Lượt 1 chỉ bịa bằng chứng ma, lượt 2 trả rỗng — đúng như Gemini đã làm."""

    def invoke(self, prompt: str, *, role: str = "") -> str:
        if role == "extractor_diff":
            self.calls.append({"role": role, "prompt": prompt})
            return PHANTOM_LUOT_1
        if role == "extractor_emergent":
            self.calls.append({"role": role, "prompt": prompt})
            return "{}"
        return super().invoke(prompt, role=role)


def test_ke_hoach_luon_ghi_ro_khi_canh_khong_co_manh_moi():
    brief = _contracts_brief_for_extract([{
        "scene_id": "CH002_S00", "pov_character": "CHAR_KAELEN",
        "location": "LOC_ORE_PORT", "location_id": "LOC_ORE_PORT",
        "scene_must_change": "x", "plant_directives": [],
        "active_characters": []}])
    assert "plant_directives: (không có)" in brief


def test_prompt_cam_dung_ma_canh_lam_clue_id():
    from novel_engine.prompts import EXTRACT_DIFF_TMPL
    assert "mã cảnh dạng" in EXTRACT_DIFF_TMPL
    assert "KHÔNG xuất `plant_evidence`" in EXTRACT_DIFF_TMPL


def test_prompt_luot_1_thay_dong_khong_co_manh_moi():
    llm = FakeLLM()
    e = build_engines(llm, db_path=":memory:")
    try:
        out = run_chapter(e, 1)
        from novel_engine.prompts import EXTRACT_DIFF_TMPL
        p = next(c["prompt"] for c in llm.calls if c["role"] == "extractor_diff")
        dong = "plant_directives: (không có)"
        # Từ Ngày 11 Scheduler cài manh mối thật, nên chỉ cảnh KHÔNG được giao
        # mới mang dòng này — mỗi cảnh như vậy đúng một lần.
        khong_co = sum(1 for c in out["contracts"] if not c["plant_directives"])
        assert 0 < khong_co < 6
        # Chính câu quy tắc trong template cũng chứa chuỗi này một lần.
        assert p.count(dong) - EXTRACT_DIFF_TMPL.count(dong) == khong_co
    finally:
        e.store.close()


def test_trich_xuat_rong_duoc_ghi_vao_delta_va_bao_cao():
    e = build_engines(_GeminiRong(), db_path=":memory:")
    try:
        st = run_chapter(e, 1)
        assert not st.get("escalated"), st.get("escalation_reason")
        d = StateDelta.model_validate(st["delta"])
        reasons = {(x["stage"], x["reason"]) for x in d.extraction_issues}
        assert ("recall", "empty_extraction") in reasons
        # Từ lượt Arc 1: giá trị đơn ở chỗ cần danh sách được ÉP KIỂU, không bị
        # loại — nên mục hỏng của bản giả này hiện ra dưới nhãn `coerced`.
        assert any(st == "parse" for st, _ in reasons)
        rep = st["extraction_report"]
        assert rep["raw"]["luot_2"] == "{}"
        assert "CH001_S00" in rep["raw"]["luot_1"]
    finally:
        e.store.close()


def test_chuong_ngan_khong_bi_bao_rong():
    """Ngưỡng 500 từ: một cảnh chuyển tiếp ngắn không có sự thật mới là bình
    thường, không phải lỗi độ phủ."""
    from novel_engine.graph.nodes import EMPTY_EXTRACTION_MIN_WORDS
    e = build_engines(_GeminiRong(), db_path=":memory:")
    e.llm.word_target = 30
    try:
        st = run_chapter(e, 1)
        total = sum(len(s["prose"].split()) for s in st["scene_outputs"])
        d = StateDelta.model_validate(st["delta"])
        rong = any(x["reason"] == "empty_extraction" for x in d.extraction_issues)
        assert rong is (total >= EMPTY_EXTRACTION_MIN_WORDS)
    finally:
        e.store.close()


def test_review_hien_canh_bao_trich_xuat(tmp_path, capsys):
    """CP-2 đọc DELTA. Vấn đề chỉ nằm trong báo cáo của `write` thì tác giả có
    thể không bao giờ thấy."""
    import cli
    db = str(tmp_path / "novel_storage.db")
    e = build_engines(_GeminiRong(), db_path=db)
    try:
        st = run_chapter(e, 1)
        e.store.append_delta(StateDelta.model_validate(st["delta"]))
    finally:
        e.store.close()
    capsys.readouterr()
    assert cli.main(["review", "--chapter", "1", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "TRÍCH XUẤT" in out and "recall:empty_extraction" in out


def test_prompt_luot_2_doi_confidence():
    """Gemini thật Chương 2: 9/9 mệnh đề lượt 2 bị loại vì thiếu `confidence` —
    prompt lượt 1 có đòi trường này, prompt lượt 2 thì quên."""
    from novel_engine.prompts import EXTRACT_EMERGENT_TMPL
    assert "`confidence` từ 0" in EXTRACT_EMERGENT_TMPL


def test_prompt_cam_dong_tu_ke_su_kien_bo_dau():
    """Model tuân thủ "snake_case không dấu" theo nghĩa đen: bỏ dấu các động từ
    kể sự kiện (`buoc_vao`, `dung_cach`) và lọt qua bộ lọc vị từ."""
    from novel_engine.prompts import EXTRACT_DIFF_TMPL, EXTRACT_EMERGENT_TMPL
    # Quyết định Ngày 8: thay luật snake_case bằng TẬP ĐÓNG từ bible.
    for t in (EXTRACT_DIFF_TMPL, EXTRACT_EMERGENT_TMPL):
        assert "DANH SÁCH VỊ TỪ HỢP LỆ" in t and "buoc_vao" in t
        assert "{predicates}" in t


def test_menh_de_thieu_confidence_bi_loai_khong_bi_gan_mac_dinh():
    """Cố ý KHÔNG đặt mặc định: `confidence` là phán đoán của model, và chín
    mệnh đề bị loại đều là lời kể sự kiện bỏ dấu — cứu chúng là nhét rác vào
    canon. Việc loại được hiện ở CP-2."""
    from novel_engine.reconcile.lenient import parse_delta_lenient
    raw = json.dumps({"assertions": [{
        "subject": "CHAR_SERENA", "predicate": "dung cach", "object": "6 met",
        "epistemic": "objective",
        "span": "Serena đứng cách đó sáu mét, sát vách ngăn"}]}, ensure_ascii=False)
    d, issues = parse_delta_lenient(raw)
    assert d.assertions == []
    assert "confidence" in issues[0]["error"]
