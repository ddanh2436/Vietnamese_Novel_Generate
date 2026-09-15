"""Quyết định Ngày 8 — bộ từ vựng vị từ cố định (`bible/predicates.yaml`).

Lượt Gemini thật ở Chương 2 lách luật "snake_case không dấu" bằng cách bỏ dấu
các động từ kể sự kiện (`dung_cach = "6 met"`). Chỉ tập đóng mới bền.
"""
from __future__ import annotations

import pytest

from novel_engine.canon.bible import load_bible
from novel_engine.canon.flashback import PERMANENT_PREDICATES
from novel_engine.canon.models import Assertion, Entity, StateDelta, item_key
from novel_engine.canon.vocabulary import (
    PredicateSpec, Vocabulary, load_vocabulary, normalize_predicates,
)
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.reconcile.classify import classify_delta
from novel_engine.reconcile.commit import reconcile


@pytest.fixture
def vocab():
    v = load_vocabulary()
    assert v is not None
    return v


@pytest.fixture
def g():
    graph, _c, _m = load_bible()
    return graph


def _a(subject, predicate, obj, span, epistemic="objective", holder=None):
    return Assertion(subject=subject, predicate=predicate, object=obj, chapter=2,
                     scene=0, span=span, confidence=0.8, epistemic=epistemic,
                     holder=holder)


def _q(out):
    return {q["key"]: q["reason"] for q in out["quarantine"]}


# ═══════════════ BIBLE ═══════════════

def test_tap_vinh_vien_khop_voi_quy_tac_hoi_uc(vocab):
    """NT-11: hai nơi khai cùng một tập sẽ lệch nhau ở lần sửa đầu tiên."""
    assert vocab.permanent == PERMANENT_PREDICATES


def test_bi_danh_quy_ve_ten_chuan(vocab):
    assert vocab.resolve("owes_money") == "owes_debt"
    assert vocab.resolve("Owes-Money") == "owes_debt"
    assert vocab.resolve("owes money") == "owes_debt"
    assert vocab.resolve("dung_cach") is None


def test_khai_trung_vi_tu_hoac_bi_danh_thi_nem():
    with pytest.raises(ValueError, match="trùng"):
        Vocabulary.from_specs([PredicateSpec(name="rank"), PredicateSpec(name="rank")])
    with pytest.raises(ValueError, match="trùng"):
        Vocabulary.from_specs([PredicateSpec(name="rank", aliases=["grade"]),
                               PredicateSpec(name="grade")])


def test_bible_khong_co_file_thi_tra_none(tmp_path):
    assert load_vocabulary(tmp_path) is None


def test_danh_sach_cho_prompt_liet_ke_moi_vi_tu(vocab):
    text = vocab.render_for_prompt()
    for name in vocab.predicates:
        assert f"- {name} —" in text


# ═══════════════ PHÂN LOẠI ═══════════════

def test_du_lieu_gemini_that_chuong_2_khong_muc_nao_vao_canon(g, vocab):
    """Chín mệnh đề Gemini trả về (đã thêm `confidence`). Mệnh đề khách quan đều
    là lời kể sự kiện bỏ dấu → cách ly. Lời khai vẫn được ghi là lời khai."""
    su_kien = [
        ("CHAR_KAELEN", "an_ap suat khoang cho", "95 kilopascal",
         "Áp suất khí quyển trong khoang chờ đang tụt xuống mức chín mươi lăm kilopascal."),
        ("CHAR_KAELEN", "nghe am thanh", "Den huynh quang nhap nhay",
         "Kaelen nghe đèn huỳnh quang trên trần nhấp nháy"),
        ("CHAR_SERENA", "dung cach", "6 met",
         "Serena đứng cách đó sáu mét, sát vách ngăn của trạm kiểm soát quặng."),
        ("CHAR_KAELEN", "nhan dien", "Tieng dong co die-zen",
         "Kaelen nghiêng đầu sang trái, thu nhận tiếng động cơ đie-zen"),
        ("CHAR_KAELEN", "buoc vao", "Khoang hang so bon",
         "Kaelen bước qua khung cửa, tiến vào vùng tối của khoang"),
    ]
    khai = [_a("CHAR_SERENA", "noi ve", "Cac ban sao lưu",
               "Tuy nhiên, các bản sao lưu hiện tại", "claimed_by", "CHAR_SERENA")]
    d = StateDelta(assertions=[_a(*x) for x in su_kien] + khai).stamp(2)
    out = classify_delta(d, g, vocabulary=vocab)
    q = _q(out)
    for a in d.assertions[:5]:
        assert item_key(a) in q, a.predicate
        assert item_key(a) not in out["classification"]
    assert out["classification"][item_key(khai[0])] == "enrichment"


def test_vi_tu_hop_le_duoc_nhan(g, vocab):
    d = StateDelta(assertions=[
        _a("FACT_CIPHER", "controls", "LOC_ORE_PORT",
           "Tổng Cục Mật Văn nắm quyền điều phối tại đây"),
        _a("CHAR_KAELEN", "rank", "Vanguard hạng hai",
           "Kaelen từng là Vanguard hạng hai")]).stamp(2)
    out = classify_delta(d, g, vocabulary=vocab)
    assert all(v == "enrichment" for v in out["classification"].values())
    assert len(out["classification"]) == 2


@pytest.mark.parametrize("assertion, reason", [
    (("OBJ_X", "rank", "cao", "Vật X có hạng cao lắm"), "subject_kind_not_allowed"),
    (("FACT_CIPHER", "controls", "LOC_PIER_3", "Tổng Cục Mật Văn kiểm soát cầu tàu"),
     "unknown_object"),
    (("FACT_CIPHER", "controls", "CHAR_VHAL", "Tổng Cục Mật Văn kiểm soát Vhal"),
     "object_type_mismatch"),
    (("OBJ_X", "is_damaged", "có", "Vật X bị hỏng một góc"), "object_type_mismatch"),
    (("CHAR_KAELEN", "dung_cach", "6 met", "Kaelen đứng cách sáu mét"),
     "predicate_not_in_vocabulary"),
])
def test_vi_pham_tu_vung_bi_cach_ly(g, vocab, assertion, reason):
    d = StateDelta(
        new_entities=[Entity(id="OBJ_X", kind="object", name="Vật X")],
        assertions=[_a(*assertion)]).stamp(2)
    out = classify_delta(d, g, vocabulary=vocab)
    assert _q(out)[item_key(d.assertions[0])] == reason


def test_khong_truyen_tu_vung_thi_quay_ve_luat_cu(g):
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "bước qua", "cửa", "Kaelen bước qua cánh cửa")]).stamp(2)
    assert _q(classify_delta(d, g))[item_key(d.assertions[0])] == "non_canonical_predicate"


def test_loi_khai_khong_bi_kiem_tu_vung(g, vocab):
    d = StateDelta(assertions=[
        _a("FACT_CIPHER", "sap_giai_the", True, "— Tổng Cục sắp giải thể, — Vhal nói.",
           "claimed_by", "CHAR_VHAL")]).stamp(2)
    out = classify_delta(d, g, vocabulary=vocab)
    assert out["classification"][item_key(d.assertions[0])] == "enrichment"


def test_quy_bi_danh_truoc_khi_sinh_khoa(vocab):
    """Khoá của Assertion chứa `predicate`. Đổi tên sau khi phân loại thì
    `delta.item(k)` không tra được."""
    d = StateDelta(assertions=[
        _a("CHAR_VHAL", "owes_money", "CHAR_SERENA", "Vhal nợ Serena một khoản lớn")]).stamp(2)
    notes = normalize_predicates(d, vocab)
    assert d.assertions[0].predicate == "owes_debt"
    assert notes[0]["action"] == "coerced" and "owes_money" in notes[0]["reason"]
    d.item(item_key(d.assertions[0]))


# ═══════════════ TÍCH HỢP ═══════════════

def test_engine_nap_tu_vung_va_giu_sau_khi_dung_lai_canon():
    e = build_engines(FakeLLM(), db_path=":memory:")
    try:
        assert e.vocabulary is not None and "has_scar" in e.vocabulary.predicates
        e.rebuild_canon()
        assert e.vocabulary is not None
    finally:
        e.store.close()


def test_prompt_trich_xuat_nhan_danh_sach_vi_tu():
    llm = FakeLLM()
    e = build_engines(llm, db_path=":memory:")
    try:
        run_chapter(e, 1)
        for role in ("extractor_diff", "extractor_emergent"):
            p = next(c["prompt"] for c in llm.calls if c["role"] == role)
            assert "- has_scar —" in p and "{predicates}" not in p
    finally:
        e.store.close()


def test_ghi_canon_dung_tu_vung(g):
    e = build_engines(FakeLLM(), db_path=":memory:")
    try:
        d = StateDelta(assertions=[
            _a("CHAR_SERENA", "dung_cach", "6 met", "Serena đứng cách đó sáu mét"),
            _a("FACT_CIPHER", "controls", "LOC_ORE_PORT",
               "Tổng Cục Mật Văn nắm quyền điều phối tại đây")]).stamp(2)
        assert reconcile(d, e)["status"] == "committed"
        assert e.graph.truth_of("CHAR_SERENA", "dung_cach") is None
        assert e.graph.truth_of("FACT_CIPHER", "controls") == "LOC_ORE_PORT"
    finally:
        e.store.close()
