"""Ngày 7 — phân loại delta (§11).

Tiêu chí của ngày: lời nói dối (`claimed_by`) không bị xếp nhầm thành
`contradiction`. Phần lớn test còn lại dựng lại ĐÚNG những gì Gemini trả về
khi chạy thật Chương 1–2 — delta thật chứa những lỗi mà ba hạng của §11 không
đủ để chứa, nên có thêm hạng cách ly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from novel_engine.canon.bible import load_bible
from novel_engine.canon.models import Assertion, Entity, Relation, StateDelta, item_key
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.planner.outline_planner import OutlinePlanner
from novel_engine.reconcile.classify import (
    affects_downstream_plan, classify_delta, conflicts_with_locked_canon,
    is_objective, retract_key,
)


@pytest.fixture
def g():
    graph, _c, _m = load_bible()
    return graph


@pytest.fixture
def planner():
    return OutlinePlanner()


def _a(subject, predicate, obj, span, epistemic="objective", holder=None):
    return Assertion(subject=subject, predicate=predicate, object=obj,
                     chapter=1, scene=0, span=span, confidence=0.9,
                     epistemic=epistemic, holder=holder)


def _fr(sid, tick, dur, locs):
    return ContinuityFrame(
        scene_id=sid,
        time=StoryTime(epoch_tick=tick, duration_ticks=dur,
                       narrative_order=int(sid[-2:]) + 1),
        locations=locs)


# Frame thật của Chương 1 (Gemini, sau `_coerce_continuity`).
FRAMES_CH1 = [
    _fr("CH001_S00", 20000, 2, {"CHAR_KAELEN": "LOC_ORE_PORT"}),
    _fr("CH001_S01", 20006, 3, {"CHAR_KAELEN": "LOC_VEDA_CHECKPOINT",
                                "CHAR_VHAL": "LOC_VEDA_CHECKPOINT"}),
    _fr("CH001_S02", 20009, 2, {"CHAR_KAELEN": "LOC_VEDA_CHECKPOINT",
                                "CHAR_VHAL": "LOC_VEDA_CHECKPOINT"}),
    _fr("CH001_S03", 20011, 2, {"CHAR_VHAL": "LOC_VEDA_CHECKPOINT"}),
    _fr("CH001_S04", 20017, 3, {"CHAR_KAELEN": "LOC_ORE_PORT",
                                "CHAR_SERENA": "LOC_ORE_PORT"}),
    _fr("CH001_S05", 20020, 2, {"CHAR_KAELEN": "LOC_ORE_PORT"}),
]


def _q(out) -> dict[str, str]:
    return {q["key"]: q["reason"] for q in out["quarantine"]}


# ═══════════════ TIÊU CHÍ NGÀY 7 — NHÂN VẬT ĐƯỢC PHÉP NÓI DỐI ═══════════════

def test_loi_noi_doi_KHONG_bi_xep_thanh_mau_thuan(g):
    """§3.4 / §11 nguyên văn: Serena nói "Hạm Đội Số 3 đã bị giải tán". Nếu
    Serena nói dối, sự thật khách quan vẫn nguyên vẹn, và mâu thuẫn là CHỦ Ý.

    C5: bản trước không xét `epistemic` → Extractor gán đúng `claimed_by` →
    `conflicts_with_locked_canon` vẫn thấy lệch canon → `contradiction` →
    escalate và dừng chương. Trong hệ thống cũ, không nhân vật nào được phép
    nói dối."""
    g.commit_truth("FLEET_3", "status", "active", tick=0)
    d = StateDelta(assertions=[
        _a("FLEET_3", "status", "disbanded", "— Hạm Đội Số 3 đã bị giải tán, — Serena nói.",
           epistemic="claimed_by", holder="CHAR_SERENA")]).stamp(1)
    out = classify_delta(d, g)
    k = item_key(d.assertions[0])
    assert out["classification"][k] == "enrichment"
    assert out["summary"]["contradictions"] == 0
    # …và chênh lệch được GHI NHẬN làm nguyên liệu mỉa mai kịch tính (M12)
    assert out["irony_seeds"][0]["holder"] == "CHAR_SERENA"
    assert out["irony_seeds"][0]["claim"] == "disbanded"


@pytest.mark.parametrize("epistemic, holder, expected", [
    ("objective", None, "contradiction"),     # người KỂ khẳng định → phá canon
    ("claimed_by", "CHAR_SERENA", "enrichment"),
    ("believed_by", "CHAR_KAELEN", "enrichment"),
])
def test_cung_menh_de_khac_epistemic_khac_hang(g, epistemic, holder, expected):
    """Cùng subject/predicate/object, chỉ khác người đứng sau mệnh đề."""
    g.commit_truth("FACT_CIPHER", "status", "active", tick=0)
    d = StateDelta(assertions=[
        _a("FACT_CIPHER", "status", "dissolved",
           "Tổng Cục Mật Văn đã bị giải thể từ mùa trước",
           epistemic=epistemic, holder=holder)]).stamp(1)
    out = classify_delta(d, g)
    assert out["classification"][item_key(d.assertions[0])] == expected


def test_cong_epistemic_dung_TRUOC_kiem_tra_vi_tri(g):
    """Nếu kiểm vị trí trước, lời nói dối về vị trí ("Kaelen đang ở cầu tàu số
    ba") bị cách ly vì `LOC_PIER_3` không tồn tại — nhân vật lại không được
    nói dối, chỉ là theo đường khác (C5)."""
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "at", "LOC_PIER_3",
           "— Kaelen đang ở cầu tàu số ba, — Serena nói.",
           epistemic="claimed_by", holder="CHAR_SERENA")]).stamp(1)
    out = classify_delta(d, g, frames=FRAMES_CH1)
    assert out["classification"][item_key(d.assertions[0])] == "enrichment"
    assert out["quarantine"] == []


def test_niem_tin_khong_co_nguoi_tin_bi_cach_ly(g):
    """`reconcile_node` §11 gặp mục này sẽ `if item.holder:` rồi `continue`
    — bỏ im lặng. Ở đây nó được đưa ra cho tác giả thấy."""
    d = StateDelta(assertions=[
        _a("FLEET_3", "status", "disbanded", "Hạm Đội Số 3 đã bị giải tán rồi",
           epistemic="believed_by", holder=None)]).stamp(1)
    out = classify_delta(d, g)
    assert _q(out)[item_key(d.assertions[0])] == "belief_without_holder"


def test_is_objective_la_nguon_duy_nhat():
    """NT-11: một quy tắc, một hàm. §10.2 (GĐ2) phải gọi đúng hàm này."""
    assert is_objective(_a("S", "p", 1, "x" * 20)) is True
    assert is_objective(_a("S", "p", 1, "x" * 20, "claimed_by", "H")) is False


# ═══════════════ DỮ LIỆU GEMINI THẬT — CHƯƠNG 1 ═══════════════

def test_du_lieu_gemini_that_chuong_1(g, planner):
    """Dựng lại đúng delta Gemini trả về khi chạy Chương 1. Không mục nào là
    mâu thuẫn canon, nhưng ghi hết vào canon thì hỏng."""
    d = StateDelta(
        assertions=[
            _a("CHAR_KAELEN", "at", "LOC_REACTOR_3",
               "Ba mươi lăm ngày lưu đày ngoại biên tính từ thời điểm niêm "
               "phong Lò Phản ứng Số 3."),
            _a("CHAR_VHAL", "located_in", "LOC_ORE_PORT",
               "Viên quan quản hạt đứng sau vách kính mờ, ngón tay cái miết "
               "đều lên mép chiếc thước kẻ"),
            _a("FACT_CIPHER", "controls", "LOC_ORE_PORT",
               "Tổng Cục Mật Văn nắm giữ bốn mươi lăm phần trăm quyền điều "
               "phối năng lượng tại đây"),
            _a("CHAR_SERENA", "at", "LOC_PIER_3",
               "Serena vừa rời đi hướng cầu tàu số ba, mang theo bản sao thứ "
               "hai của lệnh điều động"),
        ],
        new_entities=[
            Entity(id="OBJ_PERMIT_73", kind="object", name="Mẫu bảy ba"),
            Entity(id="LOC_PIER_3", kind="location", name="cầu tàu số ba"),
            Entity(id="LOC_BAY_9", kind="location", name="khoang số chín"),
        ]).stamp(1)
    out = classify_delta(d, g, planner, FRAMES_CH1)
    q, a = _q(out), d.assertions

    # Span CÓ THẬT nhưng chỉ nhắc tới Lò 3; frame nói Kaelen ở Cảng/Trạm Chốt.
    assert q[item_key(a[0])] == "contradicts_frames"
    # Vết drift từ Ngày 4: văn xuôi đưa Vhal ra cảng, frame giữ Vhal ở chốt.
    assert q[item_key(a[1])] == "contradicts_frames"
    # Người kể khẳng định, span gọi đúng tên phe → bổ sung hợp lệ.
    assert out["classification"][item_key(a[2])] == "enrichment"
    # Địa điểm con bịa ra.
    assert q[item_key(a[3])] == "unknown_location"
    assert q["E:LOC_PIER_3"] == "location_without_route"
    assert q["E:LOC_BAY_9"] == "location_without_route"
    # Dàn ý Chương 4 dùng "mẫu bảy ba" làm đạo cụ, nên vật này ĐỤNG kế hoạch hạ
    # nguồn và thuộc hạng improvement (CP-4), không phải enrichment.
    assert out["classification"]["E:OBJ_PERMIT_73"] == "improvement"
    assert out["summary"]["contradictions"] == 0


def test_dia_diem_con_neu_ghi_vao_graph_se_thanh_node_co_lap(g):
    """Lý do `location_without_route` là `major`, không chỉ là ghi chú."""
    g.upsert_entity(Entity(id="LOC_PIER_3", kind="location", name="cầu tàu số ba"))
    assert g.travel_ticks("LOC_ORE_PORT", "LOC_PIER_3") is None


def test_vi_tri_khop_frame_thi_khong_ghi_lai(g):
    """Khớp frame vẫn không commit: vị trí thay đổi theo thời gian, còn
    `_truth[(subject, predicate)]` chỉ giữ MỘT giá trị và sẽ ghi đè lịch sử."""
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "at", "LOC_ORE_PORT",
           "Kaelen dừng lại ở ranh giới khoang bốc dỡ")]).stamp(1)
    out = classify_delta(d, g, frames=FRAMES_CH1)
    assert _q(out)[item_key(d.assertions[0])] == "owned_by_frames"
    assert out["classification"] == {}


def test_vi_tri_chi_doi_chieu_frame_CUA_CHUONG_DO(g):
    frames_ch2 = [_fr("CH002_S00", 20028, 2, {"CHAR_KAELEN": "LOC_REACTOR_3"})]
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "at", "LOC_REACTOR_3", "Kaelen bước vào khoang Lõi")]).stamp(1)
    out = classify_delta(d, g, frames=frames_ch2)
    assert _q(out)[item_key(d.assertions[0])] == "no_frames_to_check"


def test_span_co_that_nhung_khong_nhac_chu_the(g):
    """Lượt 3 chỉ chứng minh span CÓ THẬT, không chứng minh span ĐỠ ĐƯỢC mệnh
    đề. Đây là ca Kaelen/Lò 3, nhưng với một vị từ không phải vị trí."""
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "has_scar", True,
           "Ba mươi lăm ngày lưu đày ngoại biên tính từ thời điểm niêm phong "
           "Lò Phản ứng Số 3.")]).stamp(1)
    out = classify_delta(d, g)
    assert _q(out)[item_key(d.assertions[0])] == "span_does_not_mention_subject"


def test_vi_tu_ke_su_kien_bi_cach_ly(g):
    """Dữ liệu thật Chương 2: `CHAR_KAELEN bước qua = cánh cửa thép`."""
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "bước qua", "cánh cửa thép",
           "Kaelen bước qua cánh cửa thép nặng bốn tấn")]).stamp(1)
    out = classify_delta(d, g)
    assert _q(out)[item_key(d.assertions[0])] == "non_canonical_predicate"


def test_chu_the_khong_ton_tai_bi_cach_ly(g):
    d = StateDelta(assertions=[
        _a("FLEET_9", "status", "active", "Hạm Đội Số 9 vẫn neo ngoài vịnh")]).stamp(1)
    assert _q(classify_delta(d, g))[item_key(d.assertions[0])] == "unknown_subject"


def test_menh_de_ve_thuc_the_moi_duoc_chap_nhan(g):
    d = StateDelta(
        new_entities=[Entity(id="OBJ_SO_CA_TRUC", kind="object", name="sổ ca trực")],
        assertions=[_a("OBJ_SO_CA_TRUC", "is_damaged", True,
                       "Sổ ca trực bị xé mất hai trang cuối")]).stamp(1)
    out = classify_delta(d, g)
    assert out["classification"][item_key(d.assertions[0])] == "enrichment"


def test_menh_de_ve_thuc_the_bi_cach_ly_cung_bi_cach_ly(g):
    d = StateDelta(
        new_entities=[Entity(id="LOC_BAY_9", kind="location", name="khoang số chín")],
        assertions=[_a("LOC_BAY_9", "is_sealed", True, "Khoang số chín đã bị niêm phong")]).stamp(1)
    assert _q(classify_delta(d, g))[item_key(d.assertions[0])] == "subject_quarantined"


# ═══════════════ THỰC THỂ ═══════════════

def test_thuc_the_da_co_cung_loai_thi_khong_ghi_lai(g):
    d = StateDelta(new_entities=[
        Entity(id="CHAR_VHAL", kind="character", name="Vhal")]).stamp(1)
    assert _q(classify_delta(d, g))["E:CHAR_VHAL"] == "already_known"


def test_thuc_the_da_co_khac_loai_la_mau_thuan(g):
    d = StateDelta(new_entities=[
        Entity(id="CHAR_VHAL", kind="object", name="Vhal")]).stamp(1)
    out = classify_delta(d, g)
    assert out["classification"]["E:CHAR_VHAL"] == "contradiction"


# ═══════════════ QUAN HỆ & KHOÁ ═══════════════

def test_is_locked_khoa_QUAN_HE_khong_khoa_theo_dau_mut(g):
    """Bản Ngày 2 coi mọi quan hệ chạm tới thực thể `canon_locked` là khoá.
    Mọi nhân vật bible đều bị khoá → Serena không bao giờ phản bội được Kaelen."""
    assert g.entity_locked("CHAR_SERENA") and g.entity_locked("CHAR_KAELEN")
    g.upsert_relation(Relation(src="CHAR_SERENA", dst="CHAR_KAELEN",
                               type="PROTECTS", provenance="extracted_ch1"))
    assert g.is_locked("CHAR_SERENA", "CHAR_KAELEN", "PROTECTS") is False
    assert g.is_locked("CHAR_SERENA", "FACT_CIPHER", "MEMBER_OF") is True


def test_phan_boi_quan_he_TAC_GIA_dat_la_mau_thuan(g):
    g.upsert_relation(Relation(src="CHAR_SERENA", dst="CHAR_KAELEN",
                               type="PROTECTS", provenance="author"))
    r = Relation(src="CHAR_SERENA", dst="CHAR_KAELEN", type="BETRAYED")
    assert conflicts_with_locked_canon(r, g) is True
    out = classify_delta(StateDelta(new_relations=[r]).stamp(1), g)
    assert out["classification"][item_key(r)] == "contradiction"


def test_phan_boi_quan_he_do_EXTRACTOR_ghi_la_dien_bien(g, planner):
    """Serena che chở Kaelen ở chương 1 (Extractor ghi), phản bội ở chương 2:
    đó là cốt truyện, không phải mâu thuẫn canon."""
    g.upsert_relation(Relation(src="CHAR_SERENA", dst="CHAR_KAELEN",
                               type="PROTECTS", provenance="extracted_ch1"))
    r = Relation(src="CHAR_SERENA", dst="CHAR_KAELEN", type="BETRAYED")
    out = classify_delta(StateDelta(new_relations=[r]).stamp(1), g, planner)
    assert out["classification"][item_key(r)] != "contradiction"


def test_quan_he_dau_mut_khong_ton_tai_bi_cach_ly(g):
    r = Relation(src="CHAR_KAELEN", dst="CHAR_EM_GAI", type="PROTECTS")
    out = classify_delta(StateDelta(new_relations=[r]).stamp(1), g)
    assert _q(out)[item_key(r)] == "dangling_endpoint"


def test_located_in_cua_nhan_vat_thuoc_ve_frame(g):
    r = Relation(src="CHAR_KAELEN", dst="LOC_ORE_PORT", type="LOCATED_IN")
    out = classify_delta(StateDelta(new_relations=[r]).stamp(1), g)
    assert _q(out)[item_key(r)] == "owned_by_frames"


# ═══════════════ THU HỒI QUAN HỆ — F7 ═══════════════

def test_thu_hoi_quan_he_tac_gia_dat_la_mau_thuan(g):
    r = Relation(src="CHAR_SERENA", dst="FACT_CIPHER", type="MEMBER_OF")
    out = classify_delta(StateDelta(retracted_relations=[r]).stamp(1), g)
    assert out["retractions"][retract_key(r)] == "contradiction"
    assert out["summary"]["contradictions"] == 1


def test_thu_hoi_quan_he_extractor_ghi_la_hop_le(g):
    g.upsert_relation(Relation(src="CHAR_VHAL", dst="CHAR_KAELEN",
                               type="PROTECTS", provenance="extracted_ch1"))
    r = Relation(src="CHAR_VHAL", dst="CHAR_KAELEN", type="PROTECTS")
    out = classify_delta(StateDelta(retracted_relations=[r]).stamp(2), g)
    assert out["retractions"][retract_key(r)] == "enrichment"


def test_thu_hoi_quan_he_da_dong_hoac_khong_co(g):
    """Kaelen MEMBER_OF Lõi Rạng đã đóng ở chương 0 (lưu đày)."""
    r1 = Relation(src="CHAR_KAELEN", dst="FACT_ARCLIGHT", type="MEMBER_OF")
    r2 = Relation(src="CHAR_KAELEN", dst="CHAR_VHAL", type="OWES_DEBT_TO")
    out = classify_delta(StateDelta(retracted_relations=[r1, r2]).stamp(1), g)
    assert _q(out)[retract_key(r1)] == "retract_missing"
    assert _q(out)[retract_key(r2)] == "retract_missing"


def test_vua_them_vua_thu_hoi_cung_quan_he(g):
    r = Relation(src="CHAR_VHAL", dst="CHAR_KAELEN", type="PROTECTS")
    out = classify_delta(StateDelta(new_relations=[r],
                                    retracted_relations=[r]).stamp(1), g)
    q = _q(out)
    assert q[item_key(r)] == "added_and_retracted"
    assert q[retract_key(r)] == "added_and_retracted"


# ═══════════════ IMPROVEMENT — đụng kế hoạch chương sau ═══════════════

def test_thuc_the_moi_duoc_dan_y_chuong_sau_nhac_toi(planner):
    """Dàn ý Chương 2: "Serena nhận ra thẻ định vị đã dừng di chuyển"."""
    assert affects_downstream_plan(
        Entity(id="OBJ_LOCATION_TAG", kind="object", name="Thẻ định vị"),
        planner, chapter=1) is True
    # Vật không chương nào nhắc tới thì không đụng kế hoạch nào.
    assert affects_downstream_plan(
        Entity(id="OBJ_COI_SUONG_MU", kind="object", name="còi sương mù"),
        planner, chapter=1) is False


def test_su_that_vinh_vien_ve_nhan_vat_con_xuat_hien(planner):
    """Ví dụ "Kaelen có một người em gái" của §11."""
    scar = _a("CHAR_KAELEN", "has_scar", True, "Kaelen có vết sẹo chéo trên trán")
    assert affects_downstream_plan(scar, planner, chapter=1) is True
    thuong = _a("CHAR_KAELEN", "is_tired", True, "Kaelen mệt rã rời sau ca trực")
    assert affects_downstream_plan(thuong, planner, chapter=1) is False


def test_chuong_cuoi_khong_co_gi_de_replan(planner):
    scar = _a("CHAR_KAELEN", "has_scar", True, "Kaelen có vết sẹo chéo trên trán")
    # Chương cuối của dàn ý (Arc 1 kết ở Chương 5) — không còn chương nào phía sau.
    assert affects_downstream_plan(scar, planner, chapter=5) is False
    assert affects_downstream_plan(scar, None, chapter=1) is False


def test_classify_xep_improvement_khi_dung_ke_hoach(g, planner):
    d = StateDelta(
        new_entities=[Entity(id="OBJ_LOCATION_TAG", kind="object", name="Thẻ định vị")],
        assertions=[_a("CHAR_KAELEN", "has_scar", True,
                       "Kaelen có vết sẹo chéo trên trán")]).stamp(1)
    out = classify_delta(d, g, planner)
    assert out["classification"]["E:OBJ_LOCATION_TAG"] == "improvement"
    assert out["classification"][item_key(d.assertions[0])] == "improvement"


# ═══════════════ BẤT BIẾN ═══════════════

def test_moi_muc_ket_thuc_o_DUNG_MOT_cho(g, planner):
    """Không mục nào bị bỏ im lặng."""
    d = StateDelta(
        new_entities=[Entity(id="OBJ_X", kind="object", name="X"),
                      Entity(id="LOC_BAY_9", kind="location", name="khoang 9")],
        new_relations=[Relation(src="CHAR_VHAL", dst="CHAR_KAELEN", type="PROTECTS"),
                       Relation(src="CHAR_KAELEN", dst="CHAR_AI_DO", type="BETRAYED")],
        retracted_relations=[Relation(src="CHAR_SERENA", dst="FACT_CIPHER",
                                      type="MEMBER_OF")],
        assertions=[
            _a("CHAR_KAELEN", "at", "LOC_REACTOR_3", "Kaelen đứng trước Lò 3"),
            _a("FLEET_3", "status", "gone", "Hạm đội số ba đã biến mất",
               "claimed_by", "CHAR_VHAL"),
            _a("CHAR_KAELEN", "bước qua", "cửa", "Kaelen bước qua cánh cửa"),
        ]).stamp(1)
    out = classify_delta(d, g, planner, FRAMES_CH1)
    keys = ([item_key(x) for x in d.new_entities + d.new_relations + d.assertions]
            + [retract_key(x) for x in d.retracted_relations])
    where = {k: [] for k in keys}
    for k in out["classification"]:
        where[k].append("cls")
    for k in out["retractions"]:
        where[k].append("retr")
    for q in out["quarantine"]:
        where[q["key"]].append("q")
    assert all(len(v) == 1 for v in where.values()), where


def test_moi_khoa_phan_loai_tra_nguoc_duoc_bang_delta_item(g, planner):
    """`reconcile_node` (GĐ3) gọi `delta.item(k)` cho từng khoá."""
    d = StateDelta(
        new_entities=[Entity(id="OBJ_X", kind="object", name="X")],
        new_relations=[Relation(src="CHAR_VHAL", dst="CHAR_KAELEN", type="PROTECTS")],
        assertions=[_a("FACT_CIPHER", "controls", "LOC_ORE_PORT",
                       "Tổng Cục Mật Văn nắm quyền điều phối tại đây")]).stamp(1)
    out = classify_delta(d, g, planner)
    assert out["classification"] and d.classification == out["classification"]
    for k in out["classification"]:
        d.item(k)                     # không ném KeyError


def test_item_key_re_export_dung_mot_ham():
    """NT-11: `reconcile.classify.item_key` là CHÍNH hàm của canon.models."""
    from novel_engine.canon import models
    from novel_engine.reconcile import classify
    assert classify.item_key is models.item_key


def test_prompt_trich_xuat_khong_con_hoi_vi_tri():
    """§12.3 CÂU 2 hỏi Extractor về vị trí trong khi §12.4 đã giao vị trí cho
    `SceneClose`. Hai nguồn cho một sự thật — Gemini trả về 3 mệnh đề vị trí
    thì 2 mâu thuẫn với frame."""
    from novel_engine.prompts import EXTRACT_DIFF_TMPL, EXTRACT_EMERGENT_TMPL
    assert "TRẠNG THÁI THỰC TẾ SAU CẢNH" not in EXTRACT_DIFF_TMPL
    assert "CÂU 3" not in EXTRACT_DIFF_TMPL
    assert "KHÔNG xuất assertion về VỊ TRÍ" in EXTRACT_DIFF_TMPL
    assert "KHÔNG xuất mệnh đề về vị trí" in EXTRACT_EMERGENT_TMPL


# ═══════════════ TÍCH HỢP ═══════════════

def test_chuong_fake_khong_co_mau_thuan():
    from novel_engine.graph.build import run_chapter
    from novel_engine.graph.engines import build_engines
    from novel_engine.llm.fake import FakeLLM
    eng = build_engines(FakeLLM(), db_path=":memory:")
    try:
        st = run_chapter(eng, 1)
        d = StateDelta.model_validate(st["delta"])
        out = classify_delta(d, eng.graph, eng.planner, st["frames"])
        assert out["summary"]["contradictions"] == 0
        # "sổ ca trực" là đạo cụ của dàn ý Chương 3 → đụng kế hoạch hạ nguồn.
        assert out["classification"]["E:OBJ_SO_CA_TRUC"] == "improvement"
    finally:
        eng.store.close()


ROOT = Path(__file__).resolve().parents[1]
_ENV = {**os.environ, "NOVEL_LLM": "fake", "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(ROOT)}


def _cli(*args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          env=_ENV, cwd=str(cwd))


@pytest.mark.slow
def test_cli_classify(tmp_path):
    assert _cli("classify", "--chapter", "1", cwd=tmp_path).returncode == 2
    assert _cli("write", "--chapter", "1", cwd=tmp_path).returncode == 0
    r = _cli("classify", "--chapter", "1", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Phân loại d_ch001" in r.stdout
    rep = json.loads((tmp_path / "output" / "reports" / "ch001_classify.json")
                     .read_text(encoding="utf-8"))
    assert rep["delta_id"] == "d_ch001"
    assert rep["classification"]["E:OBJ_SO_CA_TRUC"] == "improvement"
