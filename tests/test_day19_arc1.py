"""Ngày 19–20 — những gì lượt chạy Arc 1 thật (Gemini, 5 chương) dạy ra.

Hai lỗi dưới đây không test nào trước đó bắt được, vì FakeLLM trả JSON đúng
schema còn model thật thì không, và vì judge tiếng Việt trả lời theo cách mà
phép kiểm bằng `startswith` hiểu ngược.
"""
from __future__ import annotations

import json

from novel_engine.canon.models import StateDelta
from novel_engine.reconcile.lenient import parse_delta_lenient
from novel_engine.reconcile.verify import plan_coverage, verify_spans

PROSE = ("Số hiệu ca trực nhảy một quãng, rồi quay lại như chưa có gì. "
         "Kaelen gấp sổ lại và không nói gì thêm.")
RAW = json.dumps({"plant_evidence": [{
    "clue_id": "CLUE_MISSING_LOGS", "scene_id": "CH003_S00",
    "span": "Số hiệu ca trực nhảy một quãng, rồi quay lại như chưa có gì.",
    "carrier_used": "object",
    "concluded_by": "CHAR_KAELEN"}]}, ensure_ascii=False)   # chuỗi, không phải danh sách


def test_gia_tri_don_o_cho_can_danh_sach_duoc_EP_KIEU_khong_bi_loai():
    """Lượt Arc 1: Gemini trả `concluded_by` là một chuỗi. Cả mục `plant_evidence`
    bị loại, nên manh mối ĐÃ cài và ĐÃ trích xuất vẫn bị tính là trượt — 0/2 ở cả
    năm chương, M11 = 0.0, nợ manh mối vượt ngưỡng."""
    d, issues = parse_delta_lenient(RAW, source="luot_1")
    assert len(d.plant_evidence) == 1
    assert d.plant_evidence[0].concluded_by == ["CHAR_KAELEN"]
    assert [i["action"] for i in issues] == ["coerced"]
    assert "concluded_by" in issues[0]["reason"]


def test_muc_sai_that_su_van_bi_loai():
    """Ép kiểu chỉ chữa MỘT loại sai. Thiếu trường bắt buộc vẫn là mục hỏng."""
    raw = json.dumps({"plant_evidence": [{"clue_id": "C", "scene_id": "S"}]})
    d, issues = parse_delta_lenient(raw)
    assert d.plant_evidence == [] and issues[0]["action"] == "dropped"


def test_manh_moi_di_tron_duong_tu_parse_toi_do_phu_ke_hoach():
    d, _ = parse_delta_lenient(RAW)
    verify_spans(d, PROSE)
    assert d.plant_evidence[0].verified is True
    cov = plan_coverage([{"scene_id": "CH003_S00", "plant_directives": [
        {"clue_id": "CLUE_MISSING_LOGS", "mode": "plant", "intensity": 0.4,
         "surface_form": "số hiệu ca trực nhảy một quãng"}]}], d)
    assert cov["fulfilled_plants"] == ["CLUE_MISSING_LOGS"]
    assert cov["plan_fulfillment_rate"] == 1.0
    assert d.plant_modes == {"CLUE_MISSING_LOGS": "plant"}


def test_span_bia_van_bi_loai_sau_khi_ep_kieu():
    raw = json.dumps({"plant_evidence": [{
        "clue_id": "C", "scene_id": "S", "span": "câu này không có trong văn bản",
        "carrier_used": "object", "concluded_by": "CHAR_KAELEN"}]}, ensure_ascii=False)
    d, _ = parse_delta_lenient(raw)
    _, rejected = verify_spans(d, PROSE)
    assert d.plant_evidence[0].verified is False
    assert rejected and rejected[0]["reason"].startswith("plant_")


def test_delta_rong_van_parse_duoc():
    d, issues = parse_delta_lenient("{}")
    assert isinstance(d, StateDelta) and issues == []
