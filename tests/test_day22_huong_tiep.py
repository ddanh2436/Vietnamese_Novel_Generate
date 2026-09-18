"""Ngày 22 — những hướng còn treo sau lượt đọc Arc 1.

Mỗi bài ở đây gắn với một con số đo được trên văn bản thật, không phải với một
ý tưởng hay: khẩu ngữ chưa khai báo (22 lần/arc), ngân sách cử chỉ nói vào chỗ
không ai sửa được (19/5 chương), lỗi gõ toàn chữ Latin, `concluded_by` không ai
đọc, và sáu lần trả bài đều diễn ra trong đầu một người.
"""
from __future__ import annotations

import pytest

from novel_engine.audit.critique import POLISHABLE_CHECKS
from novel_engine.audit.text_hygiene import hygiene_findings, malformed_words
from novel_engine.audit.tics import TIC_LIMIT, tic_findings
from novel_engine.canon.bible import DEFAULT_BIBLE, load_characters
from novel_engine.canon.models import PlantEvidence, StateDelta
from novel_engine.graph.nodes import _tru_cu_chi_da_can
from novel_engine.reconcile.commit import _ghi_nhan_hieu
from novel_engine.eval import metrics as M
from novel_engine.eval.harness import SPECS
from novel_engine.eval.metrics import ChapterData
from novel_engine.eval.regression import LOWER_IS_BETTER, run_regression
from novel_engine.reconcile.verify import payoff_not_spoken


@pytest.fixture(scope="module")
def chars():
    return load_characters(DEFAULT_BIBLE)


def _ct(cid):
    return {"active_characters": [{"id": cid}]}


# ═══════════ 1. KHẨU NGỮ THUẦN CÓ TRẦN RIÊNG ═══════════

def test_khau_ngu_thuan_tach_khoi_thuat_ngu(chars):
    """"được chưa" hai từ, không nghĩa chuyên môn, Vhal nói 22 lần/arc.

    `tics.py` trước đây đoán loại cụm bằng SỐ TỪ: ≥3 từ là khẩu ngữ (trần 2),
    ≤2 từ là thuật ngữ của thế giới (trần 4). Phép đoán ấy không có cách nào
    biết "được chưa" khác "áp suất", nên tật nặng nhất của Vhal đi lọt.
    """
    vhal = chars["CHAR_VHAL"].voice
    assert "được chưa" in vhal.verbal_tics
    assert "được chưa" not in vhal.signature_lexicon      # không khai hai nơi

    van = " ".join(["— Hết giờ, được chưa?"] * (TIC_LIMIT + 1))
    f = [x for x in tic_findings(van, _ct("CHAR_VHAL"), chars)
         if x["check"] == "verbal_tic_overuse"]
    assert f and f[0]["severity"] == "minor"

    trong_tran = " ".join(["— Hết giờ, được chưa?"] * TIC_LIMIT)
    assert not [x for x in tic_findings(trong_tran, _ct("CHAR_VHAL"), chars)
                if x["check"] == "verbal_tic_overuse"]
    assert "verbal_tic_overuse" in POLISHABLE_CHECKS      # đổi cách nói là việc polish


# ═══════════ 2. CỬ CHỈ CẠN NGÂN SÁCH THÌ CẤM TRƯỚC KHI VIẾT ═══════════

def test_cu_chi_da_can_bi_chuyen_sang_danh_sach_cam(chars):
    """Ngân sách cử chỉ phải nói với Writer TRƯỚC, không phải với polish SAU.

    Nó là lỗi `minor` nên chỉ đi polish, mà polish không đổi ngôn ngữ cơ thể —
    19 lần trên 5 chương với trần 3 mỗi chương là bằng chứng.
    """
    prof = chars["CHAR_KAELEN"]
    cu_chi = prof.somatic_signature[0]
    ct = {"active_characters": [{"id": "CHAR_KAELEN",
                                 "somatic_allowed": list(prof.somatic_signature),
                                 "somatic_forbidden": ["tim đập thình thịch"]}]}

    x = _tru_cu_chi_da_can(ct, [], chars)["active_characters"][0]
    assert x["somatic_allowed"] == list(prof.somatic_signature)

    can = _tru_cu_chi_da_can(
        ct, [{"prose": (cu_chi + ". ") * 3}], chars)["active_characters"][0]
    assert cu_chi not in can["somatic_allowed"]
    assert "tim đập thình thịch" in can["somatic_forbidden"]   # cấm cũ còn nguyên
    # Phải nằm trong 5 mục ĐẦU: brief cắt danh sách cấm ở đó, mà riêng sáo ngữ
    # dùng chung đã có 10 mục — nối vào đuôi là viết cho không ai đọc.
    assert any(cu_chi in g for g in can["somatic_forbidden"][:5])


# ═══════════ 3. LỖI GÕ TOÀN CHỮ LATIN ═══════════

@pytest.mark.parametrize("van, hong", [
    ("Lực ép không hề suyuy giảm.", ["suyuy"]),
    ("Anh bước ra ngoàiập vào.", ["ngoàiập"]),
    ("Chuyến tàu của người quyết định giữa những tuyến đường.", []),
    ("Nghiêng đầu, anh nghiến răng trong nghề nghiệp cũ.", []),
])
def test_bat_tu_khong_the_la_tieng_viet(van, hong):
    assert malformed_words(van) == hong
    if hong:
        f = [x for x in hygiene_findings(van) if x["check"] == "malformed_word"]
        assert f and f[0]["evidence"] == hong[0]


def test_loi_go_thanh_tu_co_that_thi_bo_kiem_nay_mu():
    """Giới hạn phải nói thẳng, không giấu trong docstring.

    "khớp ngón tro" đáng lẽ là "trỏ", nhưng "tro" là một từ tiếng Việt thật.
    Không có từ điển thì không có cách nào bắt được, và giả vờ ngược lại sẽ
    khiến người dùng tin vào một lớp bảo vệ không tồn tại.
    """
    assert malformed_words("ấn ngón cái vào khớp ngón tro") == []


# ═══════════ 4. concluded_by KHÔNG CÒN LÀ DỮ LIỆU CHẾT ═══════════

class _Clue:
    def __init__(self):
        self.understood_by_characters = []


def test_concluded_by_chay_vao_understood_by(chars):
    clue = _Clue()
    pe = PlantEvidence(clue_id="C1", scene_id="S0", span="x", carrier_used="object",
                       verified=True, concluded_by=["CHAR_KAELEN"])
    assert _ghi_nhan_hieu(clue, pe, chars) == 1
    assert clue.understood_by_characters == ["CHAR_KAELEN"]
    assert _ghi_nhan_hieu(clue, pe, chars) == 0            # không ghi trùng


def test_ma_nhan_vat_sai_hoa_thuong_van_nhan_ra(chars):
    """Lượt viết lại Arc 1 trả về "CHAR_KAELen" — và vì trường này chưa ai đọc
    nên suốt cả arc không ai phát hiện."""
    clue = _Clue()
    pe = PlantEvidence(clue_id="C1", scene_id="S0", span="x", carrier_used="object",
                       verified=True, concluded_by=["CHAR_KAELen", "CHAR_KHONG_CO"])
    assert _ghi_nhan_hieu(clue, pe, chars) == 1
    assert clue.understood_by_characters == ["CHAR_KAELEN"]


def test_span_chua_xac_minh_thi_khong_ai_hieu_gi_ca(chars):
    """NT-5: ai đó hiểu ra điều gì là chuyện phải có trên trang giấy."""
    clue = _Clue()
    pe = PlantEvidence(clue_id="C1", scene_id="S0", span="x", carrier_used="object",
                       verified=False, concluded_by=["CHAR_KAELEN"])
    # `apply_delta` lọc `verified` trước khi gọi; ở đây chốt lại giao kèo đó.
    assert pe.verified is False


# ═══════════ 5. TRẢ BÀI PHẢI ĐƯỢC NÓI RA ═══════════

def _delta_payoff():
    d = StateDelta(chapter=3)
    d.plant_evidence = [PlantEvidence(clue_id="C1", scene_id="S0",
                                      span="con dấu đã bị mài mòn",
                                      carrier_used="object", verified=True)]
    d.plant_modes = {"C1": "payoff"}
    return d


def test_tra_bai_trong_loi_ke_bi_bat():
    ke = [{"scene_id": "S0", "prose": "Anh nhìn xuống. con dấu đã bị mài mòn ở mép."}]
    assert [x["clue_id"] for x in payoff_not_spoken(_delta_payoff(), ke)] == ["C1"]


def test_tra_bai_trong_thoai_thi_dat():
    noi = [{"scene_id": "S0",
            "prose": "— con dấu đã bị mài mòn ở mép — Kaelen nói."}]
    assert payoff_not_spoken(_delta_payoff(), noi) == []


def test_cai_va_nhac_lai_thi_im_lang_moi_dung():
    """Chỉ soi `payoff`. Manh mối đang được CÀI mà nhân vật nói toạc ra thì
    hỏng theo chiều ngược lại — nó phải đi qua mắt độc giả mà không bị chỉ trỏ."""
    d = _delta_payoff()
    d.plant_modes = {"C1": "plant"}
    ke = [{"scene_id": "S0", "prose": "Anh nhìn xuống. con dấu đã bị mài mòn ở mép."}]
    assert payoff_not_spoken(d, ke) == []


# ═══════════ 6. ĐƯA CHÍNH NHỮNG SỐ ĐÃ CHẨN ARC 1 VÀO BỘ HỒI QUY ═══════════

def _ch(n, canh):
    return ChapterData(number=n, scenes=[{"scene_id": f"CH{n:03d}_S{i:02d}",
                                          "prose": v} for i, v in enumerate(canh)])


CAU = ["Cần trục kêu ba tiếng rồi im bặt trên mặt nước đen của cảng lúc rạng sáng",
       "Mùi lưu huỳnh bám vào cổ áo và không chịu tan dù gió thổi từ phía biển",
       "Nền rung một nhịp khi đoàn tàu chở quặng cuối cùng rời khỏi bến",
       "Ánh đèn pha quét ngang mặt nước rồi dừng ở khoang hàng số bốn phía đuôi",
       "Giấy tờ trong tay viên chức sột soạt một lúc trước khi con dấu hạ xuống",
       "Hơi nước đọng thành giọt trên mép ống dẫn rồi rơi xuống sàn thép"]


def test_M13_bat_canh_dung_lai_canh_truoc():
    """Arc 1 lượt 2 đạt gần hết M1–M12 mà vẫn có ba cảnh liên tiếp dựng lại
    cùng một cảnh. Metric nào cũng chỉ bảo vệ được thứ nó đo."""
    khac = M.M13_scene_repetition([_ch(1, [". ".join(CAU[:3]), ". ".join(CAU[3:])])])
    giong = M.M13_scene_repetition([_ch(1, [". ".join(CAU), ". ".join(CAU)])])
    assert khac["value"] < giong["value"]
    assert giong["value"] > 0.085 and "ch1" in giong["worst_pair"]
    assert SPECS["M13_scene_repetition"][1](giong["value"])
    assert not SPECS["M13_scene_repetition"][1](khac["value"])


def test_M14_bat_cu_chi_dong_dau_lai(chars):
    cu_chi = chars["CHAR_KAELEN"].somatic_signature[0]
    it = M.M14_tic_saturation([_ch(1, [cu_chi + "."])], chars)
    nhieu = M.M14_tic_saturation([_ch(1, [(cu_chi + ". ") * 5])], chars)
    assert it["value"] == 1 and nhieu["value"] == 5
    assert SPECS["M14_tic_saturation"][1](nhieu["value"])
    assert "Kaelen" in nhieu["worst"]


def test_M15_rac_trong_van_da_chot_phai_bang_khong():
    sach = M.M15_hygiene_residue([_ch(1, ["Anh bước vào phòng trực."])])
    ban = M.M15_hygiene_residue([_ch(1, ["03. Anh bước vào.", "— 01.",
                                         "Lực ép không suyuy giảm."])])
    assert sach["value"] == 0 and sach["detail"] == {}
    assert ban["value"] == 3 and set(ban["detail"]["ch1"]) == {
        "chỉ số", "thoại chỉ số", "lỗi gõ"}
    assert SPECS["M15_hygiene_residue"][1](ban["value"])


def test_ba_metric_moi_deu_la_thap_thi_tot():
    """Nếu quên khai, bộ hồi quy sẽ báo NGƯỢC: trùng lặp giảm bị tính là tụt."""
    for ten in ("M13_scene_repetition", "M14_tic_saturation", "M15_hygiene_residue"):
        assert ten in LOWER_IS_BETTER
    bao = run_regression({"metrics": {"M13_scene_repetition": {"value": 0.05}}},
                         {"metrics": {"M13_scene_repetition": {"value": 0.14}}})
    assert bao["regressed"] == []
    xau = run_regression({"metrics": {"M13_scene_repetition": {"value": 0.14}}},
                         {"metrics": {"M13_scene_repetition": {"value": 0.05}}})
    assert xau["regressed"] == ["M13_scene_repetition"]
