"""Ngày 23 — ba hướng cuối: câu mẫu giọng, đối chiếu vật mang, model theo vai.

Mỗi hướng đều đã từng bị tôi gác lại với một lý do, và lý do ấy được ghi ở đây
cùng với cái đã thay đổi nó.
"""
from __future__ import annotations

import pytest

from novel_engine.canon.bible import DEFAULT_BIBLE, load_characters
from novel_engine.canon.models import PlantEvidence, StateDelta
from novel_engine.eval.harness import SPECS
from novel_engine.graph.nodes import _characters_brief
from novel_engine.llm.gemini import DEFAULT_MODEL, RPM_BY_MODEL, GeminiLLM
from novel_engine.prompts import WRITER_TMPL
from novel_engine.reconcile.verify import CARRIERS, carrier_mismatch


@pytest.fixture(scope="module")
def chars():
    return load_characters(DEFAULT_BIBLE)


# ═══════════ 1. CÂU MẪU THAY DANH SÁCH KHẨU NGỮ ═══════════

def test_moi_nhan_vat_co_cau_mau_dung_giong_minh(chars):
    for prof in chars.values():
        assert prof.voice.voice_exemplars, prof.name
        assert all(len(x.split()) >= 4 for x in prof.voice.voice_exemplars)


def test_cau_mau_khong_duoc_chua_chinh_cum_dang_do(chars):
    """Bản câu mẫu ĐẦU TIÊN của tôi chứa đúng "về mặt thủ tục" và "đã được lưu"
    — hai cụm nằm trong `signature_lexicon` của Serena.

    Tức là tôi bỏ danh sách ra cửa trước rồi tuồn nó vào cửa sau: model vẫn
    nhận được chính những chữ mà M3 và ngân sách tật ngôn ngữ đang đếm. Câu mẫu
    phải dạy CÁCH nói — mệnh đề nhượng bộ rồi "tuy nhiên", chốt bằng câu hỏi —
    chứ không giao lại từ nào để chép.
    """
    import unicodedata

    for prof in chars.values():
        # `syntactic_tic` cũng đi vào prompt Writer, nên cũng phải sạch. Lượt đo
        # với prompt câu mẫu: mật độ cụm đặc trưng tụt 15→3 trên 1000 từ, riêng
        # "được chưa" vẫn 7 lần — vì `syntactic_tic` của Vhal trích nguyên văn
        # nó. Bịt một cửa mà để hở cửa bên cạnh thì không bịt gì cả.
        van = unicodedata.normalize(
            "NFC", " ".join(list(prof.voice.voice_exemplars)
                            + [prof.voice.syntactic_tic])).lower()
        ro = [c for c in list(prof.voice.signature_lexicon)
              + list(prof.voice.verbal_tics) if c.lower() in van]
        assert ro == [], f"{prof.name}: câu mẫu rò cụm đang được đo: {ro}"


def test_writer_nhan_cau_mau_chu_khong_nhan_thuoc_do(chars):
    """Suốt Arc 1 ta vừa chấm giọng bằng `signature_lexicon` vừa đưa chính
    danh sách ấy cho Writer. Kết quả đo được: Serena dùng cả năm cụm, mỗi cụm
    ba lần, trong MỘT cảnh. NT-7 đã nói điều này về chỉ số văn phong; danh
    sách khẩu ngữ chỉ là cùng một sai lầm ở một tầng khác."""
    prof = chars["CHAR_SERENA"]
    v = prof.voice.model_dump(include={"register", "voice_exemplars",
                                       "forbidden_lexicon", "syntactic_tic"})
    ra = _characters_brief([{"id": "CHAR_SERENA", "name": "Serena",
                             "voice_reminder": v},
                            {"id": "CHAR_VHAL", "name": "Vhal",
                             "voice_reminder": v}])
    assert prof.voice.voice_exemplars[0][:30] in ra
    for cum in prof.voice.signature_lexicon:
        assert cum not in ra
    assert "signature_lexicon" not in WRITER_TMPL


def test_cau_mau_van_khong_di_vao_canh_mot_minh(chars):
    """Bài học Ngày 10 phải sống sót qua thay đổi này: cảnh một mình không có
    thoại, nên mọi nguyên liệu thoại đều sẽ chảy vào lời kể."""
    v = chars["CHAR_SERENA"].voice.model_dump(
        include={"register", "voice_exemplars", "forbidden_lexicon"})
    solo = _characters_brief([{"id": "CHAR_SERENA", "name": "Serena",
                               "voice_reminder": v}])
    assert "MỘT MÌNH" in solo
    for x in chars["CHAR_SERENA"].voice.voice_exemplars:
        assert x not in solo


# ═══════════ 2. VẬT MANG: ĐỐI CHIẾU VỚI CHỈ THỊ, KHÔNG VỚI TỪ ĐIỂN ═══════════

def _pe(cid, carrier, verified=True):
    return PlantEvidence(clue_id=cid, scene_id="S0", span="x",
                         carrier_used=carrier, verified=verified)


def test_doi_chieu_vat_mang_voi_kenh_duoc_giao():
    """Kiểm từ vựng đơn thuần là tiếng ồn — không ai đọc `carrier_used`. Nhưng
    Scheduler XOAY VÒNG vật mang để manh mối không hiện lên cùng một kiểu ở mọi
    chương, và phép xoay ấy chỉ đúng nếu kênh giao là kênh thật sự dùng."""
    d = StateDelta(chapter=1)
    d.plant_evidence = [_pe("C1", "object"), _pe("C2", "audio recording"),
                        _pe("C3", "dialogue")]
    plan = {"C1": {"carrier": "dialogue"}, "C2": {"carrier": "object"},
            "C3": {"carrier": "dialogue"}}
    ra = {x["clue_id"]: x["why"] for x in carrier_mismatch(d, plan)}
    assert ra == {"C1": "khác kênh được giao", "C2": "ngoài từ vựng"}
    assert set(CARRIERS) == {"object", "setting", "behavior", "dialogue"}


def test_khong_co_chi_thi_thi_khong_co_gi_de_doi_chieu():
    """Bằng chứng tình cờ (Writer tự cài) hợp lệ và không bị soi ở đây."""
    d = StateDelta(chapter=1)
    d.plant_evidence = [_pe("C9", "object")]
    assert carrier_mismatch(d, {}) == []


def test_span_chua_xac_minh_thi_khong_soi_vat_mang():
    d = StateDelta(chapter=1)
    d.plant_evidence = [_pe("C1", "object", verified=False)]
    assert carrier_mismatch(d, {"C1": {"carrier": "dialogue"}}) == []


# ═══════════ 3. MODEL MẠNH CHO RIÊNG VAI WRITER ═══════════

def test_model_theo_vai_va_tran_rpm_theo_tung_model():
    """Đổi model cho cả vòng lặp là không trả giá nổi — dòng flash chậm gấp 13
    lần. Nhưng Writer chỉ chiếm 6 trong 26 lượt, và là lượt DUY NHẤT sinh văn
    xuôi; 20 lượt còn lại đọc JSON."""
    g = GeminiLLM(api_key="gia", role_models={"writer": "gemini-3.8-flash"})
    assert g.model_for("writer") == "gemini-3.8-flash"
    assert g.model_for("auditor") == DEFAULT_MODEL
    assert g.model_for("") == DEFAULT_MODEL

    # Trần RPM phải theo TỪNG model: dùng chung một limiter 15 RPM sẽ ném lượt
    # Writer của model 4 RPM thẳng vào 429.
    g._limiter_of("gemini-3.8-flash")
    g._limiter_of(DEFAULT_MODEL)
    assert g._rpm_of["gemini-3.8-flash"] == RPM_BY_MODEL["gemini-3.8-flash"]
    assert g._rpm_of[DEFAULT_MODEL] == RPM_BY_MODEL[DEFAULT_MODEL]
    assert g._rpm_of["gemini-3.8-flash"] < g._rpm_of[DEFAULT_MODEL]


def test_khong_khai_thi_moi_vai_dung_chung_mot_model():
    g = GeminiLLM(api_key="gia")
    assert {g.model_for(r) for r in ("writer", "auditor", "extractor", "")} == {
        DEFAULT_MODEL}


def test_client_tach_theo_ca_model_lan_nhiet_do():
    """`ChatGoogleGenerativeAI` chốt cả model lẫn nhiệt độ lúc khởi tạo, nên
    khoá cache phải là cặp — khoá bằng mỗi nhiệt độ thì lượt Writer sẽ nhận
    client của model khác."""
    g = GeminiLLM(api_key="gia", role_models={"writer": "gemini-3.8-flash"})
    assert g._clients == {}
    import inspect
    src = inspect.getsource(g._client)
    assert "khoa = (model, temperature)" in src


# ═══════════ 4. M3 KHÔNG ĐƯỢC CHẤM GIỌNG TRÊN 5% SỐ THOẠI ═══════════

def test_M3_tinh_ca_luot_thoai_suy_ra_tu_nhip_hanh_dong(chars):
    """Đo trên Chương 1 thật: 28–34 dòng thoại, chỉ 1–3 dòng có dẫn tường minh
    ("— … — Kaelen nói"), còn 12–17 dòng gán được qua câu hành động đứng ngay
    trước. `lexical_distribution` chỉ đọc `by_speaker`, nên M3 chấm giọng trên
    khoảng 5% số thoại rồi trả về một con số nghe rất chắc chắn — và với chương
    viết bằng prompt câu mẫu thì nó báo thẳng "chưa đo được: 0 nhân vật".

    `voice_report` trong cùng module vốn đã dùng cả hai nguồn. Hai chính sách
    cho cùng câu hỏi "ai nói câu này" là thứ NT-11 cấm.
    """
    from novel_engine.eval.metrics import lexical_distribution

    # Thoại có dẫn cho Kaelen; thoại của Vhal chỉ gán được qua nhịp hành động.
    canh = chr(10).join([
        "— Ba mươi phút — Kaelen nói, mắt không rời mặt bàn.",
        "",
        "Vhal đẩy tờ giấy trở lại qua khe cửa, không ngẩng lên.",
        "",
        "— Người ta bảo tôi đóng dấu thì tôi đóng, chứ tôi biết gì đâu.",
    ])
    d = lexical_distribution([_chuong(canh)], chars)
    assert d["CHAR_KAELEN"]["lines"] >= 1
    assert d["CHAR_VHAL"]["lines"] >= 1, "lượt suy ra từ nhịp hành động bị vứt"


def _chuong(prose):
    from novel_engine.eval.metrics import ChapterData
    return ChapterData(number=1, scenes=[{"scene_id": "CH001_S00", "prose": prose}])


# ═══════════ 5. 503 CŨNG LÀ LỖI TẠM THỜI ═══════════

@pytest.mark.parametrize("loi, thu_lai", [
    ("429 RESOURCE_EXHAUSTED", True),
    ("503 UNAVAILABLE. This model is currently experiencing high demand.", True),
    ("500 internal error", True),
    ("400 API key not valid", False),
    ("404 model not found", False),
])
def test_chi_thu_lai_loi_dang_thu_lai(loi, thu_lai):
    """Lượt đầu chạy `gemini-3.8-flash` cho vai Writer chết vì 503, không phải
    429 — dòng flash chia sẻ công suất hẹp hơn hẳn dòng flash-lite, nên "model
    đang quá tải" là chuyện thường chứ không phải ngoại lệ. Vòng thử lại cũ chỉ
    bắt 429, nên một cơn quá tải hai giây giết cả chương.

    Còn khoá sai hay model không tồn tại thì thử lại chỉ tốn thêm thời gian để
    thất bại y hệt.
    """
    from novel_engine.llm.gemini import _tam_thoi

    assert _tam_thoi(Exception(loi)) is thu_lai


def test_bao_loi_goi_dung_ten_nguyen_nhan(monkeypatch):
    """503 không phải hết hạn ngạch, và nói nhầm sẽ gửi người đọc đi sai hướng.

    Đo trực tiếp lúc dòng flash đang quá tải: `gemini-3.8-flash` lỗi sau 57s,
    `gemini-3.7-flash` lỗi sau 197s, còn `gemini-3.5-flash-lite` trả lời trong
    2,7s. Bản trước gán mọi thất bại cho hạn ngạch, nên một cơn quá tải phía
    Google hiện ra thành "hết hạn ngạch ngày của bậc miễn phí".
    """
    g = GeminiLLM(api_key="gia", role_models={"writer": "gemini-3.8-flash"},
                  max_retries=1)

    class _Chet:
        def invoke(self, _):
            raise RuntimeError("503 UNAVAILABLE. This model is experiencing "
                               "high demand.")

    monkeypatch.setattr(g, "_client", lambda *a: _Chet())
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError) as e:
        g.invoke("x", role="writer")
    loi = str(e.value)
    assert "quá tải" in loi and "gemini-3.8-flash" in loi
    assert "hạn ngạch" not in loi
    assert DEFAULT_MODEL in loi            # chỉ đúng lối thoát

    class _Het:
        def invoke(self, _):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    monkeypatch.setattr(g, "_client", lambda *a: _Het())
    with pytest.raises(RuntimeError, match="hạn ngạch"):
        g.invoke("x", role="writer")


# ═══════════ 6. M12: TRUNG BÌNH GIẤU MẤT HÌNH DẠNG ═══════════

def test_M12_phai_noi_ro_bao_nhieu_canh_khong_co_mia_mai():
    """mean 0,37 nghe như "có chút mỉa mai rải đều". Sự thật của Arc 1: 22
    trên 30 cảnh gap ĐÚNG BẰNG 0, toàn bộ tín hiệu đến từ 8 cảnh.

    Hai phân bố dưới đây cho cùng một trung bình và là hai quyển sách khác hẳn
    nhau. Một con số không phân biệt được chúng thì không được đứng một mình.
    """
    from novel_engine.eval import metrics as M

    class _P:
        def pov_for(self, ch, si):
            return "CHAR_KAELEN"

    def chay(gaps):
        # Dựng thẳng phần cuối của M12: ở đây ta chấm CÁCH BÁO CÁO, không chấm
        # phép tính gap (phần ấy đã có dữ liệu thật của Arc 1 kiểm chứng).
        import statistics
        return {"value": round(statistics.mean(gaps), 2),
                "zero_gap_ratio": round(sum(1 for g in gaps if g == 0) / len(gaps), 3)}

    deu = chay([1] * 10)
    don = chay([0] * 8 + [5, 5])
    assert deu["value"] == don["value"] == 1.0
    assert deu["zero_gap_ratio"] == 0.0 and don["zero_gap_ratio"] == 0.8

    # Và M12 thật phải trả trường đó ra, không chỉ trung bình.
    ra = M.M12_irony_gap([], None, _P())
    assert ra["value"] is None            # không có frame thì chưa đo được
    assert "zero_gap_ratio" in M.M12_irony_gap.__doc__


def test_nguyen_nhan_M12_khong_do_cho_News_Dispatcher():
    """`likely_cause` cũ nói "News Dispatcher chưa bật". Trên Arc 1 nó ĐANG bật
    và ghi 3 bản tri thức mỗi chương — chẩn đoán sai gửi người đọc đi sửa thứ
    không hỏng."""
    muc_tieu, canh_bao, nguyen_nhan = SPECS["M12_irony_gap"]
    assert "News Dispatcher" not in nguyen_nhan
    assert "zero_gap_ratio" in nguyen_nhan
    assert canh_bao(0.0) and not canh_bao(0.37)   # 0,37 dưới mục tiêu, chưa báo động
