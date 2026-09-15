"""Tầng LLM — LLMPort, FakeLLM, ranh giới JSON (§9.5), chọn backend.

KHÔNG test nào ở đây gọi mạng. Test có gọi Gemini thật nằm ở
`test_gemini_live.py`, mặc định bị skip.
"""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from novel_engine.canon.timeline import SceneClose
from novel_engine.llm.env import GEMINI_KEY_NAMES, load_dotenv
from novel_engine.llm.fake import FakeLLM, ScriptedLLM
from novel_engine.llm.json_io import (
    merge_json, parse_model, strip_fences,
)
from novel_engine.llm.port import LLMPort


# ═══════════════════ LLMPort ═══════════════════

def test_fake_va_scripted_deu_thoa_man_llm_port():
    assert isinstance(FakeLLM(), LLMPort)
    assert isinstance(ScriptedLLM([]), LLMPort)


def test_fake_llm_tat_dinh():
    """Bộ hồi quy §13.2 đòi cùng seed cho cùng kết quả. `FakeLLM` dùng blake2b
    chứ không `hash()` (NT-16) nên tính này sống qua ranh giới tiến trình."""
    a, b = FakeLLM(), FakeLLM()
    p = "Người kể: Kaelen, ngôi thứ ba giới hạn."
    assert a.invoke(p, role="writer") == b.invoke(p, role="writer")


def test_fake_llm_doi_prompt_thi_doi_van():
    f = FakeLLM()
    x = f.invoke("Người kể: Kaelen", role="writer")
    y = f.invoke("Người kể: Serena", role="writer")
    assert x != y


def test_fake_llm_ghi_nhat_ky_de_test_soi():
    f = FakeLLM()
    f.invoke("p1", role="writer")
    f.invoke("p2", role="scene_digest")
    assert [c["role"] for c in f.calls] == ["writer", "scene_digest"]


# ═══════════════════ FakeLLM sinh ra thứ HỢP LỆ ═══════════════════

def test_fake_prose_dat_ngan_sach_tu():
    """Văn giả phải đủ dài, nếu không luật `word_budget` (§8.2) chặn ngay ở
    Cảnh 0 và ta không bao giờ test được phần còn lại của đồ thị."""
    prose = FakeLLM(word_target=320).invoke("Người kể: Kaelen", role="writer")
    assert 250 <= len(prose.split()) <= 600


def test_fake_prose_nhac_dung_ten_POV():
    """Đầu ra giả vẫn phải NHẮC ĐÚNG nhân vật của cảnh — nếu không
    `verify_spans` và `pov_leak_scan` chạy trên văn bản chẳng liên quan gì tới
    hợp đồng, và hai cơ chế đó thành mã chết trong mọi test."""
    prose = FakeLLM().invoke("Người kể: Kaelen, ngôi thứ ba", role="writer")
    assert "Kaelen" in prose


def test_fake_prose_khong_chua_sao_ngu_cam():
    from novel_engine.character.models import CLICHE_SOMATICS
    prose = FakeLLM().invoke("Người kể: Kaelen", role="writer")
    assert not [c for c in CLICHE_SOMATICS if c in prose]


def test_fake_prose_do_dai_cau_bien_thien():
    """σ độ dài câu phải khác 0, nếu không `rhythm.py` (§10.3.1) gắn cờ 'văn
    đều đều' ở mọi cảnh và Auditor không bao giờ trả về sạch."""
    import statistics
    prose = FakeLLM().invoke("Người kể: Kaelen", role="writer")
    lens = [len(s.split()) for s in prose.split(".") if s.strip()]
    assert len(lens) > 3 and statistics.pstdev(lens) > 1.0


def test_fake_scene_close_parse_duoc_thanh_SceneClose():
    """Đúng cái `scene_boundary_node` làm: parse_model(raw, SceneClose)."""
    raw = FakeLLM().invoke(
        "scene_id: CH001_S00\nduration_ticks dự kiến: 4\n"
        "Nhân vật có mặt: CHAR_KAELEN (Kaelen), CHAR_VHAL (Vhal)\n"
        "Địa điểm: LOC_ORE_PORT", role="scene_digest")
    close = parse_model(raw, SceneClose)
    assert close.actual_duration_ticks == 4
    # Khoá phải là MÃ ĐỊNH DANH, không phải tên. §9.2 truyền
    # `[x["name"] for x in active_characters]` vào prompt này, nhưng
    # `character_track(cid, frames)` (§3.6.1) tra bằng char_id — khoá bằng tên
    # làm MỌI luật liên tục thành mã chết: không ném lỗi, chỉ không khớp mãi
    # mãi (NT-8). SCENE_DIGEST_TMPL đã được sửa để đòi mã CHAR_*.
    assert set(close.continuity.locations) == {"CHAR_KAELEN", "CHAR_VHAL"}
    assert close.continuity.locations["CHAR_KAELEN"] == "LOC_ORE_PORT"
    assert close.unresolved


def test_fake_scene_close_boc_trong_fence():
    """Cố ý bọc ```json để `strip_fences` được chạy THẬT ở mỗi ranh giới cảnh,
    chứ không chỉ trong test riêng của nó."""
    raw = FakeLLM().invoke("duration_ticks dự kiến: 3", role="scene_digest")
    assert raw.lstrip().startswith("```")


def test_fake_polish_giu_gan_nguyen_van_goc():
    """Polish phải trả thứ GẦN GIỐNG bản gốc, nếu không `content_drifted`
    (§9.2) từ chối nó ở mọi cảnh và nhánh chấp nhận thành mã chết."""
    goc = "Cần trục kêu ba tiếng rồi im. Kaelen không nói gì."
    out = FakeLLM().invoke(f"## VĂN XUÔI\n{goc}\n", role="polish")
    assert out == goc


# ═══════════════════ ScriptedLLM ═══════════════════

def test_scripted_tra_theo_thu_tu_roi_rot_ve_fake():
    s = ScriptedLLM(["mot", "hai"])
    assert s.invoke("p") == "mot"
    assert s.invoke("p") == "hai"
    assert len(s.invoke("Người kể: Kaelen", role="writer").split()) > 50


# ═══════════════════ strip_fences (§9.5) ═══════════════════

def test_strip_fences_lot_markdown():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_lay_khoi_DAI_NHAT_khong_phai_khoi_dau():
    """Prompt Extractor có chứa ví dụ JSON, và model thỉnh thoảng nhắc lại ví
    dụ ấy trước khi xuất kết quả thật. Lấy khối đầu tiên sẽ nuốt phải ví dụ,
    parse THÀNH CÔNG, và ghi một delta rỗng vào canon — đúng kiểu lỗi âm thầm
    mà NT-5 sinh ra để chặn."""
    raw = ('Đây là ví dụ:\n```json\n{}\n```\n'
           'Và đây là kết quả:\n```json\n{"assertions": [1, 2, 3]}\n```')
    assert json.loads(strip_fences(raw))["assertions"] == [1, 2, 3]


def test_strip_fences_loi_dan_lich_su():
    assert json.loads(strip_fences('Chắc chắn rồi! {"a": 1} Hy vọng giúp ích.'))


def test_strip_fences_chuoi_rong():
    assert strip_fences("") == "" and strip_fences(None) == ""


# ═══════════════════ parse_model ═══════════════════

class _Toy(BaseModel):
    ten: str
    so: int


def test_parse_model_qua_fence():
    m = parse_model('```json\n{"ten": "x", "so": 3}\n```', _Toy)
    assert m.so == 3


def test_parse_model_mot_luot_tu_sua():
    """Một lượt sửa rẻ hơn nhiều so với viết lại cả chương."""
    repair = ScriptedLLM(['{"ten": "x", "so": 3}'])
    m = parse_model('{"ten": "x", "so": }', _Toy, repair_llm=repair)
    assert m.so == 3
    assert repair.calls[0]["role"] == "repair"


def test_parse_model_het_luot_thi_nem_ValueError_khong_phai_JSONDecodeError():
    """`safe_node` (§9.5) bắt mọi Exception, nhưng thông điệp phải nói được
    schema nào hỏng — `JSONDecodeError` trần thì không."""
    with pytest.raises(ValueError, match="_Toy"):
        parse_model("hoàn toàn không phải json", _Toy)


def test_parse_model_khong_co_repair_thi_khong_goi_them():
    with pytest.raises(ValueError):
        parse_model("{", _Toy, repair_llm=None)


# ═══════════════════ merge_json — CODE THẮNG (NT-12) ═══════════════════

def test_merge_json_chi_dien_khoa_de_trong():
    base = {"scene_id": "CH001_S00", "entry_state": "", "exit_state": ""}
    out = merge_json(base, '{"entry_state": "thế thủ", "exit_state": "đã lộ"}')
    assert out["entry_state"] == "thế thủ" and out["exit_state"] == "đã lộ"


def test_merge_json_KHONG_cho_LLM_ghi_de_truong_he_thong():
    """NT-12: `time`, `scene_id`, `plant_directives` là của code. Để model ghi
    đè là tạo hai nguồn sự thật cho cùng một đại lượng — chúng sẽ lệch nhau ở
    chương thứ ba, và không luật nào phát hiện.

    `FakeLLM._director_fill` CỐ Ý thử ghi đè cả hai để test này có răng."""
    base = {"scene_id": "CH001_S00", "time": {"epoch_tick": 6},
            "entry_state": ""}
    out = merge_json(base, FakeLLM().invoke("x", role="director"))
    assert out["scene_id"] == "CH001_S00"
    assert out["time"]["epoch_tick"] == 6
    assert out["entry_state"]                      # khoá để trống thì vẫn nhận


def test_merge_json_khong_them_khoa_la():
    out = merge_json({"a": ""}, '{"a": "x", "khoa_la": "y"}')
    assert "khoa_la" not in out


def test_merge_json_json_hong_thi_tra_base():
    base = {"a": ""}
    assert merge_json(base, "không phải json") == base
    assert merge_json(base, "[1,2,3]") == base


# ═══════════════════ .env + chọn backend ═══════════════════

def test_load_dotenv_khong_ghi_de_bien_da_co(tmp_path, monkeypatch):
    """`NOVEL_LLM=fake pytest` phải THẮNG file .env — chạy test mà vô tình đốt
    hạn ngạch vì một dòng trong file cấu hình là chuyện không nên xảy ra."""
    f = tmp_path / ".env"
    f.write_text("GEMINI_API_KEY=tu_file\nNOVEL_LLM=gemini\n", encoding="utf-8")
    monkeypatch.setenv("NOVEL_LLM", "fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    load_dotenv(f)
    import os
    assert os.environ["NOVEL_LLM"] == "fake"        # biến sẵn có thắng
    assert os.environ["GEMINI_API_KEY"] == "tu_file"  # biến chưa có thì nạp


def test_load_dotenv_che_gia_tri_trong_ket_qua(tmp_path):
    f = tmp_path / ".env"
    f.write_text("MOT_KHOA_BI_MAT=abcdefghijklmnop\n", encoding="utf-8")
    out = load_dotenv(f, override=True)
    assert "abcdefghijklmnop" not in out["MOT_KHOA_BI_MAT"]


def test_load_dotenv_file_khong_ton_tai(tmp_path):
    assert load_dotenv(tmp_path / "khong_co.env") == {}


def test_load_dotenv_bo_qua_comment_va_dong_hong(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# ghi chú\n\nDONG_HONG\nOK_KEY=1\n", encoding="utf-8")
    assert list(load_dotenv(f, override=True)) == ["OK_KEY"]


def test_build_llm_mac_dinh_la_fake(monkeypatch):
    """Mặc định an toàn CÓ CHỦ Ý: quên set biến thì chạy miễn phí, không đốt
    hạn ngạch."""
    from novel_engine.llm.gemini import build_llm
    monkeypatch.delenv("NOVEL_LLM", raising=False)
    assert isinstance(build_llm(), FakeLLM)


def test_build_llm_backend_la_thi_nem(monkeypatch):
    from novel_engine.llm.gemini import build_llm
    with pytest.raises(ValueError, match="backend"):
        build_llm("openai")


def test_gemini_thieu_khoa_thi_bao_ro_cach_sua(monkeypatch, tmp_path):
    from novel_engine.llm.gemini import GeminiLLM
    for name in GEMINI_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)              # không có .env ở đây
    with pytest.raises(RuntimeError) as ei:
        GeminiLLM()
    assert "aistudio.google.com" in str(ei.value)
    assert "FakeLLM" in str(ei.value)


def test_gemini_nhan_ca_hai_ten_bien(monkeypatch):
    from novel_engine.llm.gemini import GeminiLLM
    for name in GEMINI_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "khoa-gia")
    GeminiLLM()                               # không ném — chưa gọi mạng


# ═══════════════════ bộ điều tiết nhịp ═══════════════════

def test_rate_limiter_khong_chan_khi_duoi_han():
    from novel_engine.llm.gemini import _RateLimiter
    lim = _RateLimiter(rpm=10)
    assert all(lim.acquire() == 0.0 for _ in range(10))


def test_rate_limiter_chan_khi_vuot_han(monkeypatch):
    """Nút thắt thật của bậc miễn phí là RPM, không phải hạn ngạch ngày.
    Writer bắn 8 lượt liên tiếp trong vài giây sẽ ăn 429 từ lượt thứ 3."""
    from novel_engine.llm import gemini as G
    ngu = []
    monkeypatch.setattr(G.time, "sleep", ngu.append)
    lim = G._RateLimiter(rpm=2)
    lim.acquire(); lim.acquire()
    assert lim.acquire() > 0        # lượt thứ 3 phải chờ
    assert ngu and ngu[0] > 0


def test_rpm_suy_ra_tu_model():
    """Dùng chung một trần RPM cho mọi model là cách chắc chắn ăn 429 trên
    dòng flash (5 RPM), hoặc phí nửa thời gian chờ vô ích trên dòng flash-lite
    (~30 RPM). Đo thực: 3.8-flash mất 56s/lượt, 3.5-flash-lite mất 4,3s."""
    from novel_engine.llm.gemini import DEFAULT_MODEL, GeminiLLM

    lite = GeminiLLM(model="gemini-3.5-flash-lite", api_key="gia")
    flash = GeminiLLM(model="gemini-3.8-flash", api_key="gia")
    assert lite.rpm > flash.rpm
    assert GeminiLLM(model="model-la-hoac-moi", api_key="gia").rpm > 0
    assert GeminiLLM(api_key="gia", rpm=99).rpm == 99      # gọi chỉ định thì thắng
    assert "lite" in DEFAULT_MODEL                          # mặc định là bản nhanh
