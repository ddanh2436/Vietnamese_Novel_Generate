"""Ngày 9 — Auditor tầng thuật toán (§10.3, §10.3.1, §5.3).

Mỗi kiểm tra có hai loại test: nó BẮT được thứ phải bắt, và nó KHÔNG kêu với
thứ không sai. Loại thứ hai là loại mã §10.3 thiếu — và là lý do một nửa kiểm
tra của nó hoặc chết im lặng, hoặc báo động giả.
"""
from __future__ import annotations

import json
import re
import unicodedata

import pytest

from novel_engine.audit.deterministic import (
    audit_contract, audit_stats, deterministic_audit,
)
from novel_engine.audit.prose import prose_audit, sensory_channels
from novel_engine.audit.rhythm import (
    is_subordinate_opener, narrative_sentences, rhythm_audit, rhythm_stats,
)
from novel_engine.canon.bible import load_characters, DEFAULT_BIBLE
from novel_engine.character.voice_check import attribute_dialogue, voice_report
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM


def _checks(findings):
    return {f["check"] for f in findings}


def _cau(n: int, dau: str) -> str:
    return " ".join([dau] + ["chữ"] * (n - 1)) + "."


VARIED = " ".join(_cau(n, d) for n, d in zip(
    [3, 25, 8, 30, 4, 12, 40, 5, 18, 2, 22, 6],
    ["Tối", "Kaelen", "Gió", "Cần", "Im", "Serena", "Dưới", "Rồi", "Đèn", "Không",
     "Mùi", "Vhal"]))

MONOTONE = " ".join(_cau(15, d) for d in
                    ["Kaelen", "Gió", "Cần", "Serena", "Dưới", "Đèn", "Mùi",
                     "Vhal", "Sóng", "Bụi", "Tàu", "Người"])


# ═══════════════ KÊNH GIÁC QUAN — ranh giới từ ═══════════════

def test_mã_tai_lieu_dem_ba_kenh_cho_cau_khong_co_giac_quan():
    """Tái hiện đúng regex chuỗi con của §10.3 để ghi lại lỗi."""
    cau = "Anh cảm thấy tâm trạng rối bời ở vị trí tối đa đó."
    doc = {"thị giác": r"(nhìn|thấy|ánh|màu|sáng|tối|bóng)",
           "thính giác": r"(nghe|tiếng|âm|vang|im lặng|rít)",
           "vị giác": r"(vị|đắng|mặn|chua|ngọt|tanh)"}
    assert sum(1 for p in doc.values() if re.search(p, cau.lower())) == 3
    assert sensory_channels(cau) == []


def test_dem_dung_kenh_giac_quan_that():
    van = ("Mùi lưu huỳnh xộc vào mũi. Tiếng cần trục rít lên. Kaelen nhìn "
           "ánh đèn pha. Thanh sắt lạnh buốt dưới tay. Vị đắng bám đầu lưỡi.")
    assert set(sensory_channels(van)) == {"khứu giác", "thính giác", "thị giác",
                                          "xúc giác", "vị giác"}


@pytest.mark.parametrize("cau", [
    "Mồ hôi chảy dọc thái dương anh.",       # không phải khứu giác
    "Cô nóng lòng muốn biết kết quả.",        # không phải xúc giác
    "Một nỗi cay đắng dâng lên.",             # không phải vị giác
    "Ông ta nổi tiếng khắp cảng.",            # không phải thính giác
])
def test_nghia_bong_khong_tinh_la_giac_quan(cau):
    assert sensory_channels(cau) == []


# ═══════════════ SÁO NGỮ ═══════════════

def test_sao_ngu_bat_duoc_ca_khi_van_ban_dang_NFD():
    van = unicodedata.normalize("NFD", "Tim đập thình thịch, anh lùi lại.")
    assert "cliche" in _checks(prose_audit(van, {}))


def test_sao_ngu_theo_ranh_gioi_tu():
    """"nín thở" không được khớp bên trong một từ khác."""
    assert "cliche" not in _checks(prose_audit("Anh nín thởi gian.", {"word_budget": (0, 99)}))


def test_forbidden_cliches_cua_contract_duoc_dung():
    f = prose_audit("Bóng tối nuốt chửng cả căn phòng.",
                    {"forbidden_cliches": ["bóng tối nuốt chửng"], "word_budget": (0, 99)})
    assert any("bóng tối nuốt chửng" in x["message"] for x in f)


def test_danh_sach_sao_ngu_co_the_chi_khai_mot_noi():
    """NT-11: Writer bị cấm danh sách nào thì Auditor chấm theo đúng danh sách đó."""
    from novel_engine.audit import prose
    from novel_engine.character import models
    assert prose.CLICHE_SOMATICS is models.CLICHE_SOMATICS


def test_noi_thang_muc_tieu():
    van = "— Tôi muốn tìm kẻ đó. — Anh phải đến Lò 3. — Em cần cứu cô ấy."
    assert "on_the_nose" in _checks(prose_audit(van, {"word_budget": (0, 99)}))


def test_canh_dai_khong_thoai():
    van = " ".join(["Gió thổi qua cảng."] * 200)
    assert "no_dialogue" in _checks(prose_audit(van, {}))
    assert "no_dialogue" not in _checks(prose_audit("— Đi thôi.\n" + van, {}))


# ═══════════════ NHỊP TRẦN THUẬT ═══════════════

def test_van_deu_deu_bi_bat():
    ch = _checks(rhythm_audit(MONOTONE, {}))
    assert {"rhythm_variance", "no_staccato", "flat_run"} <= ch


def test_van_co_nhip_khong_bi_bao():
    assert rhythm_audit(VARIED, {}) == []


def test_sigma_tinh_bang_am_tiet_va_bo_gach_ngang():
    from novel_engine.audit.rhythm import word_count
    assert word_count("— Không.") == 1
    assert word_count("Tuyến ray – đã bị niêm phong.") == 6
    st = rhythm_stats(VARIED + " " + VARIED)
    assert st["n_sentences"] == 24
    assert st["sd"] > 8.5


def test_khuon_ba_menh_de_o_GIUA_doan_van():
    """Regex §10.3.1 neo `^` + re.M nên chỉ khớp đầu dòng. Gemini viết nhiều
    câu trên một dòng, nên khuôn ở giữa đoạn không bao giờ bị bắt."""
    tri = ("Khi đèn pha quét qua mặt nước, Kaelen lùi vào bóng tối, trong khi "
           "Vhal vẫn cúi đầu ghi sổ.")
    doan = VARIED + " " + " ".join([tri] * 3)
    assert "\n" not in doan
    assert re.findall(r"^(Khi|Trong khi|Sau khi|Ngay khi)\s.{5,70},\s.{5,70},\s"
                      r"(trong khi|còn|và|thì)\s", doan, flags=re.M) == []
    assert "triclause_template" in _checks(rhythm_audit(doan, {}))


@pytest.mark.parametrize("cau, la", [
    ("Khi đèn tắt, anh đứng dậy.", True),
    ("Vì anh đến muộn, cửa đã khoá.", True),
    ("Vì vậy anh quay đi.", False),
    ("Khi đó cửa đã khoá.", False),
    ("Dù sao cũng muộn rồi.", False),
])
def test_mo_dau_menh_de_phu(cau, la):
    assert is_subordinate_opener(cau) is la


def test_burstiness_chi_dem_doan_TRAN_THUAT():
    """Mỗi lượt thoại là một đoạn một câu. Đếm cả thoại thì "cảnh escalate
    phải có đoạn một câu" luôn thoả — kiểm tra chết."""
    doan_dai = " ".join(_cau(n, d) for n, d in zip(
        [3, 25, 8, 30], ["Tối", "Kaelen", "Gió", "Cần"]))
    van = "\n\n".join([doan_dai, "— Đi.", doan_dai.replace("Tối", "Im"), "— Khoan.",
                       doan_dai.replace("Gió", "Sóng")])
    ch = _checks(rhythm_audit(van, {"tension": {"mode": "escalate"}}))
    assert "paragraph_burstiness" in ch


def test_thieu_tension_hoac_van_rong_khong_no():
    assert rhythm_audit("", {}) == []
    assert isinstance(rhythm_audit(MONOTONE, {"tension": None}), list)


def test_khong_chi_so_nhip_nao_la_blocker():
    """§10.3.2: nhịp văn dở không phải lỗi thế giới quan."""
    tri = ("Khi đèn pha quét qua mặt nước, Kaelen lùi vào bóng tối, trong khi "
           "Vhal vẫn cúi đầu ghi sổ.")
    xau = MONOTONE + " " + " ".join([tri] * 6)
    assert rhythm_audit(xau, {"tension": {"mode": "escalate"}})
    assert all(f["severity"] != "blocker"
               for f in rhythm_audit(xau, {"tension": {"mode": "escalate"}}))


def test_thoai_khong_tinh_vao_cau_tran_thuat():
    assert narrative_sentences("— Đi thôi.\nGió thổi.\n– Khoan.\n“Dừng lại.”") == ["Gió thổi."]


# ═══════════════ GIỌNG NHÂN VẬT ═══════════════

@pytest.fixture(scope="module")
def chars():
    return load_characters(DEFAULT_BIBLE)


NAMES = {"Kaelen": "CHAR_KAELEN", "Serena": "CHAR_SERENA", "Vhal": "CHAR_VHAL"}


def test_gan_thoai_theo_dan_thoai_kieu_gemini():
    """Dòng thật của Gemini ở Chương 1."""
    dong = ("– Con tàu cuối cùng rời bến lúc một giờ kém mười – Serena nói, không "
            "nhìn thẳng vào bảng điều khiển – Tuyến đường sắt qua thung lũng số bảy "
            "đã bị niêm phong từ nửa đêm.")
    a = attribute_dialogue(dong, NAMES)
    assert len(a["by_speaker"]["CHAR_SERENA"]) == 2
    assert a["by_speaker"]["CHAR_KAELEN"] == [] and a["unattributed"] == []


def test_thoai_khong_co_dan_thi_khong_gan():
    a = attribute_dialogue("— Hết giờ tiếp nhận rồi, được chưa?", NAMES)
    assert a["unattributed"] and all(v == [] for v in a["by_speaker"].values())


def test_dan_thoai_neu_nhieu_ten_thi_khong_gan():
    a = attribute_dialogue("— Đi thôi, — Kaelen nói với Serena.", NAMES)
    assert a["unattributed"] == ["Đi thôi,"]


def test_vhal_KHONG_bi_bao_vi_cau_cua_serena(chars):
    """Lỗi §5.3: `extract_dialogue` gán mọi đoạn mở bằng gạch ngang cho mọi nhân
    vật. "theo thẩm quyền" là từ đặc trưng của Serena và từ CẤM của Vhal."""
    van = ("— Theo thẩm quyền, tôi e rằng hồ sơ cho thấy điều khác, — Serena nói.\n"
           "— Hết giờ tiếp nhận rồi, — Vhal nói.")
    c = {"pov_character": "CHAR_KAELEN", "pov_character_name": "Kaelen",
         "active_characters": [{"id": k, "name": n} for n, k in NAMES.items()],
         "word_budget": (0, 999)}
    eng = type("E", (), {"chars": chars})()
    voice = [f for f in deterministic_audit(van, c, 1, eng) if f["check"] == "voice"]
    assert not any("Vhal" in f["message"] for f in voice)


def test_nhan_vat_dung_tu_cam_cua_chinh_minh_la_major(chars):
    lines = attribute_dialogue("— Theo thẩm quyền, tôi không ký, — Vhal nói.",
                               NAMES)["by_speaker"]["CHAR_VHAL"]
    r = voice_report(lines, chars["CHAR_VHAL"])
    assert r["violations"][0]["severity"] == "major"
    assert "theo thẩm quyền" in r["violations"][0]["message"]


def test_mau_nho_khong_bao_ti_le_cau_hoi_hay_do_dai(chars):
    """Một lượt thoại → tỉ lệ câu hỏi là 0 hoặc 1, luôn ngoài khoảng."""
    r = voice_report(["Hồ sơ đã được lưu."], chars["CHAR_SERENA"])
    assert r["ok"] is True


def test_mau_du_lon_thi_bao_do_dai_lech(chars):
    """Kaelen giọng cộc (6–13 từ). Bốn câu dài 25 từ thì lệch."""
    dai = " ".join(["chữ"] * 25) + "."
    r = voice_report([dai] * 4, chars["CHAR_KAELEN"])
    assert any("trung bình" in v["message"] for v in r["violations"])
    assert all(v["severity"] == "minor" for v in r["violations"])


def test_gan_thoai_theo_doan_dan_lien_truoc_la_suy_ra():
    """Cách Gemini dẫn thoại phổ biến nhất: hành động ở đoạn trước, thoại trần."""
    van = "Vhal không trả lời ngay. Hắn lật sổ.\n\n— Hết giờ tiếp nhận rồi."
    a = attribute_dialogue(van, NAMES)
    assert a["inferred"]["CHAR_VHAL"] == ["Hết giờ tiếp nhận rồi."]
    assert a["by_speaker"]["CHAR_VHAL"] == []


def test_khong_suy_ra_nguoi_noi_khi_thoai_goi_ten_chinh_nguoi_do():
    """Ca sai thật ở Chương 1: đoạn trước mở bằng Kaelen, thoại gọi "Kaelen"."""
    van = "Kaelen trả lời ngay lập tức.\n\n— Đã quá giờ giao ca mười lăm phút, Kaelen."
    a = attribute_dialogue(van, NAMES)
    assert a["inferred"]["CHAR_KAELEN"] == [] and a["unattributed"]


def test_hai_dong_thoai_lien_nhau_khong_suy_ra_cho_dong_sau():
    van = "Vhal lật sổ.\n— Hết giờ rồi.\n— Không phải bốn."
    a = attribute_dialogue(van, NAMES)
    assert a["inferred"]["CHAR_VHAL"] == ["Hết giờ rồi."]
    assert a["unattributed"] == ["Không phải bốn."]


def test_dan_thoai_sau_dau_hoi_khong_can_gach_ngang():
    a = attribute_dialogue("— Sai số nằm ở đâu? Kaelen gằn giọng.", NAMES)
    assert a["by_speaker"]["CHAR_KAELEN"] == ["Sai số nằm ở đâu?"]


def test_tu_cam_chi_trong_thoai_suy_ra_la_minor(chars):
    r = voice_report([], chars["CHAR_VHAL"], ["Theo thẩm quyền, tôi không ký."])
    assert [v["severity"] for v in r["violations"]] == ["minor"]
    assert "cần xác nhận" in r["violations"][0]["message"]


def test_ti_le_cau_hoi_so_theo_so_dem(chars):
    """Sàn 0.02 của Kaelen: sáu lượt không câu hỏi nào vẫn là giọng Kaelen."""
    sau = ["Ca trực Alpha."] * 6
    assert not any("câu hỏi" in v["message"]
                   for v in voice_report(sau, chars["CHAR_KAELEN"])["violations"])
    hoi = ["Ca trực nào?"] * 6
    assert any("câu hỏi" in v["message"]
               for v in voice_report(hoi, chars["CHAR_KAELEN"])["violations"])


# ═══════════════ POV ═══════════════

@pytest.mark.parametrize("prose, pov, ro_ri", [
    # Câu thật, Gemini Chương 2, POV Serena kể ngôi thứ nhất.
    ("Tôi nhích nhẹ cổ tay để trang giấy không bị lệch, trong lòng thoáng dấy "
     "lên một mối hoài nghi mơ hồ.", "Serena", False),
    ("Kaelen siết tay, trong lòng dâng lên một nỗi ngờ.", "Kaelen", False),
    ("Kaelen nhìn Serena, trong lòng cô dâng lên một nỗi ngờ.", "Kaelen", True),
    ("Serena siết tay, trong lòng dâng lên một nỗi ngờ.", "Kaelen", True),
    ("— Tôi biết, trong lòng tôi đã rõ.", "Kaelen", True),
])
def test_noi_tam_cua_chinh_pov_khong_la_ro_ri(prose, pov, ro_ri):
    from novel_engine.character.firewall import pov_leak_scan
    assert bool(pov_leak_scan(prose, pov_name=pov)) is ro_ri

def _c_kaelen():
    return {"pov_character": "CHAR_KAELEN", "pov_character_name": "Kaelen",
            "active_characters": [], "word_budget": (0, 999)}


def test_pov_tu_nhan_khong_biet_KHONG_la_ro_ri():
    """§10.3 truyền `contract["pov_character"]` (mã CHAR_KAELEN) cho
    `pov_leak_scan`, vốn tìm TÊN trong văn bản để miễn trừ. Mã không bao giờ
    xuất hiện trong văn xuôi → câu hợp lệ thành blocker."""
    f = deterministic_audit("Kaelen không biết rằng mình đang bị theo dõi.",
                            _c_kaelen(), 1, None)
    assert "pov_leak" not in _checks(f)


def test_ro_ri_that_van_la_blocker():
    f = deterministic_audit("Anh không biết rằng phía sau cánh cửa là bẫy.",
                            _c_kaelen(), 1, None)
    assert any(x["check"] == "pov_leak" and x["severity"] == "blocker" for x in f)


# ═══════════════ HỢP ĐỒNG & CLI ═══════════════

def test_hop_dong_kiem_toan_lay_tu_outline_va_default_contract():
    from novel_engine.planner.contract import SceneContract
    e = build_engines(FakeLLM(), db_path=":memory:")
    try:
        c = audit_contract(e, 1, 1)
        assert c["pov_character_name"] == "Kaelen"
        assert {x["id"] for x in c["active_characters"]} == {"CHAR_KAELEN", "CHAR_VHAL"}
        assert c["word_budget"] == SceneContract.model_fields["word_budget"].default
        assert c["tension"]["mode"] in ("escalate", "sustain", "decompress")
    finally:
        e.store.close()


def test_so_do_hieu_chinh():
    st = audit_stats(VARIED + "\n— Đi thôi, — Kaelen nói.",
                     {"active_characters": [{"id": "CHAR_KAELEN", "name": "Kaelen"}]})
    assert st["rhythm"]["sd"] > 8.5
    assert st["dialogue_attributed"] == {"CHAR_KAELEN": 1}


def test_cli_audit(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_LLM", "fake")
    d = tmp_path / "output" / "chapters"
    d.mkdir(parents=True)
    (d / "ch001.md").write_text(
        f"# Chương 1\n\n## Cảnh 0\n\n{MONOTONE}\n\n## Cảnh 1\n\n{VARIED}\n",
        encoding="utf-8")
    assert cli.main(["audit", "--chapter", "1"]) == 0
    out = capsys.readouterr().out
    assert "CH001_S00" in out and "rhythm_variance" in out
    rep = json.loads((tmp_path / "output" / "reports" / "ch001_audit.json")
                     .read_text(encoding="utf-8"))
    assert len(rep["scenes"]) == 2
    assert cli.main(["audit", "--chapter", "9"]) == 2
