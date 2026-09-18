"""Ba lỗi đọc-thấy-ngay của Arc 1: lặp cảnh, nhồi khẩu ngữ, rác văn bản.

Ngưỡng ở `repetition.py` chọn bằng cách ĐO trên chính năm chương đã sinh, không
bằng cách đoán — test dưới đây giữ cho phép đo đó còn tách bạch: cặp cảnh dựng
lại nhau bị bắt, cặp cảnh bình thường (cùng thế giới, cùng thuật ngữ) thì không.
"""
from __future__ import annotations

import pytest

from novel_engine.audit.critique import POLISHABLE_CHECKS, routing_severity, writer_feedback
from novel_engine.audit.deterministic import audit_stats, deterministic_audit
from novel_engine.audit.repetition import (
    CONTAINMENT_MAJOR, containment, longest_shared, repetition_findings,
)
from novel_engine.audit.text_hygiene import (
    dialogue_style, foreign_letters, hygiene_findings, sanitize_prose,
    leaked_speech, so_thanh_chu, so_thoai_thanh_chu, strip_leaked_indices,
)
from novel_engine.audit.tics import (
    SOMATIC_PER_CHAPTER, numeric_opener, somatic_findings, tic_findings,
)
from novel_engine.canon.bible import DEFAULT_BIBLE, load_characters
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.prompts import WRITER_TMPL

CAU = [
    "Cần trục kêu ba tiếng rồi im bặt trên mặt nước đen của cảng quặng lúc rạng sáng",
    "Mùi lưu huỳnh bám vào cổ áo và không chịu tan ra dù gió thổi mạnh từ phía biển",
    "Nền rung một nhịp dưới gót chân khi đoàn tàu chở quặng cuối cùng rời khỏi bến",
    "Ánh đèn pha quét ngang mặt nước rồi dừng lại ở khoang hàng số bốn phía đuôi tàu",
    "Giấy tờ trong tay viên chức sột soạt một lúc lâu trước khi con dấu được hạ xuống",
    "Một tiếng động khô phía sau bức tường làm cả hai người cùng ngẩng đầu lên nhìn",
    "Không khí khô rát ở đầu lưỡi giống hệt đêm xảy ra sự cố ở lò phản ứng số ba",
    "Hơi nước đọng thành giọt trên mép ống dẫn và rơi xuống sàn thép theo từng nhịp",
]


def _canh(idx: list[int], n: int = 4) -> str:
    return " ".join(CAU[i % len(CAU)] + "." for i in idx * n)


def _scene(sid, prose):
    return {"scene_id": sid, "prose": prose}


# ═══════════════ 1. LẶP CẢNH ═══════════════

def test_canh_dung_lai_canh_truoc_bi_bat_con_canh_khac_thi_khong():
    goc = _canh([0, 1, 2, 3, 4, 5])
    dung_lai = _canh([0, 1, 2, 3, 4, 6])          # gần như cùng vật liệu
    khac = _canh([6, 7, 6, 7, 6, 7])              # cùng thế giới, khác nội dung
    assert containment(dung_lai, goc) >= CONTAINMENT_MAJOR
    assert containment(khac, goc) < CONTAINMENT_MAJOR

    f = repetition_findings(dung_lai, [_scene("CH001_S00", goc)])
    assert [x["check"] for x in f][0] == "scene_repetition"
    assert f[0]["severity"] == "major" and "CH001_S00" in f[0]["message"]
    assert repetition_findings(khac, [_scene("CH001_S00", goc)]) == []


def test_bang_chung_chi_dung_doan_bi_lap():
    goc = _canh([0, 1, 2, 3])
    assert longest_shared(_canh([0, 1]), goc).startswith("cần trục kêu ba tiếng")


def test_cau_tran_thuat_va_luot_thoai_lap_nguyen_van():
    goc = _canh([0, 1, 2, 3]) + "\n— Hết giờ tiếp nhận rồi, được chưa?"
    moi = _canh([4, 5, 6, 7]) + "\n— Hết giờ tiếp nhận rồi, được chưa?"
    checks = {x["check"] for x in repetition_findings(moi, [_scene("CH001_S00", goc)])}
    assert checks == {"dialogue_repetition"}

    lap_tran = _canh([4, 5, 6, 7]) + " " + CAU[0] + "."
    f = repetition_findings(lap_tran, [_scene("CH001_S00", goc)])
    assert any(x["check"] == "sentence_repetition" and x["severity"] == "major"
               for x in f)


def test_canh_qua_ngan_khong_bi_cham_diem():
    assert repetition_findings("Ngắn. Cụt.", [_scene("S", _canh([0, 1, 2, 3]))]) == []


def test_lap_canh_phai_VIET_LAI_khong_phai_trau_chuot():
    assert "scene_repetition" not in POLISHABLE_CHECKS
    assert "sentence_repetition" not in POLISHABLE_CHECKS
    assert "dialogue_repetition" in POLISHABLE_CHECKS
    f = [{"check": "scene_repetition", "severity": "major", "source": "code",
          "message": "trùng 14%"}]
    assert routing_severity(f) == "major" and "trùng 14%" in writer_feedback(f)


# ═══════════════ 2. NGÂN SÁCH KHẨU NGỮ ═══════════════

@pytest.fixture(scope="module")
def chars():
    return load_characters(DEFAULT_BIBLE)


def _contract(*ids):
    return {"active_characters": [{"id": i, "name": i.split("_")[1].title()} for i in ids]}


def test_cum_nhieu_tu_lap_qua_hai_lan_bi_bat(chars):
    van = chr(10).join(
        ["— Theo điều lệ, cấp trên của anh tự quyết. Biên bản không thuộc "
         "phần tôi."] * 3)
    f = tic_findings(van, _contract("CHAR_SERENA"), chars)
    qua = {x["evidence"] for x in f if x["check"] == "tic_overuse"}
    assert {"theo điều lệ", "cấp trên của anh", "không thuộc phần tôi"} <= qua
    assert any(x["check"] == "tic_budget" for x in f)


def test_thuat_ngu_cua_the_gioi_duoc_nong_tay_hon(chars):
    """"ca trực", "áp suất" vừa là khẩu ngữ của Kaelen vừa là danh từ của thế
    giới này — cấm chặt là báo động giả (P2)."""
    van = "\n".join(["— Ca trực này áp suất vẫn trong dung sai."] * 3)
    assert [x for x in tic_findings(van, _contract("CHAR_KAELEN"), chars)
            if x["check"] == "tic_overuse"] == []


def test_khau_ngu_tran_vao_loi_ke_bi_bat(chars):
    van = ("Serena chỉnh mép giấy. Theo điều lệ, mọi lộ trình đã bị khoá. "
           "Theo điều lệ, không ai được qua.")
    assert any(x["check"] == "lexicon_in_narration"
               for x in tic_findings(van, _contract("CHAR_SERENA"), chars))


@pytest.mark.parametrize("mo, co", [
    ("45. Bảy phút. Kaelen bước vào.", True),
    ("03. Hai mươi bốn.\n\nKaelen bước vào.", True),
    ("Kaelen bước vào lúc 45 phút sau.", False),
])
def test_mo_canh_bang_so_tran(mo, co):
    assert (numeric_opener(mo) is not None) is co


# ═══════════════ 3. VỆ SINH VĂN BẢN ═══════════════

def test_ky_tu_ngoai_he_latin_bi_bat_va_duoc_don():
    van = "Tiếng bánh răng nghiến vào nhau kenкет, đều đặn. Còn khoಹ್ khắc nữa."
    la = foreign_letters(van)
    assert {x["word"] for x in la} == {"kenкет,", "khoಹ್"}
    assert any(x["check"] == "foreign_script" for x in hygiene_findings(van))

    sach, ghi_chu = sanitize_prose(van)
    assert foreign_letters(sach) == [] and "ken," in sach
    assert ghi_chu and "ngoài hệ Latin" in ghi_chu[0]


def test_khong_dung_toi_van_ban_sach(chars):
    van = "Kaelen bước vào. — Đi thôi."
    sach, ghi_chu = sanitize_prose(van)
    assert sach == van and ghi_chu == []


def test_chuan_hoa_gach_dau_dong_va_bat_lan_quy_uoc():
    sach, ghi_chu = sanitize_prose("- Đi thôi.\n– Khoan đã.")
    assert sach.splitlines() == ["— Đi thôi.", "— Khoan đã."]
    assert "gạch đầu dòng" in ghi_chu[0]

    assert dialogue_style('"Đi thôi," Kaelen nói.') == "quote"
    assert dialogue_style('— Đi thôi.\n"Khoan," Serena nói.') == "mixed"
    assert dialogue_style("— Đi thôi.") == "dash"
    assert any(x["check"] == "dialogue_style"
               for x in hygiene_findings('"Đi thôi," Kaelen nói.'))


# ═══════════════ NỐI VÀO ĐƯỜNG ỐNG ═══════════════

class _LapLai(FakeLLM):
    """Writer trả CÙNG một văn xuôi ở mọi cảnh, kèm một ký tự rác."""

    def _prose(self, prompt: str) -> str:
        return _canh([0, 1, 2, 3, 4, 5]) + " Tiếng bánh răng kenкет."


def test_do_thi_bat_lap_canh_va_don_rac():
    eng = build_engines(_LapLai(), db_path=":memory:")
    try:
        out = run_chapter(eng, 1)
        assert not out.get("escalated"), out.get("escalation_reason")
        # Cảnh 0 không có gì để so; từ cảnh 1 trở đi phải bị bắt và viết lại.
        checks = [c for x in out["audit_log"] for c in x["checks"]]
        assert "scene_repetition:major" in checks
        assert any(s["audit"]["revisions"] >= 1 for s in out["scene_outputs"][1:])
        # Ký tự rác bị dọn trước khi vào văn bản chương.
        assert all("кет" not in s["prose"] for s in out["scene_outputs"])
        assert out["hygiene_notes"] and "Latin" in out["hygiene_notes"][0]["notes"][0]
    finally:
        eng.store.close()


def test_so_do_hieu_chinh_co_trung_lap_va_quy_uoc_thoai():
    st = audit_stats(_canh([0, 1, 2, 3]), {"active_characters": []},
                     previous_scenes=[_scene("S0", _canh([0, 1, 2, 3]))])
    assert st["max_repetition"] > 0.5 and st["dialogue_style"] == "none"


def test_prompt_writer_mang_ba_luat_moi():
    assert "KHÔNG mở đầu cảnh bằng một con số trần" in WRITER_TMPL
    assert "gạch đầu dòng" in WRITER_TMPL and "ngoặc kép" in WRITER_TMPL
    assert "quá hai lần trong" in WRITER_TMPL
    assert "khác các cảnh trước ở TRẠNG THÁI" in WRITER_TMPL


# ═══════════════ RÒ RỈ CHỈ SỐ & NGÂN SÁCH CỬ CHỈ ═══════════════

@pytest.mark.parametrize("van, con_lai", [
    ("45. Bảy phút. Kaelen bước vào.", "Bảy phút. Kaelen bước vào."),
    ("0.04.\n\nÁp suất tụt.", "Áp suất tụt."),
    # Chỉ số trôi xuống GIỮA cảnh và CUỐI dòng, không riêng dòng đầu.
    ("Anh dừng lại.\n\n03. Cửa hầm mở.", "Anh dừng lại."),
    ("Gót giày nện xuống nền, đếm nhịp. 04.", "Gót giày nện xuống nền, đếm nhịp."),
    # Đứng RIÊNG giữa đoạn, không ở đầu cũng không ở cuối dòng. Ca này lọt qua
    # cả ba luật kia suốt một lượt viết lại nguyên arc, và M15 vẫn báo 0 — một
    # bộ đo mù báo số 0 còn tệ hơn không đo, vì nó làm người đọc yên tâm.
    ("Gã lật lại tờ giấy. 03. Một số hiệu nhảy một quãng.",
     "Gã lật lại tờ giấy. Một số hiệu nhảy một quãng."),
    ("Áp suất tụt còn 45. Anh ghi lại.", None),   # số kết câu, không có 0 đứng đầu
    ("Hai. Hai mươi tư giờ.", None),          # tật đếm viết bằng CHỮ: của nhân vật
    ("Kaelen chờ 45 phút.", None),            # số giữa câu: không phải rò rỉ
])
def test_cat_chi_so_ro_ri_o_moi_dong(van, con_lai):
    sach, ghi = strip_leaked_indices(van)
    if con_lai is None:
        assert ghi == [] and sach == van
    else:
        assert ghi and sach.strip().startswith(con_lai.split("\n")[0])


def test_so_thanh_chu():
    assert [so_thanh_chu(n) for n in (0, 4, 10, 15, 21, 24, 45, 103)] == [
        "không", "bốn", "mười", "mười lăm", "hai mươi mốt", "hai mươi tư",
        "bốn mươi lăm", "một trăm lẻ ba"]
    assert so_thanh_chu(1000) == ""      # ngoài tầm: trả rỗng, không đọc bừa


def test_tat_dem_thanh_tieng_viet_bang_chu_khong_phai_viet_lai():
    """"— 4." là Kaelen ĐẾM THÀNH TIẾNG, không phải chỉ số beat.

    Lượt viết lại Arc 1 cho thấy luật gộp chung nổ 17 lần mà một nửa là báo
    nhầm: bản Arc 1 lượt 2 viết đúng tật ấy thành "— Bốn." rồi "— Bốn gì?".
    Bắt viết lại cả cảnh vì một chữ số là đốt tiền và xoá mất tật nhân vật.
    """
    van = "— 4." + chr(10) + "— Gì cơ?"
    sach, ghi = so_thoai_thanh_chu(van)
    assert sach.startswith("— Bốn.") and ghi
    assert leaked_speech(sach) == []

    # Câu dẫn đi sau vẫn phải còn nguyên, và dấu cách không được nuốt mất.
    sach, _ = so_thoai_thanh_chu("— 1. — Kaelen nói, giọng phẳng.")
    assert sach == "— Một. — Kaelen nói, giọng phẳng."


def test_chi_so_beat_trong_thoai_thi_phai_viet_lai():
    """Số 0 đứng đầu hoặc có phần thập phân thì không ai đếm thành tiếng.

    "— 01.", "— 0.03." là chỉ số beat trôi vào chỗ đáng lẽ có một câu thoại.
    Code không bịa được câu còn thiếu, nên đây là lỗi SÁNG TÁC: phải viết lại,
    và vì thế nó nằm ngoài danh sách lỗi polish sửa được.
    """
    for van in ("— 01.", "— 0.03. — Vhal đáp."):
        sach, ghi = so_thoai_thanh_chu(van)
        assert sach == van and ghi == []
        assert leaked_speech(van)
        f = [x for x in hygiene_findings(van)
             if x["check"] == "leaked_index_in_dialogue"]
        assert f and f[0]["severity"] == "major"
    assert "leaked_index_in_dialogue" not in POLISHABLE_CHECKS


def test_dem_cu_chi_khong_phu_thuoc_trat_tu_tu():
    """Tiếng Việt đảo trật tự thoải mái; bộ đếm phải đảo theo.

    Bible ghi "ấn ngón cái vào khớp ngón trỏ…". Bản viết lại Chương 1 diễn đạt
    nó sáu lần theo sáu kiểu, bộ đếm khớp-theo-chuỗi chỉ thấy một.
    """
    from novel_engine.audit.tics import _count_gesture

    g = "ấn ngón cái vào khớp ngón trỏ cho tới khi nghe tiếng kêu"
    for cau in ["Ngón cái Kaelen ấn mạnh vào khớp ngón trỏ bên phải.",
                "Kaelen đặt ngón cái lên khớp ngón trỏ, ép mạnh.",
                "ngón tay cái ấn chặt vào khớp ngón trỏ đến khi đốt xương trắng."]:
        assert _count_gesture(cau.lower(), g) == 1, cau
    # Bàn tay làm việc khác thì không tính — nếu không, ngân sách thành báo động giả.
    assert _count_gesture("ngón tay cái miết cạnh một con dấu cao su.".lower(), g) == 0
    assert _count_gesture("đầu ngón tay bấu lấy cán bút.".lower(), g) == 0


def test_ngan_sach_cu_chi_tinh_ca_chuong(chars):
    cu_chi = chars["CHAR_KAELEN"].somatic_signature[0]
    ct = _contract("CHAR_KAELEN")
    mot_lan = f"{cu_chi.capitalize()}. " + _canh([0, 1])
    assert somatic_findings(mot_lan, ct, chars) == []

    hai_lan = f"{cu_chi}. " * 2 + _canh([0, 1])
    assert [x["check"] for x in somatic_findings(hai_lan, ct, chars)] == ["somatic_overuse"]

    truoc = [_scene(f"CH001_S0{i}", f"{cu_chi}. " + _canh([i]))
             for i in range(SOMATIC_PER_CHAPTER)]
    f = somatic_findings(mot_lan, ct, chars, previous_scenes=truoc)
    assert [x["check"] for x in f] == ["somatic_budget"]


def test_cu_chi_lap_la_loi_POLISH_sua_duoc():
    assert {"somatic_overuse", "somatic_budget"} <= POLISHABLE_CHECKS


# ═══════════════ STATE TRACKER ═══════════════

def test_canh_sau_bat_dau_o_noi_canh_truoc_ket_thuc():
    """Không nối trạng thái thì mỗi cảnh tự nghĩ `entry_state` của riêng nó, và
    Chương 1 của Arc 1 ra ba dị bản của cùng một cuộc đối thoại."""
    eng = build_engines(FakeLLM(), db_path=":memory:")
    try:
        out = run_chapter(eng, 1)
        ct = out["contracts"]
        noi = [(ct[i - 1]["exit_state"], ct[i]["entry_state"])
               for i in range(1, len(ct)) if ct[i - 1].get("exit_state")]
        assert noi and all(a == b for a, b in noi)
        assert all(c["scene_must_change"] for c in ct)
    finally:
        eng.store.close()


def test_long_ban_tay_khong_phai_ro_ri_noi_tam():
    """"lòng bàn tay" là bộ phận cơ thể, không phải nội tâm người khác.

    Lượt viết lại Chương 1 escalate hai lần vì một câu tả bàn tay: luật đúng ý
    định nhưng sai không gian đối chiếu, và cái giá không phải là nhiễu mà là
    một chương bị chặn.
    """
    from novel_engine.character.firewall import pov_leak_scan

    assert pov_leak_scan("mép giấy bị siết chặt trong lòng bàn tay Kaelen.",
                         "Kaelen") == []
    assert pov_leak_scan("Con thuyền nằm giữa lòng sông.", "Kaelen") == []
    assert [h["check"] for h in
            pov_leak_scan("Trong lòng, Serena đã quyết định từ lâu.", "Kaelen")
            ] == ["pov_leak"]
