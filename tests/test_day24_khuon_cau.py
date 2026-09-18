"""Ngày 24 — lặp CẤU TRÚC, thứ mà bỏ danh sách khẩu ngữ không chữa được.

Bỏ `signature_lexicon` khỏi prompt làm lặp TỪ giảm tám lần. Đọc kỹ bản viết lại
thì lặp chuyển sang tầng trên: "bốn mươi lăm phần trăm" 14 lần trên 5 chương
(không hề có trong bible), lượt thoại chỉ gồm một con số đi 1 → 3 → 8, và cả ba
câu mẫu của Serena bị model hợp thành một khuôn duy nhất.

Cùng một sai lầm, một tầng cao hơn: câu mẫu dạy được CÁI KHUÔN, và model đóng
dấu cái khuôn.
"""
from __future__ import annotations

import re
import unicodedata

import pytest

from novel_engine.audit.critique import POLISHABLE_CHECKS
from novel_engine.audit.tics import (COUNTING_PER_CHAPTER, FIGURE_PER_CHAPTER,
                                     SOMATIC_PER_CHAPTER, counting_findings,
                                     counting_speech, figure_findings,
                                     somatic_findings)
from novel_engine.canon.bible import DEFAULT_BIBLE, load_characters
from novel_engine.prompts import WRITER_TMPL


@pytest.fixture(scope="module")
def chars():
    return load_characters(DEFAULT_BIBLE)


# ═══════════ 1. MỘT CON SỐ CỤ THỂ CŨNG LÀ MỘT KHẨU NGỮ ═══════════

def test_con_so_lap_lai_thanh_khau_hieu():
    """"bốn mươi lăm phần trăm" xuất hiện 14 lần trên 5 chương của bản viết
    lại, bốn lần riêng Chương 1 — và nó KHÔNG có trong bible. Model tự sinh nó
    từ câu mẫu rồi dùng làm câu mồi mỗi lần chuyển mạch tư duy."""
    truoc = [{"prose": "Bốn mươi lăm phần trăm. Anh dừng lại."},
             {"prose": "Bốn mươi lăm phần trăm khả năng là vậy."}]
    f = figure_findings("Bốn mươi lăm phần trăm. Kaelen gật đầu.", truoc)
    assert [x["evidence"] for x in f] == ["bốn mươi lăm phần trăm"]
    assert "figure_overuse" in POLISHABLE_CHECKS

    # Con số khác thì không dính, và cảnh không nhắc lại cũng không dính.
    assert figure_findings("Ba mươi phút nữa tàu rời bến.", truoc) == []
    assert figure_findings("Bốn mươi lăm phần trăm.", []) == []


def test_so_khong_kem_don_vi_thi_khong_tinh():
    """"ba người", "hai lần gõ" là văn kể bình thường; chỉ số CÓ ĐƠN VỊ mới là
    thứ nhân vật lặp lại như một khẩu hiệu."""
    truoc = [{"prose": "Ba người đã ký. Ba người khác thì không."}] * 3
    assert figure_findings("Ba người nữa bước vào.", truoc) == []


# ═══════════ 2. ĐẾM THÀNH TIẾNG PHẢI DẪN ĐI ĐÂU ĐÓ ═══════════

def test_thoai_chi_gom_mot_con_so():
    """"— Ba." rồi "— Ba người kiểm tra trước anh đã đóng dấu" là tật làm việc:
    con số dẫn vào một quan sát. "— Ba." đứng trơ thì không dẫn đi đâu."""
    van = chr(10).join(["— Ba.", "— Ba người đã ký vào góc trái.", "— Tám."])
    assert counting_speech(van) == ["— ba.", "— tám."]

    assert counting_findings("— Tám.", []) == []          # lần đầu: là nhân vật
    f = counting_findings("— Tám.", [{"prose": "— Ba."}])  # lần thứ hai: là máy
    assert f and f[0]["check"] == "counting_tic"
    assert COUNTING_PER_CHAPTER == 1
    assert "counting_tic" in POLISHABLE_CHECKS


def test_con_so_dan_vao_mot_cau_thi_khong_bi_tinh():
    assert counting_speech("— Ba. Ba người kiểm tra trước anh đã đóng dấu.") == []


# ═══════════ 3. CỬ CHỈ: MỘT LẦN MỖI CHƯƠNG ═══════════

def test_cu_chi_chi_con_mot_lan_moi_chuong(chars):
    """Trần 3 vẫn quá rộng: đọc bản viết lại, cử chỉ nhận dạng có mặt ở 80–90%
    số cảnh nên chúng thôi nhận dạng ai và thành tiếng động nền."""
    assert SOMATIC_PER_CHAPTER == 1
    cu_chi = chars["CHAR_KAELEN"].somatic_signature[0]
    ct = {"active_characters": [{"id": "CHAR_KAELEN"}]}
    f = somatic_findings(cu_chi + ".", ct, chars, previous_scenes=[{"prose": cu_chi}])
    assert [x["check"] for x in f] == ["somatic_budget"]
    assert "mất tự chủ nhất" in WRITER_TMPL


# ═══════════ 4. KHÔNG TRƯỜNG NÀO ĐI VÀO PROMPT ĐƯỢC DẠY MỘT KHUÔN ═══════════

def test_cau_mau_khong_dung_chung_mot_khuon(chars):
    """Cả ba câu mẫu cũ của Serena đều là "nhượng bộ rồi tuy nhiên, chốt bằng
    câu hỏi" — và `syntactic_tic` của cô còn trích nguyên chữ 'tuy nhiên'. Model
    hợp chúng lại thành một khuôn duy nhất và đọc như đọc script.

    Câu mẫu phải khác NHAU về hình dạng, nếu không chúng dạy một cái khuôn thay
    vì một cách nói.
    """
    for prof in chars.values():
        mau = prof.voice.voice_exemplars
        assert len(mau) >= 2, prof.name

        # Không câu nào được trích trong `syntactic_tic`, và ngược lại.
        tic = unicodedata.normalize("NFC", prof.voice.syntactic_tic).lower()
        assert "'" not in tic and '"' not in tic, f"{prof.name}: tic trích nguyên văn"

        # Không quá một nửa số câu mẫu mở đầu bằng cùng một loại cấu trúc.
        so = sum(1 for x in mau if re.match(
            r"^(Không|Một|Hai|Ba|Bốn|Năm|Sáu|Bảy|Tám|Chín|Mười)" + chr(92) + "b", x))
        assert so <= len(mau) / 2, f"{prof.name}: {so}/{len(mau)} câu mẫu mở bằng số"


def test_serena_khong_con_khuon_tuy_nhien(chars):
    v = chars["CHAR_SERENA"].voice
    van = unicodedata.normalize("NFC", " ".join(
        list(v.voice_exemplars) + [v.syntactic_tic])).lower()
    assert "tuy nhiên" not in van
    assert "anh chắc là mình muốn" not in van


# ═══════════ 5. NGÂN SÁCH PHẢI NÓI TRƯỚC KHI VIẾT ═══════════

def test_ngan_sach_da_tieu_di_vao_prompt():
    """`counting_tic` và `figure_overuse` NỔ ĐÚNG ở ch1, ch3, ch4 của lượt viết
    lại — mà số lượt thoại chỉ gồm một con số vẫn y nguyên 8.

    Vì chúng là `minor` nên đi polish, và polish không sửa được: bỏ "— Ba." đi
    thì phải VIẾT một câu thoại thay vào, mà bịa nội dung không phải việc của
    polish. Cùng một sai lầm với ngân sách cử chỉ, cùng một cách chữa: code
    đếm, prompt phát biểu, Writer tự tránh.
    """
    from novel_engine.graph.nodes import _ngan_sach_da_dung
    from novel_engine.prompts import WRITER_TMPL

    assert "{ngan_sach}" in WRITER_TMPL
    assert _ngan_sach_da_dung([]) == ""          # cảnh đầu chương: chưa tiêu gì

    truoc = [{"prose": "— Ba.\nBốn mươi lăm phần trăm. Anh dừng lại."},
             {"prose": "— Hai.\nBốn mươi lăm phần trăm khả năng là vậy."}]
    ra = _ngan_sach_da_dung(truoc)
    assert "đếm thành tiếng 2 lần" in ra
    assert "bốn mươi lăm phần trăm" in ra
    assert "KHÔNG nhắc lại" in ra


def test_ngan_sach_im_lang_khi_chua_cham_tran():
    """Nói "anh chưa tiêu gì cả" ở mọi cảnh là dạy model chú ý tới một thứ nó
    chưa làm — và nhắc một tật là cách nhanh nhất để model dùng nó."""
    from novel_engine.graph.nodes import _ngan_sach_da_dung

    assert _ngan_sach_da_dung([{"prose": "Anh bước vào phòng trực. Trời lạnh."}]) == ""
