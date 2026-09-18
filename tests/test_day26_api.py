"""Ngày 26 — mặt tiền lập trình cho giao diện.

`cli.py` trộn ba việc: truy vấn động cơ, định dạng bảng terminal, trả mã thoát.
Giao diện chỉ cần việc đầu. Nếu nó phải gọi CLI rồi bóc chữ từ stdout thì mọi
lần đổi một dòng in sẽ làm hỏng giao diện — nên `api.py` là hợp đồng, và các
bài dưới đây chốt đúng những điều một tầng UI dựa vào.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from novel_engine import api
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM


@pytest.fixture()
def du_an(tmp_path):
    """Một dự án thật, viết bằng FakeLLM: không tốn gì, và tất định."""
    db = str(tmp_path / "canon.db")
    eng = build_engines(FakeLLM(), db_path=db)
    run_chapter(eng, 1)
    ch_dir = tmp_path / "chapters"
    ch_dir.mkdir()
    st = eng.store
    van = "\n\n".join(
        f"## Cảnh {i}\n\n{d}" for i, d in
        enumerate(st.recent_scene_digests(2, 0, k=6)))
    (ch_dir / "ch001.md").write_text(f"# Chương 1\n\n{van}\n", encoding="utf-8")
    return {"db": db, "chapters_dir": str(ch_dir), "tmp": tmp_path}


# ═══════════ HỢP ĐỒNG: MỌI THỨ JSON HOÁ ĐƯỢC ═══════════

def test_moi_ham_doc_deu_json_hoa_duoc(du_an):
    """Một `Enum` hay `Path` lọt vào sẽ chỉ nổ ở tầng HTTP, xa chỗ gây ra nó.

    `ClueStatus` là ca thật: nó là Enum, và `manh_moi()` phải tự dẹp thành
    chuỗi chứ không đẩy trách nhiệm ấy cho người gọi.
    """
    db = du_an["db"]
    for ten, ra in [
        ("trang_thai", api.trang_thai(db)),
        ("danh_sach_chuong", api.danh_sach_chuong(db)),
        ("nhan_vat", api.nhan_vat(db)),
        ("manh_moi", api.manh_moi(db)),
        ("doc_chuong", api.doc_chuong(db, 1, du_an["chapters_dir"])),
    ]:
        json.dumps(ra, ensure_ascii=False)          # ném thì test đỏ, đúng ý


def test_khong_in_ra_khong_thoat_tien_trinh(capsys, du_an):
    """Một hàm thư viện in ra stdout sẽ làm hỏng mọi giao thức nào dùng stdout
    (JSON-RPC, MCP, pipe), và `sys.exit` thì giết luôn server."""
    api.trang_thai(du_an["db"])
    api.danh_sach_chuong(du_an["db"])
    api.nhan_vat(du_an["db"])
    assert capsys.readouterr().out == ""

    src = Path(api.__file__).read_text(encoding="utf-8")
    than = re.sub(r'"""(?:.|\n)*?"""', "", src)     # bỏ docstring
    assert "print(" not in than
    assert "sys.exit" not in than
    assert "argparse" not in than


# ═══════════ ĐỌC ═══════════

def test_trang_thai_bao_lien_tuc_sach(du_an):
    t = api.trang_thai(du_an["db"])
    assert t["so_chuong"] == 1 and t["so_canh"] == 6
    assert t["lien_tuc"]["sach"] is True
    assert t["chuong_da_viet"] == [1]


def test_danh_sach_chuong_liet_ke_ca_chuong_chua_viet(du_an):
    """Giao diện cần thấy CẢ outline, không chỉ phần đã viết — nếu không thì
    không có chỗ nào để bấm "viết chương 2"."""
    ds = api.danh_sach_chuong(du_an["db"])
    assert len(ds) > 1
    assert ds[0]["chapter"] == 1 and ds[0]["trang_thai"] == "đã viết"
    assert ds[1]["trang_thai"] == "chưa viết"
    assert ds[1]["so_canh_ke_hoach"] > 0
    assert ds[1]["title"]


def test_doc_chuong_khong_ton_tai_tra_ve_rong_chu_khong_nem(du_an):
    """Giao diện mở một chương chưa viết là chuyện bình thường, không phải lỗi."""
    # Chương CÓ trong dàn ý mà chưa viết: giao diện bấm "viết" được.
    chua = api.doc_chuong(du_an["db"], 2, du_an["chapters_dir"])
    assert chua["ton_tai"] is False and chua["trong_outline"] is True
    assert chua["title"] and chua["scenes"] == []

    # Chương KHÔNG có trong dàn ý: cũng rỗng, nhưng nói rõ là chuyện khác.
    lac = api.doc_chuong(du_an["db"], 99, du_an["chapters_dir"])
    assert lac["ton_tai"] is False and lac["trong_outline"] is False
    assert lac["title"] == "" and lac["so_tu"] == 0


def test_doc_chuong_kem_chi_so_doc_duoc(du_an):
    d = api.doc_chuong(du_an["db"], 1, du_an["chapters_dir"])
    assert d["ton_tai"] is True and d["scenes"]
    st = d["scenes"][0]["stats"]
    for khoa in ("words", "dialogue_style", "max_repetition"):
        assert khoa in st


def test_nhan_vat_dua_ca_thuoc_do_ra_cho_tac_gia(du_an):
    """`signature_lexicon` KHÔNG đi vào prompt Writer, nhưng PHẢI ra tới giao
    diện: tác giả cần thấy và sửa nó. Hai đường khác nhau cho hai người đọc."""
    ds = {n["name"]: n for n in api.nhan_vat(du_an["db"])}
    assert "Serena" in ds
    s = ds["Serena"]
    assert s["register"] == "clinical"
    assert s["signature_lexicon"] and s["voice_exemplars"]
    assert len(s["mean_sentence_len"]) == 2


# ═══════════ GHI ═══════════

def test_viet_chuong_phat_tien_do(tmp_path):
    """Một chương mất vài phút; `invoke()` im lặng cho tới khi xong. Giao diện
    nào cũng cần biết nó đang ở đâu."""
    su_kien = []
    ra = api.viet_chuong(str(tmp_path / "c.db"), 1, "fake",
                         on_event=su_kien.append)
    assert ra["so_canh"] == 6 and ra["escalated"] is False
    assert len(su_kien) > 5
    cuoi = su_kien[-1]
    assert cuoi["scenes_done"] == cuoi["scenes_total"] == 6
    json.dumps(su_kien, ensure_ascii=False)


def test_loi_trong_callback_khong_giet_chuong_dang_viet(tmp_path):
    """Tầng giao diện hỏng thì việc của nó; văn xuôi đã viết không được mất."""
    def no(_):
        raise RuntimeError("giao diện sập")

    ra = api.viet_chuong(str(tmp_path / "c.db"), 1, "fake", on_event=no)
    assert ra["so_canh"] == 6


def test_viet_de_phai_hoi_truoc(tmp_path):
    """Viết đè là thao tác PHÁ HUỶ — nó xoá frame, digest và delta của bản cũ.
    Giao diện phải hỏi, y như CLI bắt `--force`."""
    db = str(tmp_path / "c.db")
    api.viet_chuong(db, 1, "fake")
    with pytest.raises(FileExistsError, match="force"):
        api.viet_chuong(db, 1, "fake")
    assert api.viet_chuong(db, 1, "fake", force=True)["so_canh"] == 6


# ═══════════ SỬA BIBLE TỪ GIAO DIỆN ═══════════

@pytest.fixture()
def bible_tam(tmp_path):
    """Bản sao bible để test ghi thật mà không đụng bible dự án."""
    import shutil
    from novel_engine.canon.bible import DEFAULT_BIBLE
    d = tmp_path / "bible"
    shutil.copytree(DEFAULT_BIBLE, d)
    return d


def test_sua_giong_ghi_duoc_xuong_bible(bible_tam):
    """Khung chỉnh giọng phải GHI được, nếu không nó chỉ là khung hiển thị."""
    from novel_engine.canon.bible import load_characters

    r = api.sua_giong("CHAR_VHAL",
                      {"voice_exemplars": ["Tôi ký cái gì thì tôi chịu cái "
                                           "đó, anh hiểu không."]},
                      bible_dir=bible_tam)
    assert r["da_sua"] == ["voice_exemplars"]
    v = load_characters(bible_tam)["CHAR_VHAL"].voice
    assert v.voice_exemplars == ["Tôi ký cái gì thì tôi chịu cái đó, anh hiểu không."]
    assert v.signature_lexicon                      # trường khác không mất


def test_sua_giong_khong_ghi_gi_khi_khong_hop_le(bible_tam):
    """Ghi thẳng YAML từ ô nhập liệu là cách chắc chắn để có một bible không
    nạp được — và lúc đó MỌI lệnh đều hỏng, không riêng cái vừa bấm."""
    f = bible_tam / "characters" / "vhal.yaml"
    truoc = f.read_text(encoding="utf-8")

    with pytest.raises(Exception, match="đang đếm"):
        api.sua_giong("CHAR_VHAL",
                      {"voice_exemplars": ["Hết giờ tiếp nhận rồi, anh về đi."]},
                      bible_dir=bible_tam)
    assert f.read_text(encoding="utf-8") == truoc

    with pytest.raises(Exception):                  # min > max
        api.sua_giong("CHAR_VHAL", {"mean_sentence_len": [30, 5]},
                      bible_dir=bible_tam)
    assert f.read_text(encoding="utf-8") == truoc


def test_sua_giong_nhan_vat_khong_co(bible_tam):
    with pytest.raises(KeyError):
        api.sua_giong("CHAR_KHONG_TON_TAI", {"register": "formal"},
                      bible_dir=bible_tam)


def test_luat_ro_cum_la_rang_buoc_cua_MODEL_khong_phai_cua_test():
    """Hậu quả của lỗi này không nhìn thấy được ở chỗ gây ra: bible vẫn nạp,
    prompt vẫn chạy, văn vẫn ra — chỉ có điều thước đo và nguyên liệu là một.
    Nên nó phải chặn ngay lúc dựng model, không chờ một bài test nhớ ra."""
    from novel_engine.character.models import VoiceFingerprint

    chung = dict(mean_sentence_len=(8, 16), max_sentence_len=30,
                 register="clipped", forbidden_lexicon=[],
                 question_ratio=(0.1, 0.3), self_reference_rate=0.1,
                 under_stress_shift="ngắn lại")

    with pytest.raises(ValueError, match="đang đếm"):
        VoiceFingerprint(signature_lexicon=["theo quy trình"],
                         voice_exemplars=["Theo quy trình thì anh phải ký."],
                         syntactic_tic="", **chung)

    with pytest.raises(ValueError, match="đang đếm"):
        VoiceFingerprint(signature_lexicon=[], verbal_tics=["được chưa"],
                         voice_exemplars=[],
                         syntactic_tic="hay chốt câu bằng 'được chưa?'", **chung)

    # Dạy CÁCH nói thì được: mô tả hình dạng, không trích chữ.
    assert VoiceFingerprint(signature_lexicon=["theo quy trình"],
                            voice_exemplars=["Ai ký vào dòng này."],
                            syntactic_tic="trả lời bằng một con số trước",
                            **chung)
