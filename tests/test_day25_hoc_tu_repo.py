"""Ngày 25 — thứ học được từ các generator khác, và thứ cố tình không học.

Khảo sát AI_NovelGenerator (38 file Python), novelWriter, inkos, uncle-novel,
NovelReader. Chỉ cái đầu là generator dùng LLM; ba cái sau là ứng dụng đọc/viết
không sinh văn.

HỌC: prompt viết chương của họ nhận cả blueprint chương HIỆN TẠI lẫn chương KẾ,
nên chương không kết thúc vào khoảng không. Ta không có gì tương đương.

KHÔNG HỌC: luật chống lặp của họ là chỉ thị bằng lời trong prompt — "tương đồng
>40% thì phải tái cấu trúc", "sửa ít nhất 30%". Model không đo được tương đồng
của chính nó, nên đó là một con số để nó gật đầu chứ không phải một ràng buộc.
Ta đo bằng code (`repetition.py`) rồi mới nói.
"""
from __future__ import annotations

import pathlib

import pytest

from novel_engine.graph.engines import build_engines
from novel_engine.graph.nodes import _chuong_sau_brief
from novel_engine.llm.fake import FakeLLM
from novel_engine.prompts import WRITER_TMPL


@pytest.fixture(scope="module")
def eng():
    return build_engines(FakeLLM())


def _st(chapter=1, scene_index=5, so_canh=6):
    return {"chapter": chapter, "scene_index": scene_index,
            "contracts": [{}] * so_canh}


def test_chi_canh_cuoi_chuong_duoc_biet_chuong_sau(eng):
    """Đưa cho mọi cảnh là mời model kể trước."""
    cuoi = _chuong_sau_brief(_st(scene_index=5), eng, {})
    assert "Chương 2" in cuoi and "Đường biển" in cuoi

    for i in range(5):
        assert _chuong_sau_brief(_st(scene_index=i), eng, {}) == ""


def test_chuong_cuoi_khong_co_chuong_sau(eng):
    assert _chuong_sau_brief(_st(chapter=99), eng, {}) == ""


def test_brief_noi_ro_day_khong_phai_dieu_POV_biet(eng):
    """Cùng hàng rào mà `director_only._hard_constraint` dựng cho
    `hidden_action`. Thiếu câu này thì POV sẽ "linh cảm" về chương sau, và đó
    đúng là rò rỉ tri thức mà §5.4 sinh ra để chặn."""
    ra = _chuong_sau_brief(_st(), eng, {})
    assert "KHÔNG phải điều POV biết" in ra
    assert "không được linh cảm" in ra.lower()
    assert "{next_chapter}" in WRITER_TMPL


def test_prompt_khong_doi_model_tu_do_do_tuong_dong():
    """Luật chống lặp của AI_NovelGenerator nằm trong prompt dưới dạng ngưỡng
    phần trăm mà model phải tự áp lên chính nó. Ta không đi đường đó: ngưỡng
    sống trong `repetition.py`, và prompt chỉ nhận kết quả đã đo.
    """
    for con_so in ("40%", "30%", "20%", "tương đồng"):
        assert con_so not in WRITER_TMPL


# ═══════════ L5: NHỚ LẠI THEO NỘI DUNG, KHÔNG THEO THỨ TỰ ═══════════

def _digest(ch, si, sid, tick, van):
    return {"chapter": ch, "scene_idx": si, "scene_id": sid,
            "epoch_tick": tick, "digest": van}


class _Store:
    def __init__(self, rows):
        self.rows = rows

    def scene_digests_with_time(self):
        return list(self.rows)

    def recent_scene_digests(self, *a, **kw):
        return []

    def chapter_summaries(self, *a, **kw):
        return []

    def arc_summaries(self, *a, **kw):
        return []


class _Planner:
    def __init__(self, pov):
        self.pov = pov

    def pov_for(self, ch, si):
        return self.pov[(ch, si)]


class _Graph:
    def known_by(self, *a, **kw):
        return {"known": [], "suspected": []}

    def faction_tensions(self, *a, **kw):
        return []


def _lap_rap(rows, pov):
    from novel_engine.memory.assembler import ContextAssembler
    return ContextAssembler(_Graph(), _Store(rows), planner=_Planner(pov))


ROWS = [
    _digest(1, 0, "CH001_S00", 100, "Kaelen xem con dấu đỏ trên tờ giấy phép "
            "quá cảnh tại trạm kiểm soát Cảng Quặng."),
    _digest(1, 1, "CH001_S01", 110, "Vhal sửa sổ nhật ký ca trực rồi khoá "
            "ngăn kéo, giấu chữ ký của chính mình."),
    _digest(2, 0, "CH002_S00", 200, "Kaelen đứng trên boong tàu quặng, "
            "thuỷ triều rút dần khỏi mạn thuyền."),
]
POV = {(1, 0): "CHAR_KAELEN", (1, 1): "CHAR_VHAL", (2, 0): "CHAR_KAELEN"}


def test_L5_khong_bao_gio_tra_ve_canh_cua_POV_khac():
    """Đây là chỗ khác hẳn một vector store thông thường.

    Truy hồi tự do sẽ đưa digest cảnh POV Vhal ("Vhal sửa sổ trực") cho Writer
    trong lúc đang viết cảnh POV Kaelen — rò rỉ đúng thứ §5.4 dựng ba lớp để
    chặn. Nhân vật chỉ nhớ lại cảnh của CHÍNH MÌNH; điều người khác nói cho họ
    biết đã có L4 lo, qua `known_by`.
    """
    a = _lap_rap(ROWS, POV)
    ra = a._callbacks(5, 0, "CHAR_KAELEN", 9999, "sổ nhật ký ca trực bị sửa")
    assert [x["scene_id"] for x in ra] == ["CH001_S00"] or ra == []
    assert all(x["scene_id"] != "CH001_S01" for x in ra)


def test_L5_khong_nho_duoc_tuong_lai():
    """NT-6: lọc theo `epoch_tick`, không theo chương. Cảnh hồi ức ở Chương 2
    (tick 2480) mà lọc theo chương thì "nhớ" được Chương 5 (tick 20600)."""
    a = _lap_rap(ROWS, POV)
    assert a._callbacks(5, 0, "CHAR_KAELEN", 150, "thuỷ triều rút khỏi mạn "
                        "thuyền trên boong tàu quặng") == []
    co = a._callbacks(5, 0, "CHAR_KAELEN", 9999, "thuỷ triều rút khỏi mạn "
                      "thuyền trên boong tàu quặng")
    assert [x["scene_id"] for x in co] == ["CH002_S00"]


def test_L5_bo_qua_cua_so_L1():
    """Hai cảnh liền trước đã nằm NGUYÊN VĂN ở L1; nhắc lại chỉ tốn token và
    mời model dựng lại đúng cảnh vừa viết."""
    a = _lap_rap(ROWS, POV)
    assert a._callbacks(2, 1, "CHAR_KAELEN", 9999,
                        "boong tàu quặng thuỷ triều") == []


def test_L5_im_lang_khi_khong_co_planner():
    """Không có planner thì không biết POV của cảnh cũ — và đoán POV nghĩa là
    đoán ai được nhớ gì."""
    from novel_engine.memory.assembler import ContextAssembler
    a = ContextAssembler(_Graph(), _Store(ROWS))
    assert a._callbacks(5, 0, "CHAR_KAELEN", 9999, "con dấu đỏ") == []


def test_truy_hoi_lac_de_thi_tra_ve_rong():
    """Một callback sai còn tệ hơn không có callback: nó mời model dựng lại một
    cảnh không liên quan — đúng cái lặp cảnh mà `repetition.py` sinh ra để
    chặn."""
    from novel_engine.memory.recall import callbacks

    assert callbacks("mưa rơi trên mái nhà ở thủ đô",
                     [{"digest": r["digest"]} for r in ROWS]) == []


def test_bm25_khong_can_thu_vien_ngoai():
    """Máy đích 7,7 GB RAM, không Docker, và bộ hồi quy đòi cùng seed cho cùng
    kết quả — nên không dùng vector store như AI_NovelGenerator."""
    import novel_engine.memory.recall as r

    src = pathlib.Path(r.__file__).read_text(encoding="utf-8")
    for xau in ("chromadb", "langchain", "sentence_transformers", "openai"):
        assert xau not in src
    assert "KHÔNG bắt được diễn đạt khác chữ" in r.__doc__
