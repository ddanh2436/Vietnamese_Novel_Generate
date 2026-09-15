"""Ngày 13 — News Dispatcher & Fog of War (§5.6).

Kế hoạch: đồ thị độ trễ giữa 4 địa điểm, Dijkstra đa mục tiêu Pareto, và bộ lọc
kháng cự `accept_correction` theo `fatal_flaw`. Cộng sáu lỗi của mã §5.6 — mỗi
lỗi làm một phần cơ chế thành mã chết mà không ném lỗi nào.
"""
from __future__ import annotations

import pytest

from novel_engine.canon.bible import DEFAULT_BIBLE, load_bible
from novel_engine.canon.networkx_graph import NetworkXGraph
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.world.news import (
    CORRECTION_GAIN, Channel, NewsDispatcher, NewsItem, ParetoFrontier,
    accept_correction, interest_score, learn_tick, propagate, settle_news,
)


def _ch(lat, dist=0.0, rel=1.0, ctrl=None):
    return Channel(kind="x", latency_multiplier=lat, reliability=rel,
                   distortion_rate=dist, controlled_by=ctrl)


def _graph(*routes):
    g = NetworkXGraph()
    for u, v, t, chans, *blk in routes:
        g.add_route(u, v, base_ticks=t, terrain="plains", allowed_channels=chans,
                    blocked_by=blk[0] if blk else [])
    return g


def _news(channels, **kw):
    return NewsItem(news_id=kw.pop("nid", "N"), origin_location="A", origin_tick=0,
                    truth="SỰ THẬT", subject_entities=["CHAR_KAELEN"],
                    channels=channels, **kw)


def _locs(arr, channel=None):
    return {a["location"] for a in arr if channel is None or a["channel"] == channel}


# ═══════════════ §5.6.1 KÊNH & TUYẾN ═══════════════

def test_kenh_chi_di_tren_tuyen_cho_phep():
    """`propagate` §5.6.3 gọi `routes_from(loc)` không truyền kênh."""
    g = _graph(("A", "B", 4, ["courier"]))
    arr = propagate(_news(["rumor"]), g, 1000, {"rumor": _ch(2.4)})
    assert _locs(arr) == {"A"}


def test_kenh_thu_hai_van_lan_duoc():
    """Frontier dùng chung: kênh thứ hai bị 'trội' ngay ở điểm gốc và không đi đâu."""
    g = _graph(("A", "B", 4, ["courier"]), ("A", "C", 4, ["rumor"]))
    arr = propagate(_news(["courier", "rumor"]), g, 1000,
                    {"courier": _ch(1.0), "rumor": _ch(2.4)})
    assert "C" in _locs(arr, "rumor") and "B" in _locs(arr, "courier")


def test_phe_nam_kenh_bit_duoc_tin():
    g = _graph(("A", "B", 4, ["signal", "courier"]), ("A", "C", 4, ["courier"], ["FACT_X"]))
    chans = {"signal": _ch(0.02, ctrl="FACT_X"), "courier": _ch(1.0)}
    arr = propagate(_news(["signal", "courier"], suppressed_by=["FACT_X"]), g, 1000, chans)
    assert "signal" not in {a["channel"] for a in arr}
    assert "C" not in _locs(arr)                           # tuyến do FACT_X chặn


# ═══════════════ §5.6.3 PARETO ═══════════════

def test_frontier_giu_ban_CHINH_XAC_NHAT_khi_vuot_tran():
    """`keep[:4]` giữ 4 bản đến sớm nhất — tức vứt đúng bản sứ giả chính xác."""
    f = ParetoFrontier(cap=4)
    for t, fid in [(1, .2), (2, .3), (3, .4), (4, .5), (5, .6), (6, 1.0)]:
        f.admit("B", t, fid)
    assert (1, .2) in f.items["B"] and (6, 1.0) in f.items["B"]
    assert len(f.items["B"]) == 4


def _fast_rumor_slow_courier():
    g = _graph(("A", "B", 10, ["courier", "rumor"]))
    chans = {"rumor": _ch(0.5, dist=1.0), "courier": _ch(2.0)}
    return g, chans


@pytest.mark.parametrize("order", [["rumor", "courier"], ["courier", "rumor"]])
def test_tin_don_den_truoc_KHONG_chan_su_gia_va_dinh_chinh_khong_phu_thuoc_thu_tu(order):
    g, chans = _fast_rumor_slow_courier()
    arr = [a for a in propagate(_news(order), g, 1000, chans) if a["location"] == "B"]
    assert [(a["channel"], a["is_correction"]) for a in arr] == [
        ("rumor", False), ("courier", True)]
    assert arr[1]["payload"]["fidelity"] - arr[0]["payload"]["fidelity"] >= CORRECTION_GAIN


def test_them_tuyen_moi_khong_doi_duong_lan_tren_tuyen_cu():
    """Bốc thăm theo hash TỪNG CHẶNG, không theo một RNG chạy tuần tự (NT-16)."""
    chans = {"courier": _ch(1.0, dist=0.5, rel=0.6), "rumor": _ch(2.4, dist=0.4, rel=0.7)}
    routes = [("A", "B", 4, ["courier", "rumor"]), ("B", "C", 3, ["courier", "rumor"]),
              ("A", "D", 9, ["courier", "rumor"])]
    base = propagate(_news(["courier", "rumor"]), _graph(*routes), 1000, chans)
    more = propagate(_news(["courier", "rumor"]),
                     _graph(*routes, ("D", "Z", 2, ["courier"])), 1000, chans)
    strip = lambda arr: [(a["location"], a["tick"], a["payload"]["fidelity"])  # noqa: E731
                         for a in arr if a["location"] != "Z"]
    assert strip(base) == strip(more)


# ═══════════════ §5.6.4 AI NGHE ═══════════════

def _frame(sid, tick, locs, order=1):
    return ContinuityFrame(scene_id=sid, time=StoryTime(
        epoch_tick=tick, duration_ticks=2, narrative_order=order), locations=locs)


def test_nguoi_den_sau_van_nghe_duoc_tin():
    """`characters_at(loc, tick)` chỉ hỏi ai có mặt đúng lúc tin tới."""
    frames = [_frame("CH001_S00", 100, {"CHAR_K": "X"}),
              _frame("CH001_S01", 300, {"CHAR_K": "B"}, 2)]
    assert learn_tick("CHAR_K", frames, "B", 150, 1000) == 300
    assert learn_tick("CHAR_K", frames, "X", 120, 1000) == 120      # đang ở đó
    assert learn_tick("CHAR_K", frames, "B", 150, 250) is None       # chưa tới


def test_quan_tam_theo_nguy_co_va_ten_trong_niem_tin_khong_theo_ma():
    g, chars, _ = load_bible(DEFAULT_BIBLE)
    news = _news(["courier"])
    assert interest_score(chars["CHAR_KAELEN"], news, g) >= 1.0
    # Serena: desire bị CHAR_KAELEN đe doạ. Vhal: desire được CHAR_KAELEN thoả
    # mãn — người có thể giúp mình cũng là tin đáng nghe.
    assert interest_score(chars["CHAR_SERENA"], news, g) >= 0.5
    assert interest_score(chars["CHAR_VHAL"], news, g) >= 0.5
    loi_rang = NewsItem(news_id="F", origin_location="A", origin_tick=0, truth="x",
                        subject_entities=["FACT_ARCLIGHT"])
    assert interest_score(chars["CHAR_VHAL"], loi_rang, g) < 0.5
    # So TÊN trong câu niềm tin — §5.6.4 so mã với câu, không bao giờ khớp.
    lo3 = NewsItem(news_id="L", origin_location="A", origin_tick=0, truth="x",
                   subject_entities=["CHAR_SERENA"])
    k = chars["CHAR_KAELEN"]
    truoc = interest_score(k, lo3, g)
    k.beliefs[0].proposition += " — Serena biết điều đó"
    assert interest_score(k, lo3, g) == pytest.approx(truoc + 0.3)


# ═══════════════ §5.6.6 ĐÍNH CHÍNH ═══════════════

def test_khang_cu_dinh_chinh_theo_khuyet_diem_THAT_cua_nhan_vat():
    """§5.6.6 tra theo tên "sợ bị phản bội" — không ai mang tên đó, kháng cự = 0."""
    _, chars, _ = load_bible(DEFAULT_BIBLE)
    k = chars["CHAR_KAELEN"]
    ok, _ = accept_correction(k, 0.5184, 1.0, "courier", ["exculpates:CHAR_KAELEN"])
    assert ok is False                     # bằng chứng gỡ tội: tự kết tội, bác bỏ
    ok, _ = accept_correction(k, 0.5184, 1.0, "courier", ["incriminates:CHAR_KAELEN"])
    assert ok is True                      # tin làm nặng tội: tin ngay


def test_vhal_chi_tin_kenh_chinh_thong():
    _, chars, _ = load_bible(DEFAULT_BIBLE)
    v = chars["CHAR_VHAL"]
    assert accept_correction(v, 0.3, 1.0, "rumor")[0] is False
    assert accept_correction(v, 0.3, 1.0, "courier")[0] is True


# ═══════════════ CANON: settle_news ═══════════════

@pytest.fixture
def world():
    g, chars, _ = load_bible(DEFAULT_BIBLE)
    # Tin gỡ tội cho Kaelen: tin đồn nhanh (luôn méo), sứ giả chậm (chính xác).
    g.news = NewsDispatcher(
        [NewsItem(news_id="NEWS_T", origin_location="LOC_VEDA_CHECKPOINT",
                  origin_tick=20000, truth="Nhật ký Lò 3 bị ghi đè từ phòng điều khiển",
                  subject_entities=["CHAR_KAELEN"], channels=["rumor", "courier"],
                  tags=["exculpates:CHAR_KAELEN"])],
        {"rumor": _ch(0.25, dist=1.0), "courier": _ch(5.0)})
    frames = [_frame("CH001_S00", 20000, {"CHAR_KAELEN": "LOC_ORE_PORT"})]
    return g, chars, frames


def test_tu_choi_dinh_chinh_giu_niem_tin_SAI_va_ghi_lai(world):
    g, chars, frames = world
    r = settle_news(g, chars, frames, upto_tick=20030)
    assert r["rejected"] == 1
    k = g.news_knowledge[("CHAR_KAELEN", "NEWS_T")]
    assert k["held"]["channel"] == "rumor" and k["held"]["ops"]
    b = [x for x in chars["CHAR_KAELEN"].beliefs if x.source.startswith("news:NEWS_T")]
    assert len(b) == 1 and b[0].is_actually_true is False and b[0].confidence > 0


def test_settle_idempotent(world):
    g, chars, frames = world
    settle_news(g, chars, frames, upto_tick=20030)
    n = len(chars["CHAR_KAELEN"].beliefs)
    settle_news(g, chars, frames, upto_tick=20030)
    assert len(chars["CHAR_KAELEN"].beliefs) == n


def test_known_by_KHONG_lo_ban_that_cho_nguoi_nghe_ban_meo(world):
    g, chars, frames = world
    settle_news(g, chars, frames, upto_tick=20030)
    heard = [f for f in g.known_by("CHAR_KAELEN", 20025)["known"] if f["id"] == "NEWS_T"]
    assert heard and "sai lệch" in heard[0]["name"]
    assert "ghi đè" not in heard[0]["name"]
    assert not [f for f in g.known_by("CHAR_KAELEN", 19999)["known"] if f["id"] == "NEWS_T"]


# ═══════════════ ĐỒ THỊ ═══════════════

def test_director_giao_tin_cho_canh_ma_POV_nghe():
    eng = build_engines(FakeLLM(), db_path=":memory:")
    try:
        out = run_chapter(eng, 1)
        got = [(c["scene_index"], d["news_id"]) for c in out["contracts"]
               for d in c["news_directives"]]
        assert (0, "NEWS_KAELEN_RETURNS") in got
        assert len([x for x in got if x[1] == "NEWS_KAELEN_RETURNS"]) == 1   # chỉ lần đầu nghe
        w0 = [c["prompt"] for c in eng.llm.calls if c["role"] == "writer"][0]
        assert "TIN TỨC NGƯỜI KỂ VỪA NGHE" in w0 and "trở về Cảng Quặng" in w0
    finally:
        eng.store.close()


def test_niem_tin_tu_tin_tuc_duoc_replay_ra_nhu_cu(tmp_path):
    db = str(tmp_path / "canon.db")
    e1 = build_engines(FakeLLM(), db_path=db)
    run_chapter(e1, 1, auto_commit=True)
    truoc = [(b.proposition, b.confidence, b.is_actually_true)
             for b in e1.chars["CHAR_KAELEN"].beliefs if b.source.startswith("news:")]
    assert truoc
    e1.store.close()
    e2 = build_engines(FakeLLM(), db_path=db)
    try:
        sau = [(b.proposition, b.confidence, b.is_actually_true)
               for b in e2.chars["CHAR_KAELEN"].beliefs if b.source.startswith("news:")]
        assert sau == truoc
    finally:
        e2.store.close()
