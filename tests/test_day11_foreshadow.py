"""Ngày 11 — Foreshadowing Engine (§6.2, §6.3, §6.3.1, §6.4).

Mỗi van thoát có test BẮT ĐƯỢC và test KHÔNG KHOÁ. Loại thứ hai là loại §6.3.1
tự nói cần: "hãy tìm mọi chỗ có `continue` hoặc `return -1` và hỏi: nếu điều kiện
này đúng mãi thì sao?"
"""
from __future__ import annotations

import math

import pytest

from novel_engine.canon.bible import DEFAULT_BIBLE, load_bible
from novel_engine.canon.models import Clue, ClueStatus, PlantEvidence, StateDelta
from novel_engine.canon.networkx_graph import NetworkXGraph
from novel_engine.foreshadow.debt import narrative_debt_report
from novel_engine.foreshadow.scheduler import (
    ABANDON_AFTER, ATTENTION_BUDGET, FORCE_RESOLVE_AFTER, MAX_PLANTS_PER_CH,
    REINFORCE_AT, ForeshadowScheduler, SceneSlot, decay,
)
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.planner.outline_planner import OutlinePlanner
from novel_engine.reconcile.commit import apply_delta, derive_clue_transitions
from novel_engine.reconcile.verify import plan_coverage

S = ClueStatus


def _clue(cid, *, thr=2, dl=5, status=S.DRAFTED, prereq=(), forms=("a", "b", "c"),
          sal=0.0, touched=None, sub=0.6, carriers=None, understood=(), planted=None):
    kw = {}
    if carriers is not None:
        kw["carriers"] = list(carriers)
    return Clue(clue_id=cid, macro_event_target="EV_X", description=f"MÔ TẢ BÍ MẬT {cid}",
                payoff_threshold=thr, payoff_deadline=dl, status=status,
                prerequisites=list(prereq), surface_forms=list(forms), salience=sal,
                last_touched_chapter=touched, subtlety_target=sub,
                understood_by_characters=list(understood), planted_in_chapter=planted, **kw)


def _g(*clues):
    g = NetworkXGraph()
    for c in clues:
        g.add_clue(c)
    return g


FULL = {"setting": ["LOC"], "object": ["LOC"], "behavior": ["CHAR_B"], "dialogue": ["CHAR_B"]}
ALONE = {"setting": ["LOC"], "object": ["LOC"], "behavior": [], "dialogue": []}


def _slots(aff=FULL, povs=None, quiet=(0, 5)):
    povs = povs or ["CHAR_A"] * 6
    return [SceneSlot(i, povs[i], aff, quiet=i in quiet) for i in range(6)]


def _plan(ch, *clues, slots=None):
    g = _g(*clues)
    return g, ForeshadowScheduler(g).schedule(ch, slots or _slots())


def _ids(res):
    return [d.clue_id for d in res.directives]


# ═══════════════ §6.2 DECAY ═══════════════

def test_decay_chua_tung_cham_la_0():
    assert decay(_clue("C", sal=0.9), 5) == 0.0


def test_decay_nua_doi_khoang_3_8_chuong():
    c = _clue("C", sal=1.0, touched=0, status=S.PLANTED)
    assert decay(c, 0) == 1.0
    assert decay(c, math.log(2) / 0.18) == pytest.approx(0.5)


# ═══════════════ RÀNG BUỘC THỨ TỰ — hạn cài truyền ngược ═══════════════

def test_plant_by_keo_han_cua_tien_de_theo_manh_moi_phu_thuoc():
    g, _, _ = load_bible(DEFAULT_BIBLE)
    s = ForeshadowScheduler(g)
    assert s.plant_by(g.clues["CLUE_THIRD_SIGNATURE"]) == 4
    # Deadline riêng của nhật ký là 6 — nhưng chữ ký thứ ba cần nó trước chương 4.
    assert s.plant_by(g.clues["CLUE_MISSING_LOGS"]) == 3
    assert s.plant_by(g.clues["CLUE_SEAL_CORROSION"]) == 3


def test_qua_han_nhung_thieu_tien_de_KHONG_duoc_tra_bai():
    """Mã §6.3 trả 10+ trước khi kiểm prerequisite: chữ ký thứ ba được 'trả bài'
    khi độc giả chưa từng thấy nhật ký bị ghi đè."""
    seal = _clue("SEAL", status=S.PLANTED, touched=6, sal=0.9)
    logs = _clue("LOGS", dl=6)
    third = _clue("THIRD", thr=4, dl=5, prereq=("SEAL", "LOGS"))
    _, res = _plan(7, seal, logs, third)
    assert "THIRD" not in _ids(res)
    assert res.blocked[0]["clue_id"] == "THIRD"
    assert res.blocked[0]["waiting_on"] == ["LOGS"]
    assert "LOGS" in _ids(res)                     # tiền đề được đẩy lên thay vào


def test_manh_moi_qua_han_la_uu_tien_cao_nhat_khong_bi_loai():
    c = _clue("OLD", dl=3, status=S.PLANTED, sal=0.9, touched=3)
    g = _g(c)
    assert ForeshadowScheduler(g)._score(c, 4) >= 10.0
    assert _ids(ForeshadowScheduler(g).schedule(4, _slots())) == ["OLD"]


def test_manh_moi_chua_du_dieu_kien_khong_chan_manh_moi_xep_sau():
    """`continue`, không `break` (§6.3.1)."""
    blocked = _clue("B", dl=2, prereq=("MISSING",))
    ok = _clue("OK", dl=9, sub=0.9)
    _, res = _plan(1, blocked, ok)
    assert _ids(res) == ["OK"]


# ═══════════════ BA NẤC VAN THOÁT ═══════════════

def test_nac1_qua_han_duoc_vuot_ngan_sach_chu_y():
    cs = [_clue(f"C{i}", thr=1, dl=1, status=S.PLANTED, sal=0.9, touched=1) for i in range(3)]
    _, res = _plan(1 + FORCE_RESOLVE_AFTER, *cs)
    assert len(res.directives) == 3
    assert sum(d.weight for d in res.directives) > ATTENTION_BUDGET
    assert all(d.forced for d in res.directives)


def test_chua_qua_han_thi_ton_trong_ngan_sach_va_BAO_CAO_phan_bi_loai():
    cs = [_clue(f"C{i}", thr=1, dl=5, status=S.PLANTED, sal=0.9, touched=4) for i in range(3)]
    _, res = _plan(4, *cs)                          # vào cửa sổ trả bài, intensity 0.95
    assert len(res.directives) == 1
    assert [u["reason"] for u in res.unplaced] == ["attention_budget"] * 2


def test_nac2_bo_qua_pov_va_vat_mang_khi_qua_han_lau():
    """POV đã hiểu manh mối (pov_can_observe = False) và cảnh không có vật mang
    hợp — vẫn phải trả được bài qua lời người khác."""
    kw = dict(dl=2, status=S.PLANTED, sal=0.9, touched=2, carriers=["behavior"],
              understood=["CHAR_A"])
    _, res = _plan(2 + FORCE_RESOLVE_AFTER, _clue("C", **kw), slots=_slots(aff={
        **FULL, "behavior": []}))
    d = res.directives[0]
    assert d.forced and d.mode == "payoff" and d.carrier == "dialogue"

    _, chua = _plan(2 + 1, _clue("C", **kw), slots=_slots(aff={**FULL, "behavior": []}))
    assert chua.directives == []
    assert chua.unplaced[0]["reason"] == "no_scene_with_carrier_and_pov"


def test_nac2_canh_mot_minh_khong_ep_thoai():
    """Bài học Ngày 10: ràng buộc không thể thoả thì model phá nó."""
    c = _clue("C", dl=2, status=S.PLANTED, sal=0.9, touched=2, carriers=["dialogue"])
    _, res = _plan(2 + FORCE_RESOLVE_AFTER, c, slots=_slots(aff=ALONE))
    assert res.directives[0].carrier == "object"


def test_nac2_manh_moi_chua_tung_cai_thi_cai_manh_KHONG_tra_bai():
    c = _clue("C", thr=1, dl=2)
    _, res = _plan(2 + FORCE_RESOLVE_AFTER, c)
    assert res.directives[0].mode == "plant" and res.directives[0].intensity == 0.95


def test_nac3_qua_han_8_chuong_day_len_tac_gia():
    root = _clue("ROOT", dl=2, status=S.PLANTED, sal=0.5, touched=1, planted=1)
    child = _clue("CHILD", thr=3, dl=20, prereq=("ROOT",))
    _, res = _plan(2 + ABANDON_AFTER, root, child)
    assert "ROOT" not in _ids(res)
    esc = res.escalations[0]
    assert esc["clue_id"] == "ROOT" and esc["overdue_by"] == ABANDON_AFTER
    assert esc["retire_cost"]["orphaned_clues"] == ["CHILD"]
    assert esc["retire_cost"]["chapters_since_planted"] == 2 + ABANDON_AFTER - 1


def test_nac3_van_escalate_khi_manh_moi_ket_vi_tien_de():
    """Kiểm prerequisite trước escalation thì manh mối kẹt vì tiền đề im lặng mãi."""
    c = _clue("C", dl=2, prereq=("NEVER",))
    _, res = _plan(2 + ABANDON_AFTER, c)
    assert res.escalations and res.escalations[0]["clue_id"] == "C"


# ═══════════════ NHẮC LẠI & XẾP CẢNH ═══════════════

def test_khong_nhac_lai_khi_doc_gia_con_nho():
    c = _clue("C", thr=18, dl=20, status=S.PLANTED, sal=0.9, touched=1)
    _, res = _plan(2, c)
    assert res.directives == []
    ch = next(n for n in range(2, 20) if decay(c, n) < REINFORCE_AT)
    _, res = _plan(ch, c)
    assert res.directives[0].mode == "reinforce"


def test_xep_canh_theo_POV_CUA_TUNG_CANH_khong_theo_canh_0():
    """§9.2 kiểm POV cảnh 0 cho cả chương rồi lọc lại ở từng cảnh — manh mối rơi
    mất không ai biết. CHAR_A đã hiểu manh mối; chỉ cảnh 3 do CHAR_B kể."""
    c = _clue("C", understood=["CHAR_A"])
    povs = ["CHAR_A", "CHAR_A", "CHAR_A", "CHAR_B", "CHAR_A", "CHAR_A"]
    _, res = _plan(1, c, slots=_slots(povs=povs))
    assert res.directives[0].scene_index == 3


def test_cai_o_beat_yen_tinh_tra_bai_ve_cuoi_chuong():
    _, cai = _plan(1, _clue("P", dl=9))
    assert cai.directives[0].scene_index == 0
    _, tra = _plan(4, _clue("Q", thr=2, dl=5, status=S.PLANTED, sal=0.9, touched=3))
    assert tra.directives[0].mode == "payoff" and tra.directives[0].scene_index == 4


def test_khong_xep_duoc_thi_bao_cao_khong_im_lang():
    c = _clue("C", carriers=["dialogue"])
    _, res = _plan(1, c, slots=_slots(aff=ALONE))
    assert res.directives == [] and res.unplaced[0]["clue_id"] == "C"


def test_toi_da_ba_manh_moi_moi_chuong():
    cs = [_clue(f"C{i}", dl=20, sub=0.95) for i in range(5)]
    _, res = _plan(1, *cs)
    assert len(res.directives) == MAX_PLANTS_PER_CH
    assert {u["reason"] for u in res.unplaced} == {"max_plants_per_chapter"}
    assert len({d.scene_index for d in res.directives}) == 3      # mỗi cảnh một


def test_surface_form_chua_dung_truoc_roi_dang_dung_lau_nhat():
    c = _clue("C", status=S.PLANTED)
    g = _g(c)
    s = ForeshadowScheduler(g)
    g.record_surface_form("C", "a")
    assert s._pick_form(c) == "b"
    for f in ("b", "c", "a"):
        g.record_surface_form("C", f)
    assert s._pick_form(c) == "b"          # thứ tự dùng a,b,c,a → b lâu nhất


def test_khong_co_surface_form_thi_bi_chan_khong_crash():
    _, res = _plan(1, _clue("C", forms=()))
    assert res.blocked[0]["reason"] == "no_surface_forms"


def test_chi_thi_khong_bao_gio_chua_description():
    _, res = _plan(1, _clue("C"))
    d = res.directives[0].to_contract()
    assert "description" not in d
    assert not any("MÔ TẢ BÍ MẬT" in str(v) for v in d.values())


# ═══════════════ RECONCILE: salience, trạng thái, surface form ═══════════════

def _ev(cid, verified=True, form=None):
    return PlantEvidence(clue_id=cid, scene_id="CH002_S00", span="x" * 20,
                         carrier_used="object", verified=verified, surface_form=form)


def test_nhac_thoang_qua_KHONG_lam_doc_gia_quen_bot():
    """Bản cũ ghi đè salience = 0.45 + 0.55·0.2 = 0.56 lên manh mối đang ở 0.79."""
    c = _clue("C", status=S.PLANTED, sal=0.95, touched=1)
    g = _g(c)
    truoc = decay(c, 2)
    d = StateDelta(plant_evidence=[_ev("C")], plant_intensity={"C": 0.2}).stamp(2)
    apply_delta(d, g, {}, [])
    assert c.salience > truoc and c.last_touched_chapter == 2


def test_nhieu_bang_chung_cung_manh_moi_chi_cham_mot_lan():
    c = _clue("C")
    g = _g(c)
    d = StateDelta(plant_evidence=[_ev("C"), _ev("C")], plant_intensity={"C": 0.2}).stamp(1)
    apply_delta(d, g, {}, [])
    assert c.salience == pytest.approx(0.45 + 0.55 * 0.2)


def test_surface_form_da_len_trang_duoc_ghi_khi_commit():
    c = _clue("C")
    g = _g(c)
    apply_delta(StateDelta(plant_evidence=[_ev("C", form="b")]).stamp(1), g, {}, [])
    assert g.used_surface_forms("C") == ["b"]
    assert ForeshadowScheduler(g)._pick_form(c) == "a"


@pytest.mark.parametrize("status, mode, llm_khai, ky_vong", [
    (S.DRAFTED, "plant", None, S.PLANTED),
    (S.PLANTED, "payoff", None, S.PAID_OFF),
    (S.PLANTED, "reinforce", S.PAID_OFF, S.REINFORCED),   # LLM khai sai → code sửa
    (S.DRAFTED, "payoff", None, S.PLANTED),               # chưa cài thì không trả
    (S.PLANTED, None, S.PAID_OFF, S.REINFORCED),          # tình cờ ≠ trả bài
])
def test_trang_thai_manh_moi_do_code_suy_ra(status, mode, llm_khai, ky_vong):
    c = _clue("C", status=status)
    d = StateDelta(plant_evidence=[_ev("C")],
                   plant_modes={"C": mode} if mode else {},
                   clue_transitions={"C": llm_khai} if llm_khai else {})
    out, notes = derive_clue_transitions(d, {"C": c})
    assert out == {"C": ky_vong}
    assert bool(notes) == (llm_khai is not None)


def test_khong_co_bang_chung_hoac_da_dong_thi_khong_doi_trang_thai():
    d = StateDelta(plant_evidence=[_ev("A", verified=False), _ev("B")])
    out, _ = derive_clue_transitions(d, {"A": _clue("A"), "B": _clue("B", status=S.PAID_OFF)})
    assert out == {}


def test_plan_coverage_ghi_mode_cuong_do_va_surface_form_vao_delta():
    """Contract không được lưu: lúc `cli.py commit` chạy thì nó đã mất."""
    contracts = [{"scene_id": "CH002_S00", "plant_directives": [
        {"clue_id": "C", "mode": "payoff", "intensity": 0.95, "surface_form": "b"}]}]
    d = StateDelta(plant_evidence=[_ev("C")])
    plan_coverage(contracts, d, known_clues={"C"})
    assert d.plant_modes == {"C": "payoff"} and d.plant_intensity == {"C": 0.95}
    assert d.plant_evidence[0].surface_form == "b"


# ═══════════════ §6.4 NỢ TỰ SỰ ═══════════════

def test_bao_cao_no_khong_dung_chuong_va_thay_manh_moi_tre_han_cai():
    g, _, _ = load_bible(DEFAULT_BIBLE)
    r = narrative_debt_report(g.clues, [], 4)
    assert {x["clue"] for x in r["late_to_plant"]} == {"CLUE_SEAL_CORROSION",
                                                       "CLUE_MISSING_LOGS"}
    r = narrative_debt_report(g.clues, [], 7)
    assert all(x["severity"] == "major" for x in r["overdue"])    # NT-17


# ═══════════════ PLANNER & BIBLE ═══════════════

def test_vat_mang_la_cua_canh_canh_mot_minh_khong_co_thoai():
    p = OutlinePlanner()
    assert p.scene_affordances(2, 0)["dialogue"] == []
    assert p.scene_affordances(1, 1)["dialogue"] == ["CHAR_VHAL"]


def test_bible_khai_vat_mang_cho_tung_manh_moi():
    g, _, _ = load_bible(DEFAULT_BIBLE)
    assert g.clues["CLUE_SEAL_CORROSION"].carriers == ["object", "setting"]


# ═══════════════ TÍCH HỢP ĐỒ THỊ ═══════════════

@pytest.fixture
def eng():
    e = build_engines(FakeLLM(), db_path=":memory:")
    yield e
    e.store.close()


def test_director_cai_manh_moi_va_writer_khong_thay_description(eng):
    out = run_chapter(eng, 1)
    fs = out["foreshadow_report"]
    assert {d["clue_id"] for d in fs["directives"]} == {"CLUE_SEAL_CORROSION",
                                                        "CLUE_MISSING_LOGS"}
    assert fs["blocked"][0]["clue_id"] == "CLUE_THIRD_SIGNATURE"
    co_cai = [c for c in out["contracts"] if c["plant_directives"]]
    assert len(co_cai) == 2 and all(len(c["plant_directives"]) == 1 for c in co_cai)

    writer = [c["prompt"] for c in eng.llm.calls if c["role"] == "writer"]
    forms = [d["surface_form"] for d in fs["directives"]]
    assert all(any(f in p for p in writer) for f in forms)
    for clue in eng.graph.clues.values():
        assert not any(clue.description.strip()[:40] in p for p in writer)

    diff = next(c["prompt"] for c in eng.llm.calls if c["role"] == "extractor_diff")
    assert all(f in diff for f in forms)


def test_chuong_sau_khong_cai_lai_manh_moi_da_cai(eng):
    """E4: không cập nhật last_touched/trạng thái thì mọi chương cài lại cùng thứ."""
    run_chapter(eng, 1, auto_commit=True)
    for cid in ("CLUE_SEAL_CORROSION", "CLUE_MISSING_LOGS"):
        c = eng.graph.clues[cid]
        assert c.status is S.PLANTED and c.last_touched_chapter == 1 and c.salience > 0
        assert eng.graph.used_surface_forms(cid)

    out2 = run_chapter(eng, 2)
    plan2 = {d["clue_id"]: d["mode"] for d in out2["foreshadow_report"]["directives"]}
    assert plan2.get("CLUE_SEAL_CORROSION") != "plant"
    assert plan2.get("CLUE_MISSING_LOGS") != "plant"
    # Tiền đề đã cài → chữ ký thứ ba hết bị chặn.
    assert "CLUE_THIRD_SIGNATURE" not in {b["clue_id"]
                                          for b in out2["foreshadow_report"]["blocked"]}
