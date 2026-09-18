"""Ngày 12 — Relationship State Machine (§7.1, §7.2, §7.3).

Ba tiêu chí của kế hoạch — guard chuyển giai đoạn, sẹo đặt trần intimacy,
`chapters_in_stage` tăng một lần mỗi chương — cộng những lỗ hổng lộ ra khi đọc
mã §7: KeyError ở RUPTURE, CATHARSIS chỉ tới được qua phản bội, SEVERED không
bao giờ xảy ra, và LLM tự đặt giai đoạn qua `relationship_updates`.
"""
from __future__ import annotations

from typing import get_args

import pytest

from novel_engine.canon.bible import DEFAULT_BIBLE, load_bible
from novel_engine.canon.models import RelationshipEvent, RelationStage, StateDelta
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.graph.nodes import _relationship_brief
from novel_engine.llm.fake import FakeLLM
from novel_engine.reconcile.commit import apply_delta
from novel_engine.reconcile.lenient import parse_delta_lenient
from novel_engine.reconcile.verify import assign_scenes, verify_spans
from novel_engine.relationship.book import RelationshipBook
from novel_engine.relationship.dynamics import (
    EFFECTS, MAX_STAKE_PER_CHAPTER, SEVER_AFTER_ELAPSED, settle_chapter,
)
from novel_engine.relationship.machine import (
    MIN_CHAPTERS_IN_STAGE, advance_or_hold, can_advance,
)
from novel_engine.relationship.models import RelationshipState

R = RelationStage
SPAN = "Kaelen giữ chặt tay vịn khi sàn tàu nghiêng hẳn sang một bên."


def _st(stage=R.STRANGERS, **kw):
    return RelationshipState(a="CHAR_A", b="CHAR_B", stage=stage, **kw)


def _ev(kind, actor=None, span=SPAN, scene="CH001_S01", verified=True,
        a="CHAR_A", b="CHAR_B"):
    return RelationshipEvent(a=a, b=b, kind=kind, actor=actor, span=span,
                             scene_id=scene, verified=verified)


def test_moi_loai_su_kien_co_dung_mot_hieu_ung():
    """NT-11: loại sự kiện khai ở schema và bảng hiệu ứng phải khớp nhau."""
    kinds = get_args(RelationshipEvent.model_fields["kind"].annotation)
    assert set(kinds) == set(EFFECTS)


# ═══════════════ §7.2 GUARD ═══════════════

@pytest.mark.parametrize("stage", [R.RUPTURE, R.SEVERED, R.CATHARSIS])
def test_giai_doan_ngoai_vong_chinh_khong_nem_KeyError(stage):
    """Mã §7.2: MIN_CHAPTERS_IN_STAGE thiếu RUPTURE/SEVERED → KeyError ở dòng đầu,
    và Director gọi hàm này cho MỌI cặp."""
    d = advance_or_hold(_st(stage), 5)
    assert d["action"] == "hold"
    assert set(MIN_CHAPTERS_IN_STAGE) == set(R)


@pytest.mark.parametrize("st, ok", [
    (_st(R.STRANGERS, chapters_in_stage=1, friction=20), True),
    (_st(R.STRANGERS, chapters_in_stage=1, friction=10), False),
    (_st(R.FRICTION, chapters_in_stage=3, intimacy=25), False),          # chưa nghịch cảnh
    (_st(R.FRICTION, chapters_in_stage=3, intimacy=25, shared_ordeals=["x"]), True),
    (_st(R.FRICTION, chapters_in_stage=2, intimacy=90, shared_ordeals=["x"]), False),
    (_st(R.VULNERABILITY, chapters_in_stage=2, intimacy=50, stake_conflict=30), False),
    (_st(R.VULNERABILITY, chapters_in_stage=2, intimacy=50, stake_conflict=45), True),
    (_st(R.TRIAL, chapters_in_stage=2, stake_conflict=30), False),        # chưa trả giá
    (_st(R.TRIAL, chapters_in_stage=2, stake_conflict=30, scars=["x"]), True),
    (_st(R.TRIAL, chapters_in_stage=2, stake_conflict=60, scars=["x"]), False),
])
def test_guard_chuyen_giai_doan(st, ok):
    assert can_advance(st, 5)[0] is ok


def test_hold_KHONG_mang_rang_buoc_cho_writer():
    """WRITER_TMPL coi `scene_requirement` là ràng buộc cứng. Gợi ý mở khoá ở đó
    là ép mọi cặp đang chờ dựng nghịch cảnh ở mọi chương."""
    hold = advance_or_hold(_st(R.FRICTION, chapters_in_stage=1), 2)
    assert "scene_requirement" not in hold and hold["hint"]
    adv = advance_or_hold(_st(R.STRANGERS, chapters_in_stage=1, friction=30), 2)
    assert adv["action"] == "advance" and adv["to"] == "friction"
    assert adv["scene_requirement"]


def test_chi_thi_cho_writer_khong_co_con_so():
    """NT-7. Lý do guard chứa con số ('intimacy 10 < 20') — không được lọt vào prompt."""
    d = {"names": ["Kaelen", "Serena"],
         **advance_or_hold(_st(R.FRICTION, chapters_in_stage=3, intimacy=10,
                                shared_ordeals=["x"]), 4)}
    text = _relationship_brief(d)
    assert "CHƯA được" in text
    assert "intimacy" not in text and not any(ch.isdigit() for ch in text)


# ═══════════════ §7.3 ĐỘNG LỰC ═══════════════

def test_dem_chuong_MOT_lan_du_nhieu_su_kien_va_idempotent():
    st = _st(R.FRICTION)
    settle_chapter(st, [_ev("value_clash")] * 6, 3, interacted=True)
    assert st.chapters_in_stage == 1
    assert settle_chapter(st, [_ev("value_clash")] * 6, 3, interacted=True) == []
    assert st.chapters_in_stage == 1 and st.friction == 16


def test_phan_boi_vao_RUPTURE_va_chuong_phan_boi_khong_duoc_dem():
    st = _st(R.TRIAL, intimacy=60, chapters_in_stage=1)
    t = settle_chapter(st, [_ev("betrayal")], 5, interacted=True)
    assert [x["to"] for x in t] == ["rupture"]
    assert st.chapters_in_stage == 0 and len(st.scars) == 1
    assert st.intimacy == pytest.approx(32)
    settle_chapter(st, [], 6, interacted=True)
    assert st.chapters_in_stage == 1


def test_hy_sinh_de_lai_seo_ma_khong_do_vo_nen_toi_duoc_CATHARSIS():
    """Mã §7.3: sẹo chỉ sinh từ phản bội → mọi cặp phải phản bội nhau trước khi
    tới được catharsis."""
    st = _st(R.TRIAL, intimacy=60, stake_conflict=30, chapters_in_stage=1)
    t = settle_chapter(st, [_ev("sacrifice", actor="CHAR_A")], 7, interacted=True)
    assert [x["to"] for x in t] == ["catharsis"]
    assert len(st.scars) == 1 and st.intimacy_asym > 0


def test_seo_dat_tran_vinh_vien_cho_intimacy():
    st = _st(R.TRIAL, intimacy=50, scars=["a", "b", "c"])
    for ch in range(1, 6):
        settle_chapter(st, [_ev("acted_against_own_interest_for_other")] * 3, ch,
                       interacted=False)
    assert st.intimacy == 100 - 7 * 3


def test_xung_dot_loi_ich_bi_chan_tran_moi_chuong():
    st = _st(R.VULNERABILITY)
    settle_chapter(st, [_ev("interests_collide")] * 5, 1, interacted=True)
    assert st.stake_conflict == MAX_STAKE_PER_CHAPTER


def test_bat_doi_xung_bi_chan_bien():
    st = _st(R.FRICTION)
    for ch in range(1, 30):
        settle_chapter(st, [_ev("sacrifice", actor="CHAR_A")] * 3, ch, interacted=False)
    assert st.intimacy_asym == 50.0


def test_toi_da_mot_lan_chuyen_giai_doan_moi_chuong():
    st = _st(R.STRANGERS, chapters_in_stage=5, friction=60, intimacy=90,
             stake_conflict=60, shared_ordeals=["x"], scars=["y"])
    t = settle_chapter(st, [], 9, interacted=True)
    assert [x["to"] for x in t] == ["friction"]


def test_RUPTURE_han_gan_can_du_chuong_tuong_tac():
    st = _st(R.RUPTURE, intimacy=30, stage_entered_chapter=5, last_counted_chapter=5)
    for ch in (6, 7):
        assert settle_chapter(st, [], ch, interacted=True) == []
    t = settle_chapter(st, [], 8, interacted=True)
    assert [x["to"] for x in t] == ["trial"]


def test_SEVERED_khi_do_vo_lau_ma_khong_gap_nhau():
    """Mã §7.3 đếm chương chỉ khi có sự kiện → cặp tránh mặt nhau ở RUPTURE mãi."""
    st = _st(R.RUPTURE, intimacy=10, stage_entered_chapter=5, last_counted_chapter=5)
    assert settle_chapter(st, [], 5 + SEVER_AFTER_ELAPSED - 1, interacted=False) == []
    t = settle_chapter(st, [], 5 + SEVER_AFTER_ELAPSED, interacted=False)
    assert [x["to"] for x in t] == ["severed"]
    before = st.intimacy
    settle_chapter(st, [_ev("sacrifice")], 20, interacted=True)
    assert st.intimacy <= before and st.stage is R.SEVERED


# ═══════════════ SỔ QUAN HỆ ═══════════════

def _frame(sid, mode="present", locs=None):
    t = StoryTime(epoch_tick=20000, duration_ticks=2, narrative_order=1, mode=mode,
                  anchor_scene=None if mode == "present" else "CH001_S00")
    return ContinuityFrame(scene_id=sid, time=t,
                           locations=locs or {"CHAR_A": "LOC", "CHAR_B": "LOC"})


def test_so_quan_he_loc_su_kien_khong_hop_le():
    book = RelationshipBook()
    frames = [_frame("CH001_S01"), _frame("CH001_S02", mode="flashback")]
    known = {"CHAR_A": 1, "CHAR_B": 1}
    res = book.settle(1, [
        _ev("shared_ordeal"),
        _ev("betrayal", scene="CH001_S02"),               # hồi ức — quá khứ đã kể
        _ev("sacrifice", verified=False),
        _ev("betrayal", b="CHAR_NOBODY"),
    ], frames, known)
    assert res["applied"] == 1
    assert sorted(x["reason"] for x in res["skipped"]) == [
        "non_present_scene", "unknown_character", "unverified_span"]
    st = book.get("CHAR_B", "CHAR_A")
    assert st.stage is R.STRANGERS and st.chapters_in_stage == 1


def test_cung_canh_la_tuong_tac_du_khong_co_su_kien():
    book = RelationshipBook()
    frames = [_frame("CH002_S00", locs={"CHAR_A": "LOC1", "CHAR_B": "LOC1", "CHAR_C": "LOC2"})]
    book.settle(2, [], frames, {"CHAR_A": 1, "CHAR_B": 1, "CHAR_C": 1})
    assert book.get("CHAR_A", "CHAR_B").chapters_in_stage == 1
    assert "CHAR_A|CHAR_C" not in book.states


def test_ban_chup_trang_thai_do_LLM_viet_KHONG_BAO_GIO_vao_canon():
    g, chars, _ = load_bible(DEFAULT_BIBLE)
    d = StateDelta(relationship_updates=[RelationshipState(
        a="CHAR_KAELEN", b="CHAR_SERENA", stage=R.CATHARSIS, intimacy=100)]).stamp(1)
    apply_delta(d, g, chars, [])
    st = g.relationships.get("CHAR_KAELEN", "CHAR_SERENA")
    assert st.stage is R.FRICTION and st.intimacy == 30


def test_bible_nap_trang_thai_mo_truyen():
    g, _, _ = load_bible(DEFAULT_BIBLE)
    st = g.relationships.states["CHAR_KAELEN|CHAR_SERENA"]
    assert st.stage is R.FRICTION and st.chapters_in_stage == 0


# ═══════════════ TRÍCH XUẤT ═══════════════

def test_xac_minh_span_va_gan_canh_cho_su_kien_quan_he():
    d = StateDelta(relationship_events=[_ev("shared_ordeal", verified=False, scene="CH001_S00"),
                                        _ev("betrayal", span="câu này không có trong văn bản")])
    prose0, prose1 = "Mở đầu cảnh không liên quan gì cả.", SPAN
    verify_spans(d, prose0 + "\n\n" + prose1)
    assert [e.verified for e in d.relationship_events] == [True, False]
    assign_scenes(d, [{"scene_id": "CH001_S00", "prose": prose0},
                      {"scene_id": "CH001_S01", "prose": prose1}])
    assert d.relationship_events[0].scene_id == "CH001_S01"


def test_parse_khoan_dung_su_kien_quan_he():
    raw = ('{"relationship_events": ['
           '{"a": "CHAR_A", "b": "CHAR_B", "kind": "betrayed", "span": "x", "verified": true},'
           '{"a": "CHAR_A", "b": "CHAR_B", "kind": "fell_in_love", "span": "y"}]}')
    d, issues = parse_delta_lenient(raw)
    assert [e.kind for e in d.relationship_events] == ["betrayal"]
    assert d.relationship_events[0].verified is False       # model không tự khai được
    assert {i["action"] for i in issues} == {"coerced", "dropped"}


# ═══════════════ ĐỒ THỊ ═══════════════

@pytest.fixture
def eng():
    e = build_engines(FakeLLM(), db_path=":memory:")
    # Văn xuôi FakeLLM dựng từ cùng một kho câu nhỏ nên các cảnh CÙNG CHỖ
    # trùng nhau thật (§4.3). Tắt dò lặp ở fixture tổng hợp; test Ngày 21
    # kiểm cơ chế đó bằng một LLM lặp có chủ đích.
    e.repetition_check = False
    yield e
    e.store.close()


def test_chi_thi_quan_he_chi_o_MOT_canh_moi_cap_va_chi_khi_ca_hai_co_mat(eng):
    out = run_chapter(eng, 1)
    placed = {d["pair"]: c["scene_index"] for c in out["contracts"]
              for d in c["relationship_directives"]}
    # Kaelen–Vhal có mặt cùng nhau ở cảnh 1 và 2 → chỉ cảnh 2; Kaelen–Serena ở cảnh 4.
    assert placed == {"CHAR_KAELEN|CHAR_VHAL": 2, "CHAR_KAELEN|CHAR_SERENA": 4}
    writer = [c["prompt"] for c in eng.llm.calls if c["role"] == "writer"]
    assert "CHƯA được" in writer[4] and "CHƯA được" not in writer[0]
    assert not any("intimacy" in p for p in writer)


def test_quan_he_duoc_fold_va_replay_ra_dung_trang_thai(tmp_path):
    db = str(tmp_path / "canon.db")
    e1 = build_engines(FakeLLM(), db_path=db)
    run_chapter(e1, 1, auto_commit=True)
    snap = e1.graph.relationships.snapshot()
    kv = e1.graph.relationships.states["CHAR_KAELEN|CHAR_VHAL"]
    assert kv.chapters_in_stage == 1 and kv.last_settled_chapter == 1
    e1.store.close()

    e2 = build_engines(FakeLLM(), db_path=db)
    try:
        assert e2.graph.relationships.snapshot() == snap
    finally:
        e2.store.close()
