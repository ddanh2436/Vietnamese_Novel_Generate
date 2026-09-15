"""Ngày 8 — phân luồng ghi canon (§11).

Tiêu chí: sự thật khách quan vào graph; niềm tin vào hồ sơ nhân vật; lời khai
KHÔNG thành niềm tin của người nói; không làm bẩn canon.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from novel_engine.canon.models import (
    Assertion, ClueStatus, Entity, PlantEvidence, Relation, StateDelta,
)
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.reconcile.commit import apply_delta, reconcile, replay_committed

LOCS = ["LOC_ORE_PORT", "LOC_REACTOR_3", "LOC_RUINED_CATHEDRAL",
        "LOC_VEDA_CHECKPOINT"]


def _a(subject, predicate, obj, span, epistemic="objective", holder=None,
       scene=0, chapter=1):
    return Assertion(subject=subject, predicate=predicate, object=obj,
                     chapter=chapter, scene=scene, span=span, confidence=0.9,
                     epistemic=epistemic, holder=holder)


def _fr(sid, tick, dur, locs, mode="present", anchor=None):
    return ContinuityFrame(
        scene_id=sid,
        time=StoryTime(epoch_tick=tick, duration_ticks=dur,
                       narrative_order=int(sid[-2:]) + 1, mode=mode,
                       anchor_scene=anchor),
        locations=locs)


FR1 = [
    _fr("CH001_S00", 20000, 2, {"CHAR_KAELEN": "LOC_ORE_PORT"}),
    _fr("CH001_S01", 20006, 3, {"CHAR_KAELEN": "LOC_VEDA_CHECKPOINT",
                                "CHAR_VHAL": "LOC_VEDA_CHECKPOINT"}),
    _fr("CH001_S04", 20017, 3, {"CHAR_KAELEN": "LOC_ORE_PORT",
                                "CHAR_SERENA": "LOC_ORE_PORT"}),
]


@pytest.fixture
def eng():
    e = build_engines(FakeLLM(), db_path=":memory:")
    for f in FR1:
        e.store.put_frame(f)
    yield e
    e.store.close()


# ═══════════════ TIÊU CHÍ NGÀY 8 — PHÂN LUỒNG ═══════════════

def test_su_that_vao_graph_loi_khai_va_niem_tin_khong_ghi_de_su_that(eng):
    eng.graph.commit_truth("FACT_CIPHER", "status", "active", tick=0)
    d = StateDelta(assertions=[
        _a("FACT_CIPHER", "controls", "LOC_ORE_PORT",
           "Tổng Cục Mật Văn nắm quyền điều phối tại đây"),
        _a("FACT_CIPHER", "status", "dissolved",
           "— Tổng Cục Mật Văn đã bị giải thể, — Serena nói.",
           "claimed_by", "CHAR_SERENA", scene=4),
        _a("FACT_CIPHER", "status", "dissolved",
           "Kaelen tin rằng Tổng Cục Mật Văn đã bị giải thể",
           "believed_by", "CHAR_KAELEN", scene=1),
    ]).stamp(1)
    res = reconcile(d, eng)
    assert res["status"] == "committed"
    g = eng.graph
    assert g.truth_of("FACT_CIPHER", "controls") == "LOC_ORE_PORT"
    # F6: lời khai và niềm tin KHÔNG ghi đè sự thật khách quan
    assert g.truth_of("FACT_CIPHER", "status") == "active"
    kinds = {(b["holder"], b["kind"]) for b in g.beliefs}
    assert ("CHAR_SERENA", "claimed_by") in kinds
    assert ("CHAR_KAELEN", "believed_by") in kinds
    assert res["applied"]["truths"] == 1
    assert res["applied"]["claims"] == 1 and res["applied"]["beliefs"] == 1


def test_loi_noi_doi_KHONG_thanh_niem_tin_cua_nguoi_noi(eng):
    """§11 gọi `update_belief(char_id=item.holder)` cho cả lời khai. Serena nói
    dối → hệ thống ghi SERENA TIN điều đó → `deliberate()` (§5.2) cho cô hành
    động như thể lời nói dối là thật. Người nói dối bị biến thành người bị lừa."""
    d = StateDelta(assertions=[
        _a("FACT_CIPHER", "status", "dissolved",
           "— Tổng Cục Mật Văn đã bị giải thể, — Serena nói.",
           "claimed_by", "CHAR_SERENA")]).stamp(1)
    # Từ Ngày 13, commit còn dựng lại tri thức TIN TỨC (§5.6) — nguồn `news:` —
    # nên chỉ đếm những niềm tin KHÔNG đến từ tin tức.
    def khong_tu_tin_tuc(p):
        return [b for b in p.beliefs if not b.source.startswith("news:")]
    truoc = len(khong_tu_tin_tuc(eng.chars["CHAR_SERENA"]))
    reconcile(d, eng)
    serena = eng.chars["CHAR_SERENA"]
    assert len(khong_tu_tin_tuc(serena)) == truoc
    assert not any("dissolved" in b.proposition for b in serena.beliefs)


def test_niem_tin_sai_duoc_danh_dau_la_sai(eng):
    """Chênh lệch giữa confidence cao và `is_actually_true=False` là nguồn hiểu
    lầm TỰ NHIÊN (§5.1) — nguyên liệu của M12."""
    eng.graph.commit_truth("FACT_CIPHER", "status", "active", tick=0)
    d = StateDelta(assertions=[
        _a("FACT_CIPHER", "status", "dissolved",
           "Kaelen tin rằng Tổng Cục Mật Văn đã bị giải thể",
           "believed_by", "CHAR_KAELEN", scene=1)]).stamp(1)
    reconcile(d, eng)
    b = next(b for b in eng.chars["CHAR_KAELEN"].beliefs
             if "dissolved" in b.proposition)
    assert b.is_actually_true is False and b.confidence == pytest.approx(0.9)


def test_niem_tin_khoa_theo_epoch_cua_CANH_chua_no(eng):
    """NT-6: tri thức nhân vật khoá theo `epoch_tick`. Niềm tin sinh ở cảnh 1
    (tick 20006) không được có hiệu lực từ đầu chương."""
    d = StateDelta(assertions=[
        _a("CHAR_VHAL", "owes_money", "CHAR_SERENA",
           "Kaelen đoán Vhal nợ Serena một khoản", "believed_by",
           "CHAR_KAELEN", scene=1)]).stamp(1)
    reconcile(d, eng)
    b = next(x for x in eng.graph.beliefs if x["holder"] == "CHAR_KAELEN")
    assert b["since_tick"] == 20006


def test_cap_nhat_niem_tin_khong_chong_ban_sao(eng):
    for conf, ch in ((0.4, 1), (0.8, 2)):
        d = StateDelta(assertions=[Assertion(
            subject="CHAR_VHAL", predicate="owes_money", object="CHAR_SERENA",
            chapter=ch, scene=0, span="Kaelen đoán Vhal nợ Serena", confidence=conf,
            epistemic="believed_by", holder="CHAR_KAELEN")]).stamp(ch)
        reconcile(d, eng)
    hits = [b for b in eng.chars["CHAR_KAELEN"].beliefs if "owes_money" in b.proposition]
    assert len(hits) == 1 and hits[0].confidence == pytest.approx(0.8)


# ═══════════════ KHÔNG LÀM BẨN CANON ═══════════════

def test_muc_cach_ly_khong_vao_canon(eng):
    d = StateDelta(
        new_entities=[Entity(id="LOC_PIER_3", kind="location", name="cầu tàu số ba")],
        assertions=[_a("CHAR_KAELEN", "at", "LOC_REACTOR_3",
                       "Kaelen đứng trước Lò Phản ứng Số 3")]).stamp(1)
    res = reconcile(d, eng)
    assert res["status"] == "committed"
    assert not eng.graph.exists("LOC_PIER_3")
    assert eng.graph.location_ids() == sorted(LOCS)      # route graph nguyên vẹn
    assert eng.graph.truth_of("CHAR_KAELEN", "at") is None
    stored = eng.store.get_deltas(1)[0]
    assert stored.committed
    assert {q["reason"] for q in stored.quarantine} >= {
        "location_without_route", "contradicts_frames"}


def test_mau_thuan_chan_CA_delta_va_van_luu_quyet_dinh(eng):
    """Không ghi một phần: canon ghi nửa chương có mâu thuẫn là canon mà không
    phiên bản văn xuôi nào khớp."""
    eng.graph.commit_truth("FACT_CIPHER", "status", "active", tick=0)
    d = StateDelta(
        new_entities=[Entity(id="OBJ_X", kind="object", name="vật X")],
        assertions=[_a("FACT_CIPHER", "status", "dissolved",
                       "Tổng Cục Mật Văn đã bị giải thể từ lâu")]).stamp(1)
    res = reconcile(d, eng)
    assert res["status"] == "escalated" and "mâu thuẫn" in res["escalation_reason"]
    assert not eng.graph.exists("OBJ_X")
    stored = eng.store.get_deltas(1)[0]
    assert stored.committed is False and stored.classification


def test_khong_ghi_hai_lan(eng):
    assert reconcile(StateDelta().stamp(1), eng)["status"] == "committed"
    assert reconcile(eng.store.get_deltas(1)[0], eng)["status"] == "already_committed"


# ═══════════════ QUAN HỆ — F7 ═══════════════

def test_dong_quan_he_extractor_da_ghi(eng):
    eng.graph.upsert_relation(Relation(src="CHAR_VHAL", dst="CHAR_KAELEN",
                                       type="PROTECTS", provenance="extracted_ch1"))
    r = Relation(src="CHAR_VHAL", dst="CHAR_KAELEN", type="PROTECTS")
    res = reconcile(StateDelta(retracted_relations=[r]).stamp(1), eng)
    assert res["status"] == "committed" and res["applied"]["retracted"] == 1
    assert not eng.graph.active_relation("CHAR_VHAL", "CHAR_KAELEN", "PROTECTS")


def test_thu_hoi_quan_he_tac_gia_dat_bi_chan(eng):
    r = Relation(src="CHAR_SERENA", dst="FACT_CIPHER", type="MEMBER_OF")
    res = reconcile(StateDelta(retracted_relations=[r]).stamp(1), eng)
    assert res["status"] == "escalated"
    assert eng.graph.active_relation("CHAR_SERENA", "FACT_CIPHER", "MEMBER_OF")


def test_quan_he_moi_toi_thuc_the_moi(eng):
    d = StateDelta(
        new_entities=[Entity(id="OBJ_SO_CA_TRUC", kind="object", name="sổ ca trực")],
        new_relations=[Relation(src="CHAR_VHAL", dst="OBJ_SO_CA_TRUC", type="CONTROLS")]).stamp(1)
    res = reconcile(d, eng)
    assert res["status"] == "committed"
    assert eng.graph.active_relation("CHAR_VHAL", "OBJ_SO_CA_TRUC", "CONTROLS")
    assert eng.graph.g.edges["CHAR_VHAL", "OBJ_SO_CA_TRUC", "CONTROLS"]["provenance"] == "extracted_ch1"
    # quan hệ do Extractor ghi KHÔNG bị khoá — chương sau đổi được
    assert eng.graph.is_locked("CHAR_VHAL", "OBJ_SO_CA_TRUC", "CONTROLS") is False


# ═══════════════ IMPROVEMENT → PLANPATCH (NT-14) ═══════════════

def test_improvement_VAN_duoc_ghi_va_sinh_plan_patch(eng):
    """Văn xuôi đã viết ra là sự thật: `improvement` phải được ghi, nếu không
    kế hoạch chương sau xoay quanh một thứ không tồn tại trong canon."""
    d = StateDelta(new_entities=[
        Entity(id="OBJ_LOCATION_TAG", kind="object", name="Thẻ định vị")]).stamp(1)
    res = reconcile(d, eng)
    assert res["status"] == "committed"
    assert eng.graph.exists("OBJ_LOCATION_TAG")
    p = res["plan_patches"][0]
    assert p["from_chapter"] == 2 and p["invalidated_beats"] == ["CH002"]
    assert p["author_decision_required"] is True
    assert eng.store.get_plan_patches(1)[0]["patch_id"] == p["patch_id"]


def test_plan_patch_khong_sua_dan_y():
    """Agent không bao giờ ghi vào `bible/`."""
    from novel_engine.reconcile import commit, replan
    for mod in (commit, replan):
        src = inspect.getsource(mod)
        assert "outline.yaml\")" not in src and "write_text" not in src


# ═══════════════ MANH MỐI — E4 ═══════════════

def test_bang_chung_cai_cap_nhat_salience_theo_cuong_do(eng):
    d = StateDelta(
        plant_evidence=[
            PlantEvidence(clue_id="CLUE_SEAL_CORROSION", scene_id="CH001_S01",
                          span="x", carrier_used="object", verified=True),
            PlantEvidence(clue_id="CLUE_MISSING_LOGS", scene_id="CH001_S01",
                          span="x", carrier_used="object", verified=False)],
        clue_transitions={"CLUE_SEAL_CORROSION": ClueStatus.PLANTED}).stamp(1)
    contracts = [{"scene_id": "CH001_S01", "plant_directives": [
        {"clue_id": "CLUE_SEAL_CORROSION", "intensity": 0.2}]}]
    reconcile(d, eng, contracts=contracts)
    c = eng.graph.clues["CLUE_SEAL_CORROSION"]
    assert c.last_touched_chapter == 1
    assert c.salience == pytest.approx(0.45 + 0.55 * 0.2)
    assert c.status is ClueStatus.PLANTED and c.planted_in_chapter == 1
    assert eng.graph.clues["CLUE_MISSING_LOGS"].last_touched_chapter is None


# ═══════════════ HỒI ỨC — kiểm theo CẢNH ═══════════════

@pytest.fixture
def eng_hoi_uc(eng):
    eng.store.put_frame(_fr("CH002_S02", 2480, 4, {"CHAR_KAELEN": "LOC_REACTOR_3"},
                            mode="flashback", anchor="CH001_S01"))
    eng.store.put_frame(_fr("CH002_S04", 20054, 3, {"CHAR_KAELEN": "LOC_REACTOR_3"}))
    # canon: tại tick 20.000 Kaelen CHƯA có sẹo
    eng.graph.record_attestation("CHAR_KAELEN", "has_scar", 20000, present=False)
    return eng


def test_menh_de_canh_HIEN_TAI_khong_bi_kiem_nhu_hoi_uc(eng_hoi_uc):
    """§11 chạy `flashback_admissible` với TOÀN BỘ delta cho mỗi frame hồi ức
    → vết sẹo MỚI ở cảnh hiện tại bị kiểm như thể có từ tick 2.480 → blocker giả."""
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "has_scar", True, "Kaelen sờ vết sẹo mới trên trán",
           scene=4, chapter=2)]).stamp(2)
    res = reconcile(d, eng_hoi_uc)
    assert res["status"] == "committed", res["escalation_reason"]


def test_menh_de_trong_CANH_HOI_UC_pha_nhan_qua_bi_chan(eng_hoi_uc):
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "has_scar", True, "Cậu bé Kaelen đã có vết sẹo trên trán",
           scene=2, chapter=2)]).stamp(2)
    res = reconcile(d, eng_hoi_uc)
    assert res["status"] == "escalated" and "hồi ức" in res["escalation_reason"]


def test_menh_de_hoi_uc_ghi_o_tick_hoi_uc(eng):
    eng.store.put_frame(_fr("CH002_S02", 2480, 4, {"CHAR_KAELEN": "LOC_REACTOR_3"},
                            mode="flashback", anchor="CH001_S01"))
    eng.store.put_frame(_fr("CH002_S04", 20054, 3, {"CHAR_KAELEN": "LOC_REACTOR_3"}))
    d = StateDelta(assertions=[
        _a("CHAR_KAELEN", "bears_brand", True, "Cậu bé Kaelen mang dấu nung trên vai",
           scene=2, chapter=2)]).stamp(2)
    assert reconcile(d, eng)["status"] == "committed"
    w = eng.graph.attribute_window("CHAR_KAELEN", "bears_brand")
    assert w.attested_present == [2480]


# ═══════════════ CANON LÀ FOLD (§3.1) ═══════════════

def _snapshot(e):
    g = e.graph
    return (g.truth_of("FACT_CIPHER", "controls"),
            sorted(json.dumps(b, ensure_ascii=False, sort_keys=True) for b in g.beliefs),
            g.clues["CLUE_SEAL_CORROSION"].salience,
            g.clues["CLUE_SEAL_CORROSION"].last_touched_chapter,
            [b.proposition for b in e.chars["CHAR_KAELEN"].beliefs],
            g.exists("OBJ_SO_CA_TRUC"),
            g.active_relation("CHAR_VHAL", "OBJ_SO_CA_TRUC", "CONTROLS"))


def _delta_day_du() -> StateDelta:
    return StateDelta(
        new_entities=[Entity(id="OBJ_SO_CA_TRUC", kind="object", name="sổ ca trực")],
        new_relations=[Relation(src="CHAR_VHAL", dst="OBJ_SO_CA_TRUC", type="CONTROLS")],
        assertions=[
            _a("FACT_CIPHER", "controls", "LOC_ORE_PORT",
               "Tổng Cục Mật Văn nắm quyền điều phối tại đây"),
            _a("FACT_CIPHER", "status", "dissolved",
               "— Tổng Cục đã giải thể, — Serena nói.", "claimed_by", "CHAR_SERENA", scene=4),
            _a("CHAR_VHAL", "owes_money", "CHAR_SERENA",
               "Kaelen đoán Vhal nợ Serena", "believed_by", "CHAR_KAELEN", scene=1)],
        plant_evidence=[PlantEvidence(clue_id="CLUE_SEAL_CORROSION", scene_id="CH001_S01",
                                      span="x", carrier_used="object", verified=True)],
        plant_intensity={"CLUE_SEAL_CORROSION": 0.3}).stamp(1)


def test_fold_tai_tao_DUNG_trang_thai_o_tien_trinh_moi(tmp_path):
    """CLI chạy đa tiến trình, graph dựng lại từ bible mỗi lần. Không fold thì
    mọi thứ đã ghi ở tiến trình trước biến mất."""
    db = str(tmp_path / "canon.db")
    e1 = build_engines(FakeLLM(), db_path=db)
    for f in FR1:
        e1.store.put_frame(f)
    assert reconcile(_delta_day_du(), e1)["status"] == "committed"
    truoc = _snapshot(e1)
    e1.store.close()

    e2 = build_engines(FakeLLM(), db_path=db)      # "tiến trình mới"
    try:
        assert e2.replayed == 1
        assert _snapshot(e2) == truoc
        assert truoc[5] is True and truoc[3] == 1
    finally:
        e2.store.close()


def test_delta_chua_ghi_KHONG_duoc_fold(tmp_path):
    db = str(tmp_path / "canon.db")
    e1 = build_engines(FakeLLM(), db_path=db)
    e1.store.append_delta(_delta_day_du())          # lưu nhưng CHƯA ghi canon
    e1.store.close()
    e2 = build_engines(FakeLLM(), db_path=db)
    try:
        assert e2.replayed == 0 and not e2.graph.exists("OBJ_SO_CA_TRUC")
    finally:
        e2.store.close()


def test_rollback_la_xoa_delta_roi_fold_lai(tmp_path):
    """§3.1: "bỏ chương 18, viết lại" — không cần code rollback riêng."""
    db = str(tmp_path / "canon.db")
    e = build_engines(FakeLLM(), db_path=db)
    try:
        for f in FR1:
            e.store.put_frame(f)
        reconcile(_delta_day_du(), e)
        assert e.graph.exists("OBJ_SO_CA_TRUC")
        e.store.clear_chapter(1)
        assert e.rebuild_canon() == 0
        assert not e.graph.exists("OBJ_SO_CA_TRUC")
        assert e.graph.clues["CLUE_SEAL_CORROSION"].last_touched_chapter is None
        # firewall và assembler trỏ tới graph MỚI, không phải graph cũ
        assert e.firewall.g is e.graph and e.assembler.g is e.graph
    finally:
        e.store.close()


def test_ghi_va_phat_lai_di_qua_MOT_ham():
    """NT-11: hai bản sao luật ghi → trạng thái lúc ghi và lúc phát lại lệch nhau."""
    assert "apply_delta(" in inspect.getsource(reconcile)
    assert "apply_delta(" in inspect.getsource(replay_committed)


def test_fold_theo_epoch_khong_theo_so_chuong(tmp_path):
    """§3.6.4: hai delta cùng ghi một sự thật — bản có epoch MUỘN hơn phải thắng,
    kể cả khi nó thuộc chương số nhỏ hơn."""
    db = str(tmp_path / "canon.db")
    e = build_engines(FakeLLM(), db_path=db)
    e.store.put_frame(_fr("CH001_S00", 30000, 2, {"CHAR_KAELEN": "LOC_ORE_PORT"}))
    e.store.put_frame(_fr("CH002_S00", 10000, 2, {"CHAR_KAELEN": "LOC_ORE_PORT"}))
    for ch, val in ((1, "muộn"), (2, "sớm")):
        d = StateDelta(assertions=[Assertion(
            subject="FACT_CIPHER", predicate="motto", object=val, chapter=ch,
            scene=0, span="Tổng Cục Mật Văn có khẩu hiệu mới", confidence=0.9)]).stamp(ch)
        d.classification = {next(iter(__import__("novel_engine.canon.models",
                                                 fromlist=["item_key"]).item_key(a)
                                  for a in d.assertions)): "enrichment"}
        d.committed = True
        e.store.append_delta(d)
    try:
        e.rebuild_canon()
        assert e.graph.truth_of("FACT_CIPHER", "motto") == "muộn"
    finally:
        e.store.close()


# ═══════════════ ĐỒ THỊ ═══════════════

def test_mac_dinh_do_thi_KHONG_ghi_canon():
    """CP-2: tác giả phải thấy delta trước khi nó vào canon."""
    e = build_engines(FakeLLM(), db_path=":memory:")
    try:
        st = run_chapter(e, 1)
        assert "reconcile_report" not in st
        assert not e.graph.exists("OBJ_SO_CA_TRUC")
    finally:
        e.store.close()


def test_auto_commit_ghi_canon_trong_do_thi():
    e = build_engines(FakeLLM(), db_path=":memory:")
    try:
        st = run_chapter(e, 1, auto_commit=True)
        assert not st.get("escalated"), st.get("escalation_reason")
        assert st["reconcile_report"]["status"] == "committed"
        assert e.store.get_deltas(1)[0].committed
        assert e.graph.exists("OBJ_SO_CA_TRUC")
    finally:
        e.store.close()


# ═══════════════ CLI: write → review → commit ═══════════════

ROOT = Path(__file__).resolve().parents[1]
_ENV = {**os.environ, "NOVEL_LLM": "fake", "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(ROOT)}


def _cli(*args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          env=_ENV, cwd=str(cwd))


@pytest.mark.slow
def test_cli_write_review_commit_status(tmp_path):
    assert _cli("write", "--chapter", "1", cwd=tmp_path).returncode == 0
    r = _cli("review", "--chapter", "1", cwd=tmp_path)
    assert r.returncode == 0 and "chờ duyệt" in r.stdout and "WORLD GRAPH" in r.stdout
    c = _cli("commit", "--chapter", "1", cwd=tmp_path)
    assert c.returncode == 0, c.stdout + c.stderr
    assert "Ghi canon d_ch001" in c.stdout
    assert _cli("commit", "--chapter", "1", cwd=tmp_path).returncode == 2
    assert "ĐÃ GHI canon" in _cli("review", "--chapter", "1", cwd=tmp_path).stdout
    assert "đã ghi" in _cli("status", cwd=tmp_path).stdout


@pytest.mark.slow
def test_cli_canh_bao_khi_chuong_truoc_chua_ghi(tmp_path):
    _cli("write", "--chapter", "1", cwd=tmp_path)
    r = _cli("write", "--chapter", "2", cwd=tmp_path)
    assert r.returncode == 0 and "chưa ghi canon" in r.stderr


@pytest.mark.slow
def test_cli_viet_lai_chuong_da_ghi_thi_canon_cu_bi_go(tmp_path):
    """`write --force` xoá delta → dựng lại canon → sự thật của bản cũ biến mất
    TRƯỚC khi chương mới đọc ngữ cảnh."""
    _cli("write", "--chapter", "1", cwd=tmp_path)
    _cli("commit", "--chapter", "1", cwd=tmp_path)
    assert _cli("write", "--chapter", "1", "--force", cwd=tmp_path).returncode == 0
    from novel_engine.canon.sqlite_store import SqliteStore
    s = SqliteStore.from_file(tmp_path / "novel_storage.db")
    try:
        assert s.get_deltas(1)[0].committed is False
    finally:
        s.close()
