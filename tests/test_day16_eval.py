"""Ngày 16–17 — Eval Harness M1–M12 (§13, §13.1, §13.2, §13.3).

Mỗi metric có test cho ca ĐO ĐƯỢC và ca CHƯA ĐO ĐƯỢC. Phân biệt hai ca đó là
điều kiện để bảng điều khiển còn được người ta nhìn: một metric thiếu dữ liệu mà
hiện 0.0 sẽ bị đọc thành báo động, và báo động giả thì người dùng tắt bảng (P2).
"""
from __future__ import annotations

import json

import pytest

from novel_engine.canon.models import (
    Assertion, Clue, ClueStatus, PlantEvidence, StateDelta, item_key,
)
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.eval import metrics as M
from novel_engine.eval.harness import SPECS, evaluate, load_chapters
from novel_engine.eval.metrics import ChapterData
from novel_engine.eval.regression import run_regression
from novel_engine.eval.tension_measure import chapter_summary, measure_tension
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.relationship.book import RelationshipBook
from novel_engine.relationship.models import RelationshipState

SPAN = "Kaelen đặt tấm thẻ kim loại lên mặt bàn gỗ, không nói gì thêm."


def _a(subject="CHAR_KAELEN", predicate="status", obj="x", span=SPAN,
       epistemic="objective"):
    return Assertion(subject=subject, predicate=predicate, object=obj, chapter=1,
                     scene=0, span=span, confidence=0.9, epistemic=epistemic)


def _ch(n=1, **kw):
    return ChapterData(number=n, **kw)


# ═══════════════ ĐỘ CĂNG: vòng phản hồi bị hở ═══════════════

def test_do_cang_nam_trong_khoang_va_tang_theo_tin_hieu():
    scenes = [{"prose": "Gió thổi. " * 20} for _ in range(6)]
    yen = measure_tension(scenes, StateDelta(), [])
    don = measure_tension(scenes, StateDelta(assertions=[_a() for _ in range(18)]),
                          ["a", "b", "c", "d", "e", "f"])
    assert 0.0 <= yen < don <= 1.0


def test_tom_tat_chuong_lay_cau_dau_moi_digest():
    scenes = [{"digest": "Cảnh một xong. Chi tiết thừa."},
              {"digest": "Cảnh hai xong. Chi tiết thừa."}]
    assert chapter_summary(scenes) == "Cảnh một xong. Cảnh hai xong."
    assert len(chapter_summary([{"digest": "x" * 900}], max_chars=50)) == 50


def test_chuong_viet_xong_thi_GHI_do_cang_va_tom_tat():
    """Không ai gọi `put_chapter_summary` thì `measured_tension` trả 0.5 mãi mãi
    và tầng L2 của bộ nhớ luôn rỗng."""
    eng = build_engines(FakeLLM(), db_path=":memory:")
    try:
        out = run_chapter(eng, 1)
        assert eng.store.chapter_summaries(1, 1)
        do = eng.store.measured_tension(1)
        assert do == out["extraction_report"]["measured_tension"]
        assert 0.0 <= do <= 1.0
    finally:
        eng.store.close()


# ═══════════════ M1–M5 ═══════════════

def test_M1_doc_QUYET_DINH_da_luu_khong_hoi_lai_canon():
    """§13 hỏi canon SAU KHI đã ghi mệnh đề vào đó — canon chứa chính nó nên
    không bao giờ mâu thuẫn, metric luôn ≈1,0."""
    a1, a2 = _a(), _a(predicate="rank", obj="y")
    d = StateDelta(assertions=[a1, a2])
    d.classification = {item_key(a1): "enrichment", item_key(a2): "contradiction"}
    r = M.M1_entity_consistency([_ch(delta=d)])
    assert r["value"] == 0.5 and r["details"][0]["predicate"] == "rank"
    assert M.M1_entity_consistency([_ch()])["value"] is None


def test_M1_bo_qua_menh_de_khong_khach_quan():
    d = StateDelta(assertions=[_a(epistemic="claimed_by")])
    assert M.M1_entity_consistency([_ch(delta=d)])["value"] is None


def _clue(cid, status, dl=5, **kw):
    return Clue(clue_id=cid, macro_event_target="EV", description="d",
                payoff_threshold=1, payoff_deadline=dl, status=status, **kw)


def test_M2_tach_manh_moi_bi_BO_va_do_dung_han():
    class G:
        clues = {"A": _clue("A", ClueStatus.PAID_OFF, dl=2),
                 "B": _clue("B", ClueStatus.PLANTED, dl=2),
                 "C": _clue("C", ClueStatus.RETIRED)}
    d = StateDelta(clue_transitions={"A": ClueStatus.PAID_OFF})
    r = M.M2_clue_payoff_rate(G(), [_ch(3, delta=d)], last_chapter=3)
    assert r["value"] == 0.5                      # C bị bỏ, không tính vào mẫu
    assert r["overdue"] == ["B"] and r["paid_late"] == ["A"] and r["retired"] == ["C"]


def _chars():
    from novel_engine.canon.bible import DEFAULT_BIBLE, load_characters
    return load_characters(DEFAULT_BIBLE)


def _thoai(*cap):
    return {"scene_id": "CH001_S00",
            "prose": "\n".join(f"— {c} — {n} nói." for n, c in cap)}


def test_M3_khoang_cach_giong_khong_can_scipy():
    chars = _chars()
    khac = [_thoai(*[("Kaelen", "Sai số dung sai áp suất ca trực đúng quy trình.")] * 7,
                   *[("Serena", "Theo thẩm quyền hồ sơ cho thấy thủ tục đã được lưu.")] * 7)]
    r = M.M3_voice_distinctiveness([_ch(scenes=khac)], chars)
    assert r["value"] > 0.28

    giong = [_thoai(*[("Kaelen", "Hồ sơ cho thấy điều đó.")] * 7,
                    *[("Serena", "Hồ sơ cho thấy điều đó.")] * 7)]
    assert M.M3_voice_distinctiveness([_ch(scenes=giong)], chars)["value"] < 0.05


def test_M3_it_thoai_thi_CHUA_DO_DUOC_chu_khong_phai_0():
    r = M.M3_voice_distinctiveness([_ch(scenes=[_thoai(("Kaelen", "Ngắn."))])], _chars())
    assert r["value"] is None and "lượt thoại" in r["reason"]


def _frame(sid, locs, order=1, tick=20000):
    return ContinuityFrame(scene_id=sid, time=StoryTime(
        epoch_tick=tick, duration_ticks=2, narrative_order=order), locations=locs)


def test_M4_chi_dem_nhan_vat_phu():
    chars = _chars()
    chapters = [_ch(1, frames=[_frame("CH001_S00", {"CHAR_KAELEN": "L", "NPC_A": "L"})]),
                _ch(2, frames=[_frame("CH002_S00", {"NPC_A": "L", "NPC_B": "L"}, 2)])]
    r = M.M4_npc_reuse_ratio(chapters, chars)
    assert r["value"] == 0.5 and "CHAR_KAELEN" not in r["npcs"]


def test_M5_nhay_coc_va_khong_no_o_giai_doan_ngoai_truc():
    st = RelationshipState(a="A", b="B", stage="catharsis", history=[
        {"stage": "strangers", "chapter": 1}, {"stage": "vulnerability", "chapter": 2},
        {"stage": "rupture", "chapter": 3}, {"stage": "trial", "chapter": 6}])
    r = M.M5_relationship_pacing(RelationshipBook({"A|B": st}))
    assert r["value"] == 2                    # nhảy cóc + catharsis không sẹo
    assert any("nhảy" in x for x in r["violations"])


# ═══════════════ M6–M7 cần judge ═══════════════

class _Judge:
    def __init__(self, tra_ve):
        self.tra_ve = tra_ve
        self.calls = []

    def invoke(self, prompt, *, role=""):
        self.calls.append(prompt)
        return self.tra_ve


def test_M6_can_judge_va_cham_dung_ti_le():
    chars = _chars()
    scenes = [_thoai(*[("Kaelen", f"Sai số {i} nằm ngoài dung sai cho phép.")
                       for i in range(4)],
                     *[("Serena", f"Theo thẩm quyền, hồ sơ số {i} đã được lưu.")
                       for i in range(4)])]
    assert M.M6_blind_attribution([_ch(scenes=scenes)], chars)["value"] is None
    dung = json.dumps({str(i): ("CHAR_KAELEN" if i < 4 else "CHAR_SERENA")
                       for i in range(8)})
    r = M.M6_blind_attribution([_ch(scenes=scenes)], chars, _Judge(dung))
    assert r["value"] == 1.0 and r["n"] == 8
    r = M.M6_blind_attribution([_ch(scenes=scenes)], chars, _Judge("không JSON"))
    assert r["value"] is None


def test_M7_canh_thua_bi_dem():
    scenes = [{"scene_id": "CH001_S00", "prose": "x"}, {"scene_id": "CH001_S01", "prose": "y"}]
    assert M.M7_scene_necessity([_ch(scenes=scenes)])["value"] is None
    r = M.M7_scene_necessity([_ch(scenes=scenes)], _Judge('{"changed": false}'))
    assert r["value"] == 0.0 and len(r["unnecessary"]) == 2
    assert M.M7_scene_necessity([_ch(scenes=scenes)],
                                _Judge('{"changed": true, "what": "x"}'))["value"] == 1.0


def test_M7_khong_doan_qua_chu_dau_cau():
    """Lượt Arc 1 đầu tiên: judge trả lời "Không khí trong phòng đổi…" và
    `startswith("KHÔNG")` của §13 đếm cảnh CÓ thay đổi thành cảnh thừa."""
    scenes = [{"scene_id": "CH001_S00", "prose": "x"}]
    ans = '{"changed": true, "what": "Không khí trong phòng đổi hẳn sau câu đó"}'
    assert M.M7_scene_necessity([_ch(scenes=scenes)], _Judge(ans))["value"] == 1.0
    assert M.M7_scene_necessity([_ch(scenes=scenes)], _Judge("Không khí..."))["value"] is None


# ═══════════════ M8–M12 ═══════════════

class _Store:
    def __init__(self, do): self.do = do
    def measured_tension(self, ch): return self.do.get(ch, 0.5)


def test_M8_bao_CHUA_DO_khi_do_cang_chua_tung_duoc_ghi():
    chapters = [_ch(1), _ch(2)]
    assert M.M8_tension_mae(_Store({}), chapters, 5)["value"] is None
    r = M.M8_tension_mae(_Store({1: 0.30, 2: 0.55}), chapters, 5)
    assert r["value"] == pytest.approx(
        sum(abs(x["measured"] - x["target"]) for x in r["rows"]) / 2, abs=1e-4)


def test_M9_dung_lai_bo_do_nhip_cua_ngay_9():
    cau = " ".join(f"{'chữ ' * n}kết." for n in (2, 20, 5, 30, 3, 18, 9, 25, 7))
    r = M.M9_prose_rhythm([_ch(scenes=[{"prose": cau}])])
    assert r["sd_mean"] > 8.5 and r["ok"] is True
    assert M.M9_prose_rhythm([_ch(scenes=[{"prose": "Ngắn. Cụt."}])])["value"] is None


def test_M10_khong_tron_span_manh_moi_va_quan_he_vao_mau_so():
    rep = {"extraction": {"assertions_kept": 9, "rejected_spans": [
        {"reason": "span_not_found"}, {"reason": "plant_span_not_found"},
        {"reason": "relationship_span_too_short"}]}}
    r = M.M10_extraction_fidelity([_ch(report=rep)])
    assert r["value"] == 0.9 and r["plant_rejected"] == 1 and r["relationship_rejected"] == 1


def test_M11_chuong_khong_duoc_giao_manh_moi_khong_keo_diem_xuong():
    sach = {"extraction": {"promised_plants": [], "fulfilled_plants": []}}
    co = {"extraction": {"promised_plants": ["A", "B"], "fulfilled_plants": ["A"]}}
    assert M.M11_plan_fulfillment([_ch(report=sach)])["value"] is None
    assert M.M11_plan_fulfillment([_ch(1, report=sach), _ch(2, report=co)])["value"] == 0.5


def test_M12_do_khoang_cach_tri_thuc_doc_gia_va_nhan_vat(tmp_path):
    eng = build_engines(FakeLLM(), db_path=":memory:")
    try:
        run_chapter(eng, 1, auto_commit=True)
        # M12 chạy trên frame và canon, không cần file văn xuôi.
        chapters = load_chapters(eng, [1], chapters_dir=tmp_path, reports_dir=tmp_path)
        r = M.M12_irony_gap(chapters, eng.graph, eng.planner)
        assert r["value"] is not None and len(r["scenes"]) == 6
        # Manh mối đã cài nhưng chưa nhân vật nào HIỂU → độc giả biết trước.
        assert r["scenes"][-1]["gap"] >= 1
    finally:
        eng.store.close()


# ═══════════════ BẢNG ĐIỀU KHIỂN & HỒI QUY ═══════════════

def test_bang_dieu_khien_chay_that_va_tach_chua_do_khoi_bao_dong(tmp_path):
    eng = build_engines(FakeLLM(), db_path=":memory:")
    try:
        run_chapter(eng, 1, auto_commit=True)
        rep = evaluate(eng, [1], chapters_dir=tmp_path, reports_dir=tmp_path)
        assert set(rep["metrics"]) == set(SPECS)
        assert "M6_blind_attribution" in rep["not_measured"]      # chưa có judge
        assert all(a["metric"] not in rep["not_measured"] for a in rep["alerts"])
        assert rep["verdict"] in ("OK", "ALERT")
    finally:
        eng.store.close()


def test_hoi_quy_dao_dau_cho_metric_THAP_LA_TOT():
    """§13.2 dùng chung một dấu cho mọi metric: M8 (sai số) và M5 (số vi phạm)
    giảm là TIẾN BỘ, mà công thức gốc báo là hồi quy."""
    base = {"metrics": {"M1_entity_consistency": {"value": 1.0},
                        "M8_tension_mae": {"value": 0.20},
                        "M5_relationship_pacing": {"value": 2}}}
    cur = {"metrics": {"M1_entity_consistency": {"value": 0.80},
                       "M8_tension_mae": {"value": 0.10},
                       "M5_relationship_pacing": {"value": 4}}}
    r = run_regression(cur, base)
    assert r["verdict"] == "ROLLBACK"
    assert set(r["regressed"]) == {"M1_entity_consistency", "M5_relationship_pacing"}
    assert r["report"]["M8_tension_mae"]["delta_pct"] > 0


def test_hoi_quy_khong_coi_metric_chua_do_la_tut():
    r = run_regression({"metrics": {"M3_voice_distinctiveness": {"value": None}}},
                       {"metrics": {"M3_voice_distinctiveness": {"value": 0.4}}})
    assert r["verdict"] == "PASS" and r["not_comparable"] == ["M3_voice_distinctiveness"]


def test_cli_eval(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_LLM", "fake")
    db = str(tmp_path / "t.db")
    assert cli.main(["write", "--chapter", "1", "--llm", "fake", "--db", db]) == 0
    assert cli.main(["commit", "--chapter", "1", "--db", db]) == 0
    ma = cli.main(["eval", "--from", "1", "--to", "1", "--db", db])
    out = capsys.readouterr().out
    assert "M1_entity_consistency" in out and "chưa đo được" in out
    assert ma in (0, 3)
    rep = json.loads((tmp_path / "output" / "reports" / "eval_ch001_001.json")
                     .read_text(encoding="utf-8"))
    assert rep["chapters"] == [1] and set(rep["metrics"]) == set(SPECS)
