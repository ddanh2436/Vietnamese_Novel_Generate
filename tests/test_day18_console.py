"""Ngày 18 — Author Console: checkpoint, màn hình diff CP-2, `report-debt`.

Checkpoint là chỗ dễ có lỗi IM LẶNG nhất trong ngày này: viết lại một chương mà
không xoá thread cũ thì LangGraph chạy lại từ đầu và CỘNG DỒN lên state cũ —
chương ra 12 cảnh, `frames` trùng lặp, không một lỗi nào được ném.
"""
from __future__ import annotations

import json

import pytest

from novel_engine.canon.models import Assertion, Clue, ClueStatus, Entity, StateDelta
from novel_engine.graph.build import build_chapter_graph, run_chapter
from novel_engine.graph.checkpoint import (
    chapter_thread, drop_thread, open_checkpointer, thread_status,
)
from novel_engine.graph.engines import build_engines
from novel_engine.eval.debt import DEBT_BLOCK, story_debt
from novel_engine.llm.fake import FakeLLM


@pytest.fixture
def cp_path(tmp_path):
    return str(tmp_path / "checkpoints.db")


def _eng(db=":memory:"):
    return build_engines(FakeLLM(), db_path=db)


# ═══════════════ CHECKPOINT ═══════════════

def test_checkpoint_luu_tien_trinh_va_bao_da_xong(cp_path):
    eng = _eng()
    try:
        with open_checkpointer(cp_path) as cp:
            assert thread_status(cp, 1)["exists"] is False
            run_chapter(eng, 1, checkpointer=cp, thread_id=chapter_thread(1))
            st = thread_status(cp, 1)
            assert st["exists"] and st["done"] and st["scenes_done"] == 6
    finally:
        eng.store.close()


def test_chay_lai_thread_cu_NHAN_DOI_du_lieu_neu_khong_xoa(cp_path):
    """Cạm bẫy thật của §9.4, đo trên LangGraph 1.2: truyền state ban đầu vào một
    thread đã có checkpoint là chạy lại từ START và CỘNG DỒN lên state cũ."""
    eng = _eng()
    try:
        with open_checkpointer(cp_path) as cp:
            run_chapter(eng, 1, checkpointer=cp, thread_id=chapter_thread(1))
            lai = run_chapter(eng, 1, checkpointer=cp, thread_id=chapter_thread(1))
            assert len(lai["scene_outputs"]) == 12      # 6 cũ + 6 mới, không lỗi nào

            n = len(eng.llm.calls)
            xong = run_chapter(eng, 1, checkpointer=cp, thread_id=chapter_thread(1),
                               resume=True)
            assert len(eng.llm.calls) == n              # thread đã xong: không chạy gì
            assert len(xong["scene_outputs"]) == 12

            drop_thread(cp, 1)
            assert thread_status(cp, 1)["exists"] is False
            sach = run_chapter(eng, 1, checkpointer=cp, thread_id=chapter_thread(1))
            assert len(sach["scene_outputs"]) == 6
    finally:
        eng.store.close()


def test_chuong_dut_giua_chung_viet_tiep_duoc(cp_path):
    """Trần recursion thấp = mô phỏng một lượt chạy bị cắt ngang."""
    eng = _eng()
    try:
        with open_checkpointer(cp_path) as cp:
            cfg = {"configurable": {"engines": eng, "thread_id": chapter_thread(1)},
                   "recursion_limit": 12}
            g = build_chapter_graph(cp)
            with pytest.raises(Exception):
                g.invoke({"chapter": 1, "total_chapters": eng.total_chapters,
                          "outline_beat": eng.planner.outline_beat(1),
                          "scene_outputs": [], "frames": [], "unresolved": [],
                          "time_drift": [], "audit_log": []}, cfg)
            giua = thread_status(cp, 1)
            assert giua["exists"] and not giua["done"] and 0 < giua["scenes_done"] < 6

            out = run_chapter(eng, 1, checkpointer=cp, thread_id=chapter_thread(1),
                              resume=True)
            assert len(out["scene_outputs"]) == 6      # đi tiếp, KHÔNG cộng dồn
            assert thread_status(cp, 1)["done"] is True
    finally:
        eng.store.close()


def test_cli_write_resume_khi_chua_co_checkpoint(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_LLM", "fake")
    ma = cli.main(["write", "--chapter", "1", "--llm", "fake",
                   "--db", str(tmp_path / "t.db"),
                   "--checkpoint-db", str(tmp_path / "cp.db"), "--resume"])
    assert ma == 2 and "không có checkpoint" in capsys.readouterr().err


def test_cli_write_bo_checkpoint_cu_khi_viet_lai(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_LLM", "fake")
    db, cp = str(tmp_path / "t.db"), str(tmp_path / "cp.db")
    lenh = ["write", "--chapter", "1", "--llm", "fake", "--db", db, "--checkpoint-db", cp]
    assert cli.main(lenh) == 0
    capsys.readouterr()
    assert cli.main(lenh + ["--force"]) == 0
    out = capsys.readouterr().out
    assert "bỏ checkpoint cũ" in out and "6 cảnh" in out


# ═══════════════ CP-2: MÀN HÌNH DIFF ═══════════════

def _delta_mau():
    e = Entity(id="OBJ_X", kind="object", name="Tấm thẻ kim loại")
    a = Assertion(subject="CHAR_KAELEN", predicate="has_scar", object=True, chapter=2,
                  scene=1, span="Vết sẹo dài chạy dọc cẳng tay trái của Kaelen.",
                  confidence=0.9, epistemic="objective")
    noi = Assertion(subject="CHAR_VHAL", predicate="status", object="mệt", chapter=2,
                    scene=1, span="— Tôi mệt rồi, — Vhal nói và đẩy tập hồ sơ sang bên.",
                    confidence=0.8, epistemic="claimed_by", holder="CHAR_VHAL")
    d = StateDelta(new_entities=[e], assertions=[a, noi],
                   clue_transitions={"CLUE_SEAL_CORROSION": ClueStatus.PLANTED}).stamp(2)
    out = {"classification": {_k(x): v for x, v in
                              ((e, "enrichment"), (a, "improvement"),
                               (noi, "enrichment"))}}
    return d, out


def _k(item):
    from novel_engine.canon.models import item_key
    return item_key(item)


def test_diff_CP2_ngan_gon_va_kem_lua_chon_cho_tac_gia():
    import cli
    eng = _eng()
    try:
        d, out = _delta_mau()
        lines = cli._diff_lines(d, out, eng, 2)
        text = "\n".join(lines)
        assert "Chương 2 đề xuất ghi vào canon" in text
        assert "+ " in text and "! " in text
        assert "[G] Giữ chi tiết mới" in text and "[B] Bỏ chi tiết" in text
        assert "lời khai" in text          # claimed_by đi đúng nơi
        assert "~ Manh mối CLUE_SEAL_CORROSION" in text
    finally:
        eng.store.close()


def test_diff_bao_mau_thuan_la_chan_ca_chuong():
    import cli
    eng = _eng()
    try:
        d, out = _delta_mau()
        key = next(iter(out["classification"]))
        out["classification"][key] = "contradiction"
        text = "\n".join(cli._diff_lines(d, out, eng, 2))
        assert "✗" in text and "chặn ghi CẢ chương" in text
    finally:
        eng.store.close()


# ═══════════════ REPORT-DEBT ═══════════════

def test_no_tu_su_gom_manh_moi_quan_he_tin_tuc_va_chi_muc_treo(tmp_path):
    eng = _eng()
    try:
        eng.graph.clues["CLUE_QUA_HAN"] = Clue(
            clue_id="CLUE_QUA_HAN", macro_event_target="EV", description="d",
            payoff_threshold=1, payoff_deadline=2, status=ClueStatus.PLANTED,
            salience=0.9, last_touched_chapter=2)
        st = eng.graph.relationships.get("CHAR_KAELEN", "CHAR_SERENA")
        st.chapters_in_stage = 9
        (tmp_path / "ch001.json").write_text(
            json.dumps({"unresolved": ["một tiếng động chưa ai kiểm tra"]}),
            encoding="utf-8")
        d = story_debt(eng, last_chapter=6, reports_dir=tmp_path)
        qua_han = {x["clue"] for x in d["clues"]["overdue"]}
        # Ba manh mối trong bible có hạn 5–6 nên ở chương 6 chúng cũng quá hạn.
        assert "CLUE_QUA_HAN" in qua_han
        assert d["relationships"][0]["pair"] == "CHAR_KAELEN|CHAR_SERENA"
        assert d["unresolved_total"] == 1
        assert d["debt_load"] >= 2
    finally:
        eng.store.close()


def test_no_vuot_nguong_thi_bao_chan():
    eng = _eng()
    try:
        for i in range(DEBT_BLOCK):
            eng.graph.clues[f"C{i}"] = Clue(
                clue_id=f"C{i}", macro_event_target="EV", description="d",
                payoff_threshold=1, payoff_deadline=1, status=ClueStatus.PLANTED)
        d = story_debt(eng, last_chapter=9, reports_dir="khong_ton_tai")
        assert d["blocked"] is True and d["debt_load"] > DEBT_BLOCK
    finally:
        eng.store.close()


def test_cli_report_debt(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NOVEL_LLM", "fake")
    db = str(tmp_path / "t.db")
    assert cli.main(["write", "--chapter", "1", "--llm", "fake", "--db", db,
                     "--checkpoint-db", str(tmp_path / "cp.db")]) == 0
    assert cli.main(["commit", "--chapter", "1", "--db", db]) == 0
    ma = cli.main(["report-debt", "--db", db])
    out = capsys.readouterr().out
    assert "Nợ tự sự tới chương 1" in out and "debt_load" in out
    assert ma in (0, 3)
    d = json.loads((tmp_path / "output" / "reports" / "debt.json").read_text(encoding="utf-8"))
    assert d["last_chapter"] == 1 and "clues" in d
