"""Ngày 5 — CLI, SqliteStore xuyên tiến trình, và hồi ức.

Ba nhóm test, mỗi nhóm cho một điểm nghẽn:

1. Ranh giới TIẾN TRÌNH — WAL, roundtrip kiểu dữ liệu, idempotency.
2. Hồi ức — bất biến con trỏ, `_mode_valid`, dòng đời phân mảnh.
3. Vệ sinh ngữ cảnh — văn xuôi thô KHÔNG được rò sang chương sau.

Nhóm 1 chạy `cli.py` bằng `subprocess` thật. Trong cùng một tiến trình,
`sqlite3` giữ connection và cache trang, nên mọi lỗi thuộc loại "tiến trình
sau không đọc được thứ tiến trình trước ghi" đều không tái hiện được.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from novel_engine.audit.timeline_rules import (
    character_track, check_continuity, mode_valid, narrative_monotonic,
)
from novel_engine.canon.sqlite_store import SqliteStore
from novel_engine.canon.timeline import ContinuityFrame, StoryTime

ROOT = Path(__file__).resolve().parents[1]
_ENV = {**os.environ, "NOVEL_LLM": "fake", "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(ROOT)}


def _cli(*args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          env=_ENV, cwd=str(cwd))


@pytest.fixture
def workspace(tmp_path):
    """Thư mục làm việc riêng — CLI ghi `output/` và DB vào đây."""
    (tmp_path / "output" / "chapters").mkdir(parents=True)
    (tmp_path / "output" / "reports").mkdir(parents=True)
    return tmp_path


# ═══════════ 1. RANH GIỚI TIẾN TRÌNH ═══════════

def test_wal_duoc_bat_cho_file_khong_cho_memory(tmp_path):
    """Một tiến trình bị Ctrl-C giữa chương để lại `-wal`/`-shm`; WAL khôi
    phục sạch ở lần mở sau, journal mặc định có thể để lại lock chết."""
    s = SqliteStore.from_file(tmp_path / "x.db")
    assert s.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert s.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    s.close()
    # `:memory:` không dùng được WAL và cũng không cần — nó chết theo tiến trình
    m = SqliteStore(":memory:")
    assert m.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
    m.close()


@pytest.mark.slow
def test_hai_tien_trinh_noi_tiep_con_tro_thoi_gian(workspace):
    """Đúng kịch bản thật: `write --chapter 1` rồi TẮT, sau đó `--chapter 2`
    ở tiến trình MỚI. In-memory thì Chương 2 mất sạch `last_epoch_tick`."""
    r1 = _cli("write", "--chapter", "1", cwd=workspace)
    assert r1.returncode == 0, r1.stderr
    r2 = _cli("write", "--chapter", "2", cwd=workspace)
    assert r2.returncode == 0, r2.stderr

    s = SqliteStore.from_file(workspace / "novel_storage.db")
    ch1 = [f for f in s.get_frames(1) if f.time.mode == "present"]
    ch2 = [f for f in s.get_frames(2) if f.time.mode == "present"]
    ch2.sort(key=lambda f: f.time.narrative_order)
    assert min(f.time.epoch_tick for f in ch2) > max(f.time.end_tick for f in ch1)
    # trục ĐỌC cũng nối tiếp, không reset về 1
    assert min(f.time.narrative_order for f in ch2) > max(
        f.time.narrative_order for f in s.get_frames(1))
    s.close()


@pytest.mark.slow
def test_roundtrip_giu_nguyen_kieu_du_lieu(workspace):
    """Store lưu frame dạng JSON text. Nếu parser trả `dict` thô thay vì
    `ContinuityFrame`, `f.time.epoch_tick` ném AttributeError ở hạ nguồn —
    nhưng chỉ ở chương sau, không phải lúc ghi."""
    assert _cli("write", "--chapter", "1", cwd=workspace).returncode == 0
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    for f in s.get_frames():
        assert isinstance(f, ContinuityFrame)
        assert isinstance(f.time, StoryTime)
        assert isinstance(f.time.epoch_tick, int)
        assert f.time.mode in ("present", "flashback", "concurrent", "vision")
        assert isinstance(f.locations, dict)
        assert all(isinstance(k, str) and isinstance(v, str)
                   for k, v in f.locations.items())
    s.close()


@pytest.mark.slow
def test_idempotency_tu_choi_ghi_de_khi_khong_co_force(workspace):
    assert _cli("write", "--chapter", "1", cwd=workspace).returncode == 0
    lai = _cli("write", "--chapter", "1", cwd=workspace)
    assert lai.returncode == 2
    assert "--force" in lai.stderr


@pytest.mark.slow
def test_force_xoa_sach_ban_cu_khong_de_lai_canh_ma(workspace):
    """`INSERT OR REPLACE` một mình KHÔNG đủ: nếu lần trước có 6 cảnh và lần
    này có 4, hai cảnh cũ vẫn nằm lại với `narrative_order` cao hơn —
    `last_narrative_order` trả số của cảnh ma."""
    assert _cli("write", "--chapter", "1", cwd=workspace).returncode == 0
    db = workspace / "novel_storage.db"
    s = SqliteStore.from_file(db)
    s.put_frame(ContinuityFrame(
        scene_id="CH001_S99",
        time=StoryTime(epoch_tick=99_000, narrative_order=99),
        locations={"CHAR_KAELEN": "LOC_ORE_PORT"}))
    assert s.last_narrative_order(1) == 99
    s.close()

    assert _cli("write", "--chapter", "1", "--force", cwd=workspace).returncode == 0
    s = SqliteStore.from_file(db)
    assert "CH001_S99" not in {f.scene_id for f in s.get_frames()}
    assert s.last_narrative_order(1) == 6
    s.close()


def test_clear_chapter_khong_dung_toi_chuong_khac(tmp_path):
    s = SqliteStore.from_file(tmp_path / "x.db")
    for ch in (1, 2):
        s.put_frame(ContinuityFrame(
            scene_id=f"CH{ch:03d}_S00",
            time=StoryTime(epoch_tick=ch * 100, narrative_order=ch),
            locations={"CHAR_KAELEN": "LOC_ORE_PORT"}))
        s.put_scene_digest(ch, 0, f"digest {ch}")
    s.clear_chapter(1)
    assert not s.has_chapter(1) and s.has_chapter(2)
    assert s.recent_scene_digests(3, 0, k=5) == ["digest 2"]
    s.close()


@pytest.mark.slow
def test_cli_ghi_markdown_va_bao_cao(workspace):
    assert _cli("write", "--chapter", "1", cwd=workspace).returncode == 0
    md = workspace / "output" / "chapters" / "ch001.md"
    rp = workspace / "output" / "reports" / "ch001.json"
    assert md.exists() and rp.exists()
    assert "## Cảnh 0" in md.read_text(encoding="utf-8")
    rep = json.loads(rp.read_text(encoding="utf-8"))
    assert rep["scenes"] == 6 and rep["words"] > 500
    assert rep["continuity_findings"] == []


@pytest.mark.slow
def test_cli_status_doc_duoc_o_tien_trinh_moi(workspace):
    _cli("write", "--chapter", "1", cwd=workspace)
    r = _cli("status", cwd=workspace)
    assert r.returncode == 0
    assert "Cảng Quặng" in r.stdout and "sạch" in r.stdout


# ═══════════ 2. HỒI ỨC ═══════════

@pytest.mark.slow
def test_hoi_uc_KHONG_nhich_con_tro_dong_chinh(workspace):
    """Ca kiểm thử khắc nghiệt nhất. Chương 2 Cảnh 2 là hồi ức ở tick 2.480,
    trong khi hiện tại truyện ở tick ~20.000. Cảnh present NGAY SAU nó phải
    nhận `cursor + travel`, tuyệt đối không phải `2484 + Δt`."""
    _cli("write", "--chapter", "1", cwd=workspace)
    assert _cli("write", "--chapter", "2", cwd=workspace).returncode == 0

    s = SqliteStore.from_file(workspace / "novel_storage.db")
    fr = sorted(s.get_frames(2), key=lambda f: f.time.narrative_order)
    fb = next(f for f in fr if f.time.mode == "flashback")
    assert fb.time.epoch_tick == 2480

    truoc = [f for f in fr if f.time.mode == "present"
             and f.time.narrative_order < fb.time.narrative_order][-1]
    sau = [f for f in fr if f.time.mode == "present"
           and f.time.narrative_order > fb.time.narrative_order][0]

    assert sau.time.epoch_tick != fb.time.end_tick, "hồi ức đã nhích con trỏ"
    assert sau.time.epoch_tick > truoc.time.end_tick
    assert sau.time.epoch_tick > 20_000
    s.close()


@pytest.mark.slow
def test_dia_diem_hoi_uc_khong_thanh_diem_xuat_phat(workspace):
    """Hồi ức ở Lò 3, nhưng dòng chính đang ở Cảng Quặng. Nếu địa điểm hồi ức
    trở thành `prev_loc`, cảnh present tiếp theo sẽ được cộng quãng đường TỪ
    LÒ 3 — sai, và sai một cách khó thấy vì con số vẫn "hợp lý"."""
    _cli("write", "--chapter", "1", cwd=workspace)
    _cli("write", "--chapter", "2", cwd=workspace)
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    from novel_engine.canon.bible import load_bible
    g, _c, _m = load_bible()
    from novel_engine.planner.outline_planner import OutlinePlanner
    pl = OutlinePlanner()

    fr = sorted(s.get_frames(2), key=lambda f: f.time.narrative_order)
    fb_i = next(i for i, f in enumerate(fr) if f.time.mode == "flashback")
    truoc, sau = fr[fb_i - 1], fr[fb_i + 1]
    gap = sau.time.epoch_tick - truoc.time.end_tick
    can = g.travel_ticks(pl.location_id(2, truoc.time.narrative_order - 7),
                         pl.location_id(2, sau.time.narrative_order - 7))
    assert gap == can, f"quãng đường tính từ sai điểm xuất phát (gap={gap}, cần={can})"
    s.close()


@pytest.mark.slow
def test_luat_lien_tuc_sach_xuyen_chuong(workspace):
    """Luật chạy trên DÒNG ĐỜI nhân vật, xuyên chương (§3.6.1) — không phải
    trong phạm vi một chương."""
    _cli("write", "--chapter", "1", cwd=workspace)
    _cli("write", "--chapter", "2", cwd=workspace)
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    from novel_engine.canon.bible import load_bible
    g, _c, _m = load_bible()
    assert check_continuity(s.get_frames(), g) == []
    s.close()


@pytest.mark.slow
def test_no_teleport_tu_di_qua_ranh_gioi_hoi_uc(workspace):
    """Không cần NGOẠI LỆ cho ranh giới hồi ức: khoảng cách epoch giữa hồi ức
    (2.484) và cảnh hiện tại sớm nhất (20.000) là 17.516 tick, thừa cho mọi
    quãng đường. Sắp theo epoch làm chuyện đó thành tự nhiên."""
    _cli("write", "--chapter", "1", cwd=workspace)
    _cli("write", "--chapter", "2", cwd=workspace)
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    track = character_track("CHAR_KAELEN", s.get_frames())
    assert track[0].time.mode == "flashback"       # quá khứ đứng đầu dòng đời
    assert track[1].time.epoch_tick - track[0].time.end_tick > 10_000
    s.close()


def test_mode_valid_phan_biet_ba_tinh_huong():
    """Bản đầu gộp "không khai anchor" với "anchor không tra được" — cái thứ
    hai gần như luôn là lỗi PHẠM VI của người gọi, và báo blocker cho nó là
    báo động giả (P2)."""
    neo = ContinuityFrame(scene_id="CH001_S01",
                          time=StoryTime(epoch_tick=20_000, narrative_order=2),
                          locations={})
    anchors = {neo.scene_id: neo}

    khong_khai = ContinuityFrame(
        scene_id="CH002_S02",
        time=StoryTime(epoch_tick=2480, duration_ticks=4, narrative_order=9,
                       mode="flashback"),
        locations={})
    assert "không khai anchor_scene" in mode_valid(khong_khai, anchors)[0]

    khong_tra_duoc = khong_khai.model_copy(deep=True)
    khong_tra_duoc.time.anchor_scene = "CH009_S99"
    assert "không tra được" in mode_valid(khong_tra_duoc, anchors)[0]

    hop_le = khong_khai.model_copy(deep=True)
    hop_le.time.anchor_scene = "CH001_S01"
    assert mode_valid(hop_le, anchors) == []

    # …và hồi ức nằm SAU mốc neo vẫn phải bị bắt
    sai_moc = hop_le.model_copy(deep=True)
    sai_moc.time.epoch_tick = 30_000
    assert "SAU mốc neo" in mode_valid(sai_moc, anchors)[0]


def test_narrative_monotonic_doc_lap_thu_tu_dau_vao():
    """NT-8 lần nữa: bản gốc so `seen == sorted(seen)` trên danh sách nhận
    vào, tức giả định danh sách ĐÃ theo thứ tự đọc. Nhưng `get_frames()` trả
    theo trục EPOCH, nên một chương có hồi ức bị báo "không đơn điệu" dù hoàn
    toàn hợp lệ."""
    def fr(sid, tick, order, mode="present"):
        return ContinuityFrame(
            scene_id=sid,
            time=StoryTime(epoch_tick=tick, narrative_order=order, mode=mode,
                           anchor_scene="CH001_S01" if mode != "present" else None),
            locations={})

    # sắp theo EPOCH: hồi ức đứng đầu với narrative_order lớn — HỢP LỆ
    theo_epoch = [fr("CH002_S02", 2480, 9, "flashback"),
                  fr("CH002_S00", 20_028, 7), fr("CH002_S01", 20_030, 8)]
    assert narrative_monotonic(theo_epoch) == []

    # lỗi THẬT mà luật cần bắt: hai cảnh cùng số thứ tự đọc
    trung = [fr("CH002_S00", 20_028, 7), fr("CH002_S01", 20_030, 7)]
    out = narrative_monotonic(trung)
    assert out and "trùng nhau" in out[0] and "CH002_S00" in out[0]


# ═══════════ 3. VỆ SINH NGỮ CẢNH ═══════════

def test_van_xuoi_tho_khong_bao_gio_vao_prompt_chuong_sau():
    """Nếu Writer ở Chương 1 lỡ đưa Vhal vào một cảnh anh ta không có trong
    hợp đồng, văn xuôi đó KHÔNG được nhồi thô sang Chương 2 — nếu không model
    tiếp tục ảo giác rằng Vhal đang đứng cạnh Kaelen.

    Ngữ cảnh xuyên chương chỉ đi qua L1 digest; `scene_outputs[*].prose` được
    ghi nhưng không hàm nào đọc lại vào prompt.
    """
    import inspect

    from novel_engine.memory import assembler
    src = inspect.getsource(assembler.ContextAssembler.build)
    assert "recent_scene_digests" in src
    assert "prose" not in src

    from novel_engine.graph import nodes
    ns = inspect.getsource(nodes)
    # `prose` chỉ xuất hiện ở: biến cục bộ của scene_boundary, prompt digest
    # của CHÍNH cảnh đó, và scene_outputs. Không chỗ nào đọc ngược.
    assert "scene_outputs" not in inspect.getsource(nodes.writer_node)


@pytest.mark.slow
def test_frame_chi_ghi_nhan_nhan_vat_trong_hop_dong(workspace):
    """Hàng rào `_coerce_continuity`: dù văn xuôi có nhắc ai, `ContinuityFrame`
    chỉ ghi nhân vật CÓ TRONG hợp đồng. Canon sạch kể cả khi văn xuôi trôi."""
    _cli("write", "--chapter", "1", cwd=workspace)
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    from novel_engine.planner.outline_planner import OutlinePlanner
    pl = OutlinePlanner()
    for f in sorted(s.get_frames(1), key=lambda x: x.time.narrative_order):
        si = f.time.narrative_order - 1
        assert set(f.locations) == set(pl.present_characters(1, si))
    s.close()


# ═══════════ NGÀY 6: delta được LƯU vào log ═══════════

@pytest.mark.slow
def test_cli_luu_delta_vao_log_sau_moi_chuong(workspace):
    """`extractor_node` sinh delta, nhưng nếu CLI không gọi `append_delta` thì
    delta biến mất khi tiến trình thoát — `reconcile_node` ở GĐ3 sẽ không có gì
    để duyệt. §3.1: log chỉ ghi thêm, là lịch sử sáng tác bất biến.

    Lưu bằng chứng ≠ chấp nhận bằng chứng: delta vào LOG ngay, còn vào GRAPH
    thì phải qua CP-2."""
    assert _cli("write", "--chapter", "1", cwd=workspace).returncode == 0
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    deltas = s.get_deltas(1)
    assert len(deltas) == 1
    d = deltas[0]
    assert d.delta_id == "d_ch001" and d.chapter == 1
    assert d.committed is False                  # chưa qua reconcile
    # span bịa của FakeLLM đã bị lượt 3 loại TRƯỚC khi vào log
    assert all("không tồn tại" not in a.span for a in d.assertions)
    s.close()


@pytest.mark.slow
def test_cli_bao_cao_co_phan_trich_xuat(workspace):
    r = _cli("write", "--chapter", "1", cwd=workspace)
    assert "trích xuất:" in r.stdout
    rep = json.loads((workspace / "output" / "reports" / "ch001.json")
                     .read_text(encoding="utf-8"))
    ex = rep["extraction"]
    assert ex["assertions_kept"] >= 1
    assert any(x["reason"] == "span_not_found" for x in ex["rejected_spans"])


@pytest.mark.slow
def test_force_xoa_ca_delta_cu(workspace):
    """`clear_chapter` phải xoá cả `delta_log`, nếu không chạy lại một chương
    để lại delta của bản cũ — và bản cũ mô tả văn xuôi không còn tồn tại."""
    _cli("write", "--chapter", "1", cwd=workspace)
    assert _cli("write", "--chapter", "1", "--force", cwd=workspace).returncode == 0
    s = SqliteStore.from_file(workspace / "novel_storage.db")
    assert len(s.get_deltas(1)) == 1
    s.close()
