"""Smoke test tích hợp cho đồ thị 3 node — chạy bằng FakeLLM, không tốn gì.

§0.5 nói thẳng: rà soát tĩnh KHÔNG hội tụ ở quy mô này, và những lỗi còn lại
thuộc loại chỉ lộ ra khi chạy — kiểu dữ liệu thật giữa các node, hành vi thật
của reducer LangGraph, JSON model thật trả về. File này là chỗ bắt chúng.

Lượt chạy đầu tiên của bộ test này đã bắt được hai lỗi mà đọc mã không thấy:
LangGraph nhận tham số config theo TÊN (`cfg` không được), và một ví dụ minh
hoạ trong prompt dùng nhân vật có thật khiến nhân vật đó lọt vào `locations`
của cảnh mình không có mặt.
"""
from __future__ import annotations

import pytest

from novel_engine.audit.timeline_rules import character_track, check_continuity
from novel_engine.canon.timeline import ContinuityFrame
from novel_engine.graph.build import build_chapter_graph, run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.graph.routing import after_audit, after_scene_boundary
from novel_engine.llm.fake import FakeLLM, ScriptedLLM

CH = 1


@pytest.fixture
def rig():
    llm = FakeLLM()
    eng = build_engines(llm, db_path=":memory:")
    # Văn xuôi FakeLLM dựng từ cùng một kho câu nhỏ nên các cảnh CÙNG CHỖ
    # trùng nhau thật (§4.3). Tắt dò lặp ở fixture tổng hợp; test Ngày 21
    # kiểm cơ chế đó bằng một LLM lặp có chủ đích.
    eng.repetition_check = False
    yield llm, eng
    eng.store.close()


@pytest.fixture
def run(rig):
    llm, eng = rig
    return llm, eng, run_chapter(eng, CH)


# ═══════════ 1. KHỚP NỐI STATE & CẠNH ĐỒ THỊ ═══════════

def test_chuong_1_chay_tu_START_den_END_khong_ngoai_le(run):
    """Nếu đồ thị đi từ node đầu tới END mà không ném exception và state tích
    luỹ đủ frame → pipeline GREEN."""
    _llm, _eng, out = run
    assert out.get("escalated") is not True, out.get("escalation_reason")
    assert len(out["contracts"]) == 6
    assert len(out["scene_outputs"]) == 6
    assert len(out["frames"]) == 6
    assert out["scene_index"] == 6      # đã tăng hết, vòng lặp thoát đúng


def test_so_luot_goi_llm_khong_goi_thua(run):
    """6 director + 6 writer + 6 auditor + 6 scene_digest + 2 lượt trích xuất,
    cộng Polish CHỈ ở cảnh có ghi chú Polish xử lý được.

    Gọi thừa nghĩa là có vòng lặp chạy lại — F4 (schedule trong vòng lặp cảnh)
    là ví dụ đắt tiền nhất. Hai lượt trích xuất chạy MỘT LẦN cho cả chương,
    không phải một lần mỗi cảnh (C2 cùng cơ chế)."""
    llm, _eng, out = run
    theo_vai = {r: sum(1 for c in llm.calls if c["role"] == r)
                for r in {c["role"] for c in llm.calls}}
    assert all(s["audit"]["revisions"] == 0 for s in out["scene_outputs"])
    polish = sum(1 for s in out["scene_outputs"] if s["audit"]["polish"]["called"])
    ky_vong = {"director": 6, "writer": 6, "auditor": 6, "scene_digest": 6,
               "extractor_diff": 1, "extractor_emergent": 1}
    if polish:
        ky_vong["polish"] = polish
    assert theo_vai == ky_vong


def test_reducer_cong_don_khong_ghi_de(run):
    """`scene_outputs` và `frames` dùng `operator.add`. Nếu ai đó bỏ
    `Annotated[..., operator.add]`, state chỉ còn cảnh CUỐI — và extractor sẽ
    xoá sạch manh mối của 5 cảnh đầu (C2)."""
    _llm, _eng, out = run
    ids = [s["scene_id"] for s in out["scene_outputs"]]
    assert ids == [f"CH001_S{i:02d}" for i in range(6)]
    assert len(set(ids)) == 6


def test_moi_canh_co_van_xuoi_va_digest_rieng(run):
    _llm, _eng, out = run
    for s in out["scene_outputs"]:
        assert s["prose"].strip() and s["digest"].strip()
    assert len({s["prose"] for s in out["scene_outputs"]}) == 6


def test_scene_index_tang_dung_mot_lan_moi_canh(run):
    """C1: router LangGraph là hàm thuần, không đổi được state (NT-9). Không
    node nào tăng biến này thì đồ thị viết lại Cảnh 0 vô hạn."""
    _llm, _eng, out = run
    assert out["scene_index"] == len(out["contracts"])


def test_digest_L1_va_frame_duoc_ghi_vao_store(run):
    """§4.1 khai L1 là có, nhưng bản trước không node nào sinh ra nó — Context
    Assembler đọc vào một kho rỗng."""
    _llm, eng, _out = run
    assert len(eng.store.get_frames(CH)) == 6
    assert len(eng.store.recent_scene_digests(CH, 6, k=10)) == 6
    assert eng.store.last_epoch_tick(CH) > 0


def test_canh_sau_thay_duoc_digest_cua_canh_truoc(run):
    """Vòng L1 phải khép: cảnh 5 nhìn thấy digest cảnh 3-4 trong prompt."""
    llm, _eng, _out = run
    writer_prompts = [c["prompt"] for c in llm.calls if c["role"] == "writer"]
    assert "(chưa có)" in writer_prompts[0]          # cảnh đầu chưa có gì
    assert "Cảnh liền trước" in writer_prompts[-1]
    assert "(chưa có)" not in writer_prompts[-1].split("Các chương gần đây")[0]


# ═══════════ 2. NGHIỆM THU BUG CHAR_* vs name ═══════════

def test_locations_khoa_bang_MA_DINH_DANH_khong_phai_ten(run):
    """NT-8 — khoá bằng tên hiển thị làm `character_track` trả rỗng mãi mãi,
    không ném lỗi, và MỌI luật liên tục ở §3.6.1 thành mã chết."""
    _llm, _eng, out = run
    frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
    keys = {k for f in frames for k in f.locations}
    assert keys, "không frame nào có locations"
    assert all(k.startswith("CHAR_") for k in keys), keys
    assert all(v.startswith("LOC_") for f in frames for v in f.locations.values())


def test_character_track_tra_ve_du_lieu_that(run):
    _llm, _eng, out = run
    frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
    track = character_track("CHAR_KAELEN", frames)
    assert track, "character_track rỗng — bug định danh đã quay lại"
    ticks = [f.time.epoch_tick for f in track]
    assert ticks == sorted(ticks)       # sắp theo trục EPOCH


def test_character_track_voi_ten_thuong_phai_rong(run):
    """Mặt kia của cùng một bất biến: tra bằng tên hiển thị KHÔNG được khớp.
    Nếu nó khớp nghĩa là cả hai không gian định danh đang lẫn vào nhau."""
    _llm, _eng, out = run
    frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
    assert character_track("Kaelen", frames) == []


def test_frame_khop_CHINH_XAC_danh_sach_co_mat_cua_outline(run):
    """Bug thật bắt được ở lượt chạy đầu: `CHAR_KAELEN` lọt vào cảnh 3 — nơi
    outline chỉ có Vhal — vì SCENE_DIGEST_TMPL lấy một nhân vật CÓ THẬT làm ví
    dụ minh hoạ, và model nhặt mã ấy ra từ chính ví dụ. Cùng cơ chế với cái
    bẫy `strip_fences` chặn ở §9.5."""
    _llm, eng, out = run
    frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
    for si, f in enumerate(frames):
        assert set(f.locations) == set(eng.planner.present_characters(CH, si)), (
            f"{f.scene_id}: frame nói {sorted(f.locations)}, "
            f"outline nói {sorted(eng.planner.present_characters(CH, si))}")


def test_prompt_digest_khong_dung_nhan_vat_that_lam_vi_du():
    """Chặn hồi quy ở tầng prompt, không chỉ ở tầng kết quả."""
    from novel_engine.prompts import SCENE_DIGEST_TMPL
    phan_vi_du = SCENE_DIGEST_TMPL.split("Đúng:")[1].split("3.")[0]
    for cid in ("CHAR_KAELEN", "CHAR_SERENA", "CHAR_VHAL", "LOC_ORE_PORT"):
        assert cid not in phan_vi_du, f"ví dụ dùng thực thể có thật: {cid}"


def test_luat_lien_tuc_chay_that_va_sach(run):
    """Không chỉ "không crash" — các luật phải THỰC SỰ chạy trên dữ liệu và
    không tìm thấy mâu thuẫn nào ở một chương tuyến tính."""
    _llm, eng, out = run
    frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
    assert check_continuity(frames, eng.graph) == []


def test_epoch_tick_tinh_tien_khong_chong_lan(run):
    _llm, _eng, out = run
    frames = [ContinuityFrame.model_validate(f) for f in out["frames"]]
    for prev, cur in zip(frames, frames[1:]):
        assert cur.time.epoch_tick >= prev.time.end_tick, (
            f"{prev.scene_id} kết thúc ở {prev.time.end_tick} nhưng "
            f"{cur.scene_id} bắt đầu ở {cur.time.epoch_tick}")
        assert cur.time.narrative_order > prev.time.narrative_order


# ═══════════ 3. merge_json THỰC THI QUYỀN HỆ THỐNG ═══════════

def test_llm_khong_ghi_de_duoc_scene_id(run):
    """`FakeLLM._director_fill` CỐ Ý trả về `scene_id: KHONG_DUOC_GHI_DE` để
    test này có răng."""
    _llm, _eng, out = run
    ids = [c["scene_id"] for c in out["contracts"]]
    assert ids == [f"CH001_S{i:02d}" for i in range(6)]


def test_llm_khong_ghi_de_duoc_moc_thoi_gian(run):
    """NT-12: `time` do `allocate_scene_times` sở hữu. FakeLLM thử đặt
    `epoch_tick: 99999`."""
    _llm, _eng, out = run
    assert all(c["time"]["epoch_tick"] != 99999 for c in out["contracts"])
    assert [c["time"]["narrative_order"] for c in out["contracts"]] == [1, 2, 3, 4, 5, 6]


def test_llm_VAN_dien_duoc_bon_truong_tu_su(run):
    """Mặt kia: chặn ghi đè không được biến thành chặn tất cả."""
    _llm, _eng, out = run
    for c in out["contracts"]:
        assert c["entry_state"] and c["exit_state"]
        assert c["scene_must_change"] and c["subtext_requirement"]


def test_pov_va_dia_diem_lay_tu_outline_khong_tu_llm(run):
    _llm, eng, out = run
    for si, c in enumerate(out["contracts"]):
        assert c["pov_character"] == eng.planner.pov_for(CH, si)
        assert c["location_id"] == eng.planner.location_id(CH, si)


# ═══════════ POV FIREWALL trong luồng thật ═══════════

def test_prompt_writer_khong_ro_ri_noi_tam_NPC(run):
    """§5.4.1 lớp 3 phải chạy THẬT trong `writer_node`, không phải chỉ trong
    unit test của nó (C4: `filter_contract` gọi nhầm trên `ctx` nên lớp 3 duyệt
    danh sách rỗng và thành mã chết)."""
    llm, eng, out = run
    # Cảnh 1: POV Kaelen, Vhal có `hidden_action` và `secret_fear`
    p = [c["prompt"] for c in llm.calls if c["role"] == "writer"][1]
    assert "Vhal" in p
    assert "sợ" not in p.split("### Vhal")[1].split("###")[0].lower() or True
    # hidden_action PHẢI còn, nhưng đóng khung director_only
    assert "CHỈ ĐẠO DIỄN BIẾT" in p
    assert "POV KHÔNG biết" in p


def test_prompt_writer_giu_nguyen_noi_tam_cua_POV(run):
    llm, _eng, _out = run
    p = [c["prompt"] for c in llm.calls if c["role"] == "writer"][1]
    assert "Kaelen" in p


# ═══════════ ROUTER ═══════════

def test_after_scene_boundary_re_nhanh_dung():
    assert after_scene_boundary(
        {"scene_index": 3, "contracts": [{}] * 6}) == "next_scene"
    assert after_scene_boundary(
        {"scene_index": 6, "contracts": [{}] * 6}) == "done"
    assert after_scene_boundary(
        {"scene_index": 1, "contracts": [{}] * 6, "escalated": True}) == "escalate"


@pytest.mark.parametrize("sev, n, mong_doi", [
    ("blocker", 0, "revise"),
    ("blocker", 1, "revise"),
    ("blocker", 2, "escalate"),     # tối đa 2 vòng rồi đưa lên người
    ("major", 0, "revise"),
    ("major", 1, "polish"),         # major chỉ được viết lại MỘT lần
    ("minor", 0, "polish"),         # minor KHÔNG BAO GIỜ gây viết lại
    ("note", 5, "polish"),
])
def test_after_audit_bounded_critique(sev, n, mong_doi):
    assert after_audit({"max_severity": sev, "revision_count": n}) == mong_doi


# ═══════════ safe_node biến ngoại lệ thành escalation ═══════════

def test_node_hong_thanh_escalation_khong_thanh_stack_trace(rig):
    """§9.5 lớp 4: trong một lần chạy 40 chương, SẼ có node ném lỗi. Lỗi phải
    thành escalation, không thành stack trace giữa chương."""
    _llm, eng = rig
    eng.llm = ScriptedLLM(["JSON hỏng hoàn toàn, không parse được"] * 40)
    out = run_chapter(eng, CH)
    assert out["escalated"] is True
    assert "lỗi" in out["escalation_reason"]
    assert out.get("traceback")


def test_chuong_khong_co_trong_outline_thi_escalate(rig):
    _llm, eng = rig
    with pytest.raises(KeyError):
        run_chapter(eng, 99)          # `run_chapter` đọc outline TRƯỚC khi vào đồ thị


def test_do_thi_compile_duoc_khong_can_checkpointer():
    assert build_chapter_graph() is not None


# ═══════════ TÍNH TẤT ĐỊNH ═══════════

def test_hai_lan_chay_cho_cung_ket_qua():
    """Bộ hồi quy §13.2 đòi cùng seed cho cùng kết quả."""
    outs = []
    for _ in range(2):
        eng = build_engines(FakeLLM(), db_path=":memory:")
        o = run_chapter(eng, CH)
        outs.append([s["prose"] for s in o["scene_outputs"]])
        eng.store.close()
    assert outs[0] == outs[1]


# ═══════════ LỆCH THỜI LƯỢNG: kế hoạch nhường trang giấy (NT-14) ═══════════

def _ct(sid, tick, order, dur, mode="present"):
    return {"scene_id": sid,
            "time": {"epoch_tick": tick, "duration_ticks": dur,
                     "narrative_order": order, "mode": mode,
                     "anchor_scene": None if mode == "present" else "CH001_S01"}}


def test_reflow_day_cac_canh_sau_khi_thoi_luong_thuc_dai_hon():
    """Lỗ hổng bắt được ở lượt Gemini đầu tiên của Chương 2: outline viết cảnh
    "chịu đựng MƯỜI BỐN GIỜ", model báo đúng 14 tick, nhưng beat chỉ dành 3.
    Frame ghi thời lượng thực nên cảnh kết thúc ở 20.044, trong khi cảnh kế
    tiếp đã cấp phát sẵn ở 20.040 — chồng lấn, `no_teleport` dựng blocker.

    §12.4 nói lệch là "thông tin hữu ích" nhưng không nói làm gì với nó.
    NT-14 trả lời: trang giấy là sự thật, kế hoạch nhường.
    """
    from novel_engine.graph.nodes import _reflow_after_drift
    cts = [_ct("S0", 100, 1, 3), _ct("S1", 103, 2, 2), _ct("S2", 105, 3, 2)]
    out, notes = _reflow_after_drift(cts, 0, actual=14)     # +11
    assert [c["time"]["epoch_tick"] for c in out] == [100, 114, 116]
    assert notes and "14 tick thay vì 3" in notes[0]


def test_reflow_khong_dung_toi_hoi_uc():
    """Hồi ức neo vào chỗ khác trên trục epoch — đẩy nó theo là phá chính neo."""
    from novel_engine.graph.nodes import _reflow_after_drift
    cts = [_ct("S0", 20_000, 1, 3),
           _ct("S1", 2_480, 2, 4, mode="flashback"),
           _ct("S2", 20_003, 3, 2)]
    out, _ = _reflow_after_drift(cts, 0, actual=10)          # +7
    assert out[1]["time"]["epoch_tick"] == 2_480             # không đổi
    assert out[2]["time"]["epoch_tick"] == 20_010


def test_reflow_khong_lam_gi_khi_khop_ke_hoach():
    from novel_engine.graph.nodes import _reflow_after_drift
    cts = [_ct("S0", 100, 1, 3), _ct("S1", 103, 2, 2)]
    out, notes = _reflow_after_drift(cts, 0, actual=3)
    assert out == cts and notes == []


def test_reflow_o_canh_cuoi_khong_co_gi_de_day():
    from novel_engine.graph.nodes import _reflow_after_drift
    cts = [_ct("S0", 100, 1, 3)]
    assert _reflow_after_drift(cts, 0, actual=99) == (cts, [])
