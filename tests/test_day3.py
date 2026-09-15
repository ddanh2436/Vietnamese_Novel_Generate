"""Ngày 3 — SqliteStore, Timeline Allocator, ContextAssembler, POVFirewall.

Mọi test dùng `SqliteStore(":memory:")`: chạy trong vài mili-giây, không để
lại file rác, và mỗi test có một DB sạch hoàn toàn.
"""
from __future__ import annotations

import pytest

from novel_engine.canon.bible import load_bible
from novel_engine.canon.models import Assertion, Entity, StateDelta
from novel_engine.canon.sqlite_store import SqliteStore
from novel_engine.canon.store_port import StorePort
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.character.firewall import POVFirewall, pov_leak_scan
from novel_engine.memory.assembler import (
    ContextAssembler, ntok, render_facts, truncate_to,
)
from novel_engine.memory.hierarchy import MemoryBudget
from novel_engine.planner.timeline_alloc import (
    INTER_CHAPTER_GAP, allocate_scene_times,
)


@pytest.fixture
def store() -> SqliteStore:
    s = SqliteStore(":memory:")
    yield s
    s.close()


def _frame(scene_id: str, tick: int, order: int, dur: int = 3,
           mode: str = "present", locs: dict | None = None) -> ContinuityFrame:
    return ContinuityFrame(
        scene_id=scene_id,
        time=StoryTime(epoch_tick=tick, duration_ticks=dur,
                       narrative_order=order, mode=mode,
                       anchor_scene="CH001_S00" if mode != "present" else None),
        locations=locs or {"CHAR_KAELEN": "LOC_ORE_PORT"})


# ═══════════════════ 1. SQLITE STORE ═══════════════════

def test_sqlite_store_thoa_man_store_port(store: SqliteStore):
    assert isinstance(store, StorePort)


def test_sqlite_store_persistence_qua_tien_trinh(tmp_path):
    """Lý do tồn tại của SqliteStore: `cli.py write --chapter 1` hôm nay và
    `--chapter 2` hôm sau là HAI TIẾN TRÌNH. In-memory thì Chương 2 mất sạch
    con trỏ thời gian và mọi digest của Chương 1."""
    db = tmp_path / "novel_storage.db"
    s1 = SqliteStore.from_file(db)
    s1.put_scene_digest(1, 0, "Kaelen tới cảng, bị Vhal từ chối cấp phép.")
    s1.put_frame(_frame("CH001_S00", tick=6, order=1, dur=3))
    s1.put_chapter_summary(1, "Chương 1: Kaelen trở lại Cảng Quặng.", 0.42)
    s1.close()

    s2 = SqliteStore.from_file(db)          # tiến trình "mới"
    assert s2.last_epoch_tick(1) == 9       # 6 + 3
    assert s2.last_narrative_order(1) == 1
    assert s2.measured_tension(1) == pytest.approx(0.42)
    assert "Cảng Quặng" in s2.chapter_summaries(1, 1)[0]
    assert s2.recent_scene_digests(2, 0, k=2) == [
        "Kaelen tới cảng, bị Vhal từ chối cấp phép."]
    s2.close()


def test_last_epoch_tick_tinh_ca_thoi_luong(store: SqliteStore):
    """Trả `end_tick`, KHÔNG phải `epoch_tick` thuần.

    Cảnh cuối bắt đầu tick 100 kéo dài 8 tick ("ba tuần lênh đênh") kết thúc ở
    108. Nếu trả 100, Chương 2 bắt đầu ở 100+6 = 106 — CHỒNG LẤN 2 tick với
    cảnh vẫn đang diễn ra, và `_no_bilocation` (§3.6.1) dựng blocker ở mọi
    nhân vật có mặt cả hai nơi. Bộ cấp phát tự sinh ra báo động của chính nó.
    """
    store.put_frame(_frame("CH001_S00", tick=100, order=1, dur=8))
    assert store.last_epoch_tick(1) == 108


def test_last_epoch_tick_bo_qua_hoi_uc_va_ao_anh(store: SqliteStore):
    """Hồi ức neo vào quá khứ, ảo ảnh nhìn về tương lai — cả hai KHÔNG được
    kéo con trỏ dòng chính. Một `vision` về tick 90.000 mà tính vào đây sẽ đẩy
    Chương 2 nhảy ra khỏi câu chuyện."""
    store.put_frame(_frame("CH001_S00", tick=10, order=1, dur=3))
    store.put_frame(_frame("CH001_S01", tick=90_000, order=2, dur=1,
                           mode="vision"))
    store.put_frame(_frame("CH001_S02", tick=100, order=3, dur=2,
                           mode="flashback"))
    assert store.last_epoch_tick(1) == 13          # chỉ cảnh present
    assert store.last_narrative_order(1) == 3      # trục ĐỌC tính hết


def test_con_tro_fallback_ve_0_o_chuong_dau(store: SqliteStore):
    assert store.last_epoch_tick(0) == 0
    assert store.last_epoch_tick(-1) == 0
    assert store.last_narrative_order(0) == 0


def test_epoch_tick_of_nem_keyerror_khi_anchor_mo_coi(store: SqliteStore):
    """Cảnh hồi ức trỏ vào anchor không tồn tại là lỗi thật. Trả 0 im lặng
    khiến `_mode_valid` (§3.6.1) thấy một hồi ức "hợp lệ" ở tick 0."""
    store.put_frame(_frame("CH001_S00", tick=42, order=1))
    assert store.epoch_tick_of("CH001_S00") == 42
    with pytest.raises(KeyError):
        store.epoch_tick_of("CH009_S99")


def test_recent_scene_digests_vuot_ranh_gioi_chuong(store: SqliteStore):
    for si, txt in enumerate(["ch1s0", "ch1s1", "ch1s2"]):
        store.put_scene_digest(1, si, txt)
    store.put_scene_digest(2, 0, "ch2s0")
    # Đứng ở Chương 2 Cảnh 1: hai cảnh trước là ch2s0 và ch1s2
    assert store.recent_scene_digests(2, 1, k=2) == ["ch1s2", "ch2s0"]
    # Thứ tự CŨ → MỚI là hợp đồng với truncate_to(recency_weighted=True)
    assert store.recent_scene_digests(1, 3, k=2) == ["ch1s1", "ch1s2"]


def test_recent_scene_digests_o_canh_dau_tien_tra_rong(store: SqliteStore):
    assert store.recent_scene_digests(1, 0, k=2) == []


def test_chapter_summaries_khoang_dong_va_dung_thu_tu(store: SqliteStore):
    for ch in range(1, 6):
        store.put_chapter_summary(ch, f"tóm tắt {ch}")
    assert store.chapter_summaries(2, 4) == ["tóm tắt 2", "tóm tắt 3", "tóm tắt 4"]
    assert store.chapter_summaries(3, 1) == []      # khoảng ngược → rỗng
    assert store.chapter_summaries(-4, 0) == []     # Chương 1 gọi (ch-5, ch-1)


def test_measured_tension_mac_dinh_khi_chua_co(store: SqliteStore):
    """`director_node` gọi `measured_tension(ch-1)`; ở Chương 1 đó là chương 0."""
    assert store.measured_tension(0) == 0.5
    store.put_chapter_summary(1, "s", 0.81)
    assert store.measured_tension(1) == pytest.approx(0.81)


def test_arc_summaries_loc_theo_chuong(store: SqliteStore):
    store.put_arc_summary("ARC_1", closed_after=8, summary="Arc 1 đóng ở ch8")
    store.put_arc_summary("ARC_2", closed_after=20, summary="Arc 2 đóng ở ch20")
    assert store.arc_summaries(before_chapter=10) == ["Arc 1 đóng ở ch8"]
    assert store.arc_summaries(before_chapter=5) == []


def test_get_frames_sap_theo_truc_EPOCH_khong_theo_thu_tu_doc(store: SqliteStore):
    """B4/NT-6: các luật liên tục (§3.6.1) chạy trên DÒNG ĐỜI nhân vật, sắp
    theo epoch. Trả theo `narrative_order` là tái tạo đúng lỗi mà §3.6 sinh ra
    để chặn: hồi ức đọc sau nhưng xảy ra trước."""
    store.put_frame(_frame("CH001_S00", tick=5000, order=1))
    store.put_frame(_frame("CH001_S01", tick=500, order=2, mode="flashback"))
    assert [f.scene_id for f in store.get_frames()] == ["CH001_S01", "CH001_S00"]


def test_put_frame_la_upsert(store: SqliteStore):
    store.put_frame(_frame("CH001_S00", tick=10, order=1))
    store.put_frame(_frame("CH001_S00", tick=20, order=1))     # viết lại cảnh
    frames = store.get_frames()
    assert len(frames) == 1 and frames[0].time.epoch_tick == 20


def test_delta_log_roundtrip(store: SqliteStore):
    d = StateDelta(
        new_entities=[Entity(id="LOC_X", kind="location", name="X")],
        assertions=[Assertion(subject="S", predicate="p", object=1, chapter=3,
                              scene=0, span="bằng chứng", confidence=0.9)],
    ).stamp(3)
    store.append_delta(d)
    back = store.get_deltas(3)
    assert len(back) == 1
    assert back[0].delta_id == "d_ch003"
    assert back[0].new_entities[0].id == "LOC_X"
    assert back[0].assertions[0].span == "bằng chứng"


def test_delta_log_khong_trung_lap_khi_chay_lai(store: SqliteStore):
    """`delta_id` tất định → chạy lại cùng chương thì GHI ĐÈ, không nhân bản."""
    for _ in range(3):
        store.append_delta(StateDelta().stamp(2))
    assert len(store.get_deltas()) == 1


def test_tick_of_chapter(store: SqliteStore):
    store.put_frame(_frame("CH002_S00", tick=200, order=5))
    store.put_frame(_frame("CH002_S01", tick=210, order=6))
    assert store.tick_of_chapter(2) == 200
    assert store.tick_of_chapter(9) == 0


# ═══════════════════ 2. TIMELINE ALLOCATOR ═══════════════════

def _beats(*functions: str) -> list[dict]:
    return [{"index": i, "function": f} for i, f in enumerate(functions)]


def test_timeline_alloc_flashback_freeze(store: SqliteStore):
    """TIÊU CHÍ NGÀY 3 #2 — hồi ức KHÔNG nhích con trỏ dòng chính.

    Cảnh 0 (present, 3 tick, bắt đầu ở 6) → Cảnh 1 (hồi ức về tick 100) →
    Cảnh 2 (present) phải bắt đầu ở tick 9 = 6 + 3, không bị hồi ức đẩy lệch.
    """
    beats = _beats("va chạm", "hồi ức", "thăm dò")
    times = allocate_scene_times(
        1, beats, store,
        overrides={0: {"duration_ticks": 3},
                   1: {"mode": "flashback", "epoch_tick": 100,
                       "anchor_scene": "CH001_S00", "duration_ticks": 2}})

    # Chương ĐẦU TIÊN với store rỗng bắt đầu ở `epoch_start` (mặc định 0),
    # không phải ở INTER_CHAPTER_GAP: "khoảng nghỉ giữa hai chương" không áp
    # dụng khi chưa có chương nào trước đó.
    assert times[0].epoch_tick == 0
    assert times[1].epoch_tick == 100 and times[1].mode == "flashback"
    assert times[2].epoch_tick == 3, "hồi ức đã đẩy lệch con trỏ dòng chính"


def test_timeline_alloc_narrative_order_don_dieu_qua_moi_mode(store: SqliteStore):
    """Trục ĐỌC tịnh tiến qua MỌI cảnh, kể cả hồi ức — độc giả vẫn đọc chúng
    theo thứ tự. `_narrative_monotonic` (§3.6.1) dựa vào đúng tính chất này."""
    times = allocate_scene_times(
        1, _beats("va chạm", "hồi ức", "thăm dò"), store,
        overrides={1: {"mode": "flashback", "epoch_tick": 100,
                       "anchor_scene": "CH001_S00"}})
    orders = [t.narrative_order for t in times]
    assert orders == sorted(orders) == [1, 2, 3]


def test_timeline_alloc_concurrent_lay_moc_tu_anchor(store: SqliteStore):
    """Cảnh song song lấy mốc từ cảnh neo, nên `_mode_valid` chắc chắn đi qua:
    ràng buộc thoả mãn theo THIẾT KẾ, không nhờ may mắn."""
    store.put_frame(_frame("CH001_S00", tick=240, order=1, dur=4))
    times = allocate_scene_times(
        2, _beats("thăm dò", "va chạm"), store,
        overrides={0: {"mode": "concurrent", "anchor_scene": "CH001_S00",
                       "duration_ticks": 4}})
    assert times[0].epoch_tick == 240
    assert times[0].overlaps(StoryTime(epoch_tick=240, duration_ticks=4,
                                       narrative_order=1))
    # Con trỏ dòng chính KHÔNG nhích vì cảnh 0 là concurrent
    assert times[1].epoch_tick == store.last_epoch_tick(1) + INTER_CHAPTER_GAP


def test_timeline_alloc_noi_tiep_chuong_truoc(store: SqliteStore):
    store.put_frame(_frame("CH001_S00", tick=6, order=1, dur=3))
    times = allocate_scene_times(2, _beats("nhịp thở"), store)
    assert times[0].epoch_tick == 9 + INTER_CHAPTER_GAP == 15
    assert times[0].narrative_order == 2


def test_timeline_alloc_thoi_luong_theo_chuc_nang_beat(store: SqliteStore):
    times = allocate_scene_times(1, _beats("sinh hoạt", "va chạm"), store)
    assert times[0].duration_ticks == 8      # "ba tuần lênh đênh" là một khoảng
    assert times[1].duration_ticks == 1
    assert times[1].epoch_tick == 0 + 8


def test_timeline_alloc_gap_after(store: SqliteStore):
    times = allocate_scene_times(
        1, _beats("va chạm", "va chạm"), store,
        overrides={0: {"duration_ticks": 2, "gap_after": 10}})
    assert times[1].epoch_tick == 0 + 2 + 10


def test_timeline_alloc_hoi_uc_thieu_anchor_thi_nem(store: SqliteStore):
    """Bản gốc §8.2.1 dùng `ov["anchor_scene"]` — KeyError trần, không nói gì.
    Hồi ức không neo được thì `_mode_valid` dựng blocker ở MỌI lần chạy."""
    with pytest.raises(ValueError, match="anchor_scene"):
        allocate_scene_times(1, _beats("hồi ức"), store,
                             overrides={0: {"mode": "flashback",
                                            "epoch_tick": 10}})


def test_timeline_alloc_hoi_uc_thieu_epoch_tick_thi_nem(store: SqliteStore):
    with pytest.raises(ValueError, match="epoch_tick"):
        allocate_scene_times(1, _beats("hồi ức"), store,
                             overrides={0: {"mode": "flashback",
                                            "anchor_scene": "CH001_S00"}})


def test_timeline_alloc_concurrent_anchor_la_thi_nem_keyerror(store: SqliteStore):
    with pytest.raises(KeyError):
        allocate_scene_times(1, _beats("thăm dò"), store,
                             overrides={0: {"mode": "concurrent",
                                            "anchor_scene": "CH404_S00"}})


# ═══════════════════ 3. TRUNCATE_TO ═══════════════════

def test_truncate_to_recency_bias():
    """TIÊU CHÍ NGÀY 3 #3 — 10 digest, ngân sách đủ 3 → giữ 3 cái MỚI NHẤT.

    F1: chỉ nhận thêm `list[str]` rồi gán điểm 1.0 cho tất cả thì sai theo một
    cách TỆ HƠN CẢ CRASH — `sorted` ổn định nên khi cắt, thứ bị bỏ là phần tử
    cuối, tức chương GẦN NHẤT.
    """
    items = [f"digest số {i}" for i in range(10)]
    budget = sum(ntok(t) for t in items[-3:])
    kept = truncate_to(items, budget, recency_weighted=True)
    assert set(kept) == {"digest số 7", "digest số 8", "digest số 9"}


def test_truncate_to_khong_recency_thi_giu_dau_danh_sach():
    items = [f"digest số {i}" for i in range(10)]
    budget = sum(ntok(t) for t in items[:3])
    assert truncate_to(items, budget) == ["digest số 0", "digest số 1",
                                          "digest số 2"]


def test_truncate_to_nhan_list_str_khong_ne_TypeError_F1():
    """F1 nguyên bản: `-x[0]` trên chuỗi ném
    `TypeError: bad operand type for unary -: 'str'`."""
    assert truncate_to(["một", "hai"], 999) == ["một", "hai"]


def test_truncate_to_nhan_tuple_co_diem():
    out = truncate_to([(0.2, "thấp"), (0.9, "cao")], 999)
    assert out == ["cao", "thấp"]           # điểm cao trước


def test_truncate_to_dung_continue_khong_break():
    """Một fact DÀI ở giữa không được chặn mọi fact ngắn điểm thấp hơn — mà
    fact ngắn có thể rất quan trọng."""
    dai = "từ " * 400
    out = truncate_to([(0.9, dai), (0.5, "ngắn nhưng quan trọng")], budget=50)
    assert out == ["ngắn nhưng quan trọng"]


def test_truncate_to_ngan_sach_0_tra_rong():
    assert truncate_to(["bất cứ gì"], 0) == []


def test_truncate_to_danh_sach_rong():
    assert truncate_to([], 100) == []


def test_memory_budget_hieu_chinh_tieng_viet():
    """Tiếng Việt tốn ~2,6× token so với tiếng Anh cùng nội dung. Giữ nguyên
    con số §4.1 nghĩa là mỗi tầng chỉ chứa ~40% lượng nội dung thiết kế dự
    tính — và cắt bớt thì âm thầm."""
    en = MemoryBudget()
    vi = MemoryBudget.for_vietnamese()
    assert vi.total > en.total
    assert vi.l4_facts == int(en.l4_facts * 2.6)


# ═══════════════════ 4. CONTEXT ASSEMBLER ═══════════════════

@pytest.fixture
def rig(store: SqliteStore):
    g, _chars, _meta = load_bible()
    return g, store, ContextAssembler(g, store)


def test_assembler_build_tra_known_facts_dang_dict_co_id(rig):
    """Lỗi họ hàng F1: §4.2 cho `known_facts` qua `truncate_to` (→ list[str])
    rồi §5.4.1 chạy `f.get("id")` trên từng phần tử. `str` không có `.get` →
    AttributeError ngay Cảnh 0. Hai mục cùng tài liệu thoả thuận hai kiểu dữ
    liệu khác nhau cho cùng một khoá."""
    g, _s, asm = rig
    ctx = asm.build(chapter=1, scene_idx=0, pov_id="CHAR_KAELEN",
                    present=["CHAR_KAELEN"], location_id="LOC_REACTOR_3",
                    epoch_tick=0)
    assert all(isinstance(f, dict) and "id" in f for f in ctx["known_facts"])
    # và đường ống vẫn dẹp được thành chuỗi cho prompt, ở bước CUỐI
    assert all(isinstance(s, str) for s in render_facts(ctx["known_facts"]))


def test_assembler_noi_thang_voi_firewall_khong_no(rig):
    """Đúng chuỗi gọi của `writer_node` (§9.2): build → filter_memory."""
    g, _s, asm = rig
    ctx = asm.build(chapter=1, scene_idx=0, pov_id="CHAR_KAELEN",
                    present=["CHAR_KAELEN"], location_id="LOC_REACTOR_3",
                    epoch_tick=0)
    out = POVFirewall(g).filter_memory(ctx, "CHAR_KAELEN", 0)
    assert {f["id"] for f in out["known_facts"]} <= {"LOC_REACTOR_3"}
    assert out["hypotheses"][0]["must_render_as"].startswith("phỏng đoán")


def test_assembler_dung_epoch_tick_khong_dung_chapter(rig):
    """E5/NT-6: truyền `chapter` vào chỗ cần `epoch_tick` khiến POV mù nhận
    thức hoàn toàn (`since_tick <= 1` trong khi thế giới ở tick 2400)."""
    g, _s, asm = rig
    from novel_engine.canon.models import Relation
    g.upsert_relation(Relation(src="CHAR_KAELEN", dst="CLUE_SEAL_CORROSION",
                               type="KNOWS_ABOUT", since_tick=2000))
    som = asm.build(1, 0, "CHAR_KAELEN", ["CHAR_KAELEN"], "LOC_REACTOR_3",
                    epoch_tick=1)
    muon = asm.build(1, 0, "CHAR_KAELEN", ["CHAR_KAELEN"], "LOC_REACTOR_3",
                     epoch_tick=2400)
    ids_som = {f["id"] for f in som["known_facts"]}
    ids_muon = {f["id"] for f in muon["known_facts"]}
    assert "CLUE_SEAL_CORROSION" not in ids_som
    assert "CLUE_SEAL_CORROSION" in ids_muon


def test_assembler_uu_tien_fact_ve_nhan_vat_co_mat(rig):
    g, _s, asm = rig
    facts = ContextAssembler._score_facts(
        {"known": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
         "suspected": []}, [], present=["A"])
    diem = {f["id"]: f["score"] for f in facts}
    assert diem["A"] > diem["B"]


def test_assembler_ton_trong_ngan_sach_l4(rig):
    """Ngân sách là CỨNG. Nới ngân sách là con đường quay lại vấn đề ban đầu."""
    g, s, _asm = rig
    from novel_engine.canon.models import Entity as E, Relation
    for i in range(200):
        g.upsert_entity(E(id=f"F_{i}", kind="event", name=f"sự kiện dài dòng {i}"))
        g.upsert_relation(Relation(src="CHAR_KAELEN", dst=f"F_{i}",
                                   type="KNOWS_ABOUT", since_tick=0))
    asm = ContextAssembler(g, s, MemoryBudget(l4_facts=60))
    ctx = asm.build(1, 0, "CHAR_KAELEN", [], "LOC_REACTOR_3", epoch_tick=10)
    assert sum(ntok(t) for t in render_facts(ctx["known_facts"])) <= 60
    assert len(ctx["known_facts"]) < 200


def test_assembler_l1_l3_cat_cai_cu_truoc(rig):
    g, s, _asm = rig
    for si in range(8):
        s.put_scene_digest(1, si, f"cảnh {si} " + "chi tiết " * 20)
    asm = ContextAssembler(g, s, MemoryBudget(l1_recent_scenes=40))
    ctx = asm.build(1, 8, "CHAR_KAELEN", [], "LOC_REACTOR_3", epoch_tick=0)
    assert ctx["recent_scenes"] == [] or "cảnh 7" in ctx["recent_scenes"][0]


# ═══════════════════ 5. POV FIREWALL ═══════════════════

CONTRACT = {
    "pov_character": "CHAR_KAELEN",
    "scene_id": "CH001_S00",
    "active_characters": [
        {"id": "CHAR_KAELEN", "name": "Kaelen", "immediate_goal": "vào Lò 3",
         "secret_fear": "sợ nhớ ra mình đã ký", "internal_conflict": "…",
         "deliberation": {"why": "-0.80 flaw: tự kết tội"},
         "must_not_reveal": ["chính anh mở khoang Lõi"]},
        {"id": "CHAR_SERENA", "name": "Serena", "immediate_goal": "giữ anh lại",
         "secret_fear": "sợ bị lộ chữ ký thứ ba",
         "internal_conflict": "yêu anh nhưng phải chặn anh",
         "deliberation": {"why": "-1.30 flaw: bảo vệ bằng cách giấu"},
         "hidden_action": "gắn thiết bị theo dõi vào ba lô của Kaelen",
         "observable_behavior": "đứng hơi lâu cạnh ba lô",
         "must_not_reveal": ["chữ ký thứ ba"],
         "knows_in_this_scene": ["CLUE_THIRD_SIGNATURE"],
         "voice_reminder": {"register": "formal"},
         "somatic_allowed": ["xoay chiếc nhẫn dấu"]},
    ],
}


def test_firewall_strips_npc_deliberation():
    """TIÊU CHÍ NGÀY 3 #4 — POV là Kaelen → nội tâm Serena phải biến mất."""
    g, _c, _m = load_bible()
    out = POVFirewall(g).filter_scene_contract(CONTRACT, "CHAR_KAELEN")
    serena = next(c for c in out["active_characters"] if c["id"] == "CHAR_SERENA")

    for k in ("secret_fear", "internal_conflict", "deliberation",
              "must_not_reveal", "knows_in_this_scene"):
        assert k not in serena, f"rò rỉ nội tâm: {k}"

    kaelen = next(c for c in out["active_characters"] if c["id"] == "CHAR_KAELEN")
    assert kaelen["deliberation"]["why"].startswith("-0.80")   # POV giữ nguyên
    assert kaelen["secret_fear"]


def test_firewall_giu_hidden_action_trong_director_only():
    """`hidden_action` VẪN phải vào prompt. Xoá nó và thay bằng "có hành vi
    đáng ngờ" nghe an toàn hơn nhưng làm hỏng chính cơ chế nó bảo vệ — Writer
    không gieo được dấu vết CỤ THỂ của một hành động nó không biết là gì."""
    g, _c, _m = load_bible()
    out = POVFirewall(g).filter_scene_contract(CONTRACT, "CHAR_KAELEN")
    serena = next(c for c in out["active_characters"] if c["id"] == "CHAR_SERENA")
    assert serena["director_only"]["hidden_action"].startswith("gắn thiết bị")
    assert "KHÔNG biết" in serena["director_only"]["_hard_constraint"]
    assert serena["observable_behavior"] == "đứng hơi lâu cạnh ba lô"


def test_firewall_giu_truong_an_toan():
    g, _c, _m = load_bible()
    out = POVFirewall(g).filter_scene_contract(CONTRACT, "CHAR_KAELEN")
    serena = next(c for c in out["active_characters"] if c["id"] == "CHAR_SERENA")
    assert serena["name"] == "Serena"
    assert serena["immediate_goal"] == "giữ anh lại"
    assert serena["voice_reminder"]["register"] == "formal"
    assert serena["somatic_allowed"] == ["xoay chiếc nhẫn dấu"]


def test_firewall_khong_sua_contract_goc():
    """Hàm lọc phải thuần — sửa tại chỗ thì lượt viết lại (revise) sẽ nhận
    contract đã bị lột, và mỗi vòng lặp lột thêm một ít."""
    g, _c, _m = load_bible()
    POVFirewall(g).filter_scene_contract(CONTRACT, "CHAR_KAELEN")
    serena_goc = CONTRACT["active_characters"][1]
    assert "secret_fear" in serena_goc and "deliberation" in serena_goc


def test_firewall_doi_pov_thi_doi_ben_bi_lot():
    """Cùng contract, POV là Serena → giờ Kaelen mới là người bị lột."""
    g, _c, _m = load_bible()
    out = POVFirewall(g).filter_scene_contract(CONTRACT, "CHAR_SERENA")
    kaelen = next(c for c in out["active_characters"] if c["id"] == "CHAR_KAELEN")
    serena = next(c for c in out["active_characters"] if c["id"] == "CHAR_SERENA")
    assert "deliberation" not in kaelen and "secret_fear" not in kaelen
    assert "deliberation" in serena


def test_pov_leak_regex():
    """TIÊU CHÍ NGÀY 3 #5."""
    hits = pov_leak_scan("Anh không biết rằng phía sau cánh cửa là bẫy.",
                         pov_name="Kaelen")
    assert hits and hits[0]["severity"] == "blocker"
    assert hits[0]["check"] == "pov_leak"


def test_pov_leak_cho_phep_pov_tu_nhan_khong_biet():
    """POV tự nhận mình không biết là hợp lệ — đó là giới hạn nhận thức, không
    phải rò rỉ."""
    assert pov_leak_scan("Kaelen không biết rằng mình đang bị theo dõi.",
                         pov_name="Kaelen") == []


@pytest.mark.parametrize("prose, ro_ri", [
    # Mẫu gốc §5.4 đòi dấu cách ngay sau "thâm tâm" nên trượt đúng cách viết
    # phổ biến nhất — dấu phẩy sau trạng ngữ đầu câu là chuẩn tiếng Việt.
    ("Trong thâm tâm, Serena đã quyết định từ lâu.", True),
    ("Trong thâm tâm Serena vẫn còn ngờ vực.", True),
    ("Trong lòng: một thứ gì đó vỡ ra.", True),
    # …nhưng nới lỏng quá tay thì "của mình" hết được miễn trừ:
    ("Kaelen giấu tấm thẻ trong lòng của mình.", False),
    ("Trong đầu của tôi mọi thứ rối tung.", False),
    ("Sự thật là Serena đang che giấu điều gì đó.", True),
])
def test_pov_leak_bat_noi_tam_nguoi_khac(prose: str, ro_ri: bool):
    assert bool(pov_leak_scan(prose, pov_name="Kaelen")) is ro_ri


def test_pov_leak_bo_qua_van_xuoi_sach():
    prose = ("Kaelen ấn ngón cái vào khớp ngón trỏ. Cần trục kêu ba tiếng rồi "
             "im. Serena đứng hơi lâu cạnh ba lô, rồi bước đi.")
    assert pov_leak_scan(prose, pov_name="Kaelen") == []


def test_firewall_boundary_liet_ke_dieu_pov_chua_biet():
    g, _c, _m = load_bible()
    fw = POVFirewall(g)
    boundary = fw.boundary("CHAR_KAELEN", epoch_tick=0)
    assert boundary, "POV phải có ít nhất vài điều chưa biết ở chương 1"
    known = set(fw.known_ids("CHAR_KAELEN", 0))
    assert not ({e["name"] for e in g.entity_index() if e["id"] in known}
                & set(boundary))


# ═══════════════════ THỜI GIAN ĐI ĐƯỜNG (bổ sung so với §8.2.1) ═══════════

def test_alloc_tu_cong_thoi_gian_di_duong(store: SqliteStore):
    """§8.2.1 cấp phát thời gian mà KHÔNG biết địa lý, rồi §3.6.1 kiểm thời
    gian ĐỐI CHIẾU địa lý. Không có gì nối hai cơ chế → mọi cảnh đổi địa điểm
    sinh blocker giả. Lượt chạy thật đầu tiên dựng đúng cờ đó.

    P2 (§0.1): báo động giả là loại lỗi nguy hiểm nhất, vì nó khiến người dùng
    tắt luật — tắt xong thì mất luôn phần bảo vệ đúng đắn.
    """
    g, _c, _m = load_bible()
    beats = _beats("va chạm", "va chạm")
    locs = ["LOC_ORE_PORT", "LOC_VEDA_CHECKPOINT"]      # cần 4 tick đi đường

    khong_dia_ly = allocate_scene_times(1, beats, store)
    assert khong_dia_ly[1].epoch_tick == khong_dia_ly[0].end_tick   # gap 0

    co_dia_ly = allocate_scene_times(1, beats, store, locations=locs, graph=g)
    gap = co_dia_ly[1].epoch_tick - co_dia_ly[0].end_tick
    assert gap == g.travel_ticks(*locs) == 4


def test_alloc_khong_cong_khi_o_yen_mot_cho(store: SqliteStore):
    g, _c, _m = load_bible()
    locs = ["LOC_VEDA_CHECKPOINT"] * 3
    t = allocate_scene_times(1, _beats("va chạm", "va chạm", "va chạm"), store,
                             locations=locs, graph=g)
    assert all(b.epoch_tick == a.end_tick for a, b in zip(t, t[1:]))


def test_alloc_hoi_uc_khong_bi_cong_thoi_gian_di_duong(store: SqliteStore):
    """Hồi ức neo vào quá khứ — không có "đi từ đâu tới"."""
    g, _c, _m = load_bible()
    t = allocate_scene_times(
        1, _beats("va chạm", "hồi ức", "va chạm"), store,
        overrides={1: {"mode": "flashback", "epoch_tick": 100,
                       "anchor_scene": "CH001_S00"}},
        locations=["LOC_ORE_PORT", "LOC_REACTOR_3", "LOC_ORE_PORT"], graph=g)
    assert t[1].epoch_tick == 100
    assert t[2].epoch_tick == t[0].end_tick      # hồi ức không đổi địa điểm


def test_alloc_khong_co_graph_thi_giu_hanh_vi_cu(store: SqliteStore):
    """Tham số mới phải tuỳ chọn — mọi chỗ gọi cũ vẫn chạy."""
    t = allocate_scene_times(1, _beats("va chạm", "va chạm"), store)
    assert t[1].epoch_tick == t[0].end_tick


def test_epoch_start_cho_hoi_uc_co_cho_trong_qua_khu(store: SqliteStore):
    """Hồi ức cần chỗ TRƯỚC mốc neo. Nếu hiện tại bắt đầu ở tick 0 thì sự cố
    "hai năm trước" (17.520 tick) phải là số ÂM — `StoryTime` không cấm, nhưng
    `_mode_valid` sẽ báo "hồi ức kết thúc SAU mốc neo" cho một hồi ức hoàn
    toàn hợp lệ, và mọi log đọc như lỗi."""
    t = allocate_scene_times(1, _beats("va chạm"), store, epoch_start=20_000)
    assert t[0].epoch_tick == 20_000


def test_epoch_start_chi_ap_cho_chuong_dau(store: SqliteStore):
    """Từ chương 2 trở đi, con trỏ nối tiếp chương trước — `epoch_start` bị bỏ
    qua, nếu không mỗi chương lại nhảy về gốc."""
    store.put_frame(_frame("CH001_S00", tick=20_000, order=1, dur=3))
    t = allocate_scene_times(2, _beats("va chạm"), store, epoch_start=20_000)
    assert t[0].epoch_tick == 20_003 + INTER_CHAPTER_GAP
