"""Ngày 2 — NetworkXGraph + Unified Route Graph.

Test nặng nhất ở đây là `test_mot_nguon_dia_ly_*`: chúng không kiểm một hàm mà
kiểm BẤT BIẾN kiến trúc — sửa một cạnh địa lý phải làm đổi CẢ HAI phép chiếu.
Đó là thứ duy nhất chứng minh đồ thị thật sự thống nhất chứ không phải hai
bảng tình cờ đang khớp nhau.
"""
from __future__ import annotations

import pytest

from novel_engine.canon.bible import load_bible
from novel_engine.canon.graph_port import GraphPort
from novel_engine.canon.models import (
    Assertion, Clue, ClueStatus, Entity, Relation,
)
from novel_engine.canon.networkx_graph import SPEED_MULTIPLIERS, NetworkXGraph

PORT = "LOC_ORE_PORT"
VEDA = "LOC_VEDA_CHECKPOINT"
CATH = "LOC_RUINED_CATHEDRAL"
RE3 = "LOC_REACTOR_3"


@pytest.fixture
def g() -> NetworkXGraph:
    graph, _chars, _meta = load_bible()
    return graph


# ═══════════════════ UNIFIED ROUTE GRAPH — bất biến kiến trúc ═══════════════

def test_mot_nguon_dia_ly_doi_canh_thi_ca_hai_phep_chieu_doi(g: NetworkXGraph):
    """Bất biến trung tâm: base_ticks là MỘT nguồn sự thật.

    Tách hai bảng khoảng cách thì chúng lệch nhau ở chương thứ ba và không ai
    biết. Test này sẽ đỏ ngay khi ai đó thêm một bảng latency riêng.
    """
    before_travel = g.travel_ticks(PORT, VEDA)
    before_news = g.routes_from(PORT, channel="rumor")[0]["latency_ticks"]
    assert before_travel == before_news == 4

    g.routes.edges[PORT, VEDA]["base_ticks"] = 40      # đường bị lở, đi vòng

    assert g.travel_ticks(PORT, VEDA) != before_travel
    assert [e for e in g.routes_from(PORT) if e["to"] == VEDA
            ][0]["latency_ticks"] == 40


def test_mot_nguon_dia_ly_chan_duong_anh_huong_ca_hai(g: NetworkXGraph):
    """`blocked_by` phải hiện diện ở CẢ hai phép chiếu — di chuyển thì bị
    Dijkstra bỏ cạnh, truyền tin thì `propagate` (§5.6.3) tự đọc `blocked_by`
    từ chính dict mà `routes_from` trả về."""
    assert g.travel_ticks(PORT, CATH, hostile_to=("FACT_CIPHER",)) is None
    edge = [e for e in g.routes_from(VEDA) if e["to"] == CATH][0]
    assert edge["blocked_by"] == ["FACT_CIPHER"]


# ═══════════════════ PHÉP CHIẾU 1: travel_ticks ═══════════════════

def test_travel_ticks_chon_duong_re_nhat_khong_phai_duong_truc_tiep(g):
    """PORT→RE3 có cạnh biển trực tiếp 14 tick, nhưng đi vòng qua chốt chỉ
    hết 4+9=13. Dijkstra phải chọn 13 — nếu trả 14 thì nó chỉ đang tra cạnh
    trực tiếp chứ không tìm đường."""
    assert g.travel_ticks(PORT, RE3) == 13


def test_travel_ticks_theo_phuong_tien(g: NetworkXGraph):
    """cost = base_ticks × speed_multiplier, làm tròn LÊN một lần ở cuối."""
    assert g.travel_ticks(PORT, RE3, mode="foot") == 13          # 13 × 1.0
    assert g.travel_ticks(PORT, RE3, mode="mecha") == 8          # 13 × 0.6 = 7.8
    assert g.travel_ticks(PORT, RE3, mode="armored") == 6        # 13 × 0.4 = 5.2
    assert SPEED_MULTIPLIERS["armored"] < SPEED_MULTIPLIERS["foot"]


def test_travel_ticks_phuong_tien_nhanh_hon_thi_doi_ca_tuyen_duong(g):
    """Ở `armored`, đường biển (14×0.4=5.6) vẫn đắt hơn đường bộ (13×0.4=5.2).
    Test này chốt rằng hệ số được áp TRƯỚC khi chọn đường, không phải sau —
    áp sau thì mọi phương tiện đều đi cùng một tuyến, và cơ chế mất ý nghĩa."""
    assert g.travel_ticks(PORT, RE3, mode="armored") == 6


def test_travel_ticks_khong_co_tuyen_tra_none(g: NetworkXGraph):
    """`_no_teleport_epoch` (§3.6.1) phân biệt "đi quá nhanh" với "không có
    đường" bằng đúng giá trị None này."""
    g.upsert_entity(Entity(id="LOC_ISLAND", kind="location", name="Đảo"))
    assert g.travel_ticks(PORT, "LOC_ISLAND") is None
    assert g.travel_ticks(PORT, "LOC_KHONG_TON_TAI") is None


def test_travel_ticks_cung_cho_bang_khong(g: NetworkXGraph):
    assert g.travel_ticks(PORT, PORT) == 0


def test_travel_ticks_doi_xung_khi_duong_hai_chieu(g: NetworkXGraph):
    assert g.travel_ticks(PORT, RE3) == g.travel_ticks(RE3, PORT)


def test_route_mot_chieu_chi_them_mot_huong():
    g = NetworkXGraph()
    for i in ("A", "B"):
        g.upsert_entity(Entity(id=i, kind="location", name=i))
    g.add_route("A", "B", base_ticks=3, terrain="sea", one_way=True)
    assert g.travel_ticks("A", "B") == 3
    assert g.travel_ticks("B", "A") is None


def test_chan_duong_khong_anh_huong_phe_khac(g: NetworkXGraph):
    """Chỉ phe THÙ ĐỊCH mới bị chặn. Người của Mật Văn vẫn qua chốt của họ."""
    assert g.travel_ticks(PORT, CATH, hostile_to=("FACT_ARCLIGHT",)) == 7
    assert g.travel_ticks(PORT, CATH) == 7


# ═══════════════════ PHÉP CHIẾU 2: routes_from ═══════════════════

def test_routes_from_tra_latency_THO_chua_nhan_he_so(g: NetworkXGraph):
    """§5.6.3 tự nhân `ch.latency_multiplier`. Nhân sẵn ở đây là nhân hai lần
    — một lỗi âm thầm: tin vẫn tới nơi, chỉ là tới sai lúc mãi mãi."""
    e = [x for x in g.routes_from(VEDA) if x["to"] == RE3][0]
    assert e["latency_ticks"] == g.routes.edges[VEDA, RE3]["base_ticks"] == 9


def test_routes_from_loc_theo_allowed_channels(g: NetworkXGraph):
    """`signal` chỉ đi được trong địa phận thánh đường; `rumor` không."""
    dests = {e["to"] for e in g.routes_from(VEDA, channel="signal")}
    assert dests == {CATH}
    dests_rumor = {e["to"] for e in g.routes_from(VEDA, channel="rumor")}
    assert dests_rumor == {PORT, RE3}
    assert CATH not in dests_rumor


def test_routes_from_khong_loc_khi_khong_truyen_channel(g: NetworkXGraph):
    assert len(g.routes_from(VEDA)) == 3


def test_routes_from_dia_diem_la_tra_rong(g: NetworkXGraph):
    assert g.routes_from("LOC_KHONG_TON_TAI") == []


def test_tin_don_vong_duong_bien_khi_chot_bi_chan(g: NetworkXGraph):
    """Hệ quả tự sự của đồ thị thống nhất: phong toả chốt chặn được NGƯỜI
    nhưng không chặn được TIN ĐỒN, vì tin đồn có lối biển. Đây chính là loại
    bất đối xứng mà hai bảng rời nhau sẽ không bao giờ tạo ra nhất quán."""
    assert g.travel_ticks(PORT, CATH, hostile_to=("FACT_CIPHER",)) is None
    assert {e["to"] for e in g.routes_from(PORT, channel="rumor")} == {VEDA, RE3}


# ═══════════════════ POV FIREWALL — NT-6 ═══════════════════

def test_known_by_loc_theo_epoch_tick_khong_theo_chuong(g: NetworkXGraph):
    """E5/NT-6: truyền `chapter` vào chỗ cần `epoch_tick` khiến POV mù nhận
    thức hoàn toàn (`since_tick <= 14` trong khi thế giới ở tick 2400)."""
    g.upsert_relation(Relation(src="CHAR_KAELEN", dst="CLUE_SEAL_CORROSION",
                               type="KNOWS_ABOUT", since_tick=500))
    assert not any(f["id"] == "CLUE_SEAL_CORROSION"
                   for f in g.known_by("CHAR_KAELEN", 499)["known"])
    assert any(f["id"] == "CLUE_SEAL_CORROSION"
               for f in g.known_by("CHAR_KAELEN", 500)["known"])


def test_known_by_tach_biet_va_nghi(g: NetworkXGraph):
    out = g.known_by("CHAR_KAELEN", 0)
    assert [f["id"] for f in out["known"]] == [RE3]
    assert out["suspected"][0]["id"] == "FACT_CIPHER"
    assert out["suspected"][0]["conf"] == 0.45      # để Writer ghi là phỏng đoán


def test_known_by_nhan_vat_la_tra_rong(g: NetworkXGraph):
    out = g.known_by("CHAR_KHONG_TON_TAI", 9999)
    assert out == {"known": [], "suspected": [], "clues_held": []}


def test_known_by_clues_held_theo_understood_by(g: NetworkXGraph):
    g.clues["CLUE_SEAL_CORROSION"].understood_by_characters.append("CHAR_SERENA")
    held = g.known_by("CHAR_SERENA", 0)["clues_held"]
    assert held[0]["id"] == "CLUE_SEAL_CORROSION"
    # `form` là surface_form, KHÔNG phải description (§3.3)
    assert "axit" not in held[0]["form"]


def test_pov_can_observe_ton_trong_prerequisites(g: NetworkXGraph):
    """DAG manh mối: không thể cài chữ ký thứ ba trước khi hai manh mối nền
    đã được cài."""
    assert g.pov_can_observe("CHAR_KAELEN", "CLUE_THIRD_SIGNATURE", 1) is False
    for cid in ("CLUE_SEAL_CORROSION", "CLUE_MISSING_LOGS"):
        g.clues[cid].status = ClueStatus.PLANTED
    assert g.pov_can_observe("CHAR_KAELEN", "CLUE_THIRD_SIGNATURE", 1) is True


def test_pov_can_observe_bo_manh_moi_da_dong(g: NetworkXGraph):
    g.clues["CLUE_SEAL_CORROSION"].status = ClueStatus.PAID_OFF
    assert g.pov_can_observe("CHAR_KAELEN", "CLUE_SEAL_CORROSION", 3) is False


def test_pov_can_observe_bo_nguoi_da_hieu(g: NetworkXGraph):
    g.clues["CLUE_SEAL_CORROSION"].understood_by_characters.append("CHAR_KAELEN")
    assert g.pov_can_observe("CHAR_KAELEN", "CLUE_SEAL_CORROSION", 2) is False
    assert g.pov_can_observe("CHAR_SERENA", "CLUE_SEAL_CORROSION", 2) is True


# ═══════════════════ NIỀM TIN vs SỰ THẬT — F6 ═══════════════════

def test_commit_belief_khong_lam_ban_graph(g: NetworkXGraph):
    """F6/NT-15: lời nói dối của Serena phải vào hồ sơ NGƯỜI NÓI, không vào
    sự thật khách quan. Hai bản vá đúng (v2.3 + v2.4) hợp lại tái tạo đúng
    cái lỗi mà `epistemic` sinh ra để chặn."""
    n_before = g.g.number_of_edges()
    g.commit_belief("CHAR_SERENA", "FLEET_3", "status", "disbanded",
                    kind="claimed_by", since_tick=120)
    assert g.g.number_of_edges() == n_before        # graph KHÔNG đổi
    assert g.beliefs[-1]["holder"] == "CHAR_SERENA"
    assert ("FLEET_3", "status") not in g._truth


def test_conflicts_with_truth_gieo_irony_khong_chan(g: NetworkXGraph):
    g.commit_truth("FLEET_3", "status", "active", tick=0)
    noi_doi = Assertion(subject="FLEET_3", predicate="status", object="disbanded",
                        chapter=4, scene=1, span="Hạm Đội Số 3 đã bị giải tán.",
                        confidence=0.8, epistemic="claimed_by",
                        holder="CHAR_SERENA")
    that = Assertion(subject="FLEET_3", predicate="status", object="active",
                     chapter=4, scene=1, span="Hạm đội vẫn neo ngoài vịnh.",
                     confidence=0.9)
    assert g.conflicts_with_truth(noi_doi) is True
    assert g.conflicts_with_truth(that) is False


def test_conflicts_with_truth_chua_biet_thi_khong_mau_thuan(g: NetworkXGraph):
    a = Assertion(subject="X", predicate="y", object=1, chapter=1, scene=0,
                  span="s", confidence=0.5)
    assert g.conflicts_with_truth(a) is False


# ═══════════════════ ĐÓNG QUAN HỆ — F7 ═══════════════════

def test_close_relation_dat_until_chapter(g: NetworkXGraph):
    """F7: không có hàm này thì quan hệ cũ không bao giờ đóng."""
    g.upsert_relation(Relation(src="CHAR_VHAL", dst="FACT_CIPHER",
                               type="MEMBER_OF"))
    assert g.g.edges["CHAR_VHAL", "FACT_CIPHER", "MEMBER_OF"]["until_chapter"] is None
    g.close_relation("CHAR_VHAL", "FACT_CIPHER", "MEMBER_OF", until_chapter=4)
    assert g.g.edges["CHAR_VHAL", "FACT_CIPHER", "MEMBER_OF"]["until_chapter"] == 4


def test_close_relation_khong_ton_tai_thi_im_lang(g: NetworkXGraph):
    g.close_relation("A", "B", "PROTECTS", until_chapter=2)       # không ném


def test_upsert_relation_la_upsert_that_khong_chong_ban_sao(g: NetworkXGraph):
    """Khoá cạnh theo `type`: chương sau nhắc lại quan hệ cũ không được tạo
    cạnh song song, nếu không `faction_tensions` đếm trùng."""
    n = g.g.number_of_edges()
    for w in (0.5, 0.6, 0.7):
        g.upsert_relation(Relation(src="CHAR_VHAL", dst="FACT_ARCLIGHT",
                                   type="OWES_DEBT_TO", weight=w))
    assert g.g.number_of_edges() == n + 1
    assert g.g.edges["CHAR_VHAL", "FACT_ARCLIGHT", "OWES_DEBT_TO"]["weight"] == 0.7


# ═══════════════════ CANON & KẾ HOẠCH ═══════════════════

def test_faction_tensions_tim_phe_kiem_soat_dia_diem(g: NetworkXGraph):
    out = g.faction_tensions(RE3, chapter=1)
    assert [t["faction_id"] for t in out] == ["FACT_ARCLIGHT"]
    assert out[0]["feuds"][0]["with"] == "Tổng Cục Mật Văn"


def test_faction_tensions_ton_trong_thoi_hieu(g: NetworkXGraph):
    g.g.edges["FACT_ARCLIGHT", "FACT_CIPHER", "HAS_FEUD_WITH"]["until_chapter"] = 2
    assert g.faction_tensions(RE3, chapter=1)[0]["feuds"] != []
    assert g.faction_tensions(RE3, chapter=2)[0]["feuds"] == []   # đã đóng


def test_faction_tensions_dia_diem_la_tra_rong(g: NetworkXGraph):
    assert g.faction_tensions("LOC_KHONG_TON_TAI", 1) == []


def test_due_clues_sap_theo_slack(g: NetworkXGraph):
    for cid in ("CLUE_SEAL_CORROSION", "CLUE_MISSING_LOGS"):
        g.clues[cid].status = ClueStatus.PLANTED
    out = g.due_clues(chapter=3, lookahead=3)
    ids = [c["clue_id"] for c in out]
    assert ids == ["CLUE_SEAL_CORROSION", "CLUE_MISSING_LOGS"]   # slack 2 < 3
    assert out[0]["slack"] == 2


def test_due_clues_bo_manh_moi_chua_cai(g: NetworkXGraph):
    """DRAFTED chưa lên trang giấy thì chưa phải nợ."""
    assert g.due_clues(chapter=5, lookahead=5) == []


def test_is_locked_theo_thuc_the_va_provenance(g: NetworkXGraph):
    assert g.is_locked("CHAR_SERENA", "FACT_CIPHER", "MEMBER_OF") is True
    g.upsert_entity(Entity(id="E_FREE", kind="object", name="tự do"))
    g.upsert_entity(Entity(id="E_FREE2", kind="object", name="tự do 2"))
    assert g.is_locked("E_FREE", "E_FREE2", "DEPENDS_ON") is False


def test_conflicting_relations_phat_hien_loai_tru(g: NetworkXGraph):
    g.upsert_relation(Relation(src="CHAR_SERENA", dst="CHAR_KAELEN",
                               type="PROTECTS"))
    out = g.conflicting_relations("CHAR_SERENA", "CHAR_KAELEN", "BETRAYED")
    assert out and out[0]["existing"] == "PROTECTS"
    g.close_relation("CHAR_SERENA", "CHAR_KAELEN", "PROTECTS", until_chapter=3)
    assert g.conflicting_relations("CHAR_SERENA", "CHAR_KAELEN", "BETRAYED") == []


def test_entity_index_bo_qua_node_tran(g: NetworkXGraph):
    """Node do `upsert_relation` tự tạo không có `kind` — phải bị loại, nếu
    không Extractor nhận một danh mục lẫn rác."""
    g.upsert_relation(Relation(src="CHAR_KAELEN", dst="GHOST_NODE",
                               type="KNOWS_ABOUT"))
    idx = g.entity_index()
    assert "GHOST_NODE" not in {e["id"] for e in idx}
    assert {"CHAR_KAELEN", "FACT_CIPHER", RE3} <= {e["id"] for e in idx}


# ═══════════════════ FLASHBACK AUDIT — F8 ═══════════════════

def test_alive_after_mac_dinh_con_song(g: NetworkXGraph):
    assert g.alive_after("CHAR_KAELEN", 9999) is True
    g.commit_truth("CHAR_VHAL", "is_dead", True, tick=3000)
    assert g.alive_after("CHAR_VHAL", 2999) is True
    assert g.alive_after("CHAR_VHAL", 3000) is False


def test_exists(g: NetworkXGraph):
    assert g.exists("CHAR_KAELEN") and not g.exists("CHAR_AI_DO")


def test_attribute_window_tra_ban_sao_khong_lam_ban_canon(g: NetworkXGraph):
    """`flashback_admissible` append vào `attested_present` để THỬ giả thuyết.
    Trả bản gốc thì mỗi lần kiểm tra lại ghi chính cái tick đang bị nghi ngờ
    vào canon — và lần kiểm thứ hai sẽ thấy "nhất quán" một cách giả tạo."""
    g.record_attestation("CHAR_KAELEN", "has_scar", 5000, present=True)
    w = g.attribute_window("CHAR_KAELEN", "has_scar")
    w.attested_present.append(500)
    assert g.attribute_window("CHAR_KAELEN", "has_scar").attested_present == [5000]


def test_attribute_window_phat_hien_mau_thuan_nhan_qua(g: NetworkXGraph):
    g.record_attestation("CHAR_KAELEN", "has_scar", 500, present=False)
    g.record_attestation("CHAR_KAELEN", "has_scar", 5000, present=True)
    w = g.attribute_window("CHAR_KAELEN", "has_scar")
    assert w.onset_bounds == (500, 5000) and w.consistent is True
    w.attested_present.append(100)          # hồi ức nói đã có sẹo từ tick 100
    assert w.consistent is False


# ═══════════════════ GraphPort ═══════════════════

def test_networkx_graph_thoa_man_graph_port(g: NetworkXGraph):
    """NT-12: mỗi method trong Protocol phải có chủ sở hữu thật sự cài đặt."""
    assert isinstance(g, GraphPort)
    for m in ("upsert_entity", "upsert_relation", "close_relation",
              "commit_belief", "known_by", "exists", "alive_after",
              "attribute_window", "is_locked", "faction_tensions",
              "due_clues", "conflicts_with_truth", "entity_index",
              "pov_can_observe", "travel_ticks", "routes_from"):
        assert callable(getattr(g, m)), m


# ═══════════════════ BIBLE LOADER ═══════════════════

def test_bible_nap_dung_so_luong(g: NetworkXGraph):
    kinds: dict[str, int] = {}
    for _n, d in g.g.nodes(data=True):
        kinds[d.get("kind")] = kinds.get(d.get("kind"), 0) + 1
    assert kinds["character"] == 3
    assert kinds["faction"] == 2
    assert kinds["location"] == 4
    assert kinds["clue"] == 3


def test_bible_moi_node_deu_co_kind_va_name(g: NetworkXGraph):
    """Thứ tự nạp: thực thể TRƯỚC quan hệ. Nạp ngược thì `upsert_relation` tự
    tạo node trần và `faction_tensions` im lặng trả rỗng."""
    for n, d in g.g.nodes(data=True):
        assert d.get("kind"), f"node trần: {n}"
        assert d.get("name"), f"node thiếu name: {n}"


def test_bible_nhan_vat_hop_le():
    _g, chars, meta = load_bible()
    assert set(chars) == {"CHAR_KAELEN", "CHAR_SERENA", "CHAR_VHAL"}
    k = chars["CHAR_KAELEN"]
    assert k.voice.register == "clipped"
    assert k.fatal_flaw.arc_direction == "transmutes"
    assert {d.layer for d in k.desires} == {"want", "need"}
    assert meta["total_chapters"] == 5


def test_bible_quan_he_cu_cua_kaelen_da_dong(g: NetworkXGraph):
    """Kaelen TỪNG thuộc Lõi Rạng. Event sourcing giữ lại cạnh đã đóng thay vì
    xoá — không có nó thì không trả lời được "ở chương 0 anh ta là ai"."""
    e = g.g.edges["CHAR_KAELEN", "FACT_ARCLIGHT", "MEMBER_OF"]
    assert e["until_chapter"] == 0


def test_bible_clue_dag_khong_co_prereq_treo(g: NetworkXGraph):
    for c in g.clues.values():
        for pre in c.prerequisites:
            assert pre in g.clues, f"{c.clue_id} phụ thuộc manh mối không tồn tại: {pre}"


def test_bible_clue_deadline_hop_le(g: NetworkXGraph):
    for c in g.clues.values():
        assert c.payoff_deadline >= c.payoff_threshold
        assert len(c.surface_forms) >= 3, f"{c.clue_id}: cần 3–5 surface_form"


def test_bible_moi_dia_diem_deu_noi_duoc_voi_nhau(g: NetworkXGraph):
    """Không có địa điểm cô lập — một địa điểm không tới được là một địa điểm
    không dùng được, và lỗi chỉ lộ ra khi Director xếp cảnh vào đó."""
    locs = [n for n, d in g.g.nodes(data=True) if d.get("kind") == "location"]
    for a in locs:
        for b in locs:
            assert g.travel_ticks(a, b) is not None, f"{a} không tới được {b}"


def test_bible_bat_ky_manh_moi_nao_cung_khong_lo_description(g: NetworkXGraph):
    """§3.3 — `description` không bao giờ vào prompt. `surface_forms` phải là
    mô tả HIỆN TƯỢNG, không phải mô tả Ý NGHĨA."""
    for c in g.clues.values():
        for form in c.surface_forms:
            assert "vì" not in form and "nghĩa là" not in form, \
                f"{c.clue_id}: surface_form đang giải thích thay vì mô tả: {form}"
