"""Ngày 6 — Diff-Driven Extraction ba lượt (§10.5).

`extractor_node` là cầu nối DUY NHẤT đưa văn xuôi trở lại Canon, và là điểm
nghẽn nguy hiểm nhất của kiến trúc event-sourced: lỗi ở đây không crash,
không cảnh báo — nó chỉ âm thầm ghi sai, và hai mươi chương sau mới thấy hậu
quả mà không truy được nguồn.

Ba lượt chống ba lỗi KHÁC NHAU, và test dưới đây soi từng lượt một:

    lượt 1  prose + contract       chống SÓT mục có kế hoạch
    lượt 2  prose, KHÔNG contract  chống SÓT mục ngoài kế hoạch
    lượt 3  khớp chuỗi             chống BỊA (ảo giác xác nhận)
"""
from __future__ import annotations

import json
import unicodedata

import pytest

from novel_engine.canon.models import (
    Assertion, ClueStatus, Entity, PlantEvidence, Relation, StateDelta,
)
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM, ScriptedLLM
from novel_engine.reconcile.verify import (
    MIN_SPAN_LEN, _fuzzy_present, _norm, merge_extractions, plan_coverage,
    verify_spans,
)
from novel_engine.relationship.models import RelationshipState

PROSE = (
    "Cần trục kêu ba tiếng rồi im. Kaelen ấn ngón cái vào khớp ngón trỏ cho "
    "tới khi nghe tiếng kêu khô khốc. Mép con dấu sần lên như vỏ cam để lâu, "
    "và không ai trong phòng nhìn về phía đó.\n\n"
    "— Theo thẩm quyền, tôi e rằng hồ sơ cho thấy điều khác, — Serena nói."
)


def _a(span: str, subject: str = "CHAR_KAELEN", **kw) -> Assertion:
    base = dict(subject=subject, predicate="observed", object=True,
                chapter=1, scene=0, span=span, confidence=0.9)
    base.update(kw)
    return Assertion(**base)


# ═══════════════ LƯỢT 3 — XÁC MINH SPAN (§10.5.2) ═══════════════

def test_span_that_duoc_giu_span_bia_bi_loai():
    """TIÊU CHÍ NGÀY 6. Lượt 1 chữa lỗi SÓT nhưng mở ra lỗi ngược — hỏi "manh
    mối X đã cài chưa?" thì model có xu hướng mạnh trả lời *rồi* và bịa một
    span nghe hợp lý. Bạn vừa đổi lỗi sót lấy lỗi bịa, mà bịa còn tệ hơn: nó
    đánh dấu manh mối là đã cài trong khi độc giả chưa hề thấy gì."""
    d = StateDelta(assertions=[
        _a("Mép con dấu sần lên như vỏ cam để lâu"),
        _a("Hắn rút khẩu súng ra và bắn ba phát vào trần nhà", "CHAR_VHAL"),
    ])
    d, rejected = verify_spans(d, PROSE)
    assert len(d.assertions) == 1
    assert d.assertions[0].subject == "CHAR_KAELEN"
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "span_not_found"
    assert rejected[0]["subject"] == "CHAR_VHAL"


def test_chuan_hoa_NFC_bat_buoc_voi_tieng_viet():
    """Hai chuỗi hiện ra GIỐNG HỆT nhau trên màn hình nhưng `in` trả False:
    "ế" viết được bằng một code point (U+1EBF) hoặc bằng "e" + hai dấu tổ hợp.
    Không chuẩn hoá thì `verify_spans` loại đúng những span mà model chép lại
    CHÍNH XÁC nhất — lỗi âm thầm mà NT-5 sinh ra để chặn."""
    span_nfd = unicodedata.normalize(
        "NFD", "Mép con dấu sần lên như vỏ cam để lâu")
    assert span_nfd not in PROSE            # `in` thô: trượt
    d, rejected = verify_spans(StateDelta(assertions=[_a(span_nfd)]), PROSE)
    assert len(d.assertions) == 1 and rejected == []


def test_chuan_hoa_dau_nhay_va_gach_ngang():
    """Model hay đổi “ ” thành " " và — thành -. Đó là khác biệt kiểu chữ,
    không phải khác biệt nội dung."""
    assert _norm("“xin chào”") == _norm('"xin chào"')
    assert _norm("a — b") == _norm("a - b")
    assert _norm("A   B\n\nC") == "a b c"


def test_span_qua_ngan_bi_loai():
    """`min_len` loại các span rỗng nghĩa kiểu "anh nói" vốn khớp được ở MỌI
    nơi — chúng "xác minh" thành công mà không chứng minh gì cả."""
    d, rejected = verify_spans(StateDelta(assertions=[_a("Kaelen")]), PROSE)
    assert d.assertions == []
    assert rejected[0]["reason"] == "span_too_short"
    assert MIN_SPAN_LEN == 12


def test_span_rong_bi_loai():
    d, rejected = verify_spans(StateDelta(assertions=[_a("")]), PROSE)
    assert d.assertions == [] and rejected[0]["reason"] == "span_too_short"


def test_fuzzy_cho_phep_polish_sua_nhe():
    """Polish Agent chạy SAU Writer nhưng TRƯỚC Extractor ở GĐ2. Một dấu phẩy
    bị đổi không được làm mất cả mệnh đề."""
    hoi_khac = "Mép con dấu sần lên, như vỏ cam để lâu"
    d, rejected = verify_spans(StateDelta(assertions=[_a(hoi_khac)]), PROSE)
    assert len(d.assertions) == 1 and rejected == []


def test_fuzzy_khong_nuot_mot_cau_bia():
    """Ngưỡng 0,90 đủ rộng cho sửa chữ, đủ chặt để không nuốt câu bịa."""
    assert not _fuzzy_present(
        _norm("Con dấu hoàn toàn nguyên vẹn và sáng bóng"), _norm(PROSE))


def test_fuzzy_khong_no_khi_needle_dai_hon_haystack():
    assert _fuzzy_present("x" * 500, "ngắn") is False
    assert _fuzzy_present("", "bất cứ gì") is False


# ═══════════════ verified CHỈ do code đặt (NT-5) ═══════════════

def test_plant_evidence_verified_do_CODE_dat_khong_do_llm_khai():
    """LLM khai `verified: true` cho một span không tồn tại là đúng kịch bản
    ảo giác xác nhận. `verify_spans` là nơi DUY NHẤT được đặt trường này."""
    d = StateDelta(plant_evidence=[
        PlantEvidence(clue_id="CLUE_A", scene_id="CH001_S00",
                      span="Mép con dấu sần lên như vỏ cam để lâu",
                      carrier_used="setting", verified=False),
        PlantEvidence(clue_id="CLUE_B", scene_id="CH001_S00",
                      span="Một câu hoàn toàn bịa đặt không có trong văn bản",
                      carrier_used="object", verified=True),   # LLM tự khai!
    ])
    d, rejected = verify_spans(d, PROSE)
    ok = {e.clue_id: e.verified for e in d.plant_evidence}
    assert ok == {"CLUE_A": True, "CLUE_B": False}
    assert any(r["reason"].startswith("plant_") for r in rejected)


def test_plant_evidence_khong_bi_xoa_chi_bi_danh_dau():
    """Assertion bịa thì BỎ, nhưng plant_evidence bịa thì GIỮ LẠI với
    `verified=False` — `plan_coverage` cần biết model đã khai gì để báo cáo."""
    d = StateDelta(plant_evidence=[
        PlantEvidence(clue_id="CLUE_X", scene_id="S", span="bịa đặt hoàn toàn",
                      carrier_used="object")])
    d, _ = verify_spans(d, PROSE)
    assert len(d.plant_evidence) == 1 and d.plant_evidence[0].verified is False


# ═══════════════ plan_coverage (§10.5.3) ═══════════════

def _contract(sid: str, *clue_ids: str) -> dict:
    return {"scene_id": sid, "location_id": "LOC_ORE_PORT",
            "pov_character": "CHAR_KAELEN", "active_characters": [],
            "plant_directives": [{"clue_id": c, "mode": "plant"}
                                 for c in clue_ids]}


def test_plan_coverage_doi_chieu_qua_plant_evidence_khong_qua_assertion():
    """B1/NT-8: bản trước so `d["clue_id"]` với `{a.subject}`. Hai không gian
    định danh KHÔNG BAO GIỜ giao nhau — `subject` là thực thể trong câu văn
    ("con dấu"), `clue_id` là mã hệ thống. Phép so luôn False: mọi manh mối bị
    tính là thất bại và mọi `clue_transitions` bị xoá sạch."""
    d = StateDelta(
        assertions=[_a("Mép con dấu sần lên như vỏ cam để lâu", "con dấu")],
        plant_evidence=[PlantEvidence(clue_id="CLUE_SEAL", scene_id="CH001_S00",
                                      span="x", carrier_used="setting",
                                      verified=True)],
        clue_transitions={"CLUE_SEAL": ClueStatus.PLANTED})
    cov = plan_coverage([_contract("CH001_S00", "CLUE_SEAL")], d)
    assert cov["plan_fulfillment_rate"] == 1.0
    assert cov["fulfilled_plants"] == ["CLUE_SEAL"]
    assert d.clue_transitions == {"CLUE_SEAL": ClueStatus.PLANTED}


def test_manh_moi_hua_ma_khong_cai_bi_bao_major():
    d = StateDelta(clue_transitions={"CLUE_SEAL": ClueStatus.PLANTED})
    cov = plan_coverage([_contract("CH001_S00", "CLUE_SEAL")], d)
    assert cov["plan_fulfillment_rate"] == 0.0
    assert cov["missed_plants"][0]["clue_id"] == "CLUE_SEAL"
    assert cov["findings"][0]["severity"] == "major"
    # NT-5: không bằng chứng thì KHÔNG chuyển trạng thái, dù LLM khai gì
    assert d.clue_transitions == {}


def test_transition_KHONG_duoc_hua_ma_khong_bang_chung_cung_bi_loai():
    """Lỗ hổng còn lại của §10.5.3: bản gốc chỉ xoá transition của manh mối ĐÃ
    HỨA mà trượt. Một transition model tự bịa cho manh mối CHƯA TỪNG được hứa
    không nằm trong `missed_ids`, nên nó đi thẳng vào canon — đúng con đường ô
    nhiễm mà chính mục này tuyên bố là đã đóng.

    Hậu quả cụ thể: manh mối bị đánh dấu `planted` mà chưa từng lên trang
    giấy, và Foreshadow Scheduler từ đó không bao giờ cài lại nó nữa."""
    d = StateDelta(clue_transitions={"CLUE_MA": ClueStatus.PLANTED})
    cov = plan_coverage([_contract("CH001_S00")], d)     # không hứa gì cả
    assert d.clue_transitions == {}
    assert cov["dropped_transitions"][0]["clue_id"] == "CLUE_MA"
    assert cov["dropped_transitions"][0]["promised"] is False
    assert cov["findings"][0]["check"] == "unverified_transition"


def test_plan_coverage_khong_chia_cho_khong():
    assert plan_coverage([], StateDelta())["plan_fulfillment_rate"] == 0.0
    assert plan_coverage([_contract("S")], StateDelta())[
        "plan_fulfillment_rate"] == 0.0


def test_plan_coverage_chiu_duoc_directive_thieu_khoa():
    cts = [{"scene_id": "S", "plant_directives": [{"mode": "plant"}]}]
    assert plan_coverage(cts, StateDelta())["promised_plants"] == []


# ═══════════════ merge_extractions (§10.5.3) ═══════════════

def _json(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False)


def test_merge_gop_DU_SAU_truong_khong_chi_hai():
    """B2: bản trước chỉ gộp `assertions` và `new_entities`. Mọi
    `new_relations`, `retracted_relations`, `clue_transitions` và
    `relationship_updates` mà lượt 2 tìm được đều bị vứt IM LẶNG — một mối thù
    hay một liên minh phát sinh ngoài kế hoạch không bao giờ tới
    `reconcile_node`."""
    a = _json(assertions=[_a("span lượt một dài hơn mười hai").model_dump()])
    e = _json(
        new_entities=[Entity(id="OBJ_X", kind="object", name="X").model_dump()],
        new_relations=[Relation(src="CHAR_A", dst="CHAR_B",
                                type="BETRAYED").model_dump()],
        retracted_relations=[Relation(src="CHAR_C", dst="FACT_X",
                                      type="MEMBER_OF").model_dump()],
        clue_transitions={"CLUE_E": "planted"},
        relationship_updates=[RelationshipState(a="CHAR_A",
                                                b="CHAR_B").model_dump()],
    )
    m = merge_extractions(a, e, chapter=3)
    assert len(m.assertions) == 1
    assert [x.id for x in m.new_entities] == ["OBJ_X"]
    assert [(r.src, r.type) for r in m.new_relations] == [("CHAR_A", "BETRAYED")]
    assert [r.type for r in m.retracted_relations] == ["MEMBER_OF"]
    assert m.clue_transitions == {"CLUE_E": ClueStatus.PLANTED}
    assert len(m.relationship_updates) == 1
    assert m.chapter == 3 and m.delta_id == "d_ch003"


def test_merge_luot_1_luon_thang_khi_trung():
    """Lượt 1 có hệ quy chiếu nên đáng tin hơn hẳn."""
    a = _json(assertions=[_a("span của lượt một, dài hơn mười hai ký tự",
                             confidence=0.95).model_dump()])
    e = _json(assertions=[_a("span của lượt hai, cũng dài hơn mười hai",
                             confidence=0.3).model_dump()])
    m = merge_extractions(a, e, chapter=1)
    assert len(m.assertions) == 1            # cùng (subject, predicate)
    assert m.assertions[0].confidence == 0.95


def test_merge_clue_transitions_dung_setdefault_khong_update():
    """`update` để lượt 2 ghi đè lượt 1, đảo ngược quy tắc ưu tiên đã công bố.
    Lượt 2 không thấy contract nên phán đoán chuyển trạng thái của nó kém tin
    cậy hơn hẳn."""
    a = _json(clue_transitions={"CLUE_X": "reinforced"})
    e = _json(clue_transitions={"CLUE_X": "paid_off", "CLUE_Y": "planted"})
    m = merge_extractions(a, e, chapter=1)
    assert m.clue_transitions["CLUE_X"] is ClueStatus.REINFORCED   # lượt 1
    assert m.clue_transitions["CLUE_Y"] is ClueStatus.PLANTED      # lượt 2 bổ sung


def test_merge_quan_he_gop_THEO_CAP_khong_noi_duoi():
    """Nối đuôi khiến `apply_scene_effects` (§7.3) chạy hai lần trên cùng một
    cặp và cộng dồn intimacy — biểu hiện ra ngoài là M5 báo nhảy cóc giai đoạn
    mà nhìn vào guard thì guard đúng."""
    a = _json(relationship_updates=[
        RelationshipState(a="CHAR_A", b="CHAR_B", intimacy=10).model_dump()])
    e = _json(relationship_updates=[
        # ĐẢO thứ tự cặp — vẫn phải nhận ra là cùng một quan hệ
        RelationshipState(a="CHAR_B", b="CHAR_A", intimacy=40).model_dump()])
    m = merge_extractions(a, e, chapter=1)
    assert len(m.relationship_updates) == 1
    assert m.relationship_updates[0].intimacy == 10       # lượt 1 thắng


def test_merge_khong_trung_thuc_the():
    a = _json(new_entities=[Entity(id="OBJ_X", kind="object",
                                   name="X").model_dump()])
    e = _json(new_entities=[
        Entity(id="OBJ_X", kind="object", name="X khác").model_dump(),
        Entity(id="OBJ_Y", kind="object", name="Y").model_dump()])
    m = merge_extractions(a, e, chapter=1)
    assert [x.id for x in m.new_entities] == ["OBJ_X", "OBJ_Y"]
    assert m.new_entities[0].name == "X"          # lượt 1 thắng


def test_merge_lot_fence_va_tu_sua_json_hong():
    a = '```json\n{"assertions": []}\n```'
    m = merge_extractions(a, '{"new_entities": []}', chapter=2)
    assert m.chapter == 2


def test_merge_json_hong_khong_sua_duoc_thi_nem():
    with pytest.raises(ValueError, match="StateDelta"):
        merge_extractions("hoàn toàn không phải json", "{}", chapter=1)


# ═══════════════ TÍCH HỢP TRONG ĐỒ THỊ ═══════════════

@pytest.fixture
def rig():
    llm = FakeLLM()
    eng = build_engines(llm, db_path=":memory:")
    yield llm, eng, run_chapter(eng, 1)
    eng.store.close()


def test_extractor_chay_DUNG_MOT_LAN_cho_ca_chuong(rig):
    """F4 cùng cơ chế: gọi trong vòng lặp cảnh là 6 lần thay vì 1, và mỗi lần
    lại thấy một phần văn bản khác nhau."""
    llm, _eng, _out = rig
    theo_vai = {r: sum(1 for c in llm.calls if c["role"] == r)
                for r in {c["role"] for c in llm.calls}}
    assert theo_vai["extractor_diff"] == 1
    assert theo_vai["extractor_emergent"] == 1
    assert len(llm.calls) == 20          # 18 của GĐ1 + 2 lượt trích xuất


def test_luot_1_nhan_van_xuoi_CA_CHUONG_khong_chi_canh_cuoi(rig):
    """C2: bản trước dùng `state["polished"]`, vốn chỉ chứa văn xuôi của CẢNH
    VỪA XONG, trong khi `contracts` chứa cả 6 cảnh. `verify_spans` đi tìm span
    của cảnh 1–5 trong văn bản cảnh 6, không thấy, và `plan_coverage` xoá sạch
    manh mối của 5 cảnh đầu."""
    llm, _eng, out = rig
    p = next(c["prompt"] for c in llm.calls if c["role"] == "extractor_diff")
    for si in range(6):
        assert f"[CH001_S{si:02d}]" in p, f"thiếu cảnh {si} trong prompt lượt 1"
    for s in out["scene_outputs"]:
        assert s["prose"][:40] in p


def test_luot_1_gan_nhan_canh_de_scene_id_la_DU_LIEU(rig):
    """`PlantEvidence.scene_id` phải là dữ liệu, không phải phỏng đoán."""
    llm, _eng, _out = rig
    p = next(c["prompt"] for c in llm.calls if c["role"] == "extractor_diff")
    assert p.index("[CH001_S00]") < p.index("[CH001_S01]")


def test_luot_2_TUYET_DOI_khong_thay_contract(rig):
    """Lượt 2 mất giá trị NGAY khi nó nhìn thấy kế hoạch: model bị neo vào
    contract sẽ bỏ qua đúng những thứ nằm ngoài kế hoạch — tức toàn bộ mục
    đích của lượt này."""
    llm, _eng, out = rig
    p = next(c["prompt"] for c in llm.calls if c["role"] == "extractor_emergent")
    for tu in ("plant_directives", "must_not_reveal", "scene_must_change",
               "SceneContract", "KẾ HOẠCH"):
        assert tu not in p, f"lượt 2 nhìn thấy kế hoạch: {tu!r}"
    # …nhưng nó PHẢI thấy danh mục thực thể đã biết, nếu không nó báo mọi thứ
    # là "mới" và Extractor tạo trùng thực thể đã có dưới tên khác.
    assert "CHAR_KAELEN" in p and "LOC_ORE_PORT" in p


def test_luot_2_nhan_van_xuoi_KHONG_gan_nhan(rig):
    """Nhãn [CH001_S00] là siêu dữ liệu của hệ thống. Lượt 2 đọc như độc giả."""
    llm, _eng, _out = rig
    p = next(c["prompt"] for c in llm.calls if c["role"] == "extractor_emergent")
    assert "[CH001_S00]" not in p


def test_bao_cao_trich_xuat_ghi_nhan_span_bi_loai(rig):
    """`FakeLLM._extract_diff` CỐ Ý xuất một span bịa ở mỗi lượt chạy, để
    nhánh loại-bỏ luôn được thực thi — không có nó thì ta không biết cơ chế
    còn sống hay đã chết."""
    _llm, _eng, out = rig
    rep = out["extraction_report"]
    assert rep["assertions_kept"] >= 1
    bia = [r for r in rep["rejected_spans"] if r["reason"] == "span_not_found"]
    assert bia, "không span nào bị loại — lượt 3 có thể đã thành mã chết"
    assert "không tồn tại" in bia[0]["span"]


def test_luot_2_tim_duoc_thuc_the_ngoai_ke_hoach(rig):
    """Toàn bộ lý do tồn tại của lượt 2."""
    _llm, _eng, out = rig
    from novel_engine.canon.models import StateDelta as SD
    d = SD.model_validate(out["delta"])
    assert "OBJ_SO_CA_TRUC" in {e.id for e in d.new_entities}


def test_delta_duoc_dong_dau_chuong(rig):
    _llm, _eng, out = rig
    assert out["delta"]["chapter"] == 1
    assert out["delta"]["delta_id"] == "d_ch001"


def test_extractor_hong_thanh_escalation_khong_sap_cuoi_chuong(rig):
    """§9.5: sập ở bước CUỐI CÙNG, sau khi đã trả tiền cho toàn bộ 6 cảnh, là
    kịch bản đắt nhất."""
    _llm, eng, _out = rig
    eng.llm = ScriptedLLM([], fallback=FakeLLM())
    eng.llm.invoke = lambda p, role="": (_ for _ in ()).throw(
        RuntimeError("API sập")) if role.startswith("extractor") \
        else FakeLLM().invoke(p, role=role)
    out = run_chapter(eng, 1)
    assert out["escalated"] is True
    assert "extractor_node" in out["escalation_reason"]


def test_khong_co_canh_nao_thi_tra_delta_rong():
    eng = build_engines(FakeLLM(), db_path=":memory:")
    from novel_engine.graph.nodes import extractor_node
    out = extractor_node({"chapter": 1, "scene_outputs": [], "contracts": []},
                         {"configurable": {"engines": eng}})
    assert out["delta"]["chapter"] == 1
    assert out["extraction_report"]["skipped"]
    eng.store.close()


def test_fuzzy_khong_nhay_qua_vi_tri_khop_dung():
    """Hồi quy cho lỗi `step = n // 4` của §10.5.2.

    Bước nhảy cố định CÓ THỂ nhảy qua đúng vị trí khớp, và khi đó nó hỏng IM
    LẶNG. Đo thực trên câu 38 ký tự chỉ khác một dấu phẩy:

        vị trí đúng  i=104 → ratio 0.974  ✓
        step ghé     i= 99 → ratio 0.868  ✗
        step ghé     i=108 → ratio 0.868  ✗

    Xác suất trúng ≈ 1/9, nên lớp dự phòng mà §10.5.2 tuyên bố là có thực ra
    gần như không hoạt động. Test này dựng đúng thế lệch đó.
    """
    hay = _norm(PROSE)
    nd = _norm("Mép con dấu sần lên, như vỏ cam để lâu")
    n = len(nd)
    # chứng minh thế lệch vẫn còn nguyên: không bội số nào của n//4 rơi đúng
    assert hay.find("mép con dấu") % max(1, n // 4) != 0
    assert _fuzzy_present(nd, hay) is True


def test_fuzzy_bat_duoc_sua_VAN_PHONG_o_moi_vi_tri():
    """Polish được phép sửa dấu câu, khoảng trắng, thêm/bớt từ đệm."""
    hay = _norm(PROSE)
    for bien_the in (
        "Mép con dấu sần lên, như vỏ cam để lâu",      # thêm dấu phẩy giữa câu
        "Mép con dấu sần lên như vỏ cam để lâu rồi",   # thêm từ đệm ở cuối
        "Con dấu sần lên như vỏ cam để lâu",           # bớt từ ở đầu
    ):
        assert _fuzzy_present(_norm(bien_the), hay), bien_the


def test_fuzzy_TU_CHOI_khi_mot_tu_NOI_DUNG_bi_doi():
    """Ranh giới của cơ chế, và nó nằm đúng chỗ nên nằm.

    "vỏ cam ĐỂ lâu" → "vỏ cam PHƠI lâu" chỉ khác 2 ký tự trên 38 (ratio 0.897)
    nhưng là một KHẲNG ĐỊNH KHÁC về thế giới. Fuzzy là dự phòng cho sửa văn
    phong, không phải giấy phép cho model diễn giải lại câu văn — và §9.2 vốn
    đã có `content_drifted` cấm Polish đổi nội dung.

    Hệ quả: `verify_spans` nghiêng về phía LOẠI khi nghi ngờ. Đó là hướng sai
    đúng đắn — bỏ sót một mệnh đề thật thì Extractor chương sau nhặt lại được,
    còn ghi nhầm một mệnh đề sai thì hai mươi chương sau mới lộ.
    """
    assert not _fuzzy_present(
        _norm("Mép con dấu sần lên như vỏ cam phơi lâu"), _norm(PROSE))


def test_fuzzy_van_tu_choi_cau_khac_han():
    hay = _norm(PROSE)
    for bia in ("Serena rút khẩu súng ra khỏi bao da và ngắm thẳng",
                "Lò phản ứng số ba phát nổ lúc ba giờ sáng hôm đó"):
        assert not _fuzzy_present(_norm(bia), hay), bia


# ═══════════ Bằng chứng cài cho manh mối KHÔNG được hứa (dữ liệu Gemini thật) ═══════════

def test_bang_chung_ma_gia_bi_loai_du_da_xac_minh():
    """Gemini thật: chương không hứa manh mối nào, model vẫn xuất `plant_evidence`
    với mã tự đặt `CH001_S00_01`. Span có thật → `verified=True`. §11 sẽ gọi
    `clues.touch()` cho mọi bằng chứng đã xác minh — chạm vào manh mối không
    tồn tại."""
    d = StateDelta(plant_evidence=[
        PlantEvidence(clue_id="CH001_S00_01", scene_id="CH001_S00", span="x",
                      carrier_used="setting", verified=True)])
    cov = plan_coverage([_contract("CH001_S00")], d,
                        known_clues={"CLUE_SEAL_CORROSION"})
    assert d.plant_evidence == []
    assert cov["phantom_evidence"][0]["clue_id"] == "CH001_S00_01"
    assert any(f["check"] == "phantom_plant_evidence" for f in cov["findings"])


def test_manh_moi_that_tinh_co_duoc_cai_van_giu():
    """Writer tình cờ cài một manh mối CÓ THẬT mà Scheduler không giao — bằng
    chứng hợp lệ, không được vứt."""
    d = StateDelta(
        plant_evidence=[PlantEvidence(clue_id="CLUE_SEAL_CORROSION",
                                      scene_id="CH001_S02", span="x",
                                      carrier_used="object", verified=True)],
        clue_transitions={"CLUE_SEAL_CORROSION": ClueStatus.PLANTED})
    cov = plan_coverage([_contract("CH001_S02")], d,
                        known_clues={"CLUE_SEAL_CORROSION"})
    assert [e.clue_id for e in d.plant_evidence] == ["CLUE_SEAL_CORROSION"]
    assert cov["incidental_plants"][0]["clue_id"] == "CLUE_SEAL_CORROSION"
    assert d.clue_transitions == {"CLUE_SEAL_CORROSION": ClueStatus.PLANTED}


def test_khong_truyen_known_clues_thi_che_do_nghiem():
    d = StateDelta(plant_evidence=[
        PlantEvidence(clue_id="CLUE_SEAL_CORROSION", scene_id="S", span="x",
                      carrier_used="object", verified=True)])
    cov = plan_coverage([_contract("S")], d)
    assert d.plant_evidence == []
    assert cov["phantom_evidence"]


def test_transition_cua_manh_moi_ma_bi_loai_theo():
    d = StateDelta(
        plant_evidence=[PlantEvidence(clue_id="CH001_S00_01", scene_id="S",
                                      span="x", carrier_used="setting",
                                      verified=True)],
        clue_transitions={"CH001_S00_01": ClueStatus.PLANTED})
    plan_coverage([_contract("S")], d, known_clues=set())
    assert d.clue_transitions == {}
