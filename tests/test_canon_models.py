"""Day 1 DoD — serialize/deserialize sạch, `item_key` ổn định, validator chạy.

Ưu tiên đúng theo §0.5: Tầng 2 là deterministic nên test rẻ và bắt được gần
hết. Năm lỗi F1/F2/F3/F8/F9 đều thuộc loại mà một `pytest` hai giây tìm ra
nhanh hơn mọi vòng đọc mã.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from novel_engine.canon.models import (
    Assertion, Clue, ClueStatus, Entity, NPCRole, PlantEvidence, Relation,
    RelationStage, Severity, StateDelta, item_key,
)
from novel_engine.canon.timeline import (
    ContinuityFrame, ContinuityObservation, SceneClose, StoryTime,
)
from novel_engine.relationship.models import RelationshipState


# ───────────────────────────  ENUMS  ───────────────────────────

def test_enums_la_chuoi_serialize_duoc():
    """Mọi enum kế thừa `str` để json.dumps không cần encoder riêng."""
    assert ClueStatus.PLANTED == "planted"
    assert NPCRole.AMBIENT == "ambient"
    assert RelationStage.CATHARSIS == "catharsis"
    assert Severity.BLOCKER == "blocker"
    assert json.dumps({"s": ClueStatus.DRAFTED.value}) == '{"s": "drafted"}'


# ───────────────────────────  ENTITY / RELATION  ───────────────────────────

def test_entity_roundtrip():
    e = Entity(id="CHAR_KAELEN", kind="character", name="Kaelen",
               aliases=["Kẻ Lưu Đày"], first_appearance=1, canon_locked=True,
               attributes={"rank": "Vanguard"})
    back = Entity.model_validate_json(e.model_dump_json())
    assert back == e
    assert back.attributes["rank"] == "Vanguard"


def test_entity_kind_bi_rang_buoc():
    with pytest.raises(ValidationError):
        Entity(id="X", kind="spaceship", name="X")     # không nằm trong Literal


def test_relation_roundtrip_va_mac_dinh():
    r = Relation(src="CHAR_SERENA", dst="FACT_CIPHER", type="MEMBER_OF")
    back = Relation.model_validate_json(r.model_dump_json())
    assert back == r
    assert back.until_chapter is None        # None = còn hiệu lực
    assert back.since_tick == 0              # NT-6: trục epoch, mặc định 0
    assert back.known_by == []               # rỗng = sự thật chưa ai biết


def test_relation_weight_bi_chan_ngoai_khoang():
    with pytest.raises(ValidationError):
        Relation(src="A", dst="B", type="PROTECTS", weight=1.4)


# ───────────────────────────  CLUE  ───────────────────────────

def _clue(**kw) -> Clue:
    base = dict(clue_id="CLUE_SEAL", macro_event_target="EV_CORE_COLLAPSE",
                description="Con dấu bị ăn mòn bởi axit công nghiệp",
                payoff_threshold=5, payoff_deadline=12)
    base.update(kw)
    return Clue(**base)


def test_clue_roundtrip():
    c = _clue(surface_forms=["vết hoen trên con dấu", "mép dấu sần sùi"],
              salience=0.4, subtlety_target=0.75)
    back = Clue.model_validate_json(c.model_dump_json())
    assert back == c
    assert back.status is ClueStatus.DRAFTED
    assert back.planted_in_chapter is None
    assert back.last_touched_chapter is None


def test_payoff_deadline_phai_lon_hon_bang_threshold():
    """§3.3 — thiếu ràng buộc này thì manh mối có deadline vô nghĩa."""
    with pytest.raises(ValidationError) as ei:
        _clue(payoff_threshold=10, payoff_deadline=4)
    assert "payoff_deadline" in str(ei.value)


def test_payoff_deadline_bang_threshold_la_hop_le():
    """Biên: deadline == threshold nghĩa là chỉ có đúng một chương để trả bài."""
    c = _clue(payoff_threshold=7, payoff_deadline=7)
    assert c.payoff_deadline == 7


def test_salience_ngoai_khoang_bi_chan():
    with pytest.raises(ValidationError):
        _clue(salience=1.7)


# ───────────────────────────  ASSERTION / PLANT EVIDENCE  ───────────────────

def test_assertion_epistemic_mac_dinh_la_objective():
    a = Assertion(subject="CHAR_VHAL", predicate="located_in", object="LOC_PORT",
                  chapter=1, scene=0, span="Vhal đứng ở mép cảng.", confidence=0.9)
    assert a.epistemic == "objective"
    assert a.holder is None


def test_assertion_loi_noi_doi_giu_duoc_nguoi_noi():
    """§3.4 — thiếu `epistemic` thì mọi lời nói dối làm ô nhiễm canon."""
    a = Assertion(subject="FLEET_3", predicate="status", object="disbanded",
                  chapter=4, scene=2, span="Hạm Đội Số 3 đã bị giải tán.",
                  confidence=0.8, epistemic="claimed_by", holder="CHAR_SERENA")
    back = Assertion.model_validate_json(a.model_dump_json())
    assert back.epistemic == "claimed_by" and back.holder == "CHAR_SERENA"


def test_assertion_object_nhan_nhieu_kieu():
    for val in ("LOC_FORGE", 3, 1.5, True):
        a = Assertion(subject="S", predicate="p", object=val, chapter=1, scene=0,
                      span="x", confidence=0.5)
        assert Assertion.model_validate_json(a.model_dump_json()).object == val


def test_plant_evidence_verified_mac_dinh_false():
    """`verified` do verify_spans() đặt, KHÔNG do LLM khai."""
    p = PlantEvidence(clue_id="CLUE_SEAL", scene_id="CH001_S02",
                      span="mép con dấu sần lên", carrier_used="object")
    assert p.verified is False
    assert p.reacted_by == [] and p.concluded_by == []


# ───────────────────────────  item_key (§11, F9/NT-16)  ─────────────────────

def test_item_key_dung_blake2b_cho_assertion():
    import hashlib
    a = Assertion(subject="CHAR_KAELEN", predicate="has_scar", object=True,
                  chapter=3, scene=1, span="vết sẹo chéo trên trán",
                  confidence=0.95)
    h = hashlib.blake2b("vết sẹo chéo trên trán".encode("utf-8"),
                        digest_size=3).hexdigest()
    assert item_key(a) == f"A:CHAR_KAELEN|has_scar|{h}"
    assert len(h) == 6                      # digest_size=3 byte → 6 hex


def test_item_key_on_dinh_giua_hai_tien_trinh():
    """F9/NT-16: `hash()` ngẫu nhiên hoá theo PYTHONHASHSEED. Khoá đi vào
    checkpoint LangGraph phải sống sót qua một tiến trình MỚI, nếu không
    `delta.item(k)` ném KeyError khi chạy tiếp từ checkpoint.

    Test này chạy thật hai tiến trình con với seed khác nhau — cách duy nhất
    chứng minh được tính ổn định, vì trong cùng tiến trình `hash()` cũng nhất
    quán và bug sẽ không lộ.
    """
    code = (
        "from novel_engine.canon.models import Assertion, item_key;"
        "print(item_key(Assertion(subject='S', predicate='p', object=1,"
        " chapter=1, scene=0, span='bằng chứng', confidence=0.5)))"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, encoding="utf-8",
                           env={"PYTHONHASHSEED": seed, "PATH": "",
                                "SYSTEMROOT": __import__("os").environ.get(
                                    "SYSTEMROOT", "")})
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, f"khoá đổi theo PYTHONHASHSEED: {outs}"


def test_item_key_cho_entity_va_relation():
    assert item_key(Entity(id="LOC_FORGE", kind="location", name="Lò")) == "E:LOC_FORGE"
    r = Relation(src="A", dst="B", type="BETRAYED")
    assert item_key(r) == "R:A|BETRAYED|B"


def test_item_key_tu_choi_kieu_la():
    with pytest.raises(TypeError):
        item_key({"khong": "phai model"})


# ───────────────────────────  StateDelta  ───────────────────────────

def _delta() -> StateDelta:
    return StateDelta(
        new_entities=[Entity(id="LOC_FORGE", kind="location", name="Lò Phản ứng")],
        new_relations=[Relation(src="CHAR_KAELEN", dst="LOC_FORGE",
                                type="LOCATED_IN")],
        retracted_relations=[Relation(src="CHAR_VHAL", dst="FACT_ARCLIGHT",
                                      type="MEMBER_OF", until_chapter=1)],
        assertions=[Assertion(subject="CHAR_KAELEN", predicate="has_scar",
                              object=True, chapter=1, scene=0,
                              span="vết sẹo chéo", confidence=0.9)],
        clue_transitions={"CLUE_SEAL": ClueStatus.PLANTED},
        plant_evidence=[PlantEvidence(clue_id="CLUE_SEAL", scene_id="CH001_S00",
                                      span="mép dấu sần", carrier_used="setting")],
        relationship_updates=[RelationshipState(a="CHAR_KAELEN", b="CHAR_SERENA",
                                                intimacy=12.0, friction=40.0)],
    )


def test_statedelta_roundtrip_json_khong_loi():
    """DoD Ngày 1: serialize/deserialize thành công, không ValidationError."""
    d = _delta()
    back = StateDelta.model_validate_json(d.model_dump_json())
    assert back.new_entities[0].id == "LOC_FORGE"
    assert back.clue_transitions["CLUE_SEAL"] is ClueStatus.PLANTED
    assert back.plant_evidence[0].verified is False
    assert back.relationship_updates[0].stage is RelationStage.STRANGERS
    assert back.retracted_relations[0].until_chapter == 1


def test_statedelta_metadata_co_default_NT13():
    """NT-13: LLM không bao giờ sinh `delta_id`/`created_at`/`chapter`.
    Để chúng bắt buộc là đảm bảo `parse_model` ném ValidationError ở lượt
    trích xuất thứ hai (E3)."""
    d = StateDelta.model_validate_json("{}")        # đúng cái LLM có thể trả về
    assert d.delta_id == "" and d.chapter == 0
    assert d.created_at is not None
    assert d.committed is False
    assert d.new_entities == [] and d.assertions == []


def test_stamp_gan_metadata_tat_dinh():
    """`delta_id` phải tất định (không timestamp) để bộ hồi quy §13.2 tái lập."""
    a = StateDelta.model_validate_json("{}").stamp(7)
    b = StateDelta.model_validate_json("{}").stamp(7)
    assert a.delta_id == b.delta_id == "d_ch007"
    assert a.chapter == 7


def test_stamp_dien_chapter_cho_assertion_bo_trong():
    d = StateDelta(assertions=[Assertion(subject="S", predicate="p", object=1,
                                         chapter=0, scene=0, span="x",
                                         confidence=0.5)]).stamp(9)
    assert d.assertions[0].chapter == 9


def test_item_tra_ve_dung_phan_tu():
    d = _delta()
    assert d.item("E:LOC_FORGE").name == "Lò Phản ứng"
    assert d.item("R:CHAR_KAELEN|LOCATED_IN|LOC_FORGE").type == "LOCATED_IN"
    # retracted_relations cũng phải tra được (F7 — quan hệ cũ không bao giờ đóng)
    assert d.item("R:CHAR_VHAL|MEMBER_OF|FACT_ARCLIGHT").until_chapter == 1


def test_item_nem_keyerror_khi_khong_co():
    with pytest.raises(KeyError):
        _delta().item("E:KHONG_TON_TAI")


def test_item_tra_duoc_sau_khi_qua_json():
    """Đúng kịch bản F9: delta đi qua checkpoint rồi được tra lại bằng khoá."""
    d = _delta()
    key = item_key(d.assertions[0])
    back = StateDelta.model_validate_json(d.model_dump_json())
    assert back.item(key).predicate == "has_scar"


# ───────────────────────────  TIMELINE (§3.6)  ───────────────────────────

def test_storytime_end_tick_va_overlaps():
    a = StoryTime(epoch_tick=100, duration_ticks=5, narrative_order=1)
    b = StoryTime(epoch_tick=103, duration_ticks=4, narrative_order=2)
    c = StoryTime(epoch_tick=105, duration_ticks=2, narrative_order=3)
    assert a.end_tick == 105
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)        # biên chạm nhau KHÔNG tính là chồng lấn
    assert not c.overlaps(a)


def test_storytime_mode_mac_dinh_present():
    t = StoryTime(epoch_tick=0, narrative_order=0)
    assert t.mode == "present" and t.anchor_scene is None
    assert t.duration_ticks == 1


def test_storytime_hai_truc_doc_lap():
    """NT-6: hồi ức có narrative_order LỚN nhưng epoch_tick NHỎ. Schema phải
    cho phép điều đó — chính là thứ `_time_monotonic` của v2.0 làm sai."""
    hoi_uc = StoryTime(epoch_tick=500, duration_ticks=3, narrative_order=42,
                       mode="flashback", anchor_scene="CH010_S01")
    hien_tai = StoryTime(epoch_tick=9000, narrative_order=41)
    assert hoi_uc.narrative_order > hien_tai.narrative_order
    assert hoi_uc.epoch_tick < hien_tai.epoch_tick


def test_continuity_observation_khong_co_truong_time_F2():
    """F2: `narrative_order` bắt buộc trong schema mà LLM phải điền là
    ValidationError ở MỌI ranh giới cảnh. Sửa bằng cách cắt trường khỏi
    schema, không bằng cách dặn model (NT-13)."""
    assert "time" not in ContinuityObservation.model_fields
    assert "narrative_order" not in ContinuityObservation.model_fields
    obs = ContinuityObservation.model_validate_json(
        '{"locations": {"CHAR_KAELEN": "LOC_PORT"}}')   # đúng cái LLM trả về
    assert obs.injuries == {} and obs.possessions == {} and obs.weather is None


def test_sceneclose_parse_tu_json_kieu_llm():
    raw = json.dumps({
        "digest": "Kaelen tới cảng, gặp Vhal, bị từ chối cấp phép.",
        "continuity": {"locations": {"CHAR_KAELEN": "LOC_PORT"},
                       "injuries": {"CHAR_KAELEN": ["bầm vai trái"]},
                       "possessions": {"CHAR_KAELEN": ["thẻ Vanguard"]}},
        "actual_duration_ticks": 4,
        "unresolved": ["tiếng động trong khoang chứa chưa ai kiểm tra"],
    }, ensure_ascii=False)
    close = SceneClose.model_validate_json(raw)
    assert close.actual_duration_ticks == 4
    assert close.continuity.injuries["CHAR_KAELEN"] == ["bầm vai trái"]
    assert len(close.unresolved) == 1


def test_sceneclose_unresolved_mac_dinh_rong():
    close = SceneClose.model_validate_json(json.dumps({
        "digest": "d", "continuity": {"locations": {}},
        "actual_duration_ticks": 1}))
    assert close.unresolved == []


def test_continuity_frame_lap_tu_sceneclose_theo_9_2():
    """Đúng cách `scene_boundary_node` lắp frame: StoryTime do CODE ghép,
    model chỉ đóng góp `actual_duration_ticks` (F2)."""
    planned = StoryTime(epoch_tick=240, duration_ticks=3, narrative_order=5)
    close = SceneClose(digest="d",
                       continuity=ContinuityObservation(
                           locations={"CHAR_KAELEN": "LOC_PORT"}),
                       actual_duration_ticks=7)
    frame = ContinuityFrame(
        scene_id="CH001_S00",
        time=planned.model_copy(update={"duration_ticks":
                                        close.actual_duration_ticks}),
        **close.continuity.model_dump())
    assert frame.time.epoch_tick == 240          # mốc hệ thống giữ nguyên
    assert frame.time.duration_ticks == 7        # thời lượng THỰC từ văn bản
    assert frame.time.narrative_order == 5       # không bị model đụng vào
    assert ContinuityFrame.model_validate_json(frame.model_dump_json()) == frame


# ───────────────────────────  RELATIONSHIP  ───────────────────────────

def test_relationship_state_roundtrip_va_mac_dinh():
    st = RelationshipState(a="CHAR_KAELEN", b="CHAR_SERENA")
    back = RelationshipState.model_validate_json(st.model_dump_json())
    assert back.stage is RelationStage.STRANGERS
    assert back.chapters_in_stage == 0
    assert back.last_counted_chapter is None     # C7: chống đếm theo cảnh
    assert back.scars == []


def test_relationship_pair_key_doc_lap_thu_tu():
    x = RelationshipState(a="CHAR_A", b="CHAR_B")
    y = RelationshipState(a="CHAR_B", b="CHAR_A")
    assert x.pair_key == y.pair_key


def test_intimacy_asym_cho_phep_am():
    """Bất đối xứng: A có thể gắn bó hơn B. Chặn số âm là làm mọi quan hệ
    thành song phương hoàn hảo — cực kỳ giả."""
    st = RelationshipState(a="A", b="B", intimacy_asym=-30.0)
    assert st.intimacy_asym == -30.0
    with pytest.raises(ValidationError):
        RelationshipState(a="A", b="B", intimacy_asym=-70.0)
