"""Chỉ số cảnh do CODE suy ra từ vị trí span — không do model khai (NT-13).

Lượt Gemini thật ở Chương 2: `CHAR_KAELEN.knows_secret` được khai `scene=0`
(tick 20.028) trong khi span nói về đêm sự cố. Sai cảnh = sai mốc epoch, và
mệnh đề của cảnh hồi ức thoát khỏi kiểm tra hồi ức (§3.6.4).
"""
from __future__ import annotations

import json
import re

from novel_engine.canon.models import Assertion, PlantEvidence, StateDelta
from novel_engine.canon.timeline import ContinuityFrame, StoryTime
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.fake import FakeLLM
from novel_engine.reconcile.commit import reconcile
from novel_engine.reconcile.verify import assign_scenes

BI_MAT = ("Chính tay Kaelen đã khóa chặt van điều áp số bốn khi đồng hồ hiển thị "
          "mức giới hạn cuối cùng.")
SCENES = [
    {"scene_id": "CH002_S00", "prose": "Kaelen lên tàu quặng lúc nửa đêm, không ai ghi tên anh."},
    {"scene_id": "CH002_S02", "prose": "Đêm đó ở Lò 3. " + BI_MAT},
    {"scene_id": "CH002_S04", "prose": "Vết nứt trên vách bọc lõi kéo dài từ bệ đỡ chính."},
]


def _a(span, scene=0, predicate="knows_secret", obj="khoá van điều áp số bốn"):
    return Assertion(subject="CHAR_KAELEN", predicate=predicate, object=obj,
                     chapter=2, scene=scene, span=span, confidence=0.9)


def test_menh_de_duoc_gan_ve_canh_chua_span():
    d = StateDelta(assertions=[_a(BI_MAT, scene=0)]).stamp(2)
    notes = assign_scenes(d, SCENES)
    assert d.assertions[0].scene == 2
    assert notes[0]["reason"] == "scene_reassigned"
    assert (notes[0]["from"], notes[0]["to"]) == (0, 2)


def test_da_dung_canh_thi_khong_ghi_chu():
    d = StateDelta(assertions=[_a(BI_MAT, scene=2)]).stamp(2)
    assert assign_scenes(d, SCENES) == []


def test_span_lap_o_nhieu_canh_thi_khong_doan():
    lap = [{"scene_id": "CH002_S00", "prose": BI_MAT},
           {"scene_id": "CH002_S03", "prose": BI_MAT}]
    d = StateDelta(assertions=[_a(BI_MAT, scene=5)]).stamp(2)
    notes = assign_scenes(d, lap)
    assert d.assertions[0].scene == 5
    assert notes[0]["reason"] == "scene_ambiguous"


def test_khong_tim_thay_thi_giu_nguyen_khong_ghi_chu():
    d = StateDelta(assertions=[_a("Một câu không có ở cảnh nào trong chương này", scene=1)]).stamp(2)
    assert assign_scenes(d, SCENES) == []
    assert d.assertions[0].scene == 1


def test_polish_sua_nhe_van_gan_dung_canh():
    d = StateDelta(assertions=[_a(BI_MAT.replace("số bốn", "số bốn,"), scene=0)]).stamp(2)
    assign_scenes(d, SCENES)
    assert d.assertions[0].scene == 2


def test_bang_chung_cai_duoc_gan_ve_canh_chua_span():
    d = StateDelta(plant_evidence=[PlantEvidence(
        clue_id="CLUE_SEAL_CORROSION", scene_id="CH002_S00",
        span="Vết nứt trên vách bọc lõi kéo dài", carrier_used="setting")]).stamp(2)
    notes = assign_scenes(d, SCENES)
    assert d.plant_evidence[0].scene_id == "CH002_S04"
    assert notes[0]["reason"] == "plant_scene_reassigned"


def test_su_that_vinh_vien_cua_canh_hoi_uc_ghi_o_tick_hoi_uc():
    """Đúng triệu chứng thật: trước bản sửa, `attested_present` là [20028]."""
    e = build_engines(FakeLLM(), db_path=":memory:")
    try:
        e.store.put_frame(ContinuityFrame(
            scene_id="CH002_S00", locations={"CHAR_KAELEN": "LOC_ORE_PORT"},
            time=StoryTime(epoch_tick=20028, duration_ticks=2, narrative_order=7)))
        e.store.put_frame(ContinuityFrame(
            scene_id="CH002_S02", locations={"CHAR_KAELEN": "LOC_REACTOR_3"},
            time=StoryTime(epoch_tick=2480, duration_ticks=4, narrative_order=9,
                           mode="flashback", anchor_scene="CH001_S01")))
        d = StateDelta(assertions=[_a(BI_MAT, scene=0)]).stamp(2)
        assign_scenes(d, SCENES)
        assert reconcile(d, e)["status"] == "committed"
        w = e.graph.attribute_window("CHAR_KAELEN", "knows_secret")
        assert w.attested_present == [2480]
    finally:
        e.store.close()


class _KhaiSaiCanh(FakeLLM):
    """Lượt 1 trích đúng một câu của cảnh S03 nhưng khai `scene=0`."""

    def invoke(self, prompt: str, *, role: str = "") -> str:
        if role == "extractor_diff":
            self.calls.append({"role": role, "prompt": prompt})
            sec = prompt.split("[CH001_S03]\n", 1)[1].split("\n\n[CH001_S04]", 1)[0]
            cau = next(s.strip() for s in re.split(r"(?<=[.!?])\s+", sec)
                       if "Vhal" in s and len(s) > 40)
            return json.dumps({"assertions": [{
                "subject": "CHAR_VHAL", "predicate": "rank", "object": "quản hạt",
                "chapter": 1, "scene": 0, "span": cau, "confidence": 0.8}]},
                ensure_ascii=False)
        return super().invoke(prompt, role=role)


def test_extractor_gan_lai_canh_va_ghi_vao_extraction_issues():
    e = build_engines(_KhaiSaiCanh(), db_path=":memory:")
    try:
        st = run_chapter(e, 1)
        assert not st.get("escalated"), st.get("escalation_reason")
        d = StateDelta.model_validate(st["delta"])
        a = next(x for x in d.assertions if x.subject == "CHAR_VHAL")
        assert a.scene == 3
        assert any(x["stage"] == "scene" and x["reason"] == "scene_reassigned"
                   for x in d.extraction_issues)
    finally:
        e.store.close()
