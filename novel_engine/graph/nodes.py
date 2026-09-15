"""Các node của LangGraph (§9.2).

GĐ1 có BA node: director → writer → scene_boundary → (vòng lại writer | END).
Auditor, Polish, Extractor, Reconcile thuộc GĐ2–GĐ3.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from novel_engine.canon.models import StateDelta
from novel_engine.canon.timeline import ContinuityFrame, SceneClose, StoryTime
from novel_engine.character.models import CLICHE_SOMATICS
from novel_engine.graph.safety import safe_node
from novel_engine.graph.state import ChapterState
from novel_engine.llm.json_io import merge_json, parse_model
from novel_engine.memory.assembler import render_facts
from novel_engine.planner.beats import build_beats
from novel_engine.planner.contract import CharacterInScene, SceneContract
from novel_engine.planner.tension import tension_directive
from novel_engine.planner.timeline_alloc import allocate_scene_times
from novel_engine.prompts import (
    DIRECTOR_TMPL, EXTRACT_DIFF_TMPL, EXTRACT_EMERGENT_TMPL,
    SCENE_DIGEST_TMPL, WRITER_TMPL,
)
from novel_engine.reconcile.commit import reconcile
from novel_engine.reconcile.verify import (
    merge_extractions, plan_coverage, verify_spans,
)

SCENES_PER_CHAPTER = 6


@safe_node
def director_node(state: ChapterState, config: RunnableConfig) -> dict:
    """KHÔNG dùng LLM để quyết định logic. LLM chỉ diễn giải thành beat sheet.

    F4: lập lịch manh mối MỘT LẦN cho cả chương, không gọi trong vòng lặp cảnh.
    Trạng thái manh mối chỉ đổi ở `reconcile_node` (cuối chương), nên gọi 6 lần
    sẽ trả về CÙNG 3 manh mối và cả 6 cảnh nhận lệnh cài cùng 3 thứ đó — vỡ
    `ATTENTION_BUDGET` gấp 6. (ForeshadowScheduler thuộc GĐ3; chỗ nối đã sẵn.)
    """
    eng = config["configurable"]["engines"]
    ch = state["chapter"]
    total = state.get("total_chapters") or eng.total_chapters

    tension = tension_directive(ch, total, eng.store.measured_tension(ch - 1))

    chapter_plants: list[dict] = []          # ← ForeshadowScheduler (GĐ3)
    clue_escalations: list[dict] = []

    beats = build_beats({"tension": tension,
                         "plant_directives": chapter_plants},
                        n=SCENES_PER_CHAPTER)
    # Truyền địa lý vào bộ cấp phát: nó phải tự cộng thời gian đi đường, nếu
    # không `no_teleport` (§3.6.1) dựng blocker ở mọi cảnh đổi địa điểm —
    # báo động giả, và P2 (§0.1) nói báo động giả dẫn tới việc tắt luật.
    scene_locs = [eng.planner.location_id(ch, si) for si in range(len(beats))]
    times = allocate_scene_times(ch, beats, eng.store,
                                 overrides=eng.planner.time_overrides(ch),
                                 locations=scene_locs, graph=eng.graph,
                                 epoch_start=eng.epoch_start)

    contracts: list[dict] = []
    for si, beat in enumerate(beats):
        pov = eng.planner.pov_for(ch, si)
        present = eng.planner.present_characters(ch, si)
        epoch_tick = times[si].epoch_tick

        chars: list[CharacterInScene] = []
        for cid in present:
            prof = eng.chars.get(cid)
            if prof is None:
                continue
            chars.append(CharacterInScene(
                id=cid, name=prof.name,
                immediate_goal=eng.planner.goal_of(cid, ch, si),
                secret_fear=(prof.private_knowledge[0].proposition
                             if prof.private_knowledge else None),
                hidden_action=eng.planner.hidden_action(cid, ch, si),
                deliberation={},                      # deliberate() — GĐ2
                voice_reminder=prof.voice.model_dump(
                    include={"register", "signature_lexicon",
                             "forbidden_lexicon", "syntactic_tic",
                             "under_stress_shift"}),
                somatic_allowed=prof.somatic_signature,
                somatic_forbidden=CLICHE_SOMATICS,
                # NT-6: khoá theo epoch_tick của CHÍNH cảnh này, không theo
                # chương — nếu không, nhân vật trong hồi ức "biết" mọi thứ đã
                # học được suốt 8.600 tick sau đó.
                knows_in_this_scene=eng.firewall.known_ids(cid, epoch_tick),
                must_not_reveal=eng.planner.secrets_of(cid, ch),
            ))

        pov_prof = eng.chars.get(pov)
        raw = SceneContract(
            scene_id=f"CH{ch:03d}_S{si:02d}", chapter=ch, scene_index=si,
            time=times[si],
            location=eng.planner.location(ch, si),
            location_id=eng.planner.location_id(ch, si),
            pov_character=pov,
            pov_character_name=pov_prof.name if pov_prof else pov,
            pov_knowledge_boundary=eng.firewall.boundary(pov, epoch_tick),
            active_characters=chars,
            dramatic_question=beat["function"],
            tension=tension,
            plant_directives=[d for d in [beat.get("clue_slot")] if d],
            lore_integration=eng.graph.faction_tensions(
                eng.planner.location_id(ch, si), ch),
            subtext_requirement="",
            forbidden_cliches=list(CLICHE_SOMATICS[:5]),
        ).model_dump()

        # LLM chỉ điền bốn trường tự sự đang RỖNG. `merge_json` chặn mọi nỗ
        # lực ghi đè `time`, `scene_id`, `plant_directives` (NT-12).
        filled = eng.llm.invoke(DIRECTOR_TMPL.format_map({
            "contract": _contract_brief(raw),
            "beat_function": beat["function"],
            "outline": state.get("outline_beat", ""),
            "debt": "(chưa có — ForeshadowScheduler thuộc GĐ3)",
        }), role="director")
        contracts.append(merge_json(raw, filled))

    return {"contracts": contracts, "beats": beats,
            "scene_index": 0, "revision_count": 0,
            "findings": [], "max_severity": "note",
            "clue_escalations": clue_escalations}


@safe_node
def writer_node(state: ChapterState, config: RunnableConfig) -> dict:
    eng = config["configurable"]["engines"]
    c = state["contracts"][state["scene_index"]]
    # E1: nguồn DUY NHẤT của mốc thời gian cảnh này.
    epoch_tick = c["time"]["epoch_tick"]

    ctx = eng.assembler.build(
        chapter=state["chapter"], scene_idx=state["scene_index"],
        pov_id=c["pov_character"],
        present=[x["id"] for x in c["active_characters"]],
        location_id=c["location_id"] or c["location"], epoch_tick=epoch_tick)

    # §5.4.1 — HAI bộ lọc, không phải một. Gộp chúng là C4: lớp 3 duyệt danh
    # sách rỗng và thành mã chết, còn contract đi thẳng vào prompt không qua
    # bộ lọc nào.
    ctx = eng.firewall.filter_memory(ctx, c["pov_character"], epoch_tick)
    c_safe = eng.firewall.filter_scene_contract(c, c["pov_character"])

    feedback = ""
    if state.get("findings"):
        blockers = [f for f in state["findings"]
                    if f.get("severity") in ("blocker", "major")]
        if blockers:
            feedback = "## PHẢN HỒI CẦN SỬA\n" + "\n".join(
                f"- [{f['severity']}] {f.get('message', '')}" for f in blockers)

    prose = eng.llm.invoke(WRITER_TMPL.format_map({
        "pov_name": c.get("pov_character_name") or c["pov_character"],
        "pov_id": c["pov_character"],
        "knowledge_boundary": _bullets(c["pov_knowledge_boundary"]),
        "hypotheses": _bullets(
            [f"{h['text']} (chắc chắn {h['certainty']:.0%})"
             for h in ctx.get("hypotheses", [])]),
        "active_characters": _characters_brief(c_safe["active_characters"]),
        "dramatic_question": c["dramatic_question"],
        "entry_state": c["entry_state"], "exit_state": c["exit_state"],
        "scene_must_change": c["scene_must_change"],
        "tension_mode": c["tension"].get("mode", ""),
        "tension_note": c["tension"].get("note", ""),
        "pressure_type": c["tension"].get("pressure_type", ""),
        "plant_directives": _bullets(
            [str(d) for d in c["plant_directives"]]) or "(không có)",
        "relationship_directives": _bullets(
            [str(d) for d in c["relationship_directives"]]) or "(không có)",
        "word_min": c["word_budget"][0], "word_max": c["word_budget"][1],
        "max_explicit_goal_statements": c["max_explicit_goal_statements"],
        "sensory_channels_required": c["sensory_channels_required"],
        "subtext_requirement": c["subtext_requirement"],
        "forbidden_cliches": ", ".join(c["forbidden_cliches"]),
        "recent_scenes": _bullets(ctx["recent_scenes"]) or "(chưa có)",
        "recent_chapters": _bullets(ctx["recent_chapters"]) or "(chưa có)",
        "arc_history": _bullets(ctx["arc_history"]) or "(chưa có)",
        "known_facts": _bullets(render_facts(ctx["known_facts"])) or "(chưa có)",
        "feedback": feedback,
    }), role="writer")

    # F3: §9.4 có ghi chú về việc này nhưng thân hàm ở §9.2 thì không làm — và
    # thiếu nó thì `after_audit` không bao giờ chạm MAX_REVISIONS, vòng
    # writer↔auditor chạy vô tận. Đúng đắn được nhờ `scene_boundary_node`
    # reset `findings` về [] ở mỗi ranh giới cảnh (C6).
    return {"current_draft": prose,
            "revision_count": state.get("revision_count", 0)
                              + (1 if state.get("findings") else 0)}


@safe_node
def scene_boundary_node(state: ChapterState, config: RunnableConfig) -> dict:
    """Chốt sổ một cảnh. Node này tồn tại vì BỐN lý do, và cả bốn đều là thứ
    bản trước đánh rơi:

    1. TĂNG `scene_index`. Router LangGraph là hàm thuần — nó KHÔNG đổi được
       state (NT-9). Không node nào tăng biến này thì đồ thị viết lại Cảnh 0
       vô hạn (C1).
    2. RESET `revision_count`. Không reset thì Cảnh 0 tốn 2 lượt sửa sẽ khiến
       Cảnh 1 bị escalate ngay ở lỗi `major` đầu tiên (C6).
    3. Sinh SCENE DIGEST (L1, §4.1) — §4.1 khai là có, nhưng không node nào
       sinh ra nó, nên Context Assembler đọc vào một kho rỗng.
    4. Sinh CONTINUITY FRAME (§10.1) — cũng vậy: các luật liên tục không có
       dữ liệu để chạy.
    """
    eng = config["configurable"]["engines"]
    idx = state["scene_index"]
    c = state["contracts"][idx]
    # GĐ1 chưa có Polish, nên lấy thẳng bản nháp. GĐ2 sẽ đổi sang `polished`.
    prose = state.get("polished") or state["current_draft"]

    valid_locs = eng.graph.location_ids()
    raw = eng.llm.invoke(SCENE_DIGEST_TMPL.format_map({
        "scene_id": c["scene_id"],
        "epoch_tick": c["time"]["epoch_tick"],
        "planned_duration": c["time"]["duration_ticks"],
        "location_id": c["location_id"] or c["location"],
        "valid_locations": "\n".join(f"- {x}" for x in valid_locs),
        # Mã ĐỊNH DANH trước, tên chỉ là chú thích. §9.2 truyền tên trần vào
        # đây, trong khi `character_track(cid, …)` (§3.6.1) tra bằng char_id —
        # khoá `locations` bằng tên làm MỌI luật liên tục thành mã chết, không
        # ném lỗi, chỉ không khớp mãi mãi (NT-8).
        "present": ", ".join(f"{x['id']} ({x['name']})"
                             for x in c["active_characters"]),
        "prose": prose,
    }), role="scene_digest")
    close = parse_model(raw, SceneClose, repair_llm=eng.llm)
    close, coerced = _coerce_continuity(close, c, valid_locs)

    # F2: `StoryTime` do CODE lắp. Model chỉ đóng góp `actual_duration_ticks` —
    # thứ duy nhất phải đọc văn bản mới biết. `narrative_order` không có trong
    # schema mà model phải điền, nên model không thể quên (NT-13).
    frame = ContinuityFrame(
        scene_id=c["scene_id"],
        time=StoryTime.model_validate(c["time"]).model_copy(
            update={"duration_ticks": close.actual_duration_ticks}),
        **close.continuity.model_dump())

    eng.store.put_scene_digest(state["chapter"], idx, close.digest,
                               scene_id=c["scene_id"])
    eng.store.put_frame(frame)

    contracts, shifted = _reflow_after_drift(
        state["contracts"], idx, close.actual_duration_ticks)

    return {
        "contracts": contracts,
        "time_drift": shifted,
        "scene_index": idx + 1,
        "revision_count": 0,
        "findings": [],
        "max_severity": "note",
        "polished": "",
        "scene_outputs": [{"scene_id": c["scene_id"], "prose": prose,
                           "digest": close.digest, "coerced": coerced}],
        "frames": [frame.model_dump()],
        "unresolved": close.unresolved,
    }


def _reflow_after_drift(contracts: list[dict], idx: int,
                        actual: int) -> tuple[list[dict], list[str]]:
    """Đẩy các cảnh SAU khi thời lượng thực lệch thời lượng dự kiến.

    Lỗ hổng bắt được ở lượt chạy Gemini đầu tiên của Chương 2. Outline viết
    cảnh "chịu đựng MƯỜI BỐN GIỜ không có việc gì làm", model báo đúng
    `actual_duration_ticks = 14`, nhưng `BEAT_DURATION_TICKS` chỉ dành 3 tick
    cho beat ấy. Frame ghi thời lượng THỰC (14) nên cảnh kết thúc ở tick
    20.044 — trong khi cảnh kế tiếp đã được cấp phát từ trước ở 20.040. Hai
    cảnh chồng lấn, và `no_teleport` dựng blocker đúng.

    §12.4 nói "lệch nhiều so với dự kiến là thông tin hữu ích, không phải
    lỗi", nhưng không mục nào nói phải LÀM GÌ với thông tin đó — nó được ghi
    vào frame rồi nằm im. NT-14 trả lời: văn xuôi đã viết ra là sự thật, canon
    phải khớp với trang giấy. Nên kế hoạch nhường, không phải trang giấy nhường.

    Chỉ đẩy cảnh `present`. Hồi ức và cảnh song song neo vào chỗ khác trên
    trục epoch; đẩy chúng theo là phá chính cái neo đó.
    """
    planned = contracts[idx]["time"]["duration_ticks"]
    delta = actual - planned
    if delta == 0 or idx + 1 >= len(contracts):
        return contracts, []

    out = [dict(x) for x in contracts]
    notes: list[str] = []
    for later in out[idx + 1:]:
        t = later["time"]
        if t["mode"] != "present":
            continue
        t = dict(t)
        t["epoch_tick"] += delta
        later["time"] = t
        notes.append(later["scene_id"])
    if notes:
        notes = [f"cảnh {contracts[idx]['scene_id']} dài {actual} tick thay vì "
                 f"{planned} → đẩy {len(notes)} cảnh sau thêm {delta:+d}"]
    return out, notes


def _coerce_continuity(close: SceneClose, contract: dict,
                       valid_locs: list[str]) -> tuple[SceneClose, list[str]]:
    """Ép `continuity` về đúng canon trước khi nó thành ContinuityFrame.

    Lượt chạy THẬT đầu tiên (Gemini, Chương 1) cho thấy model bịa ra mã địa
    điểm nghe rất hợp lý — `LOC_QUAY_TIEP_TAN`, `LOC_BUONG_DEM`,
    `LOC_DOCK_EAST` — cho những góc nhỏ mà văn xuôi nhắc tới. Không mã nào tồn
    tại trong route graph, nên `travel_ticks` trả `None` và `no_teleport` dựng
    blocker ở mọi ranh giới cảnh. `FakeLLM` không bắt được vì nó chỉ chép lại
    mã có sẵn trong prompt.

    NT-5: không có bằng chứng khớp được thì KHÔNG có thay đổi trạng thái. Địa
    điểm của cảnh là quyết định của Director (NT-12), không phải của model —
    nên mã lạ bị quy về địa điểm trong hợp đồng thay vì được nhận vào canon.
    Cùng nguyên tắc "code thắng LLM" như `merge_json`.

    Prompt đã liệt kê tập đóng, nhưng prompt là lời khuyên còn đây là hàng
    rào: một quy tắc chỉ được kiểm ở một nơi thì nơi đó sẽ quên (NT-11), và
    với LLM thì "nơi đó" là chính model.
    """
    valid = set(valid_locs)
    scene_loc = contract["location_id"] or contract["location"]
    allowed_chars = {x["id"] for x in contract["active_characters"]}
    notes: list[str] = []

    locations: dict[str, str] = {}
    for cid, loc in close.continuity.locations.items():
        if cid not in allowed_chars:
            notes.append(f"bỏ nhân vật ngoài hợp đồng: {cid}")
            continue
        if loc not in valid:
            notes.append(f"{cid}: mã địa điểm lạ '{loc}' → quy về {scene_loc}")
            loc = scene_loc
        locations[cid] = loc

    # Nhân vật có trong hợp đồng mà model quên khai vị trí: điền địa điểm cảnh.
    # Thiếu họ thì `character_track` đứt quãng và `no_teleport` đọc hai cảnh
    # không liền nhau như thể chúng liền nhau.
    for cid in allowed_chars - set(locations):
        notes.append(f"{cid}: model không khai vị trí → điền {scene_loc}")
        locations[cid] = scene_loc

    obs = close.continuity.model_copy(update={
        "locations": locations,
        "injuries": {k: v for k, v in close.continuity.injuries.items()
                     if k in allowed_chars},
        "possessions": {k: v for k, v in close.continuity.possessions.items()
                        if k in allowed_chars},
    })
    return close.model_copy(update={"continuity": obs}), notes


# ───────────────────── kết xuất prompt ─────────────────────

def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items if x)


def _contract_brief(raw: dict) -> str:
    """Chỉ đưa cho Director phần nó CẦN. Đưa cả contract là mời model ghi đè
    những trường mà `merge_json` rồi sẽ chặn — tốn token cho một cuộc giằng co
    đã biết trước kết quả."""
    return "\n".join([
        f"scene_id: {raw['scene_id']}",
        f"POV: {raw['pov_character_name']} ({raw['pov_character']})",
        f"Địa điểm: {raw['location_id']}",
        f"Có mặt: {', '.join(x['name'] for x in raw['active_characters'])}",
        f"Mục tiêu: " + "; ".join(
            f"{x['name']}: {x['immediate_goal']}"
            for x in raw["active_characters"] if x["immediate_goal"]),
        f"Nhịp: {raw['tension'].get('mode')} — {raw['tension'].get('note')}",
    ])


def _characters_brief(chars: list[dict]) -> str:
    """Kết xuất nhân vật SAU khi qua `filter_scene_contract`. Các trường nội
    tâm của nhân vật không phải POV đã bị lột ở đó — ở đây chỉ định dạng."""
    out = []
    for ch in chars:
        lines = [f"### {ch.get('name')} ({ch['id']})"]
        if ch.get("immediate_goal"):
            lines.append(f"- mục tiêu ngay lúc này: {ch['immediate_goal']}")
        if ch.get("voice_reminder"):
            v = ch["voice_reminder"]
            lines.append(f"- giọng: {v.get('register')}; tật cú pháp: "
                         f"{v.get('syntactic_tic')}")
            lines.append(f"- từ hay dùng: {', '.join(v.get('signature_lexicon', [])[:6])}")
            lines.append(f"- từ CẤM: {', '.join(v.get('forbidden_lexicon', [])[:6])}")
            if v.get("under_stress_shift"):
                lines.append(f"- dưới áp lực: {v['under_stress_shift'].strip()}")
        if ch.get("somatic_allowed"):
            lines.append(f"- phản ứng cơ thể được dùng: {', '.join(ch['somatic_allowed'])}")
        if ch.get("somatic_forbidden"):
            lines.append(f"- somatic CẤM: {', '.join(ch['somatic_forbidden'][:5])}")
        if ch.get("observable_behavior"):
            lines.append(f"- POV quan sát được: {ch['observable_behavior']}")
        if ch.get("director_only"):
            d = ch["director_only"]
            lines.append(f"- [CHỈ ĐẠO DIỄN BIẾT] hidden_action: {d['hidden_action']}")
            lines.append(f"  RÀNG BUỘC CỨNG: {d['_hard_constraint']}")
        if ch.get("must_not_reveal"):
            lines.append(f"- tuyệt đối không nói ra: {', '.join(ch['must_not_reveal'])}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


@safe_node
def extractor_node(state: ChapterState, config: RunnableConfig) -> dict:
    """Luồng ghi ngược. BA LƯỢT, không phải một (§10.5).

    Yêu cầu một LLM đọc 3.000 từ rồi tự nhớ và liệt kê MỌI thực thể, quan hệ
    và chuyển trạng thái là tác vụ có tỉ lệ sót cao một cách có hệ thống: đây
    là bài toán *recall* trên không gian mở, không danh sách kiểm, không tín
    hiệu dừng. Model dừng khi thấy "đã đủ", và "đủ" biến thiên mỗi lần chạy.

        lượt 1  prose + contract      chống SÓT mục có kế hoạch        LLM
        lượt 2  prose, KHÔNG contract chống SÓT mục ngoài kế hoạch     LLM
        lượt 3  khớp chuỗi            chống BỊA (ảo giác xác nhận)      0

    Lượt 1 chữa lỗi sót nhưng MỞ RA lỗi ngược — hỏi "manh mối X đã cài chưa?"
    thì model có xu hướng mạnh trả lời *rồi* và bịa span nghe hợp lý. Lượt 3
    đóng lại đúng lỗ đó, với chi phí bằng 0.

    C2 ĐÃ SỬA: bản trước dùng `state["polished"]`, vốn chỉ chứa văn xuôi của
    CẢNH VỪA XONG, trong khi `contracts` chứa cả 6 cảnh. `verify_spans` đi tìm
    span của cảnh 1–5 trong văn bản cảnh 6, không thấy, và `plan_coverage` xoá
    sạch manh mối của 5 cảnh đầu. Toàn chương chỉ còn lại manh mối cảnh cuối.
    """
    eng = config["configurable"]["engines"]
    scenes = state["scene_outputs"]
    if not scenes:
        return {"delta": StateDelta().stamp(state["chapter"]).model_dump(),
                "extraction_report": {"skipped": "không có cảnh nào"}}

    full_prose = "\n\n".join(s["prose"] for s in scenes)
    contracts = state["contracts"]

    # LƯỢT 1 — KIỂM TOÁN VI SAI, có SceneContract làm hệ quy chiếu.
    # Truyền văn xuôi ĐÃ GẮN NHÃN CẢNH, để `PlantEvidence.scene_id` là DỮ LIỆU
    # chứ không phải phỏng đoán của model.
    labelled = "\n\n".join(f"[{s['scene_id']}]\n{s['prose']}" for s in scenes)
    vocab = getattr(eng, "vocabulary", None)
    predicates = (vocab.render_for_prompt() if vocab is not None
                  else "(bible chưa khai predicates.yaml — dùng khoá tiếng Anh "
                       "snake_case cho thuộc tính bền)")
    audited = eng.llm.invoke(EXTRACT_DIFF_TMPL.format_map({
        "contracts": _contracts_brief_for_extract(contracts),
        "chapter": state["chapter"],
        "prose": labelled,
        "predicates": predicates,
    }), role="extractor_diff")

    # LƯỢT 2 — QUÉT PHÁT SINH. KHÔNG đưa contract vào: model bị neo vào kế
    # hoạch sẽ bỏ qua đúng những thứ nằm ngoài kế hoạch.
    emergent = eng.llm.invoke(EXTRACT_EMERGENT_TMPL.format_map({
        "chapter": state["chapter"],
        "prose": full_prose,
        "known_entities": _entity_index_brief(eng.graph.entity_index()),
        "predicates": predicates,
    }), role="extractor_emergent")

    dropped: list[dict] = []
    delta = merge_extractions(audited, emergent, chapter=state["chapter"],
                              repair_llm=eng.llm, dropped=dropped)
    if vocab is not None:
        # Quy bí danh về tên chuẩn TRƯỚC khi sinh khoá phân loại (khoá chứa
        # `predicate`). Việc đổi tên được ghi lại, không làm im lặng.
        from novel_engine.canon.vocabulary import normalize_predicates
        dropped.extend(normalize_predicates(delta, vocab))

    # LƯỢT 3 — XÁC MINH SPAN trên TOÀN BỘ văn xuôi của chương.
    delta, rejected = verify_spans(delta, full_prose)
    # Chỉ số cảnh là metadata của hệ thống (NT-13): suy từ VỊ TRÍ span trong
    # văn xuôi từng cảnh, không tin số model khai. Sai cảnh nghĩa là sai mốc
    # epoch, và mệnh đề của cảnh hồi ức thoát khỏi kiểm tra hồi ức.
    from novel_engine.reconcile.verify import assign_scenes
    scene_notes = assign_scenes(delta, scenes)
    coverage = plan_coverage(contracts, delta, known_clues=set(eng.graph.clues))

    issues = ([{"stage": "parse", **i} for i in dropped]
              + [{"stage": "verify", **r} for r in rejected]
              + [{"stage": "scene", **n} for n in scene_notes]
              + [{"stage": "plan", "reason": "phantom_plant_evidence", **p}
                 for p in coverage["phantom_evidence"]])
    so_tu = len(full_prose.split())
    # Lượt Gemini thật ở Chương 2: 4.692 từ văn xuôi → 0 mệnh đề, 0 thực thể,
    # và delta rỗng được GHI vào canon không một lời cảnh báo. Delta rỗng không
    # làm bẩn canon, nhưng làm MẤT sự thật một cách im lặng.
    if (so_tu >= EMPTY_EXTRACTION_MIN_WORDS and not delta.assertions
            and not delta.new_entities and not delta.new_relations):
        issues.append({"stage": "recall", "reason": "empty_extraction",
                       "message": (f"{so_tu} từ văn xuôi nhưng không trích được "
                                   f"mệnh đề, thực thể hay quan hệ nào — xem "
                                   f"phản hồi thô trong báo cáo chương")})
    delta.extraction_issues = issues

    return {
        "delta": delta.model_dump(),
        "extraction_report": {
            "rejected_spans": rejected,
            "dropped_items": dropped,
            "assertions_kept": len(delta.assertions),
            "new_entities": len(delta.new_entities),
            "new_relations": len(delta.new_relations),
            # Phản hồi thô của model — không có nó thì một lượt trích xuất rỗng
            # là không chẩn đoán được: không biết model trả rỗng hay parse vứt.
            "raw": {"luot_1": audited[:RAW_KEEP_CHARS],
                    "luot_2": emergent[:RAW_KEEP_CHARS]},
            **coverage,
        },
    }


EMPTY_EXTRACTION_MIN_WORDS = 500
RAW_KEEP_CHARS = 8000


def _contracts_brief_for_extract(contracts: list[dict]) -> str:
    """Chỉ đưa phần Extractor cần đối chiếu. Nhét cả contract vào là đốt token
    cho `deliberation`, `voice_reminder` và `somatic_*` — những thứ không liên
    quan gì tới việc kiểm toán xem kế hoạch có được thực hiện không."""
    out = []
    for c in contracts:
        lines = [f"[{c['scene_id']}] POV {c['pov_character']} @ "
                 f"{c.get('location_id') or c['location']}",
                 f"  phải thay đổi: {c.get('scene_must_change', '')}"]
        # Luôn ghi dòng này, KỂ CẢ khi rỗng. Bản trước bỏ dòng khi không có
        # manh mối, và Gemini tự bịa `plant_evidence` với clue_id là mã cảnh
        # (`CH002_S00`) cho cả sáu cảnh — model không phân biệt được "kế hoạch
        # không có manh mối" với "kế hoạch không nói gì về manh mối".
        if c.get("plant_directives"):
            lines.append(f"  plant_directives: {c['plant_directives']}")
        else:
            lines.append("  plant_directives: (không có)")
        if c.get("relationship_directives"):
            lines.append(f"  relationship_directives: {c['relationship_directives']}")
        co_mat = []
        for x in c["active_characters"]:
            m = f"{x['id']}"
            if x.get("must_not_reveal"):
                m += f" (không được lộ: {', '.join(x['must_not_reveal'])})"
            co_mat.append(m)
        lines.append(f"  có mặt: {', '.join(co_mat)}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def _entity_index_brief(index: list[dict]) -> str:
    return "\n".join(
        f"- {e['id']} ({e['kind']}): {e['name']}"
        + (f" — còn gọi: {', '.join(e['aliases'])}" if e.get("aliases") else "")
        for e in index)


@safe_node
def reconcile_node(state: ChapterState, config: RunnableConfig) -> dict:
    """Ghi canon ngay trong đồ thị (§11). Chỉ được nối khi `auto_commit=True`.

    CLI mặc định KHÔNG dùng node này: CP-2 (§16.1) đòi tác giả thấy delta
    TRƯỚC khi bất cứ thứ gì vào canon vĩnh viễn, nên CLI tách `write` /
    `review` / `commit`. Node và CLI cùng gọi `reconcile()` — một luồng ghi.
    """
    eng = config["configurable"]["engines"]
    delta = StateDelta.model_validate(state["delta"])
    res = reconcile(delta, eng, frames=state.get("frames", []),
                    contracts=state.get("contracts", []))
    upd = {
        "delta": delta.model_dump(),
        "irony_seeds": res["classify"].get("irony_seeds", []),
        "reconcile_report": {
            "status": res["status"], "applied": res["applied"],
            "plan_patches": res["plan_patches"], "flashback": res["flashback"],
            "summary": res["classify"].get("summary", {}),
            "escalation_reason": res["escalation_reason"],
        },
    }
    if res["status"] == "escalated":
        upd["escalated"] = True
        upd["escalation_reason"] = res["escalation_reason"]
    return upd
