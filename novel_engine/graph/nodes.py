"""Các node của LangGraph (§9.2).

Vòng một cảnh: writer → auditor → (writer | polish) → scene_boundary. Phần
QUYẾT ĐỊNH của Auditor và Polish nằm ở `audit/critique.py` (hàm thuần); node ở
đây chỉ gọi LLM và ghi state.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from novel_engine.audit.critique import (
    audit_checklist, clean_polish_output, content_drifted, new_serious,
    parse_llm_findings, polish_notes, routing_severity, writer_feedback,
)
from novel_engine.audit.deterministic import deterministic_audit
from novel_engine.audit.text_hygiene import sanitize_prose
from novel_engine.audit.tics import (COUNTING_PER_CHAPTER, FIGURE_PER_CHAPTER,
                                     SOMATIC_PER_CHAPTER, _FIGURE,
                                     _count_gesture, _nfc_lower,
                                     counting_speech)
from novel_engine.audit.prose import CLICHE_PHRASES
from novel_engine.canon.models import StateDelta
from novel_engine.canon.timeline import ContinuityFrame, SceneClose, StoryTime
from novel_engine.character.models import CLICHE_SOMATICS
from novel_engine.graph.safety import safe_node
from novel_engine.graph.state import ChapterState
from novel_engine.llm.json_io import merge_json, parse_model
from novel_engine.memory.assembler import render_facts
from novel_engine.foreshadow.debt import narrative_debt_report
from novel_engine.eval.tension_measure import (
    chapter_summary, measure_tension,
)
from novel_engine.foreshadow.scheduler import ForeshadowScheduler, SceneSlot
from novel_engine.planner.beats import QUIET_BEATS, build_beats
from novel_engine.planner.contract import CharacterInScene, SceneContract
from novel_engine.planner.tension import tension_directive
from novel_engine.planner.timeline_alloc import allocate_scene_times
from novel_engine.prompts import (
    AUDITOR_TMPL, DIRECTOR_TMPL, EXTRACT_DIFF_TMPL, EXTRACT_EMERGENT_TMPL,
    POLISH_TMPL, SCENE_DIGEST_TMPL, WRITER_TMPL,
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
    `ATTENTION_BUDGET` gấp 6. Lập lịch một lần, nhưng XẾP CẢNH theo POV và vật
    mang của từng cảnh (xem `foreshadow/scheduler.py`).
    """
    eng = config["configurable"]["engines"]
    ch = state["chapter"]
    total = state.get("total_chapters") or eng.total_chapters

    tension = tension_directive(ch, total, eng.store.measured_tension(ch - 1))

    beats = build_beats({"tension": tension, "plant_directives": []},
                        n=SCENES_PER_CHAPTER)
    # POV và vật mang là của CẢNH. Scheduler xếp cảnh luôn — không để Director
    # lọc lại theo POV rồi làm rơi manh mối không ai ghi nhận (§9.2 F4).
    slots = [SceneSlot(index=si, pov=eng.planner.pov_for(ch, si),
                       affordances=eng.planner.scene_affordances(ch, si),
                       quiet=beat["function"] in QUIET_BEATS)
             for si, beat in enumerate(beats)]
    plan = ForeshadowScheduler(eng.graph).schedule(ch, slots)
    for d in plan.directives:
        beats[d.scene_index]["clue_slot"] = d.to_contract()
    # F5/NT-17: escalation manh mối là quyết định CP-4 — đi vào báo cáo, không
    # dừng chương.
    clue_escalations = plan.escalations
    debt = narrative_debt_report(eng.graph.clues, [], ch)
    # Truyền địa lý vào bộ cấp phát: nó phải tự cộng thời gian đi đường, nếu
    # không `no_teleport` (§3.6.1) dựng blocker ở mọi cảnh đổi địa điểm —
    # báo động giả, và P2 (§0.1) nói báo động giả dẫn tới việc tắt luật.
    scene_locs = [eng.planner.location_id(ch, si) for si in range(len(beats))]
    times = allocate_scene_times(ch, beats, eng.store,
                                 overrides=eng.planner.time_overrides(ch),
                                 locations=scene_locs, graph=eng.graph,
                                 epoch_start=eng.epoch_start)

    # Quan hệ: mỗi cặp một chỉ thị, ở cảnh HIỆN TẠI cuối cùng cả hai có mặt.
    rel_by_scene = eng.graph.relationships.directives_for_chapter(
        ch, eng.planner, eng.chars, [t.mode for t in times])
    # Tin tức (§5.6): tin nào chạm tới POV của từng cảnh, theo vị trí THẬT của
    # các chương đã viết cộng vị trí DỰ KIẾN của chương này.
    news_by_scene = eng.graph.news.scene_directives(
        graph=eng.graph, chars=eng.chars, past_frames=eng.store.get_frames(),
        planned=[{"scene_id": f"CH{ch:03d}_S{si:02d}", "time": times[si],
                  "location_id": eng.planner.location_id(ch, si),
                  "present": eng.planner.present_characters(ch, si),
                  "pov": eng.planner.pov_for(ch, si)}
                 for si in range(len(beats))])

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
                # `signature_lexicon` KHÔNG còn đi vào prompt: nó là thước
                # đo (M3, ngân sách tật ngôn ngữ), và suốt Arc 1 ta vừa đo vừa
                # đưa thước cho người bị đo. Xem `_characters_brief`.
                voice_reminder=prof.voice.model_dump(
                    include={"register", "voice_exemplars",
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
            plant_directives=plan.for_scene(si),
            relationship_directives=rel_by_scene.get(si, []),
            news_directives=news_by_scene.get(si, []),
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
            "debt": _debt_brief(debt),
        }), role="director")
        contract = merge_json(raw, filled)
        # Cảnh N bắt đầu ở nơi cảnh N−1 kết thúc. Không nối lại thì Director để
        # LLM tự nghĩ `entry_state` cho từng cảnh, và Arc 1 cho ra ba dị bản của
        # cùng một cuộc đối thoại trong một chương: không cảnh nào biết cảnh
        # trước đã ngã ngũ ra sao (State Tracker).
        if contracts and contracts[-1].get("exit_state"):
            truoc = contracts[-1]
            contract["entry_state"] = truoc["exit_state"]
            contract["scene_must_change"] = (
                contract.get("scene_must_change")
                or f"khác với trạng thái cuối cảnh trước: {truoc['exit_state']}")
        contracts.append(contract)

    return {"contracts": contracts, "beats": beats,
            "scene_index": 0, "revision_count": 0,
            "findings": [], "max_severity": "note",
            "clue_escalations": clue_escalations,
            "foreshadow_report": {**plan.report(), "debt": debt}}


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
        location_id=c["location_id"] or c["location"], epoch_tick=epoch_tick,
        query=_truy_van_nho_lai(c))

    # §5.4.1 — HAI bộ lọc, không phải một. Gộp chúng là C4: lớp 3 duyệt danh
    # sách rỗng và thành mã chết, còn contract đi thẳng vào prompt không qua
    # bộ lọc nào.
    ctx = eng.firewall.filter_memory(ctx, c["pov_character"], epoch_tick)
    c_safe = eng.firewall.filter_scene_contract(c, c["pov_character"])
    c_safe = _tru_cu_chi_da_can(c_safe, _scenes_truoc(state), eng.chars)

    # Chỉ số nhịp không quay lại Writer (§10.3.2), và lời giải thích của
    # Auditor LLM không vào prompt Writer — nó đọc bí mật NPC (xem critique.py).
    feedback = writer_feedback(state.get("findings") or [])

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
        "plant_directives": "\n".join(
            _plant_brief(d) for d in c["plant_directives"]) or "(không có)",
        "relationship_directives": "\n".join(
            _relationship_brief(d) for d in c["relationship_directives"]) or "(không có)",
        "word_min": c["word_budget"][0], "word_max": c["word_budget"][1],
        "max_explicit_goal_statements": c["max_explicit_goal_statements"],
        "sensory_channels_required": c["sensory_channels_required"],
        "ngan_sach": _ngan_sach_da_dung(_scenes_truoc(state)),
        "next_chapter": _chuong_sau_brief(state, eng, c),
        "subtext_requirement": c["subtext_requirement"],
        "forbidden_cliches": ", ".join(c["forbidden_cliches"]),
        "recent_scenes": _bullets(ctx["recent_scenes"]) or "(chưa có)",
        "callbacks": _bullets(ctx.get("callbacks") or []) or "(không có)",
        "recent_chapters": _bullets(ctx["recent_chapters"]) or "(chưa có)",
        "arc_history": _bullets(ctx["arc_history"]) or "(chưa có)",
        "known_facts": _bullets(render_facts(ctx["known_facts"])) or "(chưa có)",
        "news": "\n".join(_news_brief(d) for d in c.get("news_directives", []))
                or "(không có)",
        "feedback": feedback,
    }), role="writer")

    # Ký tự rác là lỗi ĐÁNH MÁY của model (Arc 1: chữ Cyrillic và Kannada lọt
    # vào giữa từ tiếng Việt), không phải lỗi sáng tác. Dọn bằng code; bắt viết
    # lại 900 từ vì ba ký tự là đốt tiền.
    prose, don_dep = sanitize_prose(prose)

    # F3: §9.4 có ghi chú về việc này nhưng thân hàm ở §9.2 thì không làm — và
    # thiếu nó thì `after_audit` không bao giờ chạm MAX_REVISIONS, vòng
    # writer↔auditor chạy vô tận. Đúng đắn được nhờ `scene_boundary_node`
    # reset `findings` về [] ở mỗi ranh giới cảnh (C6).
    return {"current_draft": prose,
            "hygiene_notes": ([{"scene_id": c["scene_id"], "notes": don_dep}]
                              if don_dep else []),
            "revision_count": state.get("revision_count", 0)
                              + (1 if state.get("findings") else 0)}


def _tru_cu_chi_da_can(contract: dict, truoc: list[dict], chars: dict) -> dict:
    """Cử chỉ đã hết ngân sách trong chương này thì CẤM ở cảnh tiếp theo.

    Ngân sách cử chỉ ở `tics.py` chỉ BÁO sau khi viết, và nó là lỗi `minor` nên
    chỉ đi polish — mà polish không đổi ngôn ngữ cơ thể. Đo trên Arc 1 viết
    lại: 19 lần trên 5 chương, trần là 3 mỗi chương. Luật đúng, nhưng nói vào
    chỗ không ai sửa được.

    Chỗ sửa được là TRƯỚC khi viết: Writer chỉ cần biết cử chỉ nào đã dùng cạn.
    Code quyết ngân sách, prompt phát biểu — LLM diễn đạt chứ không quyết định.
    """
    if not truoc:
        return contract
    van = _nfc_lower(" ".join(x.get("prose", "") for x in truoc))
    ra = dict(contract)
    ds = []
    for x in contract.get("active_characters", []):
        y = dict(x)
        prof = chars.get(x.get("id")) if chars else None
        if prof is not None and y.get("somatic_allowed"):
            can = [g for g in y["somatic_allowed"]
                   if _count_gesture(van, g) >= SOMATIC_PER_CHAPTER]
            if can:
                y["somatic_allowed"] = [g for g in y["somatic_allowed"] if g not in can]
                # Đứng ĐẦU danh sách: `_characters_brief` chỉ in 5 mục cấm
                # đầu tiên, mà riêng sáo ngữ dùng chung đã có 10 — nối vào
                # đuôi thì lệnh cấm này không bao giờ tới được prompt.
                y["somatic_forbidden"] = [
                    f"{g} (đã dùng đủ số lần cho chương này)" for g in can
                ] + list(y.get("somatic_forbidden", []))
        ds.append(y)
    ra["active_characters"] = ds
    return ra


def _truy_van_nho_lai(contract: dict) -> str:
    """Câu truy vấn cho L5, ghép từ những gì cảnh này ĐANG nói tới.

    Địa điểm và người có mặt là hai tín hiệu mạnh nhất cho một callback thật:
    người đọc nhớ lại một chỗ và một người, không nhớ lại một chủ đề.
    """
    phan = [contract.get("location_id") or contract.get("location") or "",
            contract.get("dramatic_question") or "",
            contract.get("scene_must_change") or ""]
    phan += [x.get("name") or x.get("id") or ""
             for x in contract.get("active_characters", [])]
    phan += [d.get("surface_form") or "" for d in contract.get("plant_directives", [])]
    return " ".join(x for x in phan if x)


def _ngan_sach_da_dung(truoc: list[dict]) -> str:
    """Ngân sách chương đã tiêu tới đâu — nói TRƯỚC khi viết, không phải sau.

    `counting_tic` và `figure_overuse` là lỗi `minor` nên chúng đi polish. Đo
    trên lượt viết lại Arc 1: cả hai NỔ ĐÚNG (ch1, ch3, ch4) mà số lượt thoại
    chỉ gồm một con số vẫn y nguyên 8. Polish không sửa được, vì bỏ "— Ba." đi
    thì phải VIẾT một câu thoại thay vào, mà bịa nội dung không phải việc của
    polish.

    Đây là cùng một sai lầm với ngân sách cử chỉ, và cùng một cách chữa: chuyển
    thông tin lên phía TRƯỚC. Code đếm, prompt phát biểu, Writer tự tránh.
    """
    if not truoc:
        return ""
    van = chr(10).join(x.get("prose", "") for x in truoc)
    dong = []
    n = len(counting_speech(van))
    if n >= COUNTING_PER_CHAPTER:
        dong.append(f"- Nhân vật đã đếm thành tiếng {n} lần trong chương này. "
                    f"KHÔNG dùng thêm lượt thoại chỉ gồm một con số; nếu cần "
                    f"một con số thì nó phải dẫn vào một quan sát cụ thể.")
    dem: dict[str, int] = {}
    for m in _FIGURE.finditer(_nfc_lower(van)):
        dem[m.group(0)] = dem.get(m.group(0), 0) + 1
    het = sorted(k for k, v in dem.items() if v >= FIGURE_PER_CHAPTER)
    if het:
        dong.append("- Các con số sau đã dùng hết lượt trong chương này, "
                    "KHÔNG nhắc lại: " + ", ".join(f"“{x}”" for x in het) + ".")
    if not dong:
        return ""
    return ("## NGÂN SÁCH CHƯƠNG ĐÃ TIÊU\n"
            + ("\n").join(dong) + "\n\n")


def _chuong_sau_brief(state: ChapterState, eng, contract: dict) -> str:
    """Cảnh CUỐI chương được biết chương sau mở ra ở đâu — chỉ cảnh cuối.

    Ý mượn từ AI_NovelGenerator: prompt viết chương của họ nhận cả blueprint
    chương hiện tại LẪN chương kế, nên chương không kết thúc vào khoảng không.
    Ta không có gì tương đương: Writer chỉ thấy `exit_state` của chính cảnh
    mình, nên cảnh cuối chương thường đóng lại gọn ghẽ rồi chương sau phải
    khởi động lại từ đầu.

    Hai chỗ thắt chặt hơn bản gốc của họ:

    - CHỈ cảnh cuối nhận. Đưa cho mọi cảnh là mời model kể trước.
    - Nói thẳng đây là thông tin cho NGƯỜI VIẾT, không phải điều POV biết —
      cùng một hàng rào mà `director_only._hard_constraint` dựng cho
      `hidden_action`. Thiếu câu đó thì POV sẽ "linh cảm" về chương sau, và
      đó đúng là rò rỉ tri thức mà §5.4 sinh ra để chặn.
    """
    tong = len(state.get("contracts") or [])
    if tong and state.get("scene_index") != tong - 1:
        return ""
    sau = state["chapter"] + 1
    try:
        beat = eng.planner.outline_beat(sau)
        ten = eng.planner.title(sau)
    except (KeyError, IndexError):
        return ""
    if not beat:
        return ""
    return (f"## CHƯƠNG SAU MỞ RA Ở ĐÂU (cho NGƯỜI VIẾT, KHÔNG phải điều "
            f"POV biết)\n"
            f"Chương {sau} — {ten}: {beat}\n"
            f"Kết cảnh này sao cho chương sau bắt vào được, nhưng POV KHÔNG "
            f"được linh cảm, dự đoán hay nhắc tới bất cứ điều gì ở trên."
            f"\n\n")


def _scenes_truoc(state: ChapterState) -> list[dict]:
    """Cảnh đã chốt, kèm HIỆN TRƯỜNG — dò lặp chỉ so cảnh cùng chỗ, cùng người."""
    ct = {c["scene_id"]: c for c in state.get("contracts", [])}
    out = []
    for s in state.get("scene_outputs", []):
        c = ct.get(s["scene_id"], {})
        out.append({**s,
                    "location": c.get("location_id") or c.get("location"),
                    "cast": [x.get("id") for x in c.get("active_characters", [])]})
    return out


@safe_node
def auditor_node(state: ChapterState, config: RunnableConfig) -> dict:
    """Kiểm tra HAI TẦNG: thuật toán trước (rẻ, chắc chắn), LLM sau (đắt, tinh).

    Tầng LLM chỉ chạy khi tầng thuật toán chưa thấy BLOCKER — bản nháp đó chắc
    chắn bị viết lại, thẩm định thêm là trả tiền cho nhận xét sẽ vứt đi.

    Tầng LLM hỏng (JSON vỡ, hết quota) thì GHI LẠI và đi tiếp với tầng thuật
    toán, không escalate cả chương: tầng hai là tinh chỉnh, không phải cổng chặn.
    Findings của nó phải qua `parse_llm_findings` — có trích dẫn nguyên văn mới
    được tính (NT-5), và chỉ `pov_knowledge` được là blocker.
    """
    eng = config["configurable"]["engines"]
    c = state["contracts"][state["scene_index"]]
    prose = state["current_draft"]

    findings = [{**f, "source": "code"}
                for f in deterministic_audit(prose, c, state["chapter"], eng,
                                             previous_scenes=_scenes_truoc(state))]
    issues: list[dict] = []
    llm_called = False
    if getattr(eng, "llm_audit", True) and not any(
            f["severity"] == "blocker" for f in findings):
        llm_called = True
        try:
            raw = eng.llm.invoke(AUDITOR_TMPL.format_map({
                "contract": _audit_contract_brief(c),
                "prose": prose,
                "checklist": audit_checklist(c),
            }), role="auditor")
            llm_findings, issues = parse_llm_findings(raw, prose)
            findings += llm_findings
        except Exception as e:          # noqa: BLE001 — tầng hai là tuỳ chọn
            issues = [{"stage": "llm_audit", "reason": "llm_error",
                       "message": f"{type(e).__name__}: {e}"[:200]}]

    # `max_severity` là mức ĐỊNH TUYẾN: chỉ số nhịp không kéo cảnh về Writer
    # (§10.3.2), dù chúng vẫn giữ mức major trong báo cáo.
    sev = routing_severity(findings)
    return {
        "findings": findings, "max_severity": sev,
        "audit_log": [{
            "scene_id": c["scene_id"],
            "draft": state.get("revision_count", 0),
            "max_severity": sev,
            "checks": sorted({f"{f['check']}:{f['severity']}" for f in findings}),
            # NỘI DUNG lỗi nghiêm trọng, không chỉ tên loại. Lượt Gemini đầu tiên
            # của Ngày 10 viết lại CH002_S04 vì một `pov_leak` mà báo cáo không
            # cho biết là câu nào — không phân biệt được rò rỉ thật với báo động giả.
            "serious": [{"check": f["check"], "severity": f["severity"],
                         "message": f.get("message", "")[:240],
                         "evidence": f.get("evidence", "")[:200]}
                        for f in findings if f["severity"] in ("blocker", "major")],
            "llm_called": llm_called,
            "issues": issues,
        }],
    }


@safe_node
def polish_node(state: ChapterState, config: RunnableConfig) -> dict:
    """CHỈ trau chuốt văn phong. Không đụng biến điều khiển vòng lặp — việc đó
    thuộc `scene_boundary_node` (NT-9).

    Luôn ghi `polished`: bản trau chuốt nếu được nhận, bản nháp nếu không. Ba
    trường hợp KHÔNG gọi hoặc KHÔNG nhận:

    - Không có ghi chú Polish xử lý được → không gọi. Trau chuốt không mục tiêu
      chỉ là thêm một lượt mài mòn (§9.3 quy tắc 2) và một lượt quota.
    - `content_drifted` → từ chối. Polish hay vô tình sửa nội dung, và văn xuôi
      này là thứ Extractor sẽ biến thành canon (NT-14).
    - Bản Polish sinh blocker/major MỚI (vd. một câu nội tâm NPC) → từ chối.
      §9.2 đưa bản Polish thẳng tới `scene_boundary`, không qua Auditor — không
      kiểm lại ở đây thì Polish là lối vòng qua toàn bộ vòng kiểm toán.

    LLM Polish lỗi → giữ bản nháp. Trau chuốt là tuỳ chọn; mất nó không đáng
    dừng chương.
    """
    eng = config["configurable"]["engines"]
    c = state["contracts"][state["scene_index"]]
    draft = state["current_draft"]
    findings = state.get("findings") or []
    notes = polish_notes(findings)
    report: dict = {"scene_id": c["scene_id"], "notes": len(notes)}

    if not notes:
        return {"polished": draft, "polish_report": {
            **report, "accepted": False, "called": False,
            "reason": "không có ghi chú Polish xử lý được — giữ bản nháp"}}

    banned = list(dict.fromkeys(CLICHE_SOMATICS + CLICHE_PHRASES
                                + list(c.get("forbidden_cliches", []))))
    try:
        raw = eng.llm.invoke(POLISH_TMPL.format_map({
            "notes": _bullets(notes),
            "voice_sheets": _voice_sheets(c),
            "banned": ", ".join(banned),
            "prose": draft,
        }), role="polish")
    except Exception as e:              # noqa: BLE001 — Polish là tuỳ chọn
        return {"polished": draft, "polish_report": {
            **report, "accepted": False, "called": True,
            "reason": f"lỗi LLM: {type(e).__name__}: {e}"[:200]}}

    out, _ = sanitize_prose(clean_polish_output(raw))
    why = content_drifted(draft, out, _drift_names(eng, c))
    if why is None:
        worse = new_serious(findings, deterministic_audit(
            out, c, state["chapter"], eng,
            previous_scenes=_scenes_truoc(state)))
        if worse:
            why = "bản polish sinh lỗi mới: " + ", ".join(worse)
    if why:
        # Giữ bản bị từ chối: "con số đổi: mất ['0']" chỉ kiểm lại được khi thấy
        # Polish đã viết gì thay vào.
        return {"polished": draft, "polish_report": {
            **report, "accepted": False, "called": True, "reason": why,
            "rejected_text": out[:RAW_KEEP_CHARS]}}
    return {"polished": out, "polish_report": {
        **report, "accepted": True, "called": True, "changed": out != draft}}


def _audit_contract_brief(c: dict) -> str:
    return "\n".join([
        _contract_brief(c),
        f"Câu hỏi kịch tính: {c.get('dramatic_question', '')}",
        f"Trạng thái đầu: {c.get('entry_state', '')}",
        f"Trạng thái cuối: {c.get('exit_state', '')}",
    ])


def _voice_sheets(c: dict) -> str:
    """Giọng cho Polish — KHÔNG có khoảng số (NT-7, §10.3.2)."""
    out = []
    for ch in c.get("active_characters", []):
        v = ch.get("voice_reminder") or {}
        if not v:
            continue
        out.append(f"- {ch['name']}: giọng {v.get('register')}; tật: "
                   f"{v.get('syntactic_tic')}; hay dùng: "
                   f"{', '.join(v.get('signature_lexicon', [])[:6])}; CẤM: "
                   f"{', '.join(v.get('forbidden_lexicon', []))}")
    return "\n".join(out) or "(không có)"


def _drift_names(eng, c: dict) -> list[str]:
    names = {ch["name"] for ch in c.get("active_characters", [])}
    names |= {p.name for p in eng.chars.values()}
    names |= {e["name"] for e in eng.graph.entity_index()
              if e.get("name") and e["name"] != e["id"]}
    return sorted(names)


def _scene_audit_summary(state: ChapterState) -> dict:
    fs = state.get("findings") or []
    return {
        "revisions": state.get("revision_count", 0),
        "max_severity": state.get("max_severity", "note"),
        "residual": [{"severity": f["severity"], "check": f.get("check"),
                      "message": f.get("message", "")}
                     for f in fs if f.get("severity") in ("major", "minor")],
        "polish": dict(state.get("polish_report") or {}),
    }


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
    # `polish_node` luôn ghi `polished` — bản trau chuốt nếu được nhận, bản
    # nháp nếu bị từ chối. `current_draft` chỉ là lưới an toàn.
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
        "polish_report": {},
        "scene_outputs": [{"scene_id": c["scene_id"], "prose": prose,
                           "digest": close.digest, "coerced": coerced,
                           "audit": _scene_audit_summary(state)}],
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

def _plant_brief(d: dict) -> str:
    """Một chỉ thị cài manh mối cho Writer — đúng các khoá WRITER_TMPL nhắc tới."""
    return (f"- {d.get('clue_id')}: surface_form “{d.get('surface_form')}” · "
            f"carrier {d.get('carrier')} · intensity {d.get('intensity')} · "
            f"mode {d.get('mode')}\n  instruction: {d.get('instruction')}")


def _relationship_brief(d: dict) -> str:
    """Chỉ thị quan hệ cho Writer. KHÔNG có chỉ số hay lý do guard (chứa con số):
    NT-7 — Writer viết để thoả con số nếu thấy con số."""
    a, b = d.get("names") or [d.get("a"), d.get("b")]
    line = f"- {a} ↔ {b}: đang ở giai đoạn “{d.get('stage_vi')}”."
    if d.get("action") == "advance":
        line += (f" Cảnh này là bước sang “{d.get('to_vi')}”.\n"
                 f"  scene_requirement: {d.get('scene_requirement')}")
    return line + f"\n  CHƯA được: {d.get('not_yet')}"


def _news_brief(d: dict) -> str:
    """Tin cho Writer. Bản méo mang `truth` dưới nhãn CHỈ ĐẠO DIỄN BIẾT — Writer
    cần bản thật để méo đúng toán tử, cùng khuôn với `hidden_action` (§5.4.1)."""
    kenh = d.get("channel_vi") or d.get("channel")
    if d.get("kind") == "correction_rejected":
        return (f"- Người kể nghe bản CẢI CHÍNH qua {kenh} nhưng KHÔNG TIN, giữ niềm "
                f"tin cũ. Để sự cứng đầu lộ qua hành động, không qua độc thoại giải thích.\n"
                f"  [CHỈ ĐẠO DIỄN BIẾT] nội dung cải chính: {d.get('truth')}")
    if d.get("ops"):
        line = (f"- Người kể nghe qua {kenh} một bản ĐÃ BIẾN DẠNG "
                f"({', '.join(d.get('ops_vi', []))}) và TIN nó.\n"
                f"  [CHỈ ĐẠO DIỄN BIẾT] bản thật: {d.get('truth')}\n"
                f"  Viết đúng bản méo người kể nghe. KHÔNG để lộ bản thật.")
    else:
        line = f"- Người kể nghe qua {kenh}: {d.get('truth')}"
    if d.get("kind") == "correction_accepted":
        line += "\n  Đây là beat ĐÍNH CHÍNH: người kể nhận ra điều mình tin trước đó sai."
    return line


def _debt_brief(debt: dict) -> str:
    parts = []
    for key, label in (("overdue", "quá hạn"), ("late_to_plant", "trễ hạn cài"),
                       ("at_risk", "sắp đến hạn"), ("forgotten", "độc giả đã quên")):
        if debt.get(key):
            parts.append(f"{label}: " + ", ".join(x.get("clue") or x.get("npc", "?")
                                                  for x in debt[key]))
    return "; ".join(parts) or "(không có)"


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
    tâm của nhân vật không phải POV đã bị lột ở đó — ở đây chỉ định dạng.

    Cảnh MỘT MÌNH không có thoại, nên không đưa `signature_lexicon`. Lượt Gemini
    Ngày 10 viết Chương 2 (5/6 cảnh chỉ một nhân vật) hai lần: dù WRITER_TMPL
    dặn "chỉ dùng trong THOẠI", danh sách "từ hay dùng" vẫn nằm trong prompt và
    không có chỗ nào khác để đặt — 60 rồi 66 lần các cụm như "theo thẩm quyền",
    "hồ sơ cho thấy" rải vào lời kể. Một ràng buộc không thể thoả thì model phá
    nó; bỏ nguyên liệu gây ra nó thì không cần ràng buộc. Từ CẤM vẫn giữ.
    """
    solo = len(chars) < 2
    out = []
    for ch in chars:
        lines = [f"### {ch.get('name')} ({ch['id']})"]
        if ch.get("immediate_goal"):
            lines.append(f"- mục tiêu ngay lúc này: {ch['immediate_goal']}")
        if ch.get("voice_reminder"):
            v = ch["voice_reminder"]
            lines.append(f"- giọng: {v.get('register')}; tật cú pháp: "
                         f"{v.get('syntactic_tic')}")
            if solo:
                lines.append("- cảnh này nhân vật ở MỘT MÌNH, không có ai để nói "
                             "chuyện: giọng lộ qua chi tiết được chọn và nhịp câu, "
                             "không qua khẩu ngữ hay cụm từ quen miệng")
            elif v.get("voice_exemplars"):
                lines.append("- người này nói kiểu như thế này (MẪU để bắt "
                             "chước CÁCH nói, không phải câu để chép lại):")
                lines += [f"    · “{x}”" for x in v["voice_exemplars"][:3]]
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
    if delta.relationship_updates:
        # NT-13: trạng thái quan hệ do CODE tính từ sự kiện. Bản chụp trạng thái
        # LLM viết (kể cả `stage`) không bao giờ vào canon — ghi lại rồi bỏ.
        dropped.append({"source": "merge", "field": "relationship_updates",
                        "action": "dropped", "reason": "llm_state_snapshot_ignored",
                        "item": f"{len(delta.relationship_updates)} mục"})
        delta.relationship_updates = []
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
    coverage = plan_coverage(contracts, delta,
                             known_clues=set(eng.graph.clues), scenes=scenes)

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

    # Đóng hai vòng bị hở: `measured_tension` chưa từng được GHI nên
    # `director_node` luôn đọc 0.5 mặc định (§8.1), và `chapter_summaries` rỗng
    # nên tầng L2 của bộ nhớ (§4.1 — "các chương gần đây") không bao giờ có gì.
    do_cang = measure_tension(scenes, delta, state.get("unresolved", []))
    eng.store.put_chapter_summary(state["chapter"], chapter_summary(scenes), do_cang)

    return {
        "delta": delta.model_dump(),
        "extraction_report": {
            "measured_tension": do_cang,
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
            # Mã, mode, vật mang và câu chữ ĐÃ GIAO — đủ để tìm bằng chứng.
            # Không `instruction`, không `weight`: không liên quan tới việc tìm.
            lines.append("  plant_directives:")
            for d in c["plant_directives"]:
                lines.append(f"    - {d.get('clue_id')} ({d.get('mode')}, qua "
                             f"{d.get('carrier')}): “{d.get('surface_form')}”")
        else:
            lines.append("  plant_directives: (không có)")
        if c.get("relationship_directives"):
            lines.append("  relationship_directives:")
            for d in c["relationship_directives"]:
                buoc = f" → {d.get('to')}" if d.get("action") == "advance" else ""
                lines.append(f"    - {d.get('a')} ↔ {d.get('b')}: {d.get('stage')}{buoc}")
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
