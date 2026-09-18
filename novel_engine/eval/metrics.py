"""Eval Harness — M1–M12 (§13, §13.3).

Lỗ hổng L6 là lỗ hổng nguy hiểm nhất về lâu dài: không có eval thì mỗi lần chỉnh
một câu trong prompt là một canh bạc.

Mỗi metric trả `{"value": ..., ...chi tiết}`. `value = None` nghĩa là CHƯA ĐO
ĐƯỢC (thiếu dữ liệu, thiếu judge) — khác hẳn 0.0, và bảng điều khiển phải phân
biệt hai thứ đó: một metric không đo được mà hiện 0.0 sẽ bị đọc thành báo động.

═══ CHỖ KHÁC MÃ §13 ═══════════════════════════════════════════════════════

- M1 §13 hỏi `graph.conflicting_assertion(...)` NGAY SAU KHI mệnh đề đã được
  ghi vào canon. Canon lúc đó đã chứa chính nó, nên không bao giờ mâu thuẫn:
  metric luôn ≈ 1,0. Ở đây đọc QUYẾT ĐỊNH đã lưu trong delta (`classification`,
  `quarantine`) — bản ghi lịch sử của lúc phân loại, đúng thứ event sourcing giữ.
- M2 gộp manh mối đã RETIRE vào mẫu số, nên một manh mối tác giả cố ý bỏ kéo tỉ
  lệ xuống mãi mãi; và "đúng hạn" không được đo. Ở đây tách retired, và đối chiếu
  chương trả bài với `payoff_deadline` bằng log delta.
- M3 dùng `scipy`/`numpy` (không có trong môi trường) và `lexical_distribution`
  không tồn tại. Khoảng cách Jensen-Shannon viết thuần Python; thoại lấy từ
  `attribute_dialogue` (Ngày 9) — chỉ những lượt gán được CHẮC CHẮN.
- M4 gọi `graph.npcs_in_window` không tồn tại; ở đây đếm trên frame.
- M5 `stage_index` không xử lý RUPTURE/SEVERED (ngoài trục chính) → KeyError.
- M10 chia `assertions_kept` cho TỔNG span bị loại, gồm cả span manh mối và span
  quan hệ — hai tử số khác mẫu số. Ở đây tách theo loại.
- M11 lấy trung bình `plan_fulfillment_rate`, mà chương KHÔNG có manh mối nào
  được giao có rate = 0/1 = 0 — chương sạch kéo metric xuống. Ở đây cộng dồn tử
  và mẫu, bỏ qua chương không hứa gì.
- M12 đọc `f.pov` (ContinuityFrame không có trường đó) và
  `graph.facts_known_to_reader` (không tồn tại). Ở đây POV tra từ outline, còn
  "tri thức độc giả" đo trên MỘT không gian định danh: manh mối đã cài + tin đã
  lên trang (NT-8).
"""
from __future__ import annotations

import itertools
import json
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field

from novel_engine.audit.rhythm import (
    THRESH, is_subordinate_opener, narrative_sentences, word_count,
)
from novel_engine.canon.models import ClueStatus, StateDelta
from novel_engine.canon.timeline import ContinuityFrame
from novel_engine.character.voice_check import attribute_dialogue
from novel_engine.planner.tension import target_tension
from novel_engine.relationship.machine import ORDER

MIN_LINES_FOR_VOICE = 6          # dưới mức này phân phối từ vựng là nhiễu
BLIND_SAMPLE = 10


@dataclass
class ChapterData:
    """Mọi thứ một metric cần về một chương, gom ở một chỗ."""
    number: int
    scenes: list[dict] = field(default_factory=list)      # {scene_id, prose, digest}
    delta: StateDelta | None = None
    report: dict = field(default_factory=dict)
    frames: list[ContinuityFrame] = field(default_factory=list)

    @property
    def prose(self) -> str:
        return "\n\n".join(s.get("prose", "") for s in self.scenes)


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", s or "").lower())


def _norm_dist(counts: dict[str, int]) -> dict[str, float]:
    n = sum(counts.values())
    return {k: v / n for k, v in counts.items()} if n else {}


def _js_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Khoảng cách Jensen-Shannon, log cơ số 2 → [0, 1]. Không cần scipy."""
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in set(p) | set(q)}

    def kl(a):
        return sum(v * math.log2(v / m[k]) for k, v in a.items() if v > 0)

    return math.sqrt(max(0.0, 0.5 * kl(p) + 0.5 * kl(q)))


# ═══════════════════════ M1–M8 (§13) ═══════════════════════

def M1_entity_consistency(chapters: list[ChapterData]) -> dict:
    """Tỉ lệ mệnh đề objective KHÔNG mâu thuẫn canon. Mục tiêu ≥0,98."""
    total = conflicts = 0
    details = []
    for ch in chapters:
        d = ch.delta
        if d is None:
            continue
        for a in d.assertions:
            if a.epistemic != "objective":
                continue
            total += 1
            verdict = d.classification.get(_key_of(d, a))
            if verdict == "contradiction":
                conflicts += 1
                details.append({"chapter": ch.number, "subject": a.subject,
                                "predicate": a.predicate, "span": a.span[:90]})
    if not total:
        return {"value": None, "reason": "chưa có mệnh đề objective nào",
                "objective": 0, "contradictions": 0, "details": []}
    return {"value": round(1 - conflicts / total, 4), "objective": total,
            "contradictions": conflicts, "details": details}


def _key_of(delta: StateDelta, item) -> str:
    from novel_engine.canon.models import item_key
    return item_key(item)


def M2_clue_payoff_rate(graph, chapters: list[ChapterData], last_chapter: int) -> dict:
    """Manh mối đã cài được trả bài, và trả ĐÚNG HẠN. Mục tiêu ≥0,90."""
    clues = getattr(graph, "clues", {}) or {}
    paid_at: dict[str, int] = {}
    for ch in chapters:
        if ch.delta is None:
            continue
        for cid, st in ch.delta.clue_transitions.items():
            if ClueStatus(st) == ClueStatus.PAID_OFF:
                paid_at.setdefault(cid, ch.number)

    planted = [c for c in clues.values()
               if c.status not in (ClueStatus.DRAFTED, ClueStatus.RETIRED)]
    paid = [c for c in planted if c.status == ClueStatus.PAID_OFF]
    late = [c.clue_id for c in paid
            if paid_at.get(c.clue_id, last_chapter) > c.payoff_deadline]
    overdue = [c.clue_id for c in planted
               if c.status != ClueStatus.PAID_OFF and c.payoff_deadline < last_chapter]
    if not planted:
        return {"value": None, "reason": "chưa cài manh mối nào", "overdue": [],
                "dangling": 0, "paid_late": []}
    return {"value": round(len(paid) / len(planted), 4),
            "overdue": sorted(overdue), "dangling": len(planted) - len(paid),
            "paid_late": sorted(late),
            "retired": sorted(c.clue_id for c in clues.values()
                              if c.status == ClueStatus.RETIRED)}


def lexical_distribution(chapters: list[ChapterData], chars: dict) -> dict[str, dict]:
    """Phân phối từ vựng trong THOẠI của từng nhân vật.

    Tính CẢ lượt thoại gán được qua nhịp hành động (`inferred`), không chỉ lượt
    có dẫn thoại tường minh. Đo trên Chương 1: 28–34 dòng thoại, chỉ 1–3 dòng có
    dẫn kiểu "— … — Kaelen nói", còn 12–17 dòng gán được qua câu hành động đứng
    ngay trước. Chỉ lấy `by_speaker` nghĩa là M3 chấm giọng trên khoảng 5% số
    thoại rồi trả về một con số nghe rất chắc chắn.

    `voice_report` trong cùng module vốn đã dùng cả hai — nó chỉ hạ mức nghiêm
    trọng cho lượt suy ra. Hai chính sách khác nhau cho cùng một câu hỏi "ai
    nói câu này" là thứ không nên tồn tại (NT-11).
    """
    names = {p.name: cid for cid, p in chars.items()}
    counts: dict[str, dict[str, int]] = {cid: {} for cid in chars}
    lines: dict[str, int] = {cid: 0 for cid in chars}
    for ch in chapters:
        for sc in ch.scenes:
            attr = attribute_dialogue(sc.get("prose", ""), names)
            for cid in counts:
                said = list(attr["by_speaker"].get(cid, [])) + list(
                    attr["inferred"].get(cid, []))
                lines[cid] += len(said)
                for w in _tokens(" ".join(said)):
                    counts[cid][w] = counts[cid].get(w, 0) + 1
    return {cid: {"dist": _norm_dist(counts[cid]), "lines": lines[cid]}
            for cid in chars}


def M3_voice_distinctiveness(chapters: list[ChapterData], chars: dict) -> dict:
    """Khoảng cách JS trung bình giữa phân phối từ vựng các nhân vật. ≥0,28."""
    data = lexical_distribution(chapters, chars)
    usable = {cid: d["dist"] for cid, d in data.items()
              if d["lines"] >= MIN_LINES_FOR_VOICE}
    if len(usable) < 2:
        return {"value": None,
                "reason": (f"cần ≥2 nhân vật có ≥{MIN_LINES_FOR_VOICE} lượt thoại "
                           f"gán được; hiện có {len(usable)}"),
                "lines": {cid: d["lines"] for cid, d in data.items()}}
    pairs = {f"{a}|{b}": round(_js_distance(usable[a], usable[b]), 4)
             for a, b in itertools.combinations(sorted(usable), 2)}
    return {"value": round(statistics.mean(pairs.values()), 4), "pairs": pairs,
            "lines": {cid: d["lines"] for cid, d in data.items()}}


def M4_npc_reuse_ratio(chapters: list[ChapterData], chars: dict, window: int = 10) -> dict:
    """Nhân vật phụ tái xuất hiện / tổng nhân vật phụ. Thấp = thế giới phình. ≥0,45."""
    recent = [c for c in chapters if c.number > max(0, max((x.number for x in chapters),
                                                           default=0) - window)]
    seen: dict[str, set[int]] = {}
    for ch in recent:
        for f in ch.frames:
            for cid in f.locations:
                seen.setdefault(cid, set()).add(ch.number)
    # Mẫu là nhân vật KHÔNG có trong bible — tức người do hệ thống sinh ra khi
    # viết. Nhân vật tác giả đã khai (kể cả `supporting`) vắng mặt một chương là
    # quyết định sáng tác, không phải thế giới phình ra; đếm họ vào đây là báo
    # động giả, và báo động giả thì người ta tắt bảng (P2).
    npcs = {cid: chs for cid, chs in seen.items() if cid not in chars}
    if not npcs:
        return {"value": None, "reason": "chưa có nhân vật nào ngoài bible xuất hiện",
                "npcs": {}}
    reused = sum(1 for chs in npcs.values() if len(chs) > 1)
    return {"value": round(reused / len(npcs), 4),
            "npcs": {cid: sorted(chs) for cid, chs in sorted(npcs.items())}}


def _stage_index(stage: str) -> int | None:
    try:
        return [s.value for s in ORDER].index(stage)
    except ValueError:
        return None          # rupture / severed nằm ngoài trục chính


def M5_relationship_pacing(book) -> dict:
    """Phát hiện nhảy cóc giai đoạn. Mục tiêu: 0 vi phạm."""
    bad = []
    states = getattr(book, "states", {}) or {}
    for key, st in sorted(states.items()):
        hist = list(st.history)
        for prev, cur in zip(hist, hist[1:]):
            i, j = _stage_index(prev["stage"]), _stage_index(cur["stage"])
            if i is not None and j is not None and j - i > 1:
                bad.append(f"{key}: nhảy {prev['stage']}→{cur['stage']} ở chương "
                           f"{cur['chapter']}")
        if st.stage.value == "catharsis" and not st.scars:
            bad.append(f"{key}: catharsis không có sẹo")
    return {"value": len(bad), "violations": bad, "pairs": len(states)}


def M6_blind_attribution(chapters: list[ChapterData], chars: dict, judge=None,
                         sample: int = BLIND_SAMPLE) -> dict:
    """§10.4 — che tên rồi hỏi model ai nói câu nào. Mục tiêu ≥0,70."""
    if judge is None:
        return {"value": None, "reason": "cần một LLM judge (--judge gemini)"}
    names = {p.name: cid for cid, p in chars.items()}
    items: list[tuple[str, str]] = []
    for ch in chapters:
        for sc in ch.scenes:
            attr = attribute_dialogue(sc.get("prose", ""), names)
            for cid, said in attr["by_speaker"].items():
                items += [(cid, s) for s in said if len(s.split()) >= 5]
    if len(items) < 4:
        return {"value": None, "reason": "quá ít lượt thoại gán chắc chắn"}
    items = items[:sample]
    ung_vien = sorted({cid for cid, _ in items})
    lines = "\n".join(f"{i}. {s}" for i, (_, s) in enumerate(items))
    ds = "\n".join(f"- {cid}: {chars[cid].name}" for cid in ung_vien)
    raw = judge.invoke(
        f"Dưới đây là các câu thoại đã BỎ tên người nói. Gán mỗi câu cho một "
        f"nhân vật.\n\n## NHÂN VẬT\n{ds}\n\n## THOẠI\n{lines}\n\n"
        f'Chỉ xuất JSON: {{"0": "CHAR_X", "1": "CHAR_Y"}}', role="auditor")
    try:
        from novel_engine.llm.json_io import strip_fences
        guess = json.loads(strip_fences(raw))
    except (ValueError, TypeError):
        return {"value": None, "reason": "judge trả JSON không đọc được"}
    dung = sum(1 for i, (cid, _) in enumerate(items)
               if str(guess.get(str(i), "")).strip() == cid)
    return {"value": round(dung / len(items), 4), "n": len(items),
            "candidates": ung_vien}


def M7_scene_necessity(chapters: list[ChapterData], judge=None,
                       sample: int | None = None) -> dict:
    """Tỉ lệ cảnh thực sự thay đổi trạng thái. Mục tiêu ≥0,85."""
    if judge is None:
        return {"value": None, "reason": "cần một LLM judge (--judge gemini)"}
    scenes = [(ch.number, sc) for ch in chapters for sc in ch.scenes]
    if sample:
        scenes = scenes[:sample]
    if not scenes:
        return {"value": None, "reason": "chưa có cảnh nào"}
    # §13 bảo judge "trả lời chính xác 'KHÔNG'" rồi kiểm bằng `startswith`. Trong
    # tiếng Việt, câu trả lời mở đầu bằng "Không khí trong phòng đổi…" cũng khớp,
    # nên cảnh CÓ thay đổi bị đếm là thừa — lượt Arc 1 đầu tiên cho M7 = 0.17.
    # Hỏi bằng JSON, đọc bằng khoá; không đoán qua chữ đầu câu.
    from novel_engine.llm.json_io import strip_fences
    thua, hong = [], 0
    for num, sc in scenes:
        raw = judge.invoke(
            f"Cảnh sau có làm THAY ĐỔI trạng thái câu chuyện không (quan hệ, thông "
            f"tin, vị thế, quyết định)?\n\n{sc.get('prose', '')[:2500]}\n\n"
            f'Chỉ xuất JSON: {{"changed": true|false, "what": "thay đổi gì, ngắn"}}',
            role="auditor")
        try:
            doi = bool(json.loads(strip_fences(raw)).get("changed"))
        except (ValueError, TypeError, AttributeError):
            hong += 1
            continue
        if not doi:
            thua.append(f"ch{num}:{sc.get('scene_id')}")
    n = len(scenes) - hong
    if n == 0:
        return {"value": None, "reason": "judge trả JSON không đọc được"}
    return {"value": round(1 - len(thua) / n, 4), "n": n, "unparsed": hong,
            "unnecessary": thua}


def M8_tension_mae(store, chapters: list[ChapterData], total_chapters: int) -> dict:
    """MAE giữa độ căng ĐO ĐƯỢC và đường cong mục tiêu. Mục tiêu ≤0,15."""
    rows = []
    for ch in chapters:
        do = store.measured_tension(ch.number)
        muc_tieu = target_tension(ch.number, total_chapters)
        rows.append({"chapter": ch.number, "measured": round(do, 3),
                     "target": round(muc_tieu, 3), "err": abs(do - muc_tieu)})
    if not rows:
        return {"value": None, "reason": "chưa có chương nào"}
    if all(r["measured"] == 0.5 for r in rows) and len(rows) > 1:
        # 0.5 là giá trị MẶC ĐỊNH khi chưa ai ghi số đo (§ tension_measure).
        return {"value": None, "reason": "độ căng chưa được ghi vào store",
                "rows": rows}
    return {"value": round(statistics.mean(r["err"] for r in rows), 4), "rows": rows}


# ═══════════════════════ M9–M12 (§13.3) ═══════════════════════

def M9_prose_rhythm(chapters: list[ChapterData], lang: str = "vi") -> dict:
    sds, subs = [], []
    for ch in chapters:
        s = narrative_sentences(ch.prose)
        if len(s) < 8:
            continue
        sds.append(statistics.pstdev([word_count(x) for x in s]))
        subs.append(sum(1 for x in s if is_subordinate_opener(x)) / len(s))
    if not sds:
        return {"value": None, "reason": "quá ít câu trần thuật"}
    sd, sub = statistics.mean(sds), statistics.mean(subs)
    return {"value": round(sd, 2), "sd_mean": round(sd, 2),
            "subordinate_ratio": round(sub, 3),
            "ok": sd >= THRESH[lang]["sd_floor"] and sub <= 0.30}


def M10_extraction_fidelity(chapters: list[ChapterData]) -> dict:
    """Tỉ lệ span sống sót qua `verify_spans`. Mục tiêu ≥0,95 — CẢNH BÁO SỚM
    cho ô nhiễm canon."""
    kept = rej = 0
    plant_rej = rel_rej = 0
    for ch in chapters:
        ex = ch.report.get("extraction") or {}
        kept += int(ex.get("assertions_kept", 0))
        for r in ex.get("rejected_spans", []):
            reason = str(r.get("reason", ""))
            if reason.startswith("plant_"):
                plant_rej += 1
            elif reason.startswith("relationship_"):
                rel_rej += 1
            else:
                rej += 1
    if kept + rej == 0:
        return {"value": None, "reason": "chưa có span mệnh đề nào",
                "plant_rejected": plant_rej, "relationship_rejected": rel_rej}
    return {"value": round(kept / (kept + rej), 4), "assertions_kept": kept,
            "assertions_rejected": rej, "plant_rejected": plant_rej,
            "relationship_rejected": rel_rej}


def M11_plan_fulfillment(chapters: list[ChapterData]) -> dict:
    promised = fulfilled = 0
    for ch in chapters:
        ex = ch.report.get("extraction") or {}
        promised += len(ex.get("promised_plants", []))
        fulfilled += len(ex.get("fulfilled_plants", []))
    if not promised:
        return {"value": None, "reason": "chưa chương nào được giao manh mối",
                "promised": 0, "fulfilled": 0}
    return {"value": round(fulfilled / promised, 4), "promised": promised,
            "fulfilled": fulfilled}


# ═══════════ M13–M15: ĐỌC ĐƯỢC KHÔNG (bổ sung sau lượt đọc Arc 1) ═══════════
#
# M1–M12 đo tính nhất quán, nhịp câu, độ trung thực của canon — và Arc 1 lượt 2
# đạt gần hết: M1 1.0, M2 1.0, M6 1.0, M10 1.0, M11 1.0. Rồi đọc bằng mắt thì
# thấy ba cảnh liên tiếp dựng lại cùng một cảnh, một nhân vật nói năm cụm đặc
# trưng mỗi cụm ba lần, và chỉ số beat nằm giữa trang. Không metric nào thấy.
#
# Bộ đo nào cũng chỉ bảo vệ được thứ nó đo. Ba metric dưới đây đưa chính những
# con số đã dùng để chẩn Arc 1 vào bộ hồi quy, để lần sau chúng không im lặng.


def M13_scene_repetition(chapters: list[ChapterData]) -> dict:
    """Mức trùng lặp cao nhất giữa hai cảnh CÙNG HIỆN TRƯỜNG. Thấp thì tốt."""
    from novel_engine.audit.repetition import containment

    cao, o_dau, co_cap = 0.0, "(không cặp nào trùng)", False
    for ch in chapters:
        van = [s.get("prose", "") for s in ch.scenes]
        for i in range(len(van)):
            for j in range(i + 1, len(van)):
                co_cap = True
                c = containment(van[i], van[j])
                if c > cao:
                    cao, o_dau = c, f"ch{ch.number} cảnh {i}~{j}"
    # Không cặp nào để so là CHƯA ĐO ĐƯỢC. Hai cảnh khác hẳn nhau cho 0.0, và
    # 0.0 là một phép đo — trả None ở đó là báo "mù" trong khi đang nhìn rõ.
    if not co_cap:
        return {"value": None, "reason": "chưa đủ hai cảnh để so"}
    return {"value": round(cao, 3), "worst_pair": o_dau}


def M14_tic_saturation(chapters: list[ChapterData], chars: dict) -> dict:
    """Số lần MỘT cử chỉ nhận dạng lặp nhiều nhất trong một chương. Thấp là tốt.

    Cử chỉ sinh ra để phân biệt nhân vật. Lặp mỗi cảnh thì nó không phân biệt
    ai với ai nữa — nó chỉ còn là một con dấu đóng đi đóng lại.
    """
    from novel_engine.audit.tics import _count_gesture, _nfc_lower

    cao, ai = 0, None
    for ch in chapters:
        low = _nfc_lower(ch.prose)
        for prof in (chars or {}).values():
            for g in prof.somatic_signature:
                n = _count_gesture(low, g)
                if n > cao:
                    cao, ai = n, f"{prof.name} ch{ch.number}: “{g[:30]}…”"
    if ai is None:
        return {"value": None, "reason": "không có cử chỉ nào được khai"}
    return {"value": cao, "worst": ai}


def M15_hygiene_residue(chapters: list[ChapterData]) -> dict:
    """Rác còn sót trong văn xuôi ĐÃ CHỐT: chỉ số beat, lỗi gõ, ký tự lạ.

    Đáng lẽ luôn bằng 0 — `sanitize_prose` chạy ở mỗi lượt viết. Khác 0 nghĩa
    là có một đường đi vòng qua bộ dọn, và đó là thứ cần biết ngay.
    """
    from novel_engine.audit.text_hygiene import (foreign_letters, leaked_speech,
                                                 malformed_words,
                                                 strip_leaked_indices)
    tong, chi_tiet = 0, {}
    for ch in chapters:
        van = ch.prose
        d = {"chỉ số": len(strip_leaked_indices(van)[1]),
             "thoại chỉ số": len(leaked_speech(van)),
             "lỗi gõ": len(malformed_words(van)),
             "ký tự lạ": len(foreign_letters(van))}
        n = sum(d.values())
        tong += n
        if n:
            chi_tiet[f"ch{ch.number}"] = {k: v for k, v in d.items() if v}
    return {"value": tong, "detail": chi_tiet}


def M12_irony_gap(chapters: list[ChapterData], graph, planner) -> dict:
    """Khoảng cách giữa tri thức ĐỘC GIẢ (trục đọc) và tri thức NHÂN VẬT (trục
    epoch). Bằng 0 suốt = không có mỉa mai kịch tính nào.

    SỐ TRUNG BÌNH GIẤU MẤT HÌNH DẠNG. Arc 1 cho mean 0,37 — nghe như "có chút
    mỉa mai rải đều". Sự thật: 22 trên 30 cảnh có gap ĐÚNG BẰNG 0, và toàn bộ
    tín hiệu đến từ 8 cảnh còn lại. Vì thế `zero_gap_ratio` đi kèm mọi lần trả
    về; đọc mean mà không đọc nó là hiểu ngược.

    ĐIỀU BỘ ĐO NÀY KHÔNG THỂ THẤY. Tri thức độc giả ở đây chỉ gồm manh mối đã
    cài và tin đã lan. Nhưng nguồn mỉa mai lớn nhất của Arc 1 lại là những cảnh
    POV Vhal, nơi độc giả thấy ông ta làm giả sổ trực còn Kaelen thì không —
    một MỆNH ĐỀ khách quan. Canon lưu tri thức nhân vật về sự thật khách quan ở
    mức THỰC THỂ (`KNOWS_ABOUT`), không ở mức mệnh đề, nên không có chỗ nào ghi
    "Kaelen chưa biết Vhal sửa sổ". Đo đúng chuyện đó cần một cạnh mới
    (ai học được mệnh đề nào, tại tick nào) và Extractor phải khai nó — một
    thay đổi kiến trúc, không phải một phép sửa công thức ở đây.
    """
    frames = sorted((f for ch in chapters for f in ch.frames),
                    key=lambda f: f.time.narrative_order)
    if not frames:
        return {"value": None, "reason": "chưa có cảnh nào"}

    # Độc giả thấy manh mối khi nó được cài trên trang; thấy tin khi nhân vật đầu
    # tiên nghe được nó trong một cảnh đã đọc qua.
    planted_at: dict[str, int] = {}
    order_of = {f.scene_id: f.time.narrative_order for f in frames}
    for ch in chapters:
        if ch.delta is None:
            continue
        for pe in ch.delta.plant_evidence:
            if pe.verified and pe.scene_id in order_of:
                planted_at.setdefault(pe.clue_id, order_of[pe.scene_id])
    news_at: dict[str, int] = {}
    for (_cid, nid), k in (getattr(graph, "news_knowledge", {}) or {}).items():
        first = next((f.time.narrative_order for f in frames
                      if f.time.mode == "present" and f.time.epoch_tick >= k["since_tick"]),
                     None)
        if first is not None:
            news_at[nid] = min(news_at.get(nid, first), first)

    gaps, rows = [], []
    for f in frames:
        ch_no = int(f.scene_id[2:5])
        si = int(f.scene_id.rsplit("_S", 1)[1])
        try:
            pov = planner.pov_for(ch_no, si)
        except (KeyError, IndexError):
            continue
        reader = ({c for c, o in planted_at.items() if o <= f.time.narrative_order}
                  | {n for n, o in news_at.items() if o <= f.time.narrative_order})
        known = {c.clue_id for c in getattr(graph, "clues", {}).values()
                 if pov in c.understood_by_characters}
        known |= {nid for (cid, nid), k in (getattr(graph, "news_knowledge", {}) or {}).items()
                  if cid == pov and k["since_tick"] <= f.time.epoch_tick}
        gap = len(reader - known)
        gaps.append(gap)
        rows.append({"scene_id": f.scene_id, "pov": pov, "gap": gap})
    if not gaps:
        return {"value": None, "reason": "không xác định được POV của cảnh nào"}
    mean = statistics.mean(gaps)
    run = best = 0
    for g in gaps:
        run = run + 1 if g > 6 else 0
        best = max(best, run)
    return {"value": round(mean, 2), "mean_gap": round(mean, 2),
            "zero_gap_ratio": round(sum(1 for g in gaps if g == 0) / len(gaps), 3),
            "max_sustained": best, "ok": 0.5 <= mean <= 5.0, "scenes": rows}
