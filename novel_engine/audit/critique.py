"""Bounded Critique Loop — phần QUYẾT ĐỊNH của Auditor và Polish (§9.2, §9.3, §10.3.2).

Node chỉ gọi LLM và ghi state. Mọi quyết định — mức nghiêm trọng dùng để định
tuyến, phản hồi nào được quay lại Writer, ghi chú nào đưa cho Polish, bản Polish
có được nhận không — nằm ở đây, thành hàm thuần, test được không cần đồ thị.

═══ BA LỖ HỔNG TRONG MÃ §9.2–§9.3 ═══════════════════════════════════════════

1. MÂU THUẪN GOODHART. §10.3.1 xếp `subordinate_opener` và `triclause_template`
   ở mức MAJOR. `after_audit` (§9.3) gửi mọi major về `revise` → Writer. Nhưng
   §10.3.2 quy tắc 2: "Kết quả đi vào Polish Agent, KHÔNG quay lại Writer."
   Theo mã tài liệu, một cảnh có ba câu "Khi A, B, trong khi C" bị viết lại
   toàn bộ — đúng thứ §10.3.2 cảnh báo. Ở đây chỉ số nhịp giữ nguyên mức độ
   để BÁO CÁO, nhưng không tham gia tính `max_severity` dùng để định tuyến.

2. RÒ RỈ QUA PHẢN HỒI. `writer_node` §9.2 đưa `message` của MỌI finding vào
   prompt Writer. Auditor LLM đọc hợp đồng ĐẦY ĐỦ — gồm `hidden_action` và
   `must_not_reveal` của NPC — nên lời giải thích của nó rất dễ là "cảnh để lộ
   rằng Serena đang giấu mã phê duyệt". Đưa câu đó cho Writer là tự tay xuyên
   thủng POV Firewall (§5.4.1): Writer giờ BIẾT bí mật, và lần viết lại có xác
   suất nhắc tới nó cao hơn lần đầu. Phản hồi từ LLM chỉ gồm một câu hướng dẫn
   CỐ ĐỊNH theo loại lỗi + đoạn trích từ chính bản nháp Writer đã viết — không
   có thông tin nào Writer chưa có.

3. AUDITOR LLM KHÔNG CÓ NGHĨA VỤ CHỨNG MINH. Blocker từ LLM kích hoạt viết lại
   và escalation. Một blocker bịa ra đốt hai lượt viết lại rồi dừng chương vì
   một lỗi không tồn tại (P2). NT-5 áp cho mọi mệnh đề về văn bản: finding phải
   kèm `evidence` trích nguyên văn, xác minh bằng cùng `_found` của extractor,
   và chỉ `pov_knowledge` được phép là blocker.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections import Counter

from novel_engine.llm.json_io import strip_fences
from novel_engine.reconcile.verify import MIN_SPAN_LEN, _found, _norm

SEV_RANK = {"note": 0, "minor": 1, "major": 2, "blocker": 3}
_RANK_SEV = {v: k for k, v in SEV_RANK.items()}

# §10.3.2 — chỉ số nhịp: chẩn đoán, không bao giờ quay lại Writer.
STYLE_CHECKS = frozenset({
    "rhythm_variance", "no_staccato", "flat_run", "subordinate_opener",
    "triclause_template", "opener_diversity", "paragraph_burstiness",
    "paragraph_uniformity",
})

# Polish sửa CÂU CHỮ, không thêm nội dung (POLISH_TMPL). `length`, `sensory`,
# `no_dialogue` đòi viết thêm — đưa cho Polish là mời nó bịa chi tiết, rồi
# `content_drifted` từ chối bản đó. Chúng ở lại báo cáo.
POLISHABLE_CHECKS = STYLE_CHECKS | {"cliche", "voice"}

# Loại lỗi Auditor LLM được dùng, kèm câu hướng dẫn CỐ ĐỊNH cho Writer.
LLM_CHECKS = {
    "pov_knowledge": "đoạn này tường thuật điều người kể không thể biết — viết "
                     "lại chỉ bằng những gì người kể tri giác được",
    "scene_must_change": "cảnh kết thúc mà trạng thái chưa đổi như nhiệm vụ tự sự "
                         "yêu cầu",
    "subtext": "thoại nói thẳng điều lẽ ra phải để ngầm",
    "character_too_easy": "nhân vật nhượng bộ quá dễ, không phải trả giá",
    "plant_visibility": "chi tiết cần cài quá lộ hoặc quá mờ",
    "other": "đoạn này cần viết lại",
}
LLM_BLOCKER_CHECKS = frozenset({"pov_knowledge"})

_SEV_ALIASES = {
    "critical": "blocker", "fatal": "blocker", "error": "major", "high": "major",
    "medium": "minor", "moderate": "minor", "low": "minor", "warning": "minor",
    "info": "note", "nit": "note",
}
_CHECK_HINTS = [
    ("pov_knowledge", ("pov", "knowledge", "biết", "hidden", "secret", "bí mật", "reveal")),
    ("scene_must_change", ("change", "exit", "thay đổi", "state")),
    ("subtext", ("subtext", "ẩn ý", "on_the_nose", "explicit")),
    ("character_too_easy", ("easy", "dễ", "concede", "nhượng")),
    ("plant_visibility", ("plant", "clue", "foreshadow", "manh mối")),
]


def _clip(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _canon_sev(s: str) -> str:
    s = (s or "").strip().lower()
    return s if s in SEV_RANK else _SEV_ALIASES.get(s, "minor")


def _canon_check(s: str) -> str:
    s = (s or "").strip().lower()
    if s in LLM_CHECKS:
        return s
    for name, keys in _CHECK_HINTS:
        if any(k in s for k in keys):
            return name
    return "other"


# ═══════════════════════ định tuyến ═══════════════════════

def routing_severity(findings: list[dict]) -> str:
    """Mức nghiêm trọng mà `after_audit` đọc — bỏ qua chỉ số nhịp (§10.3.2)."""
    rank = max((SEV_RANK.get(f.get("severity"), 0) for f in findings
                if f.get("check") not in STYLE_CHECKS), default=0)
    return _RANK_SEV[rank]


def writer_feedback(findings: list[dict]) -> str:
    lines: list[str] = []
    for f in findings:
        if f.get("severity") not in ("blocker", "major") or f.get("check") in STYLE_CHECKS:
            continue
        if f.get("source") == "llm":
            text = LLM_CHECKS.get(f.get("check"), LLM_CHECKS["other"])
        else:
            text = f.get("message", "")
        line = f"- [{f['severity']}] {text}"
        if f.get("evidence"):
            line += f"\n  đoạn cần sửa: “{_clip(f['evidence'])}”"
        lines.append(line)
    lines = list(dict.fromkeys(lines))
    if not lines:
        return ""
    return ("## PHẢN HỒI TỪ LẦN VIẾT TRƯỚC — BẮT BUỘC SỬA\n" + "\n".join(lines)
            + "\nViết lại TOÀN BỘ cảnh. Không nhắc tới phản hồi này trong văn xuôi.")


def polish_notes(findings: list[dict]) -> list[str]:
    notes = []
    for f in findings:
        if f.get("source") == "llm" or f.get("check") not in POLISHABLE_CHECKS:
            continue
        note = f.get("message", "")
        if f.get("evidence"):
            note += f" — ví dụ: “{_clip(f['evidence'])}”"
        notes.append(note)
    return list(dict.fromkeys(notes))


# ═══════════════════════ Auditor LLM ═══════════════════════

def audit_checklist(c: dict) -> str:
    pov = c.get("pov_character")
    others = [x for x in c.get("active_characters", []) if x.get("id") != pov]
    hidden = [f"{x['name']}: {x['hidden_action']}" for x in others if x.get("hidden_action")]
    secrets = [f"{x['name']}: {s}" for x in others for s in x.get("must_not_reveal", [])]
    boundary = c.get("pov_knowledge_boundary") or []

    items = [
        "- pov_knowledge: văn xuôi có TƯỜNG THUẬT trực tiếp điều người kể "
        f"({c.get('pov_character_name') or pov}) không thể biết không? Chỉ dấu vết "
        "quan sát được là hợp lệ."
        + (f"\n  Người kể KHÔNG biết: {'; '.join(boundary[:12])}" if boundary else "")
        + (f"\n  Hành động ngầm người kể không thấy: {'; '.join(hidden)}" if hidden else "")
        + (f"\n  Bí mật không được nói ra: {'; '.join(secrets)}" if secrets else ""),
    ]
    if c.get("scene_must_change"):
        items.append(f"- scene_must_change: cảnh có thực hiện thay đổi này không? "
                     f"“{c['scene_must_change']}”"
                     + (f" (trạng thái cuối mong đợi: {c['exit_state']})"
                        if c.get("exit_state") else ""))
    items.append("- subtext: thoại có nói thẳng động cơ, cảm xúc lẽ ra phải để ngầm không?")
    items.append("- character_too_easy: có nhân vật nào nhượng bộ mà không có lý do hay giá phải trả?")
    if c.get("plant_directives"):
        items.append("- plant_visibility: chi tiết cần cài có quá lộ (được bình luận, "
                     "nhấn mạnh) hoặc quá mờ (không xuất hiện) không?")
    return "\n".join(items)


def parse_llm_findings(raw: str, prose: str) -> tuple[list[dict], list[dict]]:
    """→ (findings đã xác minh, issues). Không bao giờ ném lỗi: tầng LLM là tuỳ
    chọn, nó hỏng thì tầng thuật toán vẫn đứng."""
    issues: list[dict] = []
    # Thử nguyên chuỗi TRƯỚC: `strip_fences` lấy khối `{…}` dài nhất, nên với
    # một MẢNG trần `[{…}, {…}]` nó trả về một phần tử và làm rơi phần còn lại.
    data = None
    for candidate in ((raw or "").strip(), strip_fences(raw or "")):
        try:
            data = json.loads(candidate)
            break
        except (ValueError, TypeError) as e:
            err = e
    if data is None:
        return [], [{"stage": "llm_audit", "reason": "unparseable",
                     "message": _clip(f"{type(err).__name__}: {err}", 200)}]
    if isinstance(data, dict) and "findings" not in data and "severity" in data:
        data = [data]                   # một finding đơn lẻ, không bọc
    items = data.get("findings", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], [{"stage": "llm_audit", "reason": "findings_not_list",
                     "message": _clip(raw, 200)}]

    hay = _norm(prose)
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            issues.append({"stage": "llm_audit", "reason": "not_object",
                           "message": _clip(it, 200)})
            continue
        sev = _canon_sev(str(it.get("severity", "")))
        check = _canon_check(str(it.get("check", "")))
        msg = str(it.get("message", "")).strip()
        ev = str(it.get("evidence") or "").strip().strip("“”\"")
        ok, why = _found(ev, hay, MIN_SPAN_LEN) if ev else (False, "no_evidence")
        if not ok:
            issues.append({"stage": "llm_audit", "reason": why, "check": check,
                           "severity": sev, "message": _clip(msg, 200)})
            continue
        if sev == "blocker" and check not in LLM_BLOCKER_CHECKS:
            issues.append({"stage": "llm_audit", "reason": "blocker_capped",
                           "check": check, "message": _clip(msg, 200)})
            sev = "major"
        out.append({"severity": sev, "check": check, "message": msg,
                    "evidence": ev, "source": "llm"})
    return out, issues


# ═══════════════════════ Polish ═══════════════════════

_WORD = re.compile(r"\w+")


def _tokens(s: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", s or "").lower())


def _mentions(text: str, name: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(unicodedata.normalize("NFC", name)) + r"(?!\w)",
                     unicodedata.normalize("NFC", text)) is not None


def clean_polish_output(raw: str) -> str:
    s = (raw or "").strip()
    m = re.fullmatch(r"```[\w-]*\s*\n(.*?)\n?```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"\A\s*#{1,6}\s*VĂN XUÔI[^\n]*\n", "", s)
    # Model hay chép lại câu hướng dẫn cuối của POLISH_TMPL. Dòng đó ngắn nên
    # `content_drifted` để lọt, và nó thành văn xuôi — rồi thành span canon.
    s = re.sub(r"\n\s*Chỉ xuất văn xuôi[^\n]*\s*\Z", "", s)
    return s.strip()


def content_drifted(draft: str, polished: str, names=(),
                    threshold: float = 0.25) -> str | None:
    """Lý do từ chối bản Polish, hoặc `None` nếu nhận được.

    §9.2 gọi `content_drifted` nhưng không định nghĩa. Polish được sửa câu chữ;
    thứ nó KHÔNG được đổi đo được bằng code: ai có mặt (tên), con số, và tỉ lệ
    từ ngữ giữ nguyên. Tên và con số là hai thứ Extractor sẽ biến thành canon.
    """
    a, b = _tokens(draft), _tokens(polished)
    if not b:
        return "bản polish rỗng"
    if not a:
        return None
    lo = 1 - threshold
    ratio = len(b) / len(a)
    if not (lo <= ratio <= 1 / lo):
        return f"độ dài đổi còn {ratio:.0%} bản nháp"
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    if sim < lo:
        return f"chỉ giống bản nháp {sim:.0%} (< {lo:.0%}) — đã viết lại, không phải trau chuốt"
    na = {n for n in names if n and _mentions(draft, n)}
    nb = {n for n in names if n and _mentions(polished, n)}
    if na != nb:
        return (f"tên riêng đổi: thêm {sorted(nb - na)}, mất {sorted(na - nb)}")
    da, db = set(re.findall(r"\d+", draft)), set(re.findall(r"\d+", polished))
    if da != db:
        return f"con số đổi: thêm {sorted(db - da)}, mất {sorted(da - db)}"
    return None


def new_serious(before: list[dict], after: list[dict]) -> list[str]:
    """Lỗi blocker/major xuất hiện ở `after` nhiều hơn ở `before`."""
    def count(fs):
        return Counter((f.get("check"), f.get("severity")) for f in fs
                       if f.get("severity") in ("blocker", "major"))
    b, a = count(before), count(after)
    return [f"{c}:{s}" for (c, s), n in sorted(a.items()) if n > b.get((c, s), 0)]
