"""POV Firewall — lỗ hổng nghiêm trọng nhất (L4, §5.4, §5.4.1).

Nếu Writer thấy toàn bộ world state, nó SẼ rò rỉ. Không phải vì model kém, mà
vì bất cứ thông tin nào trong context đều làm tăng xác suất xuất hiện ở output.
Một câu vô hại như "Kaelen không biết rằng phía sau bức tường là xưởng đúc bị
niêm phong" đã phá hỏng toàn bộ suspense — vì nó vừa nói ra điều đáng lẽ ẩn.

Ba lớp, nhưng HAI HÀM — và đó là điểm mấu chốt:

    lớp 1  bỏ fact POV không biết          ─┐ filter_memory (ngữ cảnh L1–L4)
    lớp 2  điều POV chỉ NGHI → gắn nhãn    ─┘
    lớp 3  xoá nội tâm nhân vật không phải POV  → filter_scene_contract

Gộp làm một chính là nguyên nhân của C4: `filter_contract` được gọi trên `ctx`
(một dict không có khoá `active_characters`) nên lớp 3 luôn duyệt danh sách
rỗng và trở thành mã chết — trong khi `SceneContract` đi thẳng vào prompt
không qua bộ lọc nào, mang theo `deliberation`, `secret_fear` và
`must_not_reveal` của mọi nhân vật.
"""
from __future__ import annotations

import re

# Trường nội tâm của nhân vật KHÔNG phải POV — xoá hẳn khỏi contract.
_INTERNAL_FIELDS = ("secret_fear", "internal_conflict", "deliberation",
                    "must_not_reveal", "knows_in_this_scene")

# Trường an toàn để Writer nhìn thấy ở nhân vật không phải POV.
_SAFE_FIELDS = ("id", "name", "immediate_goal", "skill_deployed",
                "voice_reminder", "somatic_allowed", "somatic_forbidden")

LEAK_PATTERNS = [
    r"(?:anh|cô|hắn|nàng) không (?:hề )?biết rằng",
    r"(?:điều|thứ) mà .{1,25} (?:không|chưa) (?:hề )?(?:biết|hay)",
    # Nội tâm người khác. Mẫu gốc ở §5.4 đòi một DẤU CÁCH ngay sau "thâm tâm",
    # nên nó trượt đúng cách viết phổ biến nhất: "Trong thâm tâm, Serena đã…".
    # Dấu phẩy sau trạng ngữ đầu câu là chuẩn tiếng Việt, không phải ngoại lệ.
    # Dấu phân cách phải tiêu thụ ÍT NHẤT một ký tự (`\s+` hoặc dấu câu), nếu
    # không `\s*` khớp rỗng ngay sau "lòng" rồi lookahead kiểm tại " của mình"
    # — thấy dấu cách chứ không thấy "của", nên guard không bao giờ chặn và
    # "giấu tấm thẻ trong lòng của mình" bị báo nhầm là rò rỉ nội tâm.
    r"trong (?:thâm tâm|đầu|lòng)(?:\s*[,:;]\s*|\s+)(?!của (?:mình|tôi|anh ấy)\b)",
    r"(?:thực ra|sự thật là) .{1,40} đang (?:nói dối|che giấu)",
]
INTERIOR_PATTERN = LEAK_PATTERNS[2]


def _interior_is_pov(prose: str, m: re.Match, pov_name: str) -> bool:
    """Mẫu nội tâm thuộc về CHÍNH POV thì không phải rò rỉ.

    Gemini Chương 2, cảnh Serena kể ngôi thứ nhất: "Tôi nhích nhẹ cổ tay…, trong
    lòng thoáng dấy lên một mối hoài nghi". Mẫu nội tâm bắt câu này thành
    BLOCKER, và ở Ngày 10 blocker đẩy cảnh vào vòng viết lại — vì một câu đúng.

    Miễn trừ chỉ khi chủ ngữ rõ ràng là POV: câu nêu tên POV, hoặc câu trần
    thuật mở bằng "Tôi"; VÀ không có tên riêng nào khác trong câu. Không có chủ
    ngữ nào ("Trong lòng: một thứ gì đó vỡ ra.") thì vẫn báo — thà hỏi lại.
    """
    start = max(prose.rfind(ch, 0, m.start()) for ch in ".!?…\n") + 1
    ends = [i for i in (prose.find(ch, m.end()) for ch in ".!?…\n") if i != -1]
    sentence = prose[start: min(ends) if ends else len(prose)].strip()
    line_start = prose.rfind("\n", 0, m.start()) + 1
    in_dialogue = prose[line_start:].lstrip()[:1] in ("—", "–", "“", '"')
    pov = pov_name.strip()
    first_person = not in_dialogue and re.match(r"Tôi(?!\w)", sentence) is not None
    named = bool(pov) and re.search(rf"(?<!\w){re.escape(pov)}(?!\w)", sentence) is not None
    if not (first_person or named):
        return False
    # `w[0].isupper()` chứ không phải `[A-ZÀ-Ỹ]`: dải À–Ỹ của Unicode xen kẽ
    # chữ HOA và chữ thường ("đ", "ẹ", "ổ"), nên "để", "nhẹ" bị tính là tên riêng.
    proper = {w for w in re.findall(r"\w+", sentence) if w[0].isupper()}
    return proper <= {pov, "Tôi"}


class POVFirewall:
    def __init__(self, graph):
        self.g = graph

    # ───────────────── LỚP 1 + 2: ngữ cảnh bộ nhớ ─────────────────

    def filter_memory(self, ctx: dict, pov_id: str, epoch_tick: int) -> dict:
        """Lọc NGỮ CẢNH BỘ NHỚ (L1–L4). Khoá theo `epoch_tick`, không theo
        chương (NT-6, §3.6.2).

        `ctx["known_facts"]` là `list[dict]` có khoá `id` — xem ghi chú trong
        `ContextAssembler.build` về vì sao nó KHÔNG phải `list[str]`.
        """
        ep = self.g.known_by(pov_id, epoch_tick)
        allowed = {f["id"] for f in ep["known"]}
        out = dict(ctx)
        out["known_facts"] = [f for f in ctx.get("known_facts", [])
                              if f.get("id") in allowed]
        out["hypotheses"] = [
            {"text": s["name"], "certainty": s["conf"],
             "must_render_as": "phỏng đoán, không phải sự thật"}
            for s in ep["suspected"]]
        return out

    # ───────────────── LỚP 3: hợp đồng cảnh ─────────────────

    def filter_scene_contract(self, contract: dict, pov_id: str) -> dict:
        """Lọc HỢP ĐỒNG CẢNH. Việc khác hẳn lớp 1–2: ở đây ta không lọc sự
        thật về thế giới mà lọc NỘI TÂM của các nhân vật không phải POV."""
        out = dict(contract)
        cleaned = []
        for ch in contract.get("active_characters", []):
            if ch["id"] == pov_id:
                cleaned.append(ch)                  # POV giữ nguyên mọi thứ
                continue

            safe = {k: v for k, v in ch.items() if k in _SAFE_FIELDS}

            # BỎ HẲN: đây là nội tâm, và là thứ model sẽ tường thuật thẳng nếu
            # nhìn thấy. `deliberation.why` đặc biệt nguy hiểm vì nó là chuỗi
            # giải thích nội bộ kiểu "-1.30 flaw: sợ phản bội" — đưa vào prompt
            # là mời model viết luận về tâm lý nhân vật.
            for k in _INTERNAL_FIELDS:
                safe.pop(k, None)

            # GIỮ LẠI có kiểm soát: hành động ngầm. Writer CẦN biết nó, vì dấu
            # vết quan sát được phải là dấu vết CỦA CHÍNH hành động đó. Thay
            # bằng một câu chung chung ("có vẻ đáng ngờ") sẽ cho ra văn xuôi
            # chung chung — tức là làm hỏng đúng cơ chế mà nó định bảo vệ.
            if ch.get("hidden_action"):
                safe["director_only"] = {
                    "hidden_action": ch["hidden_action"],
                    "_hard_constraint": (
                        "POV KHÔNG biết và KHÔNG suy ra được điều này. Chỉ "
                        "được gieo MỘT dấu vết vật lý mà POV nhìn thấy nhưng "
                        "hiểu sai hoặc bỏ qua. Cấm mọi câu tường thuật nội "
                        "tâm, ý định hay cảm xúc của nhân vật này."),
                }
            # Chỉ hành động do POV quan sát được mới ở dạng trần
            safe["observable_behavior"] = ch.get("observable_behavior", "")
            cleaned.append(safe)

        out["active_characters"] = cleaned
        return out

    # ───────────────── tiện ích cho Director (§9.2) ─────────────────

    def known_ids(self, cid: str, epoch_tick: int) -> list[str]:
        return [f["id"] for f in self.g.known_by(cid, epoch_tick)["known"]]

    def boundary(self, pov_id: str, epoch_tick: int) -> list[str]:
        """`pov_knowledge_boundary` — điều POV KHÔNG biết, để Writer né.

        Liệt kê tên thực thể tồn tại trong canon mà POV chưa biết tới. §8.2
        dặn: "nếu cần nhắc, chỉ được nhắc qua sự hiểu sai của POV".
        """
        known = set(self.known_ids(pov_id, epoch_tick))
        known.add(pov_id)
        return [e["name"] for e in self.g.entity_index()
                if e["id"] not in known and e["kind"] in ("event", "doctrine",
                                                          "object", "clue")]


def pov_leak_scan(prose: str, pov_name: str) -> list[dict]:
    """Bộ kiểm tra SAU khi viết — lớp lọc prompt vẫn có thể bị model vượt qua.

    Heuristic, không hoàn hảo, nhưng bắt được phần lớn ca kinh điển với chi phí
    bằng không. Ca tinh vi để Auditor Agent xử lý (§10).
    """
    hits = []
    for pat in LEAK_PATTERNS:
        for m in re.finditer(pat, prose, flags=re.IGNORECASE):
            ctx = prose[max(0, m.start() - 60): m.end() + 60]
            if pov_name.lower() in ctx.lower() and "không biết" in ctx:
                continue      # POV tự nhận mình không biết → hợp lệ
            if pat == INTERIOR_PATTERN and _interior_is_pov(prose, m, pov_name):
                continue      # nội tâm của CHÍNH POV → hợp lệ
            hits.append({"pattern": pat, "span": ctx, "severity": "blocker",
                         "check": "pov_leak"})
    return hits
