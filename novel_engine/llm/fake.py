"""`FakeLLM` — chạy cả pipeline, tất định, miễn phí.

Đây KHÔNG phải mock cho có. Nhờ NT-1 (LLM chỉ diễn đạt, không quyết định),
mọi quyết định của hệ thống — cấp phát thời gian, chọn manh mối, guard quan
hệ, chấm nhịp văn, phân loại delta — vẫn là quyết định THẬT khi chạy với
FakeLLM. Thứ giả duy nhất là câu chữ.

Vì vậy nó bắt được đúng loại lỗi mà §0.5 nói rà soát tĩnh không bao giờ bắt
được: kiểu dữ liệu thật giữa các node, hành vi thật của reducer LangGraph,
vòng lặp vô hạn, JSON không parse được. Chạy trong pytest, vài giây, không
cần mạng, không cần ví.

Văn xuôi nó sinh ra CỐ Ý dở nhưng CỐ Ý hợp lệ: đúng ngân sách từ, có đủ số
kênh giác quan, không chứa sáo ngữ trong danh sách cấm — để Auditor tầng
thuật toán (§10.3) chạy thật chứ không bị chặn ngay ở luật đầu tiên.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

# Kho câu để ghép văn xuôi giả. Chọn theo hash của prompt nên CÙNG prompt luôn
# cho CÙNG kết quả — bộ hồi quy §13.2 cần tính tất định này.
_FRAGMENTS = [
    "Cần trục kêu ba tiếng rồi im",
    "Mùi lưu huỳnh bám vào cổ áo",
    "Nền rung một nhịp dưới gót chân",
    "Ánh đèn pha quét ngang mặt nước",
    "Giấy tờ trong tay ai đó sột soạt",
    "Một tiếng động khô phía sau bức tường",
    "Không khí khô rát ở đầu lưỡi",
    "Tiếng vọng dội lại ba lần rồi tắt",
    "Hơi nước đọng thành giọt trên mép ống",
    "Bụi quặng đỏ lắng xuống vai áo",
]

_DIALOGUE = [
    "— Anh xếp hàng bên kia.",
    "— Tôi ký. Đúng giờ đó.",
    "— Về mặt thủ tục, điều đó không cần thiết.",
    "— Sai số bao nhiêu?",
    "— Hết giờ tiếp nhận rồi.",
    "— Tôi e rằng hồ sơ cho thấy điều khác.",
]


def _seed(text: str) -> int:
    """blake2b, không dùng `hash()` — NT-16: khoá sống lâu hơn một tiến trình
    phải tất định qua mọi PYTHONHASHSEED, và test hồi quy chạy ở tiến trình
    khác với lúc sinh."""
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _field(prompt: str, label: str) -> str:
    """Móc một giá trị ra khỏi prompt đã render, để đầu ra giả vẫn NHẮC ĐÚNG
    tên nhân vật và địa điểm của cảnh. Không có nó thì `verify_spans` và
    `pov_leak_scan` chạy trên văn bản chẳng liên quan gì tới hợp đồng."""
    m = re.search(rf"{re.escape(label)}\s*[:=]\s*([^\n,}}\]]+)", prompt)
    return m.group(1).strip().strip("'\"") if m else ""


def _line(prompt: str, label: str) -> str:
    """Như `_field` nhưng đọc TRỌN DÒNG — `_field` dừng ở dấu phẩy, nên nó
    cắt mất mọi nhân vật sau người đầu tiên trong "Nhân vật có mặt"."""
    m = re.search(rf"{re.escape(label)}\s*[:=]\s*([^\n]+)", prompt)
    return m.group(1).strip() if m else ""


class FakeLLM:
    """Triển khai `LLMPort`. Không gọi mạng, không tốn tiền, tất định."""

    def __init__(self, *, word_target: int = 320) -> None:
        self.word_target = word_target
        self.calls: list[dict[str, str]] = []      # nhật ký, để test soi

    def invoke(self, prompt: str, *, role: str = "") -> str:
        self.calls.append({"role": role, "prompt": prompt})
        handler = {
            "writer": self._prose,
            "polish": self._polish,
            "scene_digest": self._scene_close,
            "director": self._director_fill,
            "auditor": self._findings,
            "extractor_diff": self._extract_diff,
            "extractor_emergent": self._extract_emergent,
        }.get(role, self._prose)
        return handler(prompt)

    # ───────────────────── văn xuôi ─────────────────────

    def _prose(self, prompt: str) -> str:
        """Văn xuôi giả nhưng HỢP LỆ với Auditor tầng thuật toán: đủ độ dài,
        ≥3 kênh giác quan, độ dài câu biến thiên (chống cờ 'văn đều đều'
        §10.3.1), không chứa sáo ngữ trong `CLICHE_SOMATICS`."""
        n = _seed(prompt)
        pov = _field(prompt, "Người kể") or "Nhân vật"
        pov = pov.split(",")[0].strip()

        # `random.Random(n)` thay cho `(n + i) % len`: chỉ số tuần tự cho mỗi POV
        # đúng 30 biến thể văn xuôi, nên hai cảnh cùng POV trùng nhau NGUYÊN VĂN
        # khi hash prompt rơi cùng lớp đồng dư — xảy ra thật khi WRITER_TMPL
        # thêm một dòng ở Ngày 10. Vẫn tất định: cùng prompt, cùng seed.
        rng = random.Random(n)
        out: list[str] = []
        words = 0
        i = 0
        while words < self.word_target:
            frag = rng.choice(_FRAGMENTS)
            if i % 4 == 3:
                # Câu ngắn xen vào — σ độ dài câu phải khác 0, nếu không
                # `rhythm.py` gắn cờ đúng ở lần chạy đầu tiên.
                sent = f"{frag}."
            elif i % 5 == 2:
                out.append(rng.choice(_DIALOGUE))
                words += 6
                i += 1
                continue
            else:
                sent = (f"{frag}, và {pov} ghi nhận điều đó mà không để lộ ra "
                        f"ngoài nét mặt, bởi vì việc để lộ ra sẽ khiến người "
                        f"đối diện biết mình vừa chú ý tới cái gì.")
            out.append(sent)
            words += len(sent.split())
            i += 1
        return " ".join(out)

    def _polish(self, prompt: str) -> str:
        """Polish phải trả về thứ GẦN GIỐNG bản gốc, nếu không
        `content_drifted` (§9.2) sẽ từ chối nó ở mọi cảnh và ta không bao giờ
        test được nhánh chấp nhận."""
        # Dừng TRƯỚC câu hướng dẫn cuối của POLISH_TMPL — không thì dòng
        # "Chỉ xuất văn xuôi…" lọt vào văn xuôi của mọi cảnh.
        m = re.search(r"## VĂN XUÔI\s*\n(.*?)(?:\n##|\n\s*Chỉ xuất văn xuôi|\Z)",
                      prompt, re.S)
        if m:
            return m.group(1).strip()
        return self._prose(prompt)

    # ───────────────────── JSON ─────────────────────

    def _scene_close(self, prompt: str) -> str:
        """`SceneClose`. Cố ý bọc trong ```json để `strip_fences` (§9.5) được
        chạy thật ở mỗi ranh giới cảnh, chứ không chỉ ở test riêng của nó."""
        # Chỉ đọc DÒNG "Nhân vật có mặt", không quét cả prompt. Bản đầu quét
        # toàn prompt và nhặt phải mã trong phần ví dụ minh hoạ — Kaelen hiện
        # ra ở một cảnh anh ta không có mặt. Một model thật cũng mắc đúng lỗi
        # đó, nên quét cả prompt vừa sai vừa che mất bug của prompt.
        line = _line(prompt, "Nhân vật có mặt")
        present = list(dict.fromkeys(re.findall(r"CHAR_[A-Z0-9_]+", line)))
        m_loc = re.search(r"LOC_[A-Z0-9_]+", prompt)
        loc = m_loc.group(0) if m_loc else "LOC_ORE_PORT"
        planned = _field(prompt, "duration_ticks dự kiến")
        payload: dict[str, Any] = {
            "digest": ("Cảnh diễn ra đúng theo hợp đồng. Các nhân vật có mặt "
                       "trao đổi ngắn rồi tách ra. Một chi tiết vật lý được "
                       "ghi nhận mà chưa ai diễn giải. Trạng thái cuối cảnh: "
                       "chưa ai rời khỏi khu vực."),
            "continuity": {
                "locations": {p: loc for p in present} or
                             {"CHAR_KAELEN": loc},
                "injuries": {},
                "possessions": {},
                "weather": None,
            },
            # F2: model CHỈ đóng góp thời lượng thực — `narrative_order` và
            # `epoch_tick` là của code, và schema không có chỗ cho chúng.
            "actual_duration_ticks": int(planned) if planned.isdigit() else 3,
            "unresolved": ["một tiếng động phía sau bức tường chưa ai kiểm tra"],
        }
        return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

    def _director_fill(self, prompt: str) -> str:
        """Director chỉ được điền các khoá mà code CỐ Ý để trống (§9.2).
        `merge_json` sẽ bỏ qua mọi thứ khác — kể cả khi FakeLLM cố ghi đè."""
        return json.dumps({
            "scene_must_change": "Cán cân thông tin giữa hai người phải đổi chiều",
            "entry_state": "Hai bên còn giữ thế thủ",
            "exit_state": "Một bên đã để lộ thứ mình định giấu",
            "subtext_requirement": "Không ai nói thẳng điều mình muốn",
            # Thử ghi đè trường của hệ thống — merge_json PHẢI chặn:
            "scene_id": "KHONG_DUOC_GHI_DE",
            "time": {"epoch_tick": 99999},
        }, ensure_ascii=False)

    def _findings(self, _prompt: str) -> str:
        return json.dumps({"findings": []}, ensure_ascii=False)

    def _extract_diff(self, prompt: str) -> str:
        """Lượt 1 — kiểm toán vi sai. Trích span THẬT từ văn bản trong prompt.

        Một mock trả về `{"assertions": []}` làm `verify_spans` không có gì để
        xác minh, và lượt 3 — cơ chế quan trọng nhất của §10.5 — thành mã chết
        trong mọi test. Ở đây ta chép nguyên văn một câu có thật, đúng như một
        extractor tốt phải làm.
        """
        prose = self._prose_of(prompt)
        cau = self._sentences(prose)
        if not cau:
            return json.dumps({"assertions": []}, ensure_ascii=False)
        n = _seed(prompt)
        scene_ids = re.findall(r"\[(CH\d+_S\d+)\]", prompt) or ["CH001_S00"]
        chars = list(dict.fromkeys(re.findall(r"CHAR_[A-Z0-9_]+", prompt)))
        return "```json\n" + json.dumps({
            "assertions": [
                {"subject": chars[0] if chars else "CHAR_KAELEN",
                 "predicate": "observed", "object": True,
                 "chapter": 0, "scene": 0,
                 "span": cau[n % len(cau)],          # span CÓ THẬT
                 "confidence": 0.9, "epistemic": "objective"},
                # …và một span BỊA, để `verify_spans` có việc làm ở mọi lượt
                # chạy. Không có nó thì nhánh loại-bỏ không bao giờ được thực
                # thi và ta không biết nó còn sống hay đã chết.
                {"subject": chars[-1] if chars else "CHAR_VHAL",
                 "predicate": "claimed", "object": "điều không có thật",
                 "chapter": 0, "scene": 0,
                 "span": "Câu này tuyệt đối không tồn tại trong văn bản gốc.",
                 "confidence": 0.7, "epistemic": "claimed_by",
                 "holder": chars[-1] if chars else "CHAR_VHAL"},
            ],
            "plant_evidence": [
                {"clue_id": cid, "scene_id": scene_ids[0],
                 "span": cau[(n + i) % len(cau)], "carrier_used": "setting"}
                for i, cid in enumerate(
                    dict.fromkeys(re.findall(r"CLUE_[A-Z0-9_]+", prompt)))
            ],
        }, ensure_ascii=False) + "\n```"

    def _extract_emergent(self, prompt: str) -> str:
        """Lượt 2 — quét phát sinh. KHÔNG thấy contract, nên nó chỉ biết văn
        bản và danh mục thực thể đã có."""
        prose = self._prose_of(prompt)
        cau = self._sentences(prose)
        if not cau:
            return json.dumps({"new_entities": []}, ensure_ascii=False)
        n = _seed(prompt + "emergent")
        return "```json\n" + json.dumps({
            "new_entities": [
                {"id": "OBJ_SO_CA_TRUC", "kind": "object",
                 "name": "sổ ca trực", "aliases": []},
            ],
            "assertions": [
                {"subject": "OBJ_SO_CA_TRUC", "predicate": "exists_at",
                 "object": "LOC_ORE_PORT", "chapter": 0, "scene": 0,
                 "span": cau[n % len(cau)], "confidence": 0.8,
                 "epistemic": "objective"},
            ],
        }, ensure_ascii=False) + "\n```"

    @staticmethod
    def _prose_of(prompt: str) -> str:
        """Lấy phần VĂN BẢN ra khỏi prompt extractor."""
        m = re.search(r"## VĂN BẢN[^\n]*\n(.*?)(?:\n## |\Z)", prompt, re.S)
        return m.group(1) if m else ""

    @staticmethod
    def _sentences(prose: str) -> list[str]:
        """Câu đủ dài để vượt `MIN_SPAN_LEN`. Bỏ nhãn [SCENE_ID]."""
        clean = re.sub(r"\[CH\d+_S\d+\]", " ", prose)
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean)
                if len(s.strip()) >= 20][:40]

    def _delta(self, prompt: str) -> str:
        return json.dumps({"assertions": [], "new_entities": [],
                           "new_relations": [], "plant_evidence": []},
                          ensure_ascii=False)


class ScriptedLLM:
    """Trả về các phản hồi đã soạn sẵn, theo thứ tự. Dùng cho test cần dựng
    một tình huống CỤ THỂ (Auditor trả blocker hai lần liên tiếp, JSON hỏng,
    văn xuôi rò rỉ POV) mà `FakeLLM` không tự sinh ra."""

    def __init__(self, responses: list[str], fallback: "FakeLLM | None" = None):
        self.responses = list(responses)
        self.fallback = fallback or FakeLLM()
        self.calls: list[dict[str, str]] = []

    def invoke(self, prompt: str, *, role: str = "") -> str:
        self.calls.append({"role": role, "prompt": prompt})
        if self.responses:
            return self.responses.pop(0)
        return self.fallback.invoke(prompt, role=role)
