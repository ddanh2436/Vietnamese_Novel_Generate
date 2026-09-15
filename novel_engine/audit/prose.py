"""Kiểm tra văn phong bằng code (§10.3).

Việc code làm hoàn hảo — đếm, so khớp, tra bảng — thì không giao cho LLM.

═══ SO KHỚP THEO RANH GIỚI TỪ, KHÔNG THEO CHUỖI CON ═══════════════════════

Mã §10.3 dò kênh giác quan bằng `re.search(r"(nhìn|thấy|ánh|màu|...)", low)` —
chuỗi con. Với tiếng Việt, gần như MỌI câu đều khớp:

    "thấy" ⊂ "cảm thấy"    "âm"  ⊂ "tâm", "năm", "cầm"
    "vị"   ⊂ "vị trí"      "tối" ⊂ "tối đa"    "ánh" ⊂ "tránh", "cánh"

Câu "Anh cảm thấy tâm trạng rối bời ở vị trí đó" không có chi tiết giác quan
nào mà được tính ba kênh. Kiểm tra `sensory` vì thế không bao giờ kích hoạt —
một kiểm tra chết, chết im lặng. Ở đây mọi cụm so khớp theo ranh giới từ, cụm
nhiều âm tiết được ưu tiên cho từ đa nghĩa, và có danh sách loại trừ cho các
nghĩa bóng hay gặp ("nóng lòng", "mồ hôi", "cay đắng").
"""
from __future__ import annotations

import re
import unicodedata

from novel_engine.character.models import CLICHE_SOMATICS

# NT-11: sáo ngữ CƠ THỂ chỉ khai một nơi — `character.models.CLICHE_SOMATICS`,
# vốn đã đi vào `somatic_forbidden` của mọi SceneContract. §10.3 khai một bản
# thứ hai khác nội dung; hai bản sẽ lệch nhau và Writer bị cấm một danh sách
# còn Auditor chấm theo danh sách khác.
CLICHE_PHRASES = [
    "không khí trở nên nặng nề", "thời gian như ngừng lại",
    "một nụ cười nhếch mép", "ánh mắt sắc như dao", "im lặng đến đáng sợ",
    "một cảm giác khó tả", "không thể tin vào mắt mình",
    "tim như ngừng đập", "hơi thở dồn dập", "nín thở",
]

SENSORY_CUES: dict[str, list[str]] = {
    "thị giác": ["nhìn", "nhìn thấy", "trông thấy", "ánh sáng", "ánh đèn",
                 "màu", "sáng rực", "tối om", "bóng tối", "lấp lánh", "chói",
                 "mờ mịt", "nheo mắt", "le lói", "loá"],
    "thính giác": ["nghe", "tiếng", "âm thanh", "vang", "rít", "lách cách",
                   "ken két", "im bặt", "tĩnh lặng", "gầm", "rền", "ù ù",
                   "thì thầm", "rì rầm", "chát chúa"],
    "khứu giác": ["mùi", "thơm", "hôi", "khét", "tanh", "hăng", "ngai ngái",
                  "nồng", "thum thủm", "hắc"],
    "xúc giác": ["chạm", "lạnh buốt", "giá lạnh", "hơi lạnh", "nóng", "rát",
                 "nhám", "ẩm", "ướt", "nhớp", "rung", "tê", "buốt", "sần",
                 "trơn", "ráp", "đau nhói", "ram ráp", "run lên"],
    "vị giác": ["vị đắng", "vị mặn", "vị chua", "vị ngọt", "vị tanh",
                "mùi vị", "đắng", "mặn", "chua", "ngọt", "chát", "cay",
                "đầu lưỡi"],
}

# Nghĩa bóng / từ ghép KHÔNG phải chi tiết giác quan. Bị xoá khỏi văn bản trước
# khi dò, để "nóng lòng" không thành xúc giác và "mồ hôi" không thành khứu giác.
SENSORY_EXCLUDE = [
    "nổi tiếng", "tiếng việt", "tiếng anh", "danh tiếng", "mồ hôi",
    "nồng nhiệt", "nồng hậu", "nóng lòng", "nóng giận", "nóng nảy",
    "rung động", "cay đắng", "ngọt ngào", "chua chát", "tê tái", "mặn mà",
    "nghe nói", "nghe đồn", "vang danh",
]

EXPLICIT_GOAL_RE = re.compile(
    r"(?<!\w)(?:tôi|ta|anh|em) (?:muốn|cần|phải) (?:tìm|giết|cứu|lấy|đến)(?!\w)")
DIALOGUE_LINE_RE = re.compile(r"^\s*[—–“\"]", flags=re.M)


def nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


def contains_phrase(text_nfc_lower: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(nfc_lower(phrase)) + r"(?!\w)",
                     text_nfc_lower) is not None


def sensory_channels(prose: str) -> list[str]:
    low = nfc_lower(prose)
    for ex in SENSORY_EXCLUDE:
        low = re.sub(r"(?<!\w)" + re.escape(ex) + r"(?!\w)", " ", low)
    return [ch for ch, cues in SENSORY_CUES.items()
            if any(contains_phrase(low, c) for c in cues)]


def prose_audit(prose: str, c: dict) -> list[dict]:
    f: list[dict] = []
    words = len(prose.split())
    lo, hi = c.get("word_budget", (900, 1600))
    if not (lo <= words <= hi):
        f.append({"severity": "minor", "check": "length",
                  "message": f"{words} từ, ngoài khoảng [{lo},{hi}]"})

    # NFC trước khi so: cùng lý do với `verify_spans` — "ế" dựng sẵn và "ế" tổ
    # hợp hiện ra giống nhau nhưng không bằng nhau.
    low = nfc_lower(prose)
    for cl in CLICHE_SOMATICS + CLICHE_PHRASES + list(c.get("forbidden_cliches", [])):
        if contains_phrase(low, cl):
            f.append({"severity": "minor", "check": "cliche",
                      "message": f"sáo ngữ: “{cl}”"})

    need = c.get("sensory_channels_required", 3)
    used = sensory_channels(prose)
    if len(used) < need:
        f.append({"severity": "minor", "check": "sensory",
                  "message": f"chỉ {len(used)} kênh giác quan {used}, cần ≥{need}"})

    trần = c.get("max_explicit_goal_statements", 1)
    explicit = len(EXPLICIT_GOAL_RE.findall(low))
    if explicit > trần:
        f.append({"severity": "major", "check": "on_the_nose",
                  "message": f"{explicit} lần nói thẳng mục tiêu, trần là {trần}"})

    if not DIALOGUE_LINE_RE.search(prose) and words > 700:
        f.append({"severity": "minor", "check": "no_dialogue",
                  "message": "cảnh dài không có thoại"})
    return f
