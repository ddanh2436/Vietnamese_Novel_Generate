"""Vệ sinh văn bản — bắt rác mà mọi tầng kiểm tra khác bỏ qua.

Arc 1 lượt 2, đọc bằng mắt mới thấy:

    Chương 3: "kenкет"  — ba chữ Cyrillic lọt vào giữa một từ tiếng Việt
    Chương 4: "khoಹ್"   — chữ Kannada
    Chương 3, 4, 5: năm cảnh dùng ngoặc kép cho thoại, phần còn lại dùng gạch
                    đầu dòng — có chương lẫn cả hai kiểu

Không luật nào ở §10 nhìn thấy những thứ này: chúng không phải sáo ngữ, không
phải nhịp câu, không phải rò rỉ POV. Nhưng người đọc thấy ngay, và một ký tự
Kannada giữa trang văn tiếng Việt phá tan ảo giác hơn bất kỳ lỗi nhịp nào.

Ký tự lạ được DỌN bằng code chứ không bắt viết lại cả cảnh: đây là lỗi đánh máy
của model, không phải lỗi sáng tác — bắt LLM viết lại 900 từ vì ba ký tự là đốt
tiền. Việc dọn được ghi lại để tác giả thấy (NT-13: code sửa thì code khai).
"""
from __future__ import annotations

import re
import unicodedata

_DIALOGUE_START = ("—", "–", "“", '"')
_ALLOWED_SCRIPTS = ("LATIN",)


def _is_foreign_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    name = unicodedata.name(ch, "")
    return not any(s in name for s in _ALLOWED_SCRIPTS)


def foreign_letters(text: str) -> list[dict]:
    """Chữ cái ngoài hệ Latin, kèm từ chứa nó."""
    out = []
    for m in re.finditer(r"\S*\S", text or ""):
        tu = m.group(0)
        la = [c for c in tu if _is_foreign_letter(c)]
        if la:
            out.append({"word": tu, "chars": sorted({unicodedata.name(c, "?") for c in la})})
    return out


_LEAKED_INDEX = re.compile(r"^\s*\d+(?:[.,]\d+)*\s*[.:]\s+(?=\S)")
# Dòng chỉ có mỗi con số ("0.04.", "45.") — kiểu rò rỉ trắng trợn nhất.
_LEAKED_LINE = re.compile(r"^\s*\d+(?:[.,]\d+)*\s*[.:]?\s*$")
# Chỉ số đứng RIÊNG như một câu, ở cuối dòng hoặc giữa đoạn:
#   "…gót giày lại nện xuống nền sắt, đếm nhịp. 04."
#   "…lật qua lật lại tờ giấy. 03. Một số hiệu vị trí nhảy một quãng…"
# Ca giữa đoạn lọt qua cả ba luật kia và M15 báo "sạch" — một bộ đo mù báo số 0
# thì tệ hơn không đo, vì nó còn làm người đọc yên tâm.
# Bắt buộc có số 0 đứng đầu: "04." là chỉ số, còn "Anh chờ 45." có thể là văn.
_LEAKED_MID = re.compile(r"(?<=[.!?])\s+0\d*(?:[.,]\d+)*\s*[.:](?=\s|$)")
# Lượt thoại mà cả lời nói chỉ là một con số. Phải tách làm hai, vì hai thứ
# trông giống hệt nhau mà bản chất ngược nhau — bài học từ lượt viết lại Arc 1,
# nơi luật gộp chung nổ 17 lần và một nửa là báo nhầm:
#
#   "— 4."     Kaelen ĐẾM THÀNH TIẾNG. Bản Arc 1 lượt 2 viết "— Bốn." rồi
#              "— Bốn gì?" — đó là tật của nhân vật, không phải rác. Chữa bằng
#              cách viết số thành chữ, không bắt viết lại cảnh.
#   "— 01."    Chỉ số beat: số 0 đứng đầu hoặc có phần thập phân. Không nhân vật
#   "— 0.03."  nào nói "không phẩy không ba" ở giữa một cuộc cãi vã.
# Câu dẫn đi sau vẫn tính: "— 1. — Kaelen nói" cũng là một lượt thoại rỗng.
_SPEECH_NUM = re.compile(r"^(\s*[—–]\s*)(\d+(?:[.,]\d+)*)\s*([.:]?)\s*(?=[—–]|$)")


def _la_chi_so(so: str) -> bool:
    """Số 0 đứng đầu hoặc có phần thập phân — không ai đếm thành tiếng như vậy."""
    return "." in so or "," in so or (len(so) > 1 and so.startswith("0"))


_DON = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


def so_thanh_chu(n: int) -> str:
    """Số thành chữ tiếng Việt, đủ dùng cho 0–999.

    Tật đếm của Kaelen không bao giờ đếm tới nghìn; quá phạm vi thì trả rỗng để
    nơi gọi giữ nguyên chữ số thay vì bịa ra một cách đọc sai.
    """
    if n < 10:
        return _DON[n]
    if n < 100:
        chuc, don = divmod(n, 10)
        dau = "mười" if chuc == 1 else f"{_DON[chuc]} mươi"
        if don == 0:
            return dau
        duoi = {1: "mốt", 4: "tư", 5: "lăm"}.get(don, _DON[don]) if chuc > 1 else (
            "lăm" if don == 5 else _DON[don])
        return f"{dau} {duoi}"
    if n < 1000:
        tram, con = divmod(n, 100)
        if con == 0:
            return f"{_DON[tram]} trăm"
        if con < 10:
            return f"{_DON[tram]} trăm lẻ {_DON[con]}"
        return f"{_DON[tram]} trăm {so_thanh_chu(con)}"
    return ""


def so_thoai_thanh_chu(text: str) -> tuple[str, list[str]]:
    """Lượt thoại chỉ có một con số nguyên → viết số ấy bằng chữ."""
    ghi: list[str] = []
    ra = []
    for line in (text or "").split("\n"):
        m = _SPEECH_NUM.match(line)
        if m and not _la_chi_so(m.group(2)):
            chu = so_thanh_chu(int(m.group(2)))
            if chu:
                ghi.append(f"viết số thoại “{m.group(2)}” thành chữ")
                con = line[m.end():]
                # `\s*` đã nuốt dấu cách trước câu dẫn — trả lại, nếu không
                # thành "— Một.— Kaelen nói".
                line = (f"{m.group(1)}{chu.capitalize()}{m.group(3) or '.'}"
                        + (" " if con[:1] in ("—", "–") else "") + con)
        ra.append(line)
    return "\n".join(ra), ghi


def leaked_speech(prose: str) -> list[str]:
    """Lượt thoại mà cả lời nói chỉ là một CHỈ SỐ: "— 01.", "— 0.03."."""
    out = []
    for l in (prose or "").split("\n"):
        m = _SPEECH_NUM.match(l)
        if m and _la_chi_so(m.group(2)):
            out.append(l.strip()[:60])
    return out




def strip_leaked_indices(text: str) -> tuple[str, list[str]]:
    """Cắt chỉ số beat rò rỉ ở MỌI dòng, không riêng dòng đầu cảnh.

    Bản đầu chỉ quét dòng đầu tiên, vì 16/30 cảnh của Arc 1 mở đầu bằng một con
    số trần. Lượt viết lại Chương 1 cho thấy chỗ rò dịch đi chứ không biến mất:
    chỉ số trôi xuống giữa cảnh ("…đếm nhịp. 04.") và vào cả trong thoại.

    Ba dạng cắt được vì chúng CHẮC CHẮN là rác. Dạng thứ tư — lượt thoại mà cả
    lời nói chỉ là một con số — thì KHÔNG: ở đó model đáng lẽ phải viết một câu
    thoại và nó đã không viết. Code không bịa được câu còn thiếu, nên việc đó
    thuộc về `hygiene_findings` và phải viết lại (NT-13).
    """
    ghi: list[str] = []
    ra: list[str] = []
    for line in (text or "").split("\n"):
        if line.strip() and _LEAKED_LINE.match(line):
            ghi.append(f"bỏ dòng chỉ số “{line.strip()}”")
            continue
        m = _LEAKED_INDEX.match(line)
        if m:
            ghi.append(f"cắt chỉ số đầu dòng “{m.group(0).strip()}”")
            line = line[:m.start()] + line[m.end():]
        for t in reversed(list(_LEAKED_MID.finditer(line))):
            ghi.append(f"cắt chỉ số “{t.group(0).strip()}” đứng riêng giữa câu")
            line = line[:t.start()] + line[t.end():]
        ra.append(line)
    while ra and not ra[0].strip():
        ra.pop(0)
    return "\n".join(ra), ghi


def sanitize_prose(text: str) -> tuple[str, list[str]]:
    """Dọn những thứ CHẮC CHẮN là rác. Trả `(văn bản sạch, ghi chú)`.

    Bốn việc, không đụng tới câu chữ: cắt chỉ số rò rỉ ở mọi dòng, bỏ chữ cái
    ngoài hệ Latin, đưa gạch đầu dòng thoại về một ký tự "—", bỏ khoảng trắng thừa.
    """
    text, notes = strip_leaked_indices(text)
    text, chu = so_thoai_thanh_chu(text)
    notes += chu
    la = foreign_letters(text)
    if la:
        text = "".join("" if _is_foreign_letter(c) else c for c in text)
        notes.append("bỏ ký tự ngoài hệ Latin: "
                     + ", ".join(f"“{x['word']}”" for x in la[:4]))

    dong, doi_gach = [], 0
    for line in (text or "").split("\n"):
        s = line.rstrip()
        m = re.match(r"^(\s*)([-–])(\s)", s)
        if m:
            s = f"{m.group(1)}—{m.group(3)}{s[m.end():]}"
            doi_gach += 1
        dong.append(s)
    if doi_gach:
        notes.append(f"chuẩn hoá {doi_gach} gạch đầu dòng thoại về “—”")
    return "\n".join(dong), notes


_NGUYEN_AM = set("aăâeêioôơuưy")
MAX_NGUYEN_AM = 3         # "uyê", "oai", "ươi" — tiếng Việt không có cụm dài hơn


def _bo_dau(tu: str) -> str:
    """Bỏ dấu thanh, GIỮ dấu tạo chữ (ă â ê ô ơ ư) — chúng là chữ cái riêng."""
    ra = "".join(c for c in unicodedata.normalize("NFD", tu.lower())
                 if unicodedata.category(c) != "Mn"
                 or c in "\u0306\u0302\u031b")
    return unicodedata.normalize("NFC", ra)


def malformed_words(text: str) -> list[str]:
    """Từ không thể là tiếng Việt vì chuỗi nguyên âm quá dài.

    Bắt được thứ mà `foreign_letters` mù, vì nó toàn chữ Latin: "suyuy giảm",
    "ngoàiập" — lỗi gõ của model, hai từ dính vào nhau hoặc một âm bị nhân đôi.

    Ngưỡng đo trên 42 nghìn từ của mười chương: cụm ba nguyên âm là bình thường
    ("chuyến", "người", "quyết"), cụm bốn thì chỉ có đúng hai từ hỏng và không
    một từ thật nào. Luật phụ âm thì ngược lại — "ngh" hợp lệ, "hertz", "inch",
    "atmosphere" đều là từ mượn thật — nên không đặt luật ở đó.

    Giới hạn phải nói thẳng: lỗi gõ thành một âm CÓ THẬT thì bộ này mù. "khớp
    ngón tro" (đáng lẽ "trỏ") lọt qua, vì "tro" là một từ tiếng Việt.
    """
    xau = []
    for tu in re.findall(r"[^\W\d_]+", text or ""):
        dem = 0
        for c in _bo_dau(tu):
            dem = dem + 1 if c in _NGUYEN_AM else 0
            if dem > MAX_NGUYEN_AM:
                xau.append(tu)
                break
    return xau


def dialogue_style(prose: str) -> str:
    """`dash` | `quote` | `mixed` | `none` — quy ước thoại của một cảnh."""
    dash = len(re.findall(r"^\s*[—–]", prose or "", flags=re.M))
    quote = len(re.findall(r'^\s*["“]', prose or "", flags=re.M))
    if dash and quote:
        return "mixed"
    if quote:
        return "quote"
    return "dash" if dash else "none"


def hygiene_findings(prose: str) -> list[dict]:
    out = []
    for x in foreign_letters(prose):
        out.append({"severity": "major", "check": "foreign_script",
                    "message": f"ký tự ngoài hệ Latin trong “{x['word']}”: "
                               f"{', '.join(x['chars'])}",
                    "evidence": x["word"]})
    for l in leaked_speech(prose):
        out.append({"severity": "major", "check": "leaked_index_in_dialogue",
                    "message": ("cả lượt thoại chỉ là một chỉ số beat — nhân vật "
                                "phải NÓI một câu ở chỗ này; code không bịa được "
                                "câu còn thiếu nên cảnh phải viết lại"),
                    "evidence": l})
    for tu in malformed_words(prose):
        out.append({"severity": "minor", "check": "malformed_word",
                    "message": (f"“{tu}” không phải một từ tiếng Việt — nhiều khả "
                                f"năng là lỗi gõ hoặc hai từ dính vào nhau"),
                    "evidence": tu})
    kieu = dialogue_style(prose)
    if kieu in ("quote", "mixed"):
        out.append({"severity": "minor", "check": "dialogue_style",
                    "message": (f"thoại dùng kiểu '{kieu}' — cả tiểu thuyết dùng "
                                f"gạch đầu dòng “—”; lẫn kiểu làm người đọc vấp")})
    return out
