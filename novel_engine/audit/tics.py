"""Ngân sách tật ngôn ngữ — chống nhồi cụm đặc trưng (§5.3, §10.3.2).

`VoiceFingerprint.signature_lexicon` sinh ra để giọng nhân vật nhận ra được. Đo
trên Arc 1 lượt 2 thì nó đã vượt qua điểm hữu ích:

    Chương 1 cảnh 4 — Serena dùng CẢ NĂM cụm đặc trưng, mỗi cụm ×3 trong một cảnh
    Chương 3 cảnh 4 — Vhal: "hết giờ tiếp nhận rồi" ×4, "không phải việc của tôi" ×4
    Chương 3 cảnh 2 — cụm thoại của Vhal tràn vào LỜI KỂ ở POV của chính ông ta
    16/30 cảnh      — mở đầu bằng một con số trần ("Hai.", "03. Hai mươi bốn.")

Người đọc không thấy "giọng riêng" nữa, chỉ thấy một cái khuôn đóng dấu.

Hai mức ngân sách, vì hai loại cụm hành xử khác nhau:

- Cụm NHIỀU TỪ ("hết giờ tiếp nhận rồi") là khẩu ngữ. Quá hai lần là nhại.
- Cụm NGẮN ("ca trực", "áp suất", "dung sai") vừa là khẩu ngữ vừa là danh từ
  chuyên ngành của thế giới này; cấm chặt sẽ thành báo động giả (P2), nên ngưỡng
  rộng hơn.

Tổng số lần một nhân vật dùng cụm đặc trưng trong một cảnh cũng có trần: năm cụm
khác nhau mỗi cụm hai lần vẫn là nhồi, dù không cụm nào phạm ngưỡng riêng.
"""
from __future__ import annotations

import re
import unicodedata

TIC_LIMIT = 2             # khẩu ngữ THUẦN (`verbal_tics`): quá hai lần là nhại
PHRASE_LIMIT = 2          # cụm ≥3 từ
SHORT_LIMIT = 4           # cụm ≤2 từ — cũng là thuật ngữ của thế giới
TOTAL_LIMIT = 5           # tổng lần dùng của MỘT nhân vật trong MỘT cảnh
NARRATION_LIMIT = 1       # khẩu ngữ trong lời kể
SOMATIC_PER_SCENE = 1     # cử chỉ nhận dạng: một lần là nhấn, hai lần là tật
SOMATIC_PER_CHAPTER = 1   # MỘT lần mỗi chương, và chỉ ở chỗ căng nhất
# Trần 3 vẫn quá rộng. Đọc bản viết lại Arc 1: cử chỉ nhận dạng có mặt ở 80–90%
# số cảnh — Kaelen bẻ khớp ngón tay, Serena xoay nhẫn đúng ba vòng, Vhal hà hơi
# lên con dấu — nên chúng thôi nhận dạng ai và thành tiếng động nền. Một cử chỉ
# chỉ đáng nhớ nếu nó hiếm: dùng một lần, ở cảnh nhân vật mất tự chủ nhất.
_DIALOGUE_START = ("—", "–", "“", '"')
_NUMERIC_OPEN = re.compile(r"^\s*[\d]+(?:[.,]\d+)*\s*[.:]?\s+\S")


def _nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


def _count(text_low: str, phrase: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(_nfc_lower(phrase)) + r"(?!\w)",
                          text_low))


def split_narration_dialogue(prose: str) -> tuple[str, str]:
    tran, thoai = [], []
    for line in unicodedata.normalize("NFC", prose or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        (thoai if s[0] in _DIALOGUE_START else tran).append(s)
    return "\n".join(tran), "\n".join(thoai)


def numeric_opener(prose: str) -> str | None:
    """Câu mở cảnh là một con số trần — tật đếm của POV biến thành khuôn."""
    for line in unicodedata.normalize("NFC", prose or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s[:60] if _NUMERIC_OPEN.match(s) else None
    return None


SOMATIC_CORE_WORDS = 5    # số từ lõi dùng làm dấu nhận dạng cử chỉ
SOMATIC_CORE_HITS = 4     # bấy nhiêu từ lõi cùng nằm trong MỘT câu là một lần
# Ngưỡng đo trên văn bản thật, không chọn bằng cảm tính. Với 3 từ lõi thì bắt
# đủ 6/6 câu đúng nhưng báo nhầm 2/5 câu không phải cử chỉ ("nền rung một nhịp
# dưới gót chân khi đoàn tàu rời bến"). Với 4 thì bắt 4/6 và báo nhầm 0/5, mà
# tật nặng nhất vẫn bị đếm 4–7 lần mỗi chương — vượt xa trần 3, nên luật vẫn
# nổ đúng chỗ cần nổ. P2 nói rõ: báo động giả khiến người ta tắt luật, còn bỏ
# sót một lần lặp thì không.

# Hư từ tiếng Việt: có mặt ở mọi câu nên không phân biệt được cử chỉ nào với
# cử chỉ nào. Giữ chúng trong lõi thì ngưỡng bị thổi phồng bằng nhiễu.
_HU_TU = {
    "vào", "cho", "tới", "đến", "khi", "một", "và", "của", "với", "rồi", "dù",
    "đã", "như", "đang", "ở", "kể", "cả", "không", "cần", "trước", "sau",
    "ngay", "thì", "mà", "là", "trên", "dưới", "ra", "lên", "xuống", "bằng",
    "cách", "những", "các", "được", "bị", "này", "đó", "nó", "có", "trong",
}


def gesture_core(phrase: str) -> list[str]:
    """Từ lõi của một cử chỉ, đã bỏ hư từ và bỏ trùng."""
    thay: set[str] = set()
    loi: list[str] = []
    for t in _nfc_lower(phrase).split():
        t = t.strip(".,;:!?—–\"'()")
        if not t or t in _HU_TU or t in thay:
            continue
        thay.add(t)
        loi.append(t)
        if len(loi) == SOMATIC_CORE_WORDS:
            break
    return loi


def _count_gesture(text_low: str, phrase: str) -> int:
    """Đếm cử chỉ theo TỪ LÕI CÙNG CÂU, không theo trật tự từ.

    Bản trước khớp lõi theo đúng thứ tự, cho phép chen vài từ. Nó vẫn là mã
    chết: bible ghi "ấn ngón cái vào khớp ngón trỏ", model viết "Ngón cái
    Kaelen ấn mạnh vào khớp ngón trỏ" — đảo chủ ngữ ra trước động từ là trượt.
    Đo trên bản viết lại Chương 1: bộ đếm báo 1, đếm tay ra 6.

    Tiếng Việt cho phép đảo trật tự thoải mái mà nghĩa không đổi, nên dấu nhận
    dạng đúng là TẬP HỢP từ lõi trong cùng một câu, không phải chuỗi.
    """
    loi = gesture_core(phrase)
    if not loi:
        return 0
    can = min(len(loi), SOMATIC_CORE_HITS)
    n = 0
    for cau in re.split(r"[.!?\u2026\n;]+", text_low):
        tu = set(re.findall(r"\w+", cau))
        if sum(1 for t in loi if t in tu) >= can:
            n += 1
    return n


def somatic_findings(prose: str, contract: dict, chars: dict,
                     previous_scenes=()) -> list[dict]:
    """Ngân sách CỬ CHỈ đặc trưng, tính cả chương (§5.3).

    `somatic_signature` sinh ra để thay sáo ngữ cơ thể dùng chung. Nhưng Arc 1
    cho thấy nó thành một sáo ngữ RIÊNG: Kaelen ấn ngón cái vào khớp ngón trỏ ở
    gần như cả ba mươi cảnh, Serena xoay nhẫn đúng ba vòng, Vhal lật cùng một
    trang sổ. Một cử chỉ nhận dạng lặp mỗi cảnh không còn nhận dạng ai cả.

    Trần: MỘT lần mỗi cảnh, BA lần mỗi chương cho cùng một cử chỉ.
    """
    out: list[dict] = []
    truoc = " ".join(p.get("prose", "") for p in (previous_scenes or []))
    low, low_truoc = _nfc_lower(prose), _nfc_lower(truoc)
    for x in contract.get("active_characters", []):
        prof = chars.get(x.get("id")) if chars else None
        if prof is None:
            continue
        for cu_chi in prof.somatic_signature:
            trong_canh = _count_gesture(low, cu_chi)
            ca_chuong = trong_canh + _count_gesture(low_truoc, cu_chi)
            if trong_canh > SOMATIC_PER_SCENE:
                out.append({"severity": "minor", "check": "somatic_overuse",
                            "message": (f"{prof.name}: cử chỉ “{cu_chi}” ×{trong_canh} "
                                        f"trong một cảnh (trần {SOMATIC_PER_SCENE})"),
                            "evidence": cu_chi})
            elif ca_chuong > SOMATIC_PER_CHAPTER:
                out.append({"severity": "minor", "check": "somatic_budget",
                            "message": (f"{prof.name}: cử chỉ “{cu_chi}” đã dùng "
                                        f"{ca_chuong} lần trong chương (trần "
                                        f"{SOMATIC_PER_CHAPTER}) — dùng ngôn ngữ cơ "
                                        f"thể khác cho cảnh này"),
                            "evidence": cu_chi})
    return out


FIGURE_PER_CHAPTER = 2    # một con số cụ thể nhắc quá hai lần là khẩu hiệu

_SO_CHU = ("không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|lăm|mươi|trăm|"
           "nghìn|tư|mốt|linh|lẻ")
_DON_VI = ("phần trăm|phút|giây|giờ|lần|mét|độ|bar|ki-lô-mét|cây số")
# "bốn mươi lăm phần trăm", "mười hai phần trăm" — số viết bằng chữ, kèm đơn vị.
_FIGURE = re.compile("(?:(?:" + _SO_CHU + ")(?:\\s+(?:" + _SO_CHU + "))*)"
                     "\\s+(?:" + _DON_VI + ")")


COUNTING_PER_CHAPTER = 1  # lượt thoại CHỈ gồm một con số

_SPEECH_ONLY_NUM = re.compile(
    "^[—–]\\s*(?:" + _SO_CHU + ")(?:\\s+(?:" + _SO_CHU + "))*\\s*[.!?]?\\s*$")


def counting_speech(prose: str) -> list[str]:
    """Lượt thoại mà cả lời nói chỉ là một con số đếm: "— Ba.", "— Tám."."""
    return [l.strip() for l in _nfc_lower(prose).splitlines()
            if _SPEECH_ONLY_NUM.match(l.strip())]


def counting_findings(prose: str, previous_scenes=()) -> list[dict]:
    """Ngân sách cho tật ĐẾM THÀNH TIẾNG, tính cả chương.

    "— Ba." rồi "— Ba người kiểm tra trước anh đã đóng dấu" là tật nhân vật làm
    việc: con số dẫn vào một quan sát. "— Ba." đứng trơ một mình thì không dẫn
    đi đâu, và lặp lại thì thành máy móc chứ không thành bí ẩn.

    Đo trên ba đời Arc 1, số lượt thoại chỉ gồm một con số: 1 → 3 → 8. Chính
    việc tôi đưa câu mẫu "Hai. Hai lần anh nói là không biết rồi đấy." đã dạy
    model cái khuôn số-trước-câu, rồi nó rụng mất phần câu.
    """
    truoc = sum(len(counting_speech(x.get("prose", "")))
                for x in (previous_scenes or []))
    nay = counting_speech(prose)
    if not nay or truoc + len(nay) <= COUNTING_PER_CHAPTER:
        return []
    return [{"severity": "minor", "check": "counting_tic",
             "message": (f"lượt thoại chỉ gồm một con số, lần thứ "
                         f"{truoc + len(nay)} trong chương (trần "
                         f"{COUNTING_PER_CHAPTER}) — cho con số dẫn vào một "
                         f"quan sát, hoặc thay bằng hành động nhìn/đếm ngầm"),
             "evidence": nay[0]}]


def figure_findings(prose: str, previous_scenes=()) -> list[dict]:
    """Ngân sách cho MỘT CON SỐ CỤ THỂ, tính cả chương.

    Đọc bản viết lại Arc 1: "bốn mươi lăm phần trăm" xuất hiện 14 lần trên 5
    chương — bốn lần riêng Chương 1 — mà nó KHÔNG có trong bible. Model tự sinh
    ra từ câu mẫu "Mười hai phần trăm. Tôi cần tên người đã sửa dòng đó." rồi
    dùng làm câu mồi mỗi lần chuyển mạch tư duy.

    Đây là cùng một sai lầm với `signature_lexicon`, chỉ ở tầng cao hơn: câu mẫu
    dạy được CÁI KHUÔN, và model đóng dấu cái khuôn. Bỏ danh sách khẩu ngữ làm
    lặp TỪ giảm tám lần nhưng đẻ ra lặp CẤU TRÚC, nên chỗ này cần bộ đếm riêng
    chứ không chờ luật cũ bắt hộ.
    """
    truoc = " ".join(x.get("prose", "") for x in (previous_scenes or []))
    low = _nfc_lower(prose)
    dem: dict[str, int] = {}
    for van in (_nfc_lower(truoc), low):
        for m in _FIGURE.finditer(van):
            dem[m.group(0)] = dem.get(m.group(0), 0) + 1
    out = []
    for cum, n in sorted(dem.items()):
        if n > FIGURE_PER_CHAPTER and _count(low, cum):
            out.append({"severity": "minor", "check": "figure_overuse",
                        "message": (f"“{cum}” đã dùng {n} lần trong chương (trần "
                                    f"{FIGURE_PER_CHAPTER}) — con số lặp lại thành "
                                    f"khẩu hiệu; thay bằng một quan sát cụ thể"),
                        "evidence": cum})
    return out


def tic_findings(prose: str, contract: dict, chars: dict,
                 previous_scenes=()) -> list[dict]:
    out: list[dict] = somatic_findings(prose, contract, chars, previous_scenes)
    out += figure_findings(prose, previous_scenes)
    out += counting_findings(prose, previous_scenes)
    mo = numeric_opener(prose)
    if mo:
        out.append({"severity": "minor", "check": "numeric_opener",
                    "message": ("cảnh mở đầu bằng một con số trần — 16/30 cảnh của "
                                "Arc 1 làm vậy; mở bằng hành động hoặc chi tiết"),
                    "evidence": mo})

    tran, _ = split_narration_dialogue(prose)
    low_all, low_tran = _nfc_lower(prose), _nfc_lower(tran)
    for x in contract.get("active_characters", []):
        prof = chars.get(x.get("id")) if chars else None
        if prof is None:
            continue
        tong = 0
        # Khẩu ngữ thuần có trần riêng, chặt hơn, vì không cụm nào trong đây
        # mang nghĩa chuyên môn để mà cần dùng lại. Đo trên Arc 1: 24/30 cảnh
        # không dùng lần nào, còn cảnh nào dùng thì dùng 3–8 lần — nên trần 2
        # không đụng vào cảnh viết bình thường.
        for phrase in getattr(prof.voice, "verbal_tics", []):
            n = _count(low_all, phrase)
            tong += n
            if n > TIC_LIMIT:
                out.append({
                    "severity": "minor", "check": "verbal_tic_overuse",
                    "message": (f"{prof.name}: “{phrase}” ×{n} trong một cảnh "
                                f"(trần {TIC_LIMIT}) — đây là khẩu ngữ thuần, "
                                f"lặp quá thì thành nhại chứ không thành giọng"),
                    "evidence": phrase})
        for phrase in prof.voice.signature_lexicon:
            n = _count(low_all, phrase)
            tong += n
            tran_n = _count(low_tran, phrase)
            gioi_han = PHRASE_LIMIT if len(phrase.split()) >= 3 else SHORT_LIMIT
            if n > gioi_han:
                out.append({
                    "severity": "minor", "check": "tic_overuse",
                    "message": (f"{prof.name}: “{phrase}” ×{n} trong một cảnh "
                                f"(trần {gioi_han}) — thay bằng cách nói khác của "
                                f"cùng con người đó"),
                    "evidence": phrase})
            if tran_n > NARRATION_LIMIT:
                out.append({
                    "severity": "minor", "check": "lexicon_in_narration",
                    "message": (f"khẩu ngữ của {prof.name} (“{phrase}”) xuất hiện "
                                f"{tran_n} lần trong LỜI KỂ — giọng nhân vật thuộc "
                                f"về thoại, lời kể mang giọng qua chi tiết được chọn"),
                    "evidence": phrase})
        if tong > TOTAL_LIMIT:
            out.append({
                "severity": "minor", "check": "tic_budget",
                "message": (f"{prof.name} dùng cụm đặc trưng {tong} lần trong một "
                            f"cảnh (trần {TOTAL_LIMIT}) — nhận ra giọng cần vài dấu "
                            f"vết, không cần một danh mục")})
    return out
