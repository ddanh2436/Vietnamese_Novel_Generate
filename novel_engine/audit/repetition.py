"""Chống lặp cảnh — mảnh còn thiếu của §4.3 (self-plagiarism).

Đọc Arc 1 lượt 2 thấy ngay: Chương 1 có ba dị bản của cùng một cảnh (Kaelen xin
giấy, Vhal từ chối), Chương 3 lặp lại cuộc đối chất ở cảnh 1 và cảnh 4, Chương 5
lặp ở cảnh 1 và cảnh 4. Truyện đứng yên trong khi chữ vẫn chạy — thứ làm người
đọc bỏ sách sớm nhất, mà không luật nào trong §10 bắt được.

═══ CHỌN THƯỚC ĐO BẰNG CÁCH ĐO, KHÔNG BẰNG CÁCH ĐOÁN ═══════════════════════

Đo trên chính 5 chương đã sinh, 15 cặp cảnh mỗi chương:

    thước đo          cặp LẶP (đọc thấy)      cặp bình thường
    5-gram            0.045 – 0.052           ≤ 0.014      ← quá nhỏ, khó đặt ngưỡng
    cosine tần suất   0.704 – 0.783           0.60 – 0.74  ← chồng lấn, thuật ngữ
                                                             kỹ thuật dùng chung
                                                             thổi lên
    3-gram (chọn)     0.089 – 0.139           ≤ 0.078      ← tách bạch

Cảnh lặp ở đây lặp NHỊP và NỘI DUNG, không lặp câu chữ: trùng 5-gram chỉ 0.05 ở
đúng cặp cảnh mà người đọc thấy là một. Đó là lý do ngưỡng đặt trên 3-gram.

Hai tín hiệu phụ, hiển nhiên hơn và giải thích được ngay cho Writer: câu trần
thuật lặp NGUYÊN VĂN giữa hai cảnh, và lượt thoại lặp nguyên văn (Vhal nói "hết
giờ tiếp nhận rồi, được chưa?" ở bốn cảnh khác nhau).
"""
from __future__ import annotations

import difflib
import re
import unicodedata

NGRAM = 3
CONTAINMENT_MAJOR = 0.085     # đo được: cặp lặp ≥0.089, cặp thường ≤0.078
MIN_TOKENS = 120              # cảnh quá ngắn thì tỉ lệ trùng là nhiễu
REPEAT_SENTENCE_MIN_WORDS = 6
QUOTE_EVIDENCE_TOKENS = 14
_DIALOGUE_START = ("—", "–", "“", '"')


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", s or "").lower())


def _grams(tokens: list[str], n: int = NGRAM) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def containment(a: str, b: str, n: int = NGRAM) -> float:
    """Tỉ lệ n-gram của cảnh NGẮN hơn nằm trong cảnh kia.

    Dùng `min` chứ không phải hợp (Jaccard): một cảnh ngắn viết lại nguyên một
    cảnh dài vẫn phải bị bắt, và Jaccard sẽ pha loãng đúng ca đó.
    """
    ga, gb = _grams(_tokens(a), n), _grams(_tokens(b), n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def longest_shared(a: str, b: str) -> str:
    """Đoạn dài nhất hai cảnh dùng chung — để Writer thấy CHỖ NÀO lặp."""
    ta, tb = _tokens(a), _tokens(b)
    m = difflib.SequenceMatcher(None, ta, tb, autojunk=False).find_longest_match(
        0, len(ta), 0, len(tb))
    return " ".join(ta[m.a: m.a + min(m.size, QUOTE_EVIDENCE_TOKENS)])


def _sentences(text: str) -> list[tuple[str, bool]]:
    """(câu đã chuẩn hoá, có phải thoại không)."""
    out = []
    for line in unicodedata.normalize("NFC", text or "").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        thoai = line[0] in _DIALOGUE_START
        for c in re.split(r"(?<=[.!?…])\s+", line):
            norm = " ".join(_tokens(c))
            if len(norm.split()) >= REPEAT_SENTENCE_MIN_WORDS:
                out.append((norm, thoai))
    return out


def _cung_hien_truong(a: dict, b: dict) -> bool:
    """Hai cảnh có phải cùng CHỖ và cùng NGƯỜI không.

    Chỉ so những cảnh như vậy. Hai cảnh ở hai nơi với hai nhóm nhân vật khác
    nhau không phải là "cùng một cảnh viết hai lần" dù câu chữ có giống — cái
    giống đó là vấn đề văn phong, đã có `rhythm` và `tic` lo. Thu hẹp thế này
    nhắm đúng thứ đọc thấy ở Arc 1: Chương 1 cảnh 0–1–2 cùng ở quầy Vhal.
    """
    if a.get("location") and b.get("location") and a["location"] != b["location"]:
        return False
    ca, cb = set(a.get("cast") or ()), set(b.get("cast") or ())
    return not (ca and cb) or bool(ca & cb)


def repetition_findings(draft: str, previous: list[dict], *,
                        location: str | None = None, cast=()) -> list[dict]:
    """`previous`: cảnh ĐÃ chốt của chương này — `[{scene_id, prose, location, cast}]`."""
    out: list[dict] = []
    hien_truong = {"location": location, "cast": cast}
    if len(_tokens(draft)) < MIN_TOKENS:
        return out
    cau_moi = _sentences(draft)
    moi_tran = {c for c, thoai in cau_moi if not thoai}
    moi_thoai = {c for c, thoai in cau_moi if thoai}

    for prev in previous:
        van = prev.get("prose", "")
        if len(_tokens(van)) < MIN_TOKENS or not _cung_hien_truong(hien_truong, prev):
            continue
        sid = prev.get("scene_id", "?")
        r = containment(draft, van)
        if r >= CONTAINMENT_MAJOR:
            out.append({
                "severity": "major", "check": "scene_repetition",
                "message": (f"cảnh này trùng {r:.0%} khối ba từ với {sid} — viết lại "
                            f"để nó đẩy tình huống sang trạng thái KHÁC, đừng dựng "
                            f"lại cùng một cuộc đối thoại"),
                "evidence": longest_shared(draft, van)})
        cu = _sentences(van)
        lap_tran = moi_tran & {c for c, thoai in cu if not thoai}
        if lap_tran:
            out.append({
                "severity": "major", "check": "sentence_repetition",
                "message": f"{len(lap_tran)} câu trần thuật lặp nguyên văn từ {sid}",
                "evidence": sorted(lap_tran, key=len, reverse=True)[0]})
        lap_thoai = moi_thoai & {c for c, thoai in cu if thoai}
        if lap_thoai:
            out.append({
                "severity": "minor", "check": "dialogue_repetition",
                "message": (f"{len(lap_thoai)} lượt thoại lặp nguyên văn từ {sid} — "
                            f"nhân vật nói lại y hệt câu đã nói"),
                "evidence": sorted(lap_thoai, key=len, reverse=True)[0]})
    return out
