"""Cưỡng chế Voice Fingerprint (§5.3) — kiểm tra thuần code sau khi viết.

═══ GÁN THOẠI CHO ĐÚNG NGƯỜI NÓI ═══════════════════════════════════════════

`extract_dialogue` của §5.3 lấy một đoạn nếu tên nhân vật xuất hiện trong đó
HOẶC đoạn mở đầu bằng gạch ngang. Vế "hoặc" làm MỌI lượt thoại của MỌI nhân vật
bị gán cho từng nhân vật. Hệ quả:

- Vhal bị báo dùng từ cấm "theo thẩm quyền" — đó là câu của Serena.
- Kaelen (giọng cộc) bị báo độ dài câu vượt khoảng vì nhận câu dài của Serena.
- Mọi nhân vật trong cảnh có CÙNG thống kê giọng — bộ kiểm tra chống làm phẳng
  giọng (Stereotype Flattening) tự làm phẳng chính nó.

Ở đây gán theo HAI mức tin cậy:

1. `by_speaker` — dẫn thoại TRÊN CÙNG DÒNG nêu đích danh đúng một nhân vật:
   "– Con tàu rời bến – Serena nói – Tuyến ray đã bị niêm phong." hoặc
   "— Sai số nằm ở đâu? Kaelen gằn giọng." Gần như không sai.
2. `inferred` — dòng thoại không có dẫn, nhưng đoạn trần thuật NGAY TRƯỚC mở
   đầu bằng tên một nhân vật ("Vhal không trả lời ngay. Hắn…" rồi "— Hết giờ
   tiếp nhận rồi."). Đây là cách Gemini dẫn thoại phổ biến nhất. Soát tay 26 ca
   ở Chương 1–2: 24 đúng. Một ca sai là câu thoại GỌI TÊN chính người đó ("—
   Đã quá giờ, Kaelen.") — không ai gọi tên mình, nên loại. Ca còn lại không
   chặn được bằng cú pháp.

Vì mức 2 có sai số, từ cấm tìm thấy CHỈ trong thoại mức 2 là `minor` — không
kích hoạt viết lại vì một câu có thể không phải của nhân vật đó (P2).

═══ KÍCH THƯỚC MẪU ═══════════════════════════════════════════════════════

§5.3 tính tỉ lệ câu hỏi trên mọi số lượt thoại. Với MỘT lượt, tỉ lệ là 0 hoặc 1
— luôn nằm ngoài khoảng như [0.18, 0.34] của Serena. Và sàn 0.02 của Kaelen chỉ
kiểm được khi có ≥50 lượt: sáu lượt không câu hỏi nào vẫn là giọng Kaelen. Nên
tỉ lệ được so theo SỐ ĐẾM, dung sai một lượt.
"""
from __future__ import annotations

import re
import statistics
import unicodedata

from novel_engine.character.models import CharacterProfile

MIN_LINES_FOR_STATS = 4
MIN_LINES_FOR_SIGNATURE = 6

_DIALOGUE_MARKS = "—–“\""
_DASH_SPLIT = re.compile(r"\s[—–]\s")
_QUOTE_RE = re.compile(r"[“\"](.+?)[”\"]")


def _nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


def _has_word(text_low: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(_nfc_lower(phrase)) + r"(?!\w)",
                     text_low) is not None


def _split_line(line: str, names: dict[str, str]) -> tuple[list[str], list[str]] | None:
    """(lời nói, dẫn thoại) của một dòng; `None` nếu dòng không phải thoại."""
    if line[0] in "—–":
        body = line.lstrip("—– ").strip()
        parts = _DASH_SPLIT.split(body)
        if len(parts) == 1 and names:
            alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
            m = re.search(rf"[.?!…]\s+(?=(?:{alt})(?!\w))", body)
            if m:
                parts = [body[:m.start() + 1], body[m.end():]]
        return parts[0::2], parts[1::2]
    speech = _QUOTE_RE.findall(line)
    return (speech, [_QUOTE_RE.sub(" ", line)]) if speech else None


def _beat_speaker(prev: str, names: dict[str, str], speech: list[str]) -> str | None:
    if not prev or prev[0] in _DIALOGUE_MARKS:
        return None
    for name, cid in names.items():
        if re.match(re.escape(name) + r"(?!\w)", prev):
            spoken = _nfc_lower(" ".join(speech))
            return None if _has_word(spoken, name) else cid
    return None


def attribute_dialogue(prose: str, names: dict[str, str]) -> dict:
    """`names`: tên hiển thị → char_id. Trả `{by_speaker, inferred, unattributed}`."""
    by_speaker: dict[str, list[str]] = {cid: [] for cid in names.values()}
    inferred: dict[str, list[str]] = {cid: [] for cid in names.values()}
    unattributed: list[str] = []
    prev = ""
    for raw in unicodedata.normalize("NFC", prose or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        split = _split_line(line, names)
        if split is None:
            prev = line
            continue
        speech = [s.strip() for s in split[0] if s.strip()]
        tags = split[1]
        if speech:
            speakers = {cid for name, cid in names.items()
                        if any(_has_word(_nfc_lower(t), name) for t in tags)}
            beat = None if speakers else _beat_speaker(prev, names, speech)
            if len(speakers) == 1:
                by_speaker[next(iter(speakers))].extend(speech)
            elif beat:
                inferred[beat].extend(speech)
            else:
                unattributed.extend(speech)
        prev = line
    return {"by_speaker": by_speaker, "inferred": inferred,
            "unattributed": unattributed}


def voice_report(lines: list[str], char: CharacterProfile,
                 inferred: list[str] = ()) -> dict:
    """Chấm giọng trên các lượt thoại đã gán cho nhân vật này.

    Mức độ: từ cấm trong thoại có DẪN là `major` — so khớp chính xác, gần như
    không nhiễu. Từ cấm chỉ có trong thoại SUY RA, và mọi chỉ số thống kê, là
    `minor` — đi vào Polish thay vì kích hoạt viết lại (§10.3.2).
    """
    all_lines = list(lines) + list(inferred)
    if not all_lines:
        return {"ok": True, "violations": [], "stats": {"n_lines": 0}}
    v = char.voice
    # `mean_sentence_len` là từ/CÂU, không phải từ/LƯỢT: một lượt thoại của Vhal
    # ở Chương 1 dài 119 từ nhưng gồm nhiều câu. Đo theo lượt thì mọi nhân vật
    # nói dài hơn một câu đều "vượt trần".
    sents = [x for s in all_lines for x in re.split(r"(?<=[.!?…])\s+", s.strip())
             if re.search(r"\w", x)]
    if not sents:
        return {"ok": True, "violations": [], "stats": {"n_lines": len(all_lines)}}
    lens = [sum(1 for w in x.split() if re.search(r"\w", w)) for x in sents]
    tagged_low = _nfc_lower(" ".join(lines))
    inferred_low = _nfc_lower(" ".join(inferred))
    out: list[dict] = []

    forbidden = [w for w in v.forbidden_lexicon if _has_word(tagged_low, w)]
    if forbidden:
        out.append({"severity": "major", "message": f"dùng từ cấm: {forbidden}"})
    maybe = [w for w in v.forbidden_lexicon
             if w not in forbidden and _has_word(inferred_low, w)]
    if maybe:
        out.append({"severity": "minor",
                    "message": f"có thể dùng từ cấm {maybe} "
                               f"(thoại gán theo đoạn dẫn, cần xác nhận)"})
    if max(lens) > v.max_sentence_len:
        out.append({"severity": "minor",
                    "message": f"có câu {max(lens)} từ > trần {v.max_sentence_len}"})

    n, ns = len(all_lines), len(sents)
    mean_len = statistics.mean(lens)
    qn = sum(1 for x in sents if x.rstrip().endswith("?"))
    if n >= MIN_LINES_FOR_STATS:
        lo, hi = v.mean_sentence_len
        if not (lo <= mean_len <= hi):
            out.append({"severity": "minor",
                        "message": f"độ dài câu trung bình {mean_len:.1f} ngoài [{lo},{hi}]"})
        qlo, qhi = v.question_ratio
        if qn < qlo * ns - 1 or qn > qhi * ns + 1:
            out.append({"severity": "minor",
                        "message": f"{qn}/{ns} câu thoại là câu hỏi, giọng này "
                                   f"khoảng {qlo:.0%}–{qhi:.0%}"})

    all_low = _nfc_lower(" ".join(all_lines))
    sig_hits = sum(1 for w in v.signature_lexicon if _has_word(all_low, w))
    if n >= MIN_LINES_FOR_SIGNATURE and sig_hits == 0:
        out.append({"severity": "minor",
                    "message": f"không có dấu vết lexicon đặc trưng trong {n} lượt thoại"})

    return {"ok": not out, "violations": out,
            "stats": {"n_lines": n, "n_inferred": len(inferred),
                      "mean_len": round(mean_len, 1),
                      "n_sentences": ns, "q_ratio": round(qn / ns, 2),
                      "sig_hits": sig_hits}}
