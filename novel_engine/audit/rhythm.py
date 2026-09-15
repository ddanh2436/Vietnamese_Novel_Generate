"""Nhịp điệu trần thuật — chống văn xuôi đều đều (§10.3.1).

`VoiceFingerprint` chỉ kiểm soát tầng THOẠI. Tầng trần thuật là nơi LLM lộ
điểm yếu dai dẳng nhất: cấu trúc câu ghép an toàn lặp đi lặp lại — "Khi
[mệnh đề A], anh [hành động B], trong khi [mệnh đề C]". Từng câu một không sai.
Đọc liên tục năm cảnh thì ru ngủ.

§10.3.2 — cảnh báo Goodhart: đây là công cụ CHẨN ĐOÁN, không phải mục tiêu cho
Writer. Không chỉ số nào ở đây được lên `blocker`; kết quả đi vào Polish. Polish
cần ghi chú CỤ THỂ ("đoạn 4 có 6 câu liên tiếp dài 18–22 từ"), nên các finding
chỉ ra được chỗ nào thì kèm `evidence` là câu trích nguyên văn.

Bốn chỗ khác mã §10.3.1, cả bốn đều làm một kiểm tra chết hoặc kêu sai:

1. `TRICLAUSE_RE` của tài liệu neo `^` với `re.M`, tức chỉ khớp ở ĐẦU DÒNG.
   Đoạn văn của Gemini chứa nhiều câu trên một dòng, nên khuôn "Khi A, B,
   trong khi C" ở giữa đoạn không bao giờ bị bắt. Ở đây khớp trên TỪNG CÂU.
2. `paragraph_burstiness` đếm cả đoạn THOẠI — mỗi lượt thoại là một đoạn một
   câu, nên "cảnh escalate phải có đoạn một câu" luôn được thoả và không bao
   giờ kích hoạt. Ở đây chỉ đếm đoạn TRẦN THUẬT.
3. "Vì vậy", "Khi đó", "Dù sao" bắt đầu bằng từ của `SUBORDINATE_OPENERS` nhưng
   không mở mệnh đề phụ. Đếm chúng là thổi phồng chỉ số MAJOR duy nhất.
4. Đếm từ bỏ các token không có chữ (gạch ngang "—"), và `contract["tension"]`
   thiếu thì không ném KeyError.
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter

# Ngưỡng theo NGÔN NGỮ. Tiếng Việt viết rời âm tiết ("nghiên cứu" = 2 token),
# nên mọi thống kê đếm-từ bị thổi lên ~1,3–1,5× so với tiếng Anh.
THRESH = {
    "en": {"sd_floor": 6.5, "short_max": 4, "short_ratio_min": 0.10},
    "vi": {"sd_floor": 8.5, "short_max": 5, "short_ratio_min": 0.08},
}

SUBORDINATE_OPENERS = [
    "Khi", "Sau khi", "Trước khi", "Trong khi", "Trong lúc", "Ngay khi",
    "Cho đến khi", "Dù", "Mặc dù", "Nếu", "Bởi vì", "Vì", "Để", "Kể từ khi",
]
NOT_SUBORDINATE = ("Vì vậy", "Vì thế", "Dù sao", "Dù vậy", "Nếu vậy",
                   "Khi ấy", "Khi đó", "Để rồi")

TRICLAUSE_RE = re.compile(
    r"(?:Khi|Trong khi|Sau khi|Ngay khi)\s.{5,70},\s.{5,70},\s(?:trong khi|còn|và|thì)\s")

_DIALOGUE_START = ("—", "–", "“", '"')
_SPLIT_SENT = re.compile(r"(?<=[.!?…])\s+")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def narrative_sentences(prose: str) -> list[str]:
    """Chỉ lấy câu TRẦN THUẬT — bỏ thoại (VoiceFingerprint lo) và tiêu đề."""
    body = "\n".join(l for l in _nfc(prose).split("\n")
                     if not l.strip().startswith(_DIALOGUE_START)
                     and not l.strip().startswith("#"))
    return [s.strip() for s in _SPLIT_SENT.split(body) if s.strip()]


def word_count(sentence: str) -> int:
    return sum(1 for w in sentence.split() if re.search(r"\w", w))


def _opener(sentence: str) -> str:
    for w in sentence.split():
        w = re.sub(r"^\W+|\W+$", "", w)
        if w:
            return w.lower()
    return ""


def is_subordinate_opener(sentence: str) -> bool:
    if sentence.startswith(NOT_SUBORDINATE):
        return False
    return any(sentence.startswith(o + " ") for o in SUBORDINATE_OPENERS)


def rhythm_stats(prose: str, lang: str = "vi") -> dict | None:
    """Số đo thô. `None` khi quá ít câu trần thuật để thống kê có nghĩa."""
    T = THRESH[lang]
    sents = narrative_sentences(prose)
    if len(sents) < 8:
        return None
    lens = [word_count(s) for s in sents]
    run = max_run = 1
    run_start = best_start = 0
    for i, (a, b) in enumerate(zip(lens, lens[1:])):
        if abs(a - b) <= 3:
            run += 1
        else:
            run, run_start = 1, i + 1
        if run > max_run:
            max_run, best_start = run, run_start
    openers = Counter(_opener(s) for s in sents if _opener(s))
    paras = [p for p in re.split(r"\n\s*\n", _nfc(prose)) if p.strip()]
    para_counts = [n for n in (len(narrative_sentences(p)) for p in paras) if n > 0]
    subs = [s for s in sents if is_subordinate_opener(s)]
    tris = [s for s in sents if TRICLAUSE_RE.match(s)]
    return {
        "n_sentences": len(sents),
        "mean_len": round(statistics.mean(lens), 1),
        "sd": round(statistics.pstdev(lens), 2),
        "short_ratio": round(sum(1 for n in lens if n <= T["short_max"]) / len(lens), 3),
        "max_flat_run": max_run,
        "flat_run_at": sents[best_start],
        "sub_ratio": round(len(subs) / len(sents), 3),
        "sub_examples": subs[:3],
        "triclause": len(tris),
        "triclause_examples": tris[:3],
        "opener_ratio": round(len(openers) / len(sents), 3),
        "top_opener": openers.most_common(1)[0] if openers else ("", 0),
        "para_sentence_counts": para_counts,
    }


def rhythm_audit(prose: str, contract: dict, lang: str = "vi") -> list[dict]:
    T = THRESH[lang]
    st = rhythm_stats(prose, lang)
    if st is None:
        return []
    out: list[dict] = []

    if st["sd"] < T["sd_floor"]:
        out.append({"severity": "minor", "check": "rhythm_variance",
                    "message": f"σ độ dài câu = {st['sd']:.1f} < {T['sd_floor']} — "
                               f"nhịp trần thuật đang đều đều"})
    # σ một mình GAMEABLE: phân phối hai cực cho σ đẹp mà đọc như máy.
    if st["short_ratio"] < T["short_ratio_min"]:
        out.append({"severity": "minor", "check": "no_staccato",
                    "message": f"chỉ {st['short_ratio']:.0%} câu ≤{T['short_max']} từ — "
                               f"thiếu nhịp dứt"})
    if st["max_flat_run"] > 5:
        out.append({"severity": "minor", "check": "flat_run",
                    "message": f"{st['max_flat_run']} câu liên tiếp dài xấp xỉ nhau, "
                               f"bắt đầu từ câu được trích",
                    "evidence": st["flat_run_at"]})
    # MAJOR theo §10.3.1 — dấu vân tay của LLM. Vẫn không bao giờ là blocker,
    # và `routing_severity` không để nó kéo cảnh về Writer (§10.3.2).
    if st["sub_ratio"] > 0.30:
        out.append({"severity": "major", "check": "subordinate_opener",
                    "message": f"{st['sub_ratio']:.0%} câu mở đầu bằng mệnh đề phụ "
                               f"(trần 30%) — đưa chủ ngữ hoặc hành động lên đầu",
                    "evidence": st["sub_examples"][0]})
    if st["triclause"] >= 3:
        out.append({"severity": "major", "check": "triclause_template",
                    "message": f"{st['triclause']} câu theo khuôn 'Khi A, B, trong khi C' "
                               f"— tách hoặc đảo cấu trúc",
                    "evidence": st["triclause_examples"][0]})
    if st["opener_ratio"] < 0.55:
        top, n = st["top_opener"]
        out.append({"severity": "minor", "check": "opener_diversity",
                    "message": f"từ mở câu lặp nhiều, '{top}' xuất hiện "
                               f"{n}/{st['n_sentences']} lần"})

    pc = st["para_sentence_counts"]
    mode = (contract.get("tension") or {}).get("mode")
    if mode == "escalate" and pc and min(pc) > 1:
        out.append({"severity": "minor", "check": "paragraph_burstiness",
                    "message": "cảnh escalate không có đoạn TRẦN THUẬT một câu — "
                               "mắt độc giả không có chỗ tăng tốc"})
    if len(pc) >= 5 and statistics.pstdev(pc) < 1.2:
        out.append({"severity": "minor", "check": "paragraph_uniformity",
                    "message": "mọi đoạn trần thuật dài xấp xỉ nhau"})
    return out
