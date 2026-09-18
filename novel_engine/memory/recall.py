"""L5 — nhớ lại cảnh CŨ theo nội dung, không theo thứ tự (§4.1 mở rộng).

L1–L3 là bộ nhớ theo VỊ TRÍ: hai cảnh liền trước, năm chương gần nhất, các arc
đã đóng. Với truyện dài, một chi tiết ở Chương 3 được gọi lại ở Chương 37 nằm
ngoài tầm với của cả ba tầng — trừ khi nó đã được đăng ký thành manh mối.
AI_NovelGenerator giải bằng Chroma + embedding; ở đây chọn khác, và nói rõ vì
sao:

- KHÔNG thêm phụ thuộc nặng. Máy đích 7,7 GB RAM, không Docker. Một vector store
  cộng model embedding là cái giá lớn cho một kho ba mươi cảnh.
- PHẢI TẤT ĐỊNH. Bộ hồi quy §13.2 đòi cùng seed cho cùng kết quả; embedding gọi
  qua API thì mỗi lần chạy lại tốn hạn ngạch và không lặp lại được.
- Kho nhỏ. BM25 trên âm tiết tiếng Việt đủ để bắt lại đúng cái cần bắt: tên
  riêng, địa điểm, vật thể — những thứ mà một callback thật sự dựa vào.

Đánh đổi phải nói thẳng: BM25 KHÔNG bắt được diễn đạt khác chữ ("con dấu bị
mài" ↔ "mép dấu sần lên"). Embedding sẽ bắt được. Khi kho vượt vài trăm cảnh
hoặc khi có ngân sách cho embedding cục bộ thì đổi lõi chấm điểm ở đây là đủ —
phần lọc POV và epoch ở dưới không đổi.
"""
from __future__ import annotations

import math
import re
import unicodedata

K1 = 1.5          # BM25: độ bão hoà tần suất
B = 0.75          # BM25: mức chuẩn hoá theo độ dài
MIN_SCORE = 1.0   # dưới ngưỡng này thì thà không nhớ gì còn hơn nhớ nhầm


def tokens(s: str) -> list[str]:
    """Âm tiết + cặp âm tiết liền nhau.

    Tiếng Việt viết rời âm tiết nên unigram một mình quá nhiễu: "cảng" khớp
    mọi cảnh có cảng. Cặp ("cảng quặng", "con dấu") mới là đơn vị mang nghĩa.
    """
    tu = re.findall(r"\w+", unicodedata.normalize("NFC", s or "").lower())
    return tu + [f"{a} {b}" for a, b in zip(tu, tu[1:])]


def bm25(query: str, docs: list[str]) -> list[float]:
    """Điểm BM25 của `query` với từng tài liệu. Không phụ thuộc thư viện ngoài."""
    dt = [tokens(d) for d in docs]
    if not dt:
        return []
    do_dai = [len(d) for d in dt]
    tb = sum(do_dai) / len(dt)
    df: dict[str, int] = {}
    for d in dt:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    n = len(dt)
    diem = []
    for d, dl in zip(dt, do_dai):
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in set(tokens(query)):
            if t not in df:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = tf.get(t, 0)
            s += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / max(tb, 1)))
        diem.append(s)
    return diem


def callbacks(query: str, ung_vien: list[dict], k: int = 2) -> list[dict]:
    """`k` cảnh cũ liên quan nhất. Mỗi ứng viên cần `digest`.

    Dưới `MIN_SCORE` thì bỏ: một callback sai còn tệ hơn không có callback, vì
    nó mời model dựng lại một cảnh không liên quan — đúng cái lặp cảnh mà
    `repetition.py` sinh ra để chặn.
    """
    if not ung_vien:
        return []
    diem = bm25(query, [x.get("digest", "") for x in ung_vien])
    xep = sorted(zip(diem, range(len(ung_vien))), key=lambda p: (-p[0], p[1]))
    return [{**ung_vien[i], "score": round(d, 3)}
            for d, i in xep[:k] if d >= MIN_SCORE]
