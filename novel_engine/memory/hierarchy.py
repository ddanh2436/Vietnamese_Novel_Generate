"""Narrative Memory Hierarchy — năm tầng nén (§4.1).

Chống drift không phải bằng cách nhớ NHIỀU hơn, mà bằng cách nhớ ở ĐÚNG ĐỘ
PHÂN GIẢI. Chương 37 cần chi tiết chương 36, cần tóm tắt chương 30–35, và chỉ
cần biết sự kiện nào đã xảy ra ở chương 1–10.

    L0  văn xuôi thô       ~3.000 từ/chương   xuất bản, tra nguyên văn
    L1  scene digest       ~200 từ/chương     chương kế tiếp (chi tiết cao)
    L2  chapter summary    ~120 từ/chương     cửa sổ 5 chương gần nhất
    L3  arc summary        ~300 từ/arc        toàn bộ quá khứ xa
    L4  standing facts     GRAPH              luôn luôn, qua truy vấn

L4 không phải văn bản — nó là graph. Nghĩa là nó không tiêu tốn context theo
tuyến tính: Context Assembler chỉ kéo ra đúng những fact liên quan tới cảnh
sắp viết.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryBudget:
    """Ngân sách token cho từng tầng. Tổng phải ≤ ngưỡng an toàn.

    ⚠ HIỆU CHỈNH CHO TIẾNG VIỆT — các con số ở §4.1 tính theo văn tiếng Anh.
    Đo thực trên cl100k_base: một câu tiếng Việt tốn ~2,6× token so với câu
    tiếng Anh cùng nội dung (tiếng Việt viết rời âm tiết, và phần lớn âm tiết
    có dấu bị tách thành nhiều token). Giữ nguyên con số của §4.1 nghĩa là mỗi
    tầng chỉ chứa được khoảng 40% lượng nội dung mà thiết kế dự tính — và cắt
    bớt thì âm thầm, không ai thấy.

    Vì vậy `for_vietnamese()` nhân toàn bộ lên. Ngân sách vẫn là CỨNG: nếu L4
    trả về nhiều hơn, Assembler cắt theo điểm liên quan, không nới ngân sách.
    Nới ngân sách là con đường quay lại đúng vấn đề ban đầu.
    """
    l1_recent_scenes: int = 1200      # 2 cảnh liền trước, nguyên digest
    l2_recent_chapters: int = 900     # 5 chương gần nhất
    l3_arcs: int = 700                # toàn bộ arc đã đóng
    l4_facts: int = 1500              # fact truy vấn theo cảnh
    l5_callbacks: int = 500           # cảnh cũ gọi lại theo nội dung
    character_sheets: int = 1800      # hồ sơ nhân vật có mặt
    clue_directives: int = 400
    style_exemplars: int = 800        # trích đoạn văn phong mẫu

    @property
    def total(self) -> int:
        return sum(v for v in self.__dict__.values() if isinstance(v, int))

    @classmethod
    def for_vietnamese(cls, factor: float = 2.6) -> "MemoryBudget":
        base = cls()
        return cls(**{k: int(v * factor) for k, v in base.__dict__.items()
                      if isinstance(v, int)})
