"""Hồ sơ nhân vật — từ tính từ sang ràng buộc (§5.1).

Năm trường của v1 (Want/Need, Fatal Flaw, Private Agenda, Skill & Blindspot,
Private KB) đúng hướng nhưng chưa đủ để CƯỠNG CHẾ. Tầng thêm vào đây là tầng
kiểm chứng được bằng máy — đó mới là thứ chống Stereotype Flattening, không
phải mô tả tính cách hay hơn.
"""
from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# §10.3.3 — khoảng `mean_sentence_len` hiệu chỉnh cho TIẾNG VIỆT.
# Tiếng Việt viết rời âm tiết ("nghiên cứu" = 2 token khi tách theo khoảng
# trắng), nên mọi thống kê đếm-từ bị thổi lên ~1,3–1,5× so với tiếng Anh.
# Lấy thẳng con số từ tài liệu tiếng Anh thì MỌI nhân vật bị gắn cờ vi phạm
# ngay chương đầu. Bảng này là điểm khởi đầu; cách chắc chắn hơn là chạy
# statistics.mean/pstdev trên 2.000 từ văn mẫu rồi lấy ±1,5σ.
REGISTER_SENTENCE_LEN_VI: dict[str, tuple[int, int]] = {
    "clipped": (6, 13),      # cộc lốc, quân đội
    "vernacular": (10, 20),  # đời thường
    "formal": (17, 31),      # nghi thức, giáo sĩ
    "ornate": (21, 39),      # hoa mỹ, cổ
    "clinical": (13, 25),    # kỹ thuật, lạnh
}

# Sáo ngữ cơ thể dùng chung — CẤM với mọi nhân vật. Đây là danh sách đi vào
# `somatic_forbidden` của mọi SceneContract (§9.2).
CLICHE_SOMATICS: list[str] = [
    "tim đập thình thịch", "tim đập nhanh", "mồ hôi lạnh", "nuốt khan",
    "máu chảy ngược", "sống lưng lạnh toát", "tay run lẩy bẩy",
    "nghẹn đắng nơi cổ họng", "thở phào nhẹ nhõm", "tim thắt lại",
]


# `VoiceFingerprint.register` che `ABCMeta.register` (đăng ký virtual subclass)
# mà Pydantic kế thừa. Trường hoạt động đúng — `v.register` trả giá trị chứ
# không trả method — và ta không bao giờ gọi `register()` trên một model. Giữ
# nguyên tên vì §12.1 và `model_dump(include={"register", ...})` ở §9.2 bám
# vào nó; đổi tên ở đây là đổi luôn hợp đồng với prompt.
# Lọc HẸP theo đúng message: mọi shadowing khác vẫn phải kêu.
warnings.filterwarnings(
    "ignore", message=r'Field name "register" .* shadows an attribute',
    category=UserWarning)


class VoiceFingerprint(BaseModel):
    """Ràng buộc ĐỊNH LƯỢNG ở tầng câu chữ.

    NT-7: đây là công cụ CHẨN ĐOÁN, không phải mục tiêu đưa cho Writer. Đưa
    ngưỡng thống kê vào prompt sẽ sinh ra văn xuôi thoả mãn con số và đọc như
    máy (§10.3.2). Writer chỉ nhận `register`, `signature_lexicon`,
    `forbidden_lexicon`, `syntactic_tic` — không nhận các khoảng số.
    """
    mean_sentence_len: tuple[int, int]          # (min, max) từ/câu
    max_sentence_len: int
    register: Literal["formal", "clipped", "ornate", "vernacular", "clinical"]
    signature_lexicon: list[str]                # 8–15 từ/cụm nhân vật hay dùng
    forbidden_lexicon: list[str]                # từ nhân vật KHÔNG BAO GIỜ dùng
    syntactic_tic: str                          # vd "hay bỏ lửng câu bằng '—'"
    question_ratio: tuple[float, float]         # tỉ lệ câu hỏi trong thoại
    self_reference_rate: float                  # tần suất nói về bản thân
    # Dưới áp lực thì giọng đổi thế nào — đây là chỗ nhân vật trở nên "người"
    under_stress_shift: str

    @model_validator(mode="after")
    def _khoang_hop_le(self) -> "VoiceFingerprint":
        lo, hi = self.mean_sentence_len
        if lo >= hi:
            raise ValueError(f"mean_sentence_len phải là (min, max) với min < max, nhận {self.mean_sentence_len}")
        if self.max_sentence_len <= hi:
            raise ValueError("max_sentence_len phải lớn hơn cận trên của mean_sentence_len")
        if set(self.signature_lexicon) & set(self.forbidden_lexicon):
            raise ValueError(
                "một từ vừa nằm trong signature_lexicon vừa trong "
                f"forbidden_lexicon: {set(self.signature_lexicon) & set(self.forbidden_lexicon)}")
        return self


class Flaw(BaseModel):
    name: str                                   # "kiêu ngạo trí thức"
    # KHÔNG mô tả bằng tính từ — mô tả bằng luật biến dạng quyết định
    distortion_rule: str
    trigger_conditions: list[str]
    cost_already_paid: list[str] = Field(default_factory=list)
    arc_direction: Literal["deepens", "heals", "transmutes", "static"]
    # §5.6.6 — kháng cự đính chính, do TÁC GIẢ khai bằng số. §5.6.6 tra
    # `FLAW_RESISTANCE[fatal_flaw.name]` bằng tên ("sợ bị phản bội"...) mà không
    # nhân vật nào mang, nên kháng cự bằng 0 với mọi người. Tên là văn xuôi,
    # không phải khoá.
    correction_resistance: float = Field(default=0.0, ge=0.0, le=0.6)
    resists_tags: list[str] = Field(default_factory=list)   # "{self}" = mã nhân vật
    channel_trust: dict[str, float] = Field(default_factory=dict)


class BeliefState(BaseModel):
    """Belief trong BDI — tách BIẾT khỏi TIN."""
    proposition: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str                                 # ai/cái gì khiến tin điều này
    is_actually_true: bool | None = None        # None = hệ thống chưa quyết
    # Chênh lệch giữa confidence cao và is_actually_true=False chính là nguồn
    # hiểu lầm TỰ NHIÊN, không cần tình tiết gượng ép.


class Desire(BaseModel):
    label: str
    layer: Literal["want", "need"]              # bề nổi vs nội tâm
    urgency: float = Field(ge=0.0, le=1.0)
    satisfied_by: list[str] = Field(default_factory=list)
    threatened_by: list[str] = Field(default_factory=list)


class CharacterProfile(BaseModel):
    id: str
    name: str
    role_tier: Literal["protagonist", "major", "supporting", "npc"]

    # ── BDI ──
    beliefs: list[BeliefState] = Field(default_factory=list)
    desires: list[Desire] = Field(default_factory=list)
    intentions: list[str] = Field(default_factory=list)

    # ── Năm trường của v1 ──
    want: str
    need: str
    fatal_flaw: Flaw
    private_agenda: str | None = None
    skills: dict[str, float] = Field(default_factory=dict)
    blindspots: list[str] = Field(default_factory=list)
    private_knowledge: list[BeliefState] = Field(default_factory=list)

    # ── v2 bổ sung ──
    voice: VoiceFingerprint
    somatic_signature: list[str] = Field(default_factory=list)
    moral_line: str
    moral_line_breached: bool = False
    leverage_over: dict[str, str] = Field(default_factory=dict)
    debt_to: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _co_ca_want_va_need(self) -> "CharacterProfile":
        """§5.1 nói "want và need phải mâu thuẫn — validator kiểm tra ở
        CharacterProfile", nhưng mâu thuẫn NGỮ NGHĨA thì code không kiểm được;
        nhận là kiểm được sẽ cho cảm giác an toàn giả.

        Cái kiểm được — và là lỗi thật hay gặp — là hồ sơ chỉ khai một tầng:
        có `want` mà không có `need` thì nhân vật không có nội tâm, và mọi
        cảnh của họ sẽ đọc như nhiệm vụ. Mâu thuẫn thật là việc của tác giả,
        duyệt ở CP-1 (§16.1).
        """
        layers = {d.layer for d in self.desires}
        if self.desires and layers != {"want", "need"}:
            raise ValueError(
                f"{self.id}: `desires` phải có CẢ HAI tầng want và need, "
                f"nhận {sorted(layers) or 'rỗng'}")
        if set(self.somatic_signature) & set(CLICHE_SOMATICS):
            raise ValueError(
                f"{self.id}: somatic_signature chứa sáo ngữ dùng chung: "
                f"{sorted(set(self.somatic_signature) & set(CLICHE_SOMATICS))}")
        return self
