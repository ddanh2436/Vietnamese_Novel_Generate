"""Schema cốt lõi của Canon Store (§3.2, §3.3, §3.4).

Canon là kết quả fold các `StateDelta`, không phải một bảng bị ghi đè (§3.1).
Mọi thứ trong file này là *dữ liệu*, không có logic nghiệp vụ — logic nằm ở
`novel_engine/reconcile/` và `novel_engine/audit/`.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:                      # chỉ để type checker thấy
    from novel_engine.relationship.models import RelationshipState  # noqa: F401


def _utcnow() -> datetime:
    """`datetime.utcnow` bị deprecate từ Python 3.12 và trả về naive datetime.

    Tài liệu §3.4 viết `default_factory=datetime.utcnow`; trên 3.13 điều đó
    sinh DeprecationWarning ở MỌI lần khởi tạo StateDelta. Giữ nguyên ngữ
    nghĩa (UTC), đổi cách lấy.
    """
    return datetime.now(timezone.utc)


# ─────────────────────────────  ENUMS  ─────────────────────────────

class ClueStatus(str, Enum):
    DRAFTED         = "drafted"          # đã thiết kế, chưa xuất hiện
    PLANTED         = "planted"          # đã cài, độc giả đã thấy
    REINFORCED      = "reinforced"       # đã nhắc lại ≥1 lần
    PARTIALLY_READ  = "partially_read"   # có nhân vật hiểu một phần
    PAID_OFF        = "paid_off"         # đã trả bài
    RETIRED         = "retired"          # bỏ, không dùng nữa (phải ghi lý do)


class NPCRole(str, Enum):
    NARRATIVE_CATALYST  = "narrative_catalyst"   # giữ một mảnh manh mối
    THEMATIC_MIRROR     = "thematic_mirror"      # phản chiếu góc khuất của main
    FUTURE_CONSEQUENCE  = "future_consequence"   # hành động với NPC dội ngược sau
    AMBIENT             = "ambient"              # chỉ tạo texture — CÓ HẠN NGẠCH


class RelationStage(str, Enum):
    STRANGERS        = "strangers"
    FRICTION         = "friction"              # GĐ1: va chạm giá trị
    VULNERABILITY    = "vulnerability"         # GĐ2: bộc lộ tổn thương
    TRIAL            = "trial"                 # GĐ3: thử thách lòng tin
    CATHARSIS        = "catharsis"             # GĐ4: gắn kết mang tính trả giá
    RUPTURE          = "rupture"               # đổ vỡ (có thể quay lại TRIAL)
    SEVERED          = "severed"               # chấm dứt vĩnh viễn


class Severity(str, Enum):
    BLOCKER = "blocker"   # mâu thuẫn canon / rò rỉ POV → BẮT BUỘC viết lại
    MAJOR   = "major"     # lệch tính cách / manh mối hỏng → nên viết lại
    MINOR   = "minor"     # vấn đề văn phong → để Polish xử lý
    NOTE    = "note"      # ghi nhận, không hành động


# ─────────────────────────────  WORLD  ─────────────────────────────

class Entity(BaseModel):
    """Node nền cho mọi thực thể trong graph."""
    id: str                                   # CHAR_KAELEN, FACT_CHURCH, LOC_FORGE
    kind: Literal["character", "faction", "location", "object",
                  "event", "doctrine", "resource", "clue"]
    name: str
    aliases: list[str] = Field(default_factory=list)
    first_appearance: int | None = None        # số chương
    canon_locked: bool = False                 # True = tác giả chốt, agent cấm sửa
    attributes: dict[str, Any] = Field(default_factory=dict)


RelationType = Literal[
    "HAS_FEUD_WITH", "DEPENDS_ON", "PRACTICES_HERESY", "ORIGINATED_FROM",
    "CONTROLS", "MEMBER_OF", "OWES_DEBT_TO", "BETRAYED", "PROTECTS",
    "KNOWS_ABOUT", "SUSPECTS", "LOCATED_IN", "EVIDENCE_FOR",
]


class Relation(BaseModel):
    """Cạnh có hướng, có trọng số và có thời hiệu."""
    src: str
    dst: str
    type: RelationType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    since_chapter: int = 0
    until_chapter: int | None = None            # None = còn hiệu lực
    # NT-6 (§3.6.2): tri thức nhân vật khoá theo epoch_tick, KHÔNG theo chương.
    # `known_by(pov, epoch_tick)` lọc trên trường này; `since_chapter` chỉ dùng
    # cho truy vấn theo trục ĐỌC.
    since_tick: int = 0
    # Quan hệ này được ai biết? Rỗng = sự thật khách quan chưa ai biết.
    known_by: list[str] = Field(default_factory=list)
    provenance: str = "author"                  # author | extracted_ch12 | inferred


class Clue(BaseModel):
    clue_id: str
    macro_event_target: str                  # node đích trong Event DAG
    description: str                         # mô tả nội bộ (KHÔNG đưa cho Writer)

    # --- v1 đã có ---
    status: ClueStatus = ClueStatus.DRAFTED
    planted_in_chapter: int | None = None
    payoff_threshold: int                     # chương SỚM NHẤT được trả bài
    understood_by_characters: list[str] = Field(default_factory=list)

    # --- v2 bổ sung ---
    payoff_deadline: int                      # chương MUỘN NHẤT — thiếu trường
                                              # này thì manh mối treo vĩnh viễn
    prerequisites: list[str] = Field(default_factory=list)
                                              # clue_id phải PLANTED trước
    surface_forms: list[str] = Field(default_factory=list)
                                              # 3–5 cách hiện hình khác nhau, để
                                              # re-plant không lặp từ
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
                                              # độ "còn trong trí nhớ độc giả"
    last_touched_chapter: int | None = None
    subtlety_target: float = Field(default=0.6, ge=0.0, le=1.0)
                                              # 0 = đập vào mặt, 1 = gần như ẩn
    retire_reason: str | None = None

    @field_validator("payoff_deadline")
    @classmethod
    def _deadline_after_threshold(cls, v: int, info):
        thr = info.data.get("payoff_threshold")
        if thr is not None and v < thr:
            raise ValueError("payoff_deadline phải ≥ payoff_threshold")
        return v


# ───────────────────────  LUỒNG GHI NGƯỢC (§3.4)  ───────────────────────

class Assertion(BaseModel):
    """Một mệnh đề được trích từ văn xuôi vừa viết."""
    subject: str
    predicate: str
    object: str | int | float | bool
    chapter: int
    scene: int
    span: str                      # trích dẫn nguyên văn làm bằng chứng
    confidence: float = Field(ge=0.0, le=1.0)
    # Mệnh đề này là sự thật khách quan, hay chỉ là điều nhân vật TIN?
    epistemic: Literal["objective", "believed_by", "claimed_by"] = "objective"
    holder: str | None = None      # ai tin/ai nói, nếu không phải objective


class PlantEvidence(BaseModel):
    """Bằng chứng một plant_directive đã thực sự xuất hiện trên trang giấy.

    Đây là kênh RIÊNG, không dùng chung với `Assertion` (B1, NT-8). Hai lý do:
    (1) không gian định danh khác nhau — `Assertion.subject` là thực thể trong
        câu văn, `clue_id` là mã hệ thống; trộn chúng là nguồn của một lớp lỗi
        so khớp luôn-sai;
    (2) quan trọng hơn: PHẦN LỚN manh mối được cài KHÔNG sinh ra assertion nào
        cả. Một con dấu hoen gỉ tả trong bối cảnh không khẳng định sự thật mới
        nào về thế giới, nên Extractor sẽ không xuất assertion cho nó — nhưng
        nó ĐÃ được cài. Đo độ phủ qua assertion sẽ đếm thiếu một cách có hệ
        thống, và đếm thiếu đúng ở những manh mối tinh tế nhất.
    """
    clue_id: str
    scene_id: str
    span: str                       # trích dẫn nguyên văn, bắt buộc
    carrier_used: str               # object | dialogue | setting | behavior
    verified: bool = False          # do verify_spans() đặt, KHÔNG do LLM
    reacted_by: list[str] = Field(default_factory=list)
    concluded_by: list[str] = Field(default_factory=list)
                                    # ai đã RÚT RA KẾT LUẬN — khác với ai
                                    # nhìn thấy; dùng để chặn lộ bài sớm


def item_key(item: Entity | Relation | Assertion) -> str:
    """Khoá định danh ổn định cho một phần tử của StateDelta (§11).

    ĐẶT Ở ĐÂY, không ở `reconcile/classify.py` như tài liệu ghi, vì
    `StateDelta.item()` gọi nó — để ở reconcile thì canon phải import reconcile
    và reconcile phải import canon, thành vòng import. NT-11 đòi quy tắc nằm
    trong MỘT hàm mà mọi nơi cùng gọi; `reconcile.classify` re-export hàm này
    chứ không viết lại.

    F9/NT-16: `hash()` của Python ngẫu nhiên hoá theo PYTHONHASHSEED ở mỗi lần
    khởi động tiến trình. Một chương dừng ở checkpoint LangGraph rồi chạy tiếp
    ở tiến trình khác sẽ sinh khoá khác → `delta.item(k)` ném KeyError. Mọi
    khoá sống lâu hơn một tiến trình phải dùng hashlib. Dùng blake2b thay md5
    vì md5 không khả dụng ở chế độ FIPS.
    """
    if isinstance(item, Entity):
        return f"E:{item.id}"
    if isinstance(item, Relation):
        return f"R:{item.src}|{item.type}|{item.dst}"
    if isinstance(item, Assertion):
        h = hashlib.blake2b(item.span.encode("utf-8"), digest_size=3).hexdigest()
        return f"A:{item.subject}|{item.predicate}|{h}"
    raise TypeError(f"không biết khoá cho {type(item).__name__}")


class StateDelta(BaseModel):
    # NT-13: metadata hệ thống PHẢI có default. LLM không bao giờ sinh
    # `delta_id` hay `created_at`, nên để chúng bắt buộc là đảm bảo
    # `parse_model` ném ValidationError ở lượt trích xuất thứ hai.
    # `delta_id` dẫn xuất tất định (không dùng timestamp) để bộ hồi quy
    # §13.2 chạy cùng seed cho ra cùng id.
    delta_id: str = ""             # code gán sau khi parse: f"d_ch{chapter:03d}"
    chapter: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    new_entities: list[Entity] = Field(default_factory=list)
    new_relations: list[Relation] = Field(default_factory=list)
    retracted_relations: list[Relation] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    clue_transitions: dict[str, ClueStatus] = Field(default_factory=dict)
    relationship_updates: list["RelationshipState"] = Field(default_factory=list)
    plant_evidence: list[PlantEvidence] = Field(default_factory=list)

    # Kết quả phân loại ở Tầng 5
    classification: dict[str, Literal["contradiction", "enrichment",
                                      "improvement"]] = Field(default_factory=dict)
    # Quan hệ bị THU HỒI dùng khoá riêng `X:…` (xem `retract_key`), nên quyết
    # định của chúng không nằm chung `classification`, vốn tra bằng `item()`.
    retraction_classification: dict[str, Literal["contradiction", "enrichment"]] = \
        Field(default_factory=dict)
    # Mục bị cách ly — không ghi vào canon nhưng phải được LƯU, để tác giả thấy
    # ở CP-2 và để lần xem lại sau khi đã ghi không phải đoán lại.
    quarantine: list[dict] = Field(default_factory=list)
    # Vấn đề phát sinh TRƯỚC khi phân loại: mục bị loại lúc parse, span bị loại
    # lúc xác minh, bằng chứng manh mối ma, trích xuất rỗng. Lưu vào delta vì
    # CP-2 đọc delta — báo cáo của `write` thì tác giả có thể không bao giờ mở.
    extraction_issues: list[dict] = Field(default_factory=list)
    # Cường độ cài manh mối lấy từ contract lúc ghi. Lưu vào delta để fold phát
    # lại ra ĐÚNG salience như lúc ghi — contract không được lưu lại.
    plant_intensity: dict[str, float] = Field(default_factory=dict)
    committed: bool = False

    def item(self, key: str) -> Entity | Relation | Assertion:
        """Tra một phần tử theo khoá do item_key() sinh (§11). Bản trước gọi
        `delta.item(k)` trong reconcile_node mà không định nghĩa hàm này —
        AttributeError ngay cuối chương đầu tiên."""
        for coll in (self.new_entities, self.new_relations,
                     self.retracted_relations, self.assertions):
            for obj in coll:
                if item_key(obj) == key:
                    return obj
        raise KeyError(key)

    def stamp(self, chapter: int) -> "StateDelta":
        """NT-13: code ghi metadata SAU khi parse, không để LLM đoán.
        `delta_id` tất định theo chương để bộ hồi quy §13.2 tái lập được."""
        self.chapter = chapter
        self.delta_id = f"d_ch{chapter:03d}"
        for a in self.assertions:
            if a.chapter == 0:
                a.chapter = chapter
        return self


def _resolve_forward_refs() -> None:
    """Nối `StateDelta.relationship_updates` với `RelationshipState` thật.

    KHÔNG import ở module level — kể cả ở cuối file. `relationship.models`
    import ngược `RelationStage` từ đây, nên import hai chiều ở module level
    chỉ chạy được khi `canon.models` tình cờ được nạp trước; nạp
    `relationship.models` trước thì ném `ImportError: partially initialized
    module`. Đó là một lỗi phụ thuộc thứ tự import — đúng loại lỗi âm thầm mà
    một test import canon trước sẽ không bao giờ bắt được.

    Hàm này được gọi từ `novel_engine/__init__.py`: package `__init__` luôn
    chạy trước mọi submodule, nên forward-ref được giải đúng một lần bất kể
    entry point là gì.
    """
    from novel_engine.relationship.models import RelationshipState

    globals()["RelationshipState"] = RelationshipState
    StateDelta.model_rebuild()
