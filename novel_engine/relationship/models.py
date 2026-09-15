"""Không gian trạng thái quan hệ ba chiều (§7.1).

`Ideological_Friction` của v1 bị tách đôi vì hai đại lượng hành xử khác nhau:
`friction` hoà giải được bằng đối thoại, `stake_conflict` thì không — chỉ đổi
khi hoàn cảnh bên ngoài đổi. Gộp chúng lại sẽ sinh ra những cảnh hoà giải rẻ
tiền.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from novel_engine.canon.models import RelationStage


class RelationshipState(BaseModel):
    a: str
    b: str
    stage: RelationStage = RelationStage.STRANGERS
    intimacy: float = Field(default=0.0, ge=0.0, le=100.0)
    friction: float = Field(default=0.0, ge=0.0, le=100.0)
    stake_conflict: float = Field(default=0.0, ge=0.0, le=100.0)

    # Bất đối xứng — A có thể yêu B nhiều hơn B yêu A. Thiếu điều này,
    # mọi quan hệ đều là quan hệ song phương hoàn hảo, cực kỳ giả.
    intimacy_asym: float = Field(default=0.0, ge=-50.0, le=50.0)

    shared_ordeals: list[str] = Field(default_factory=list)
    unresolved_debts: list[str] = Field(default_factory=list)
    scars: list[str] = Field(default_factory=list)   # tổn thương KHÔNG lành
    stage_entered_chapter: int = 0
    chapters_in_stage: int = 0
    last_counted_chapter: int | None = None   # chống đếm theo cảnh (C7)
    # Chương đã CHỐT (áp sự kiện + xét guard). Fold chạy lại một delta, hay
    # một chương được chốt hai lần, không được cộng dồn hai lần.
    last_settled_chapter: int | None = None
    history: list[dict] = Field(default_factory=list)

    @property
    def pair_key(self) -> str:
        """Khoá ổn định, độc lập thứ tự — (A,B) và (B,A) là cùng một quan hệ."""
        return "|".join(sorted((self.a, self.b)))
