"""Forward-Reachability Audit cho cảnh hồi ức (§3.6.4).

Fold theo `epoch_tick` nghĩa là một delta hồi ức được chèn vào GIỮA lịch sử đã
có, nên mọi trạng thái sau nó đều có thể bị thay đổi. Đó chính là mối nguy:
chương 15 kể Kaelen năm 10 tuổi và vô tình tả cậu bé ĐÃ CÓ vết sẹo mà chương
10 nói là có từ tick 5000 — canon lặng lẽ đổi, và nếu vết sẹo ấy là manh mối
thì manh mối vừa bị vô hiệu hoá mà không luật nào phát hiện.

Ràng buộc: hồi ức chỉ được CỘNG THÊM sự thật về quá khứ, không được phủ định
sự thật đã ghi.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from novel_engine.canon.models import StateDelta
    from novel_engine.canon.timeline import ContinuityFrame

# Thuộc tính VĨNH VIỄN — một khi có thì không mất, nên thời điểm khởi phát
# của chúng là kiểm chứng được.
PERMANENT_PREDICATES = {
    "has_scar", "is_dead", "lost_limb", "bears_brand", "holds_title",
    "sworn_oath", "is_exiled", "knows_secret",
}


class AttributeWindow(BaseModel):
    """Cửa sổ thời gian mà một thuộc tính vĩnh viễn có thể đã khởi phát."""
    entity: str
    attribute: str
    attested_present: list[int] = Field(default_factory=list)  # tick CÓ bằng chứng
    attested_absent: list[int] = Field(default_factory=list)   # tick bằng chứng CHƯA có

    @property
    def onset_bounds(self) -> tuple[int | None, int | None]:
        lo = max(self.attested_absent) if self.attested_absent else None
        hi = min(self.attested_present) if self.attested_present else None
        return lo, hi

    @property
    def consistent(self) -> bool:
        lo, hi = self.onset_bounds
        return lo is None or hi is None or lo < hi


def flashback_admissible(delta: "StateDelta", frame: "ContinuityFrame",
                         canon) -> list[dict]:
    """Delta hồi ức có phá vỡ tương lai không?

    Chạy trong `reconcile_node` (§11), TRƯỚC mọi thao tác ghi. Mọi `blocker` ở
    đây đi thẳng tới escalation cho tác giả — không tự động sửa, vì hai phiên
    bản quá khứ mâu thuẫn nhau là quyết định sáng tác, không phải lỗi kỹ thuật.
    """
    if frame.time.mode not in ("flashback", "vision"):
        return []
    out: list[dict] = []
    tick = frame.time.epoch_tick

    # (a) Thuộc tính vĩnh viễn: kiểm tra cửa sổ khởi phát còn hợp lệ không
    for a in delta.assertions:
        if a.predicate not in PERMANENT_PREDICATES:
            continue
        w = canon.attribute_window(a.subject, a.predicate)
        w.attested_present.append(tick)
        if not w.consistent:
            lo, _hi = w.onset_bounds
            out.append({
                "severity": "blocker", "check": "flashback_causality",
                "message": (f"hồi ức tại tick {tick} khẳng định "
                            f"{a.subject}.{a.predicate}, nhưng canon đã ghi "
                            f"thuộc tính này CHƯA có tại tick {lo} "
                            f"(muộn hơn) — mâu thuẫn nhân quả"),
                "evidence": a.span,
            })

    # (b) Hồi ức KHÔNG được huỷ quan hệ — huỷ là thao tác của hiện tại
    if delta.retracted_relations:
        out.append({
            "severity": "blocker", "check": "flashback_retraction",
            "message": ("cảnh hồi ức cố huỷ quan hệ "
                        f"{[(r.src, r.type, r.dst) for r in delta.retracted_relations]}"
                        " — hồi ức chỉ được bổ sung, không được phủ định"),
        })

    # (c) Hồi ức không được giết ai còn sống ở hiện tại, và ngược lại
    for a in delta.assertions:
        if a.predicate == "is_dead" and canon.alive_after(a.subject, tick):
            out.append({
                "severity": "blocker", "check": "flashback_mortality",
                "message": f"{a.subject} chết ở tick {tick} nhưng còn sống sau đó",
            })

    # (d) Thực thể mới do hồi ức tạo ra phải khai báo số phận ở hiện tại
    for e in delta.new_entities:
        if e.kind == "character" and not canon.exists(e.id):
            out.append({
                "severity": "note", "check": "flashback_orphan",
                "message": (f"hồi ức giới thiệu nhân vật mới '{e.name}' — "
                            f"cần quyết định người này hiện ở đâu, hoặc "
                            f"đăng ký vào Chekhov Registry (§5.5)"),
            })
    return out
