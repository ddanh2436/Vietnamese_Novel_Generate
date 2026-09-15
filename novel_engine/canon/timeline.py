"""Timeline Engine — thời gian phi tuyến và Multi-POV song song (§3.6, §12.4).

Hệ thống dùng HAI trục độc lập, và trộn chúng là nguồn của cả một lớp lỗi:

| Trục           | Trả lời                                  | Khoá theo         |
|----------------|------------------------------------------|-------------------|
| Story time     | Sự việc xảy ra khi nào trong thế giới?   | `epoch_tick`      |
| Narrative time | Độc giả biết điều đó ở thời điểm nào?    | `narrative_order` |

NT-6: tri thức nhân vật khoá theo `epoch_tick`; tri thức độc giả khoá theo
`narrative_order`. Đừng bao giờ trộn hai trục.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TICKS_PER_HOUR = 1     # 1 tick = 1 giờ truyện. Đổi granularity DUY NHẤT ở đây;
                       # đừng rải hằng số thời gian ra khắp codebase.


class StoryTime(BaseModel):
    epoch_tick: int                    # trục TUYỆT ĐỐI của thế giới
    duration_ticks: int = 1
    narrative_order: int               # thứ tự độc giả đọc (đơn điệu tăng)
    mode: Literal["present", "flashback", "concurrent", "vision"] = "present"
    anchor_scene: str | None = None    # flashback/concurrent neo vào cảnh nào

    @property
    def end_tick(self) -> int:
        return self.epoch_tick + self.duration_ticks

    def overlaps(self, other: "StoryTime") -> bool:
        return self.epoch_tick < other.end_tick and other.epoch_tick < self.end_tick


class ContinuityObservation(BaseModel):
    """Phần trạng thái vật lý mà CHỈ ĐỌC VĂN BẢN mới biết. Cố ý KHÔNG có
    trường `time` (§12.4).

    F2: bản trước cho `SceneClose.continuity` là `ContinuityFrame`, vốn chứa
    `StoryTime` với `narrative_order` bắt buộc. LLM không có cách nào biết
    `narrative_order` là gì, nên Pydantic ném ValidationError ở mọi ranh giới
    cảnh. Sửa bằng cách CẮT TRƯỜNG ĐÓ KHỎI SCHEMA thay vì dặn model trong
    prompt: schema không có thì model không thể quên (NT-13).
    """
    locations: dict[str, str]              # char_id → location_id
    injuries: dict[str, list[str]] = Field(default_factory=dict)
    possessions: dict[str, list[str]] = Field(default_factory=dict)
    weather: str | None = None


class ContinuityFrame(BaseModel):
    """Trạng thái vật lý tại cuối mỗi cảnh — khoá theo epoch, KHÔNG theo chương."""
    scene_id: str
    time: StoryTime
    locations: dict[str, str]              # char_id → location_id
    injuries: dict[str, list[str]] = Field(default_factory=dict)
    possessions: dict[str, list[str]] = Field(default_factory=dict)
    weather: str | None = None


class SceneClose(BaseModel):
    """Kết quả chốt sổ một cảnh: bản tóm tắt L1 + trạng thái vật lý (§12.4)."""
    digest: str                           # 5–8 câu, dùng cho L1 (§4.1)
    continuity: ContinuityObservation
    actual_duration_ticks: int            # thời lượng THỰC so với dự kiến
    unresolved: list[str] = Field(default_factory=list)
