"""Scene Contract (§8.2).

Ba trường đáng nhấn mạnh:

- `scene_must_change` — cảnh nào không thay đổi trạng thái thì là cảnh thừa.
  Ràng buộc này một mình đã loại bỏ phần lớn các cảnh "đi lại nói chuyện" vô
  nghĩa mà LLM rất hay sinh ra.
- `max_explicit_goal_statements` — chống bệnh "nhân vật tự thuyết minh động
  cơ". Đặt là 1 hoặc 0.
- `pov_knowledge_boundary` — liệt kê rõ điều POV KHÔNG biết, kèm chỉ thị "nếu
  cần nhắc, chỉ được nhắc qua sự hiểu sai của POV".
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from novel_engine.canon.timeline import StoryTime


class CharacterInScene(BaseModel):
    id: str
    name: str
    immediate_goal: str = ""
    # v1 đã có
    secret_fear: str | None = None
    internal_conflict: str | None = None
    hidden_action: str | None = None
    skill_deployed: str | None = None
    observable_behavior: str = ""
    # v2 bổ sung
    deliberation: dict = Field(default_factory=dict)   # kết quả deliberate() §5.2
    voice_reminder: dict = Field(default_factory=dict)  # trích từ VoiceFingerprint
    somatic_allowed: list[str] = Field(default_factory=list)
    somatic_forbidden: list[str] = Field(default_factory=list)
    knows_in_this_scene: list[str] = Field(default_factory=list)  # đã qua firewall
    must_not_reveal: list[str] = Field(default_factory=list)


class SceneContract(BaseModel):
    scene_id: str
    chapter: int
    scene_index: int
    # NT-12: chủ sở hữu trường này là `allocate_scene_times` trong Director.
    # Thêm trường mà không chỉ định chủ sở hữu là cách tạo NameError ở hạ nguồn.
    time: StoryTime
    location: str
    location_id: str = ""
    pov_character: str
    pov_character_name: str = ""
    pov_knowledge_boundary: list[str] = Field(default_factory=list)

    active_characters: list[CharacterInScene] = Field(default_factory=list)

    # v2: mục tiêu tự sự ĐO ĐƯỢC. Bốn trường này cố ý để rỗng — Director LLM
    # điền, và `merge_json` chỉ nhận khoá nào đang rỗng (NT-12).
    dramatic_question: str = ""
    scene_must_change: str = ""
    entry_state: str = ""
    exit_state: str = ""

    tension: dict = Field(default_factory=dict)
    relationship_directives: list[dict] = Field(default_factory=list)
    plant_directives: list[dict] = Field(default_factory=list)
    lore_integration: list[dict] = Field(default_factory=list)
    # Chủ sở hữu: NewsDispatcher (§5.6) qua Director. Tin POV nghe TRONG cảnh này.
    news_directives: list[dict] = Field(default_factory=list)

    # v2: ràng buộc văn phong đo được
    word_budget: tuple[int, int] = (900, 1600)
    forbidden_cliches: list[str] = Field(default_factory=list)
    subtext_requirement: str = ""
    max_explicit_goal_statements: int = 1
    sensory_channels_required: int = 3
