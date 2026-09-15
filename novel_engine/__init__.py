"""Novel Engine v2.5.

Package `__init__` chỉ làm đúng một việc: giải các forward-ref bắc cầu giữa
hai package có quan hệ hai chiều về KIỂU nhưng một chiều về IMPORT
(`canon.models.StateDelta` tham chiếu `relationship.models.RelationshipState`,
trong khi `relationship.models` import `RelationStage` từ `canon.models`).

Đặt ở đây vì package `__init__` luôn chạy trước mọi submodule — nhờ vậy kết
quả không phụ thuộc vào việc module nào được import trước.
Đừng thêm import nặng (LLM client, tiktoken...) vào file này.
"""
from novel_engine.canon.models import _resolve_forward_refs as _rfr

_rfr()
del _rfr

__version__ = "2.5.0"
