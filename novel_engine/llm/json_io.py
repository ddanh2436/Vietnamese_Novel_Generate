"""Ranh giới JSON — chỗ pipeline gãy sớm nhất (§9.5).

`StateDelta.model_validate_json(raw)` gọi thẳng trên đầu ra của LLM là dòng
code sẽ hỏng đầu tiên khi chạy thật. Dù prompt có ghi "Chỉ xuất JSON", model
vẫn thường bọc đầu ra trong ```json … ```, đôi khi kèm một câu mở đầu lịch sự.
`JSONDecodeError` ném ra giữa `extractor_node` sẽ làm sập cả luồng LangGraph —
và sập ở bước CUỐI CÙNG, sau khi đã trả tiền cho toàn bộ 6 cảnh.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")

M = TypeVar("M", bound=BaseModel)


def strip_fences(raw: str) -> str:
    """Lột markdown fence và lời dẫn quanh JSON.

    Lấy khối DÀI NHẤT, không phải khối đầu tiên. Điều này quan trọng hơn vẻ
    ngoài: prompt của Extractor có chứa ví dụ JSON, và model thỉnh thoảng nhắc
    lại ví dụ ấy trước khi xuất kết quả thật. Lấy khối đầu tiên sẽ nuốt phải
    ví dụ, parse THÀNH CÔNG, và ghi một delta rỗng vào canon — đúng kiểu lỗi
    âm thầm mà NT-5 sinh ra để chặn.
    """
    raw = (raw or "").strip()
    blocks = FENCE_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        return raw[i:j + 1]
    return raw


JSON_REPAIR_TMPL = """JSON dưới đây không hợp lệ so với schema. Sửa và chỉ
xuất JSON đã sửa, không giải thích, không bọc trong dấu ```.

SCHEMA: {schema}
LỖI: {error}
JSON HỎNG:
{broken}
"""


def parse_model(raw: str, model_cls: type[M], repair_llm=None,
                max_repair: int = 1) -> M:
    """Lớp 2 (lột fence) + lớp 3 (một lượt tự sửa).

    Lớp 1 — structured output của chính nhà cung cấp — mới là cách sửa thật,
    và nên dùng ở đâu có. Hai lớp này là phòng thủ chiều sâu cho chỗ không
    dùng được.
    """
    text = strip_fences(raw)
    last_err: Exception | None = None
    for attempt in range(max_repair + 1):
        try:
            return model_cls.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            if repair_llm is None or attempt == max_repair:
                break
            # Một lượt sửa rẻ hơn nhiều so với viết lại cả chương.
            text = strip_fences(repair_llm.invoke(JSON_REPAIR_TMPL.format(
                schema=json.dumps(model_cls.model_json_schema())[:4000],
                broken=text[:6000], error=str(e)[:1200]), role="repair"))
    raise ValueError(f"không parse được {model_cls.__name__}: {last_err}")


def merge_json(base: dict, filled_raw: str) -> dict:
    """Trộn phần LLM điền vào khung do code dựng.

    Một chiều, và chiều đó quan trọng: code THẮNG ở mọi khoá mà code đã đặt.
    `time`, `scene_id`, `plant_directives` là của hệ thống (NT-12) — để model
    ghi đè chúng là tạo ra hai nguồn sự thật cho cùng một đại lượng, và chúng
    sẽ lệch nhau ở chương thứ ba.
    """
    try:
        filled = json.loads(strip_fences(filled_raw))
    except (json.JSONDecodeError, TypeError):
        return base
    if not isinstance(filled, dict):
        return base
    out = dict(base)
    for k, v in filled.items():
        # Chỉ nhận khoá mà code CỐ Ý để trống — không nhận khoá code đã điền.
        if k in out and out[k] in ("", None, [], {}):
            out[k] = v
    return out
