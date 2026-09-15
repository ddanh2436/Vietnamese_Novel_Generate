"""Nạp `.env` — không thêm dependency.

`python-dotenv` làm đúng việc này, nhưng đây là ~20 dòng và ta đang giữ danh
sách phụ thuộc ngắn nhất có thể. Quan trọng hơn: bộ nạp này KHÔNG ghi đè biến
môi trường đã có sẵn, nên `NOVEL_LLM=fake pytest` vẫn thắng file `.env` —
chạy test mà vô tình đốt hạn ngạch vì file cấu hình là chuyện không nên xảy ra.
"""
from __future__ import annotations

import os
from pathlib import Path

# Tên biến chấp nhận cho khoá Gemini, theo thứ tự ưu tiên. Google dùng cả hai
# trong tài liệu của chính họ ở các thời điểm khác nhau.
GEMINI_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict:
    """Đọc `.env` vào `os.environ`. Trả về các cặp đã nạp (giá trị đã che)."""
    p = Path(path) if path else Path.cwd() / ".env"
    if not p.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if not k:
            continue
        if override or k not in os.environ:
            os.environ[k] = v
        loaded[k] = f"{v[:4]}…{v[-3:]}" if len(v) > 12 else "…"
    return loaded


def gemini_api_key() -> str | None:
    """Khoá Gemini từ bất kỳ tên biến nào được chấp nhận."""
    for name in GEMINI_KEY_NAMES:
        v = os.environ.get(name)
        if v:
            return v
    return None
