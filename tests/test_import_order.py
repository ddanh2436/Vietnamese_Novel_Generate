"""Hồi quy: thứ tự import không được làm vỡ forward-ref.

`canon.models.StateDelta` tham chiếu `relationship.models.RelationshipState`,
còn `relationship.models` import ngược `RelationStage` từ `canon.models`.
Bản đầu tiên giải forward-ref bằng một import ở cuối `canon/models.py`. Nó
chạy được — nhưng CHỈ khi `canon.models` tình cờ được nạp trước. Vì mọi test
khác đều import canon trước, bug đi qua suite sạch sẽ.

Mỗi test dưới đây chạy trong TIẾN TRÌNH RIÊNG. Đó là điểm mấu chốt: trong
cùng một tiến trình, `sys.modules` đã có sẵn cả hai module từ test trước nên
thứ tự import không còn ý nghĩa và bug không thể tái hiện.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
}

_CHECK = (
    "from novel_engine.canon.models import StateDelta;"
    "from novel_engine.relationship.models import RelationshipState;"
    "d = StateDelta(relationship_updates=[RelationshipState(a='A', b='B')]);"
    "back = StateDelta.model_validate_json(d.model_dump_json());"
    "assert isinstance(back.relationship_updates[0], RelationshipState);"
    "print('OK')"
)


def _run(first_import: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"import {first_import}\n{_CHECK}"],
        capture_output=True, text=True, encoding="utf-8", env=_ENV)


@pytest.mark.parametrize("first", [
    "novel_engine.canon.models",
    "novel_engine.relationship.models",   # thứ tự làm vỡ bản đầu tiên
    "novel_engine",
])
def test_moi_thu_tu_import_deu_chay(first: str):
    r = _run(first)
    assert r.returncode == 0, f"import {first} trước thì vỡ:\n{r.stderr}"
    assert r.stdout.strip().endswith("OK")
