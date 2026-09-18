"""Bộ hồi quy (§13.2) — so bộ chỉ số hiện tại với một mốc đã duyệt.

"Mỗi lần thay đổi prompt hoặc tham số, sinh lại các chương vàng với cùng seed và
so metric. Tụt quá 10% thì rollback."

Hai chỗ khác §13.2:

- §13.2 tính `delta = (v - b) / |b|` rồi báo động khi `delta < -0.10` cho MỌI
  metric. Nhưng M8 (sai số đường cong căng thẳng) và M5 (số vi phạm) là THẤP THÌ
  TỐT: tụt giá trị ở đó là tiến bộ, còn tăng mới là hồi quy. Dùng chung một dấu
  cho cả hai loại là báo ngược đúng những metric đáng lo nhất.
- Metric CHƯA ĐO ĐƯỢC (`value = None`) không phải hồi quy, và cũng không phải
  đạt. Nó được liệt kê riêng để người đọc biết mốc so sánh đang thiếu mảng nào.
"""
from __future__ import annotations

LOWER_IS_BETTER = {"M5_relationship_pacing", "M8_tension_mae",
                   "M13_scene_repetition", "M14_tic_saturation",
                   "M15_hygiene_residue"}
TOLERANCE = 0.10


def _value(entry) -> float | None:
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def run_regression(current: dict, baseline: dict, tolerance: float = TOLERANCE) -> dict:
    cur_m = current.get("metrics", current)
    base_m = baseline.get("metrics", baseline)
    report: dict[str, dict] = {}
    regressed, missing = [], []

    for name in sorted(set(cur_m) | set(base_m)):
        v, b = _value(cur_m.get(name)), _value(base_m.get(name))
        if v is None or b is None:
            missing.append(name)
            report[name] = {"baseline": b, "current": v, "delta_pct": None}
            continue
        delta = (v - b) / max(abs(b), 1e-6)
        if name in LOWER_IS_BETTER:
            delta = -delta                      # thấp hơn là tốt hơn
        report[name] = {"baseline": b, "current": v,
                        "delta_pct": round(delta * 100, 1)}
        if delta < -tolerance:
            regressed.append(name)

    return {"report": report, "regressed": regressed, "not_comparable": missing,
            "verdict": "ROLLBACK" if regressed else "PASS"}
