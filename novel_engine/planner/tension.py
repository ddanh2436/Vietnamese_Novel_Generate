"""Tension Curve Controller (§8.1).

Nhịp truyện không thể phó mặc cho model. Nếu không kiểm soát, LLM có xu hướng
đẩy MỌI chương lên cao trào — đọc 10 chương là mệt, đọc 40 chương là tê liệt.
Cần một đường cong mục tiêu và một bộ điều khiển kéo chương về đường cong đó.
"""
from __future__ import annotations

import math

# Đổi LOẠI áp lực là kỹ thuật chống bào mòn quan trọng nhất: nếu mọi chương
# đều là nguy hiểm thể chất, độc giả chai lì. Xen kẽ áp lực đạo đức (phải chọn
# điều sai), nhận thức (biết quá ít / quá muộn) và thời gian tạo cảm giác căng
# thẳng đa dạng mà KHÔNG cần tăng cường độ.
PRESSURE_TYPES = ["physical", "social", "moral", "epistemic", "temporal"]


def target_tension(chapter: int, total: int, acts: int = 3) -> float:
    """Đường cong mục tiêu 0..1. Dạng răng cưa đi lên: mỗi act có cao trào
    riêng, cao trào sau cao hơn cao trào trước, và SAU mỗi cao trào có một
    vùng trũng để độc giả thở."""
    p = chapter / total
    macro = 0.25 + 0.62 * (p ** 1.35)              # nền đi lên phi tuyến
    act_len = 1.0 / acts
    within = (p % act_len) / act_len               # vị trí trong act, 0..1
    micro = 0.22 * math.sin(math.pi * within ** 1.5)
    # Vùng trũng bắt buộc ngay sau mỗi cao trào act
    if within < 0.12 and chapter > total / acts * 0.5:
        micro -= 0.18
    # Cao trào cuối
    if p > 0.90:
        macro = 0.92 + 0.08 * (p - 0.90) / 0.10
    return max(0.05, min(1.0, macro + micro))


def tension_directive(chapter: int, total: int, measured_prev: float) -> dict:
    tgt = target_tension(chapter, total)
    prev_tgt = target_tension(chapter - 1, total) if chapter > 1 else 0.2
    delta = tgt - measured_prev

    if delta > 0.18:
        mode = "escalate"
        note = "Chương trước hạ nhiệt hơn kế hoạch — cần leo thang rõ."
    elif delta < -0.18:
        mode = "decompress"
        note = ("Chương trước quá căng. Chương này PHẢI có vùng lặng: sinh "
                "hoạt, hồi ức, hoặc một cảnh hoàn toàn không có nguy hiểm "
                "vật lý.")
    else:
        mode = "sustain"
        note = "Giữ nhịp, đổi LOẠI áp lực thay vì tăng cường độ."

    return {
        "target": round(tgt, 3), "previous_measured": round(measured_prev, 3),
        "mode": mode, "note": note,
        "rising": tgt > prev_tgt,
        "pressure_type": PRESSURE_TYPES[chapter % len(PRESSURE_TYPES)],
    }
