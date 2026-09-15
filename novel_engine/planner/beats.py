"""Beat Sheet generation (§8.3)."""
from __future__ import annotations

BEAT_ARCHETYPES: dict[str, list[str]] = {
    "escalate":   ["đặt cược", "va chạm", "leo thang", "mất kiểm soát", "hậu quả"],
    "sustain":    ["nhịp thở", "thăm dò", "phát hiện", "hiểu lầm", "rẽ hướng"],
    "decompress": ["hậu chấn", "sinh hoạt", "hồi ức", "gắn kết", "mầm mống mới"],
}

# Manh mối cài trong lúc cao trào sẽ bị NUỐT MẤT — độc giả lướt qua chi tiết
# khi đang đọc nhanh. Manh mối phải đặt ở chỗ độc giả đang đọc CHẬM.
QUIET_BEATS = {"nhịp thở", "thăm dò", "sinh hoạt", "hậu chấn", "đặt cược"}


def build_beats(contract_seed: dict, n: int = 6) -> list[dict]:
    mode = contract_seed["tension"]["mode"]
    arch = BEAT_ARCHETYPES[mode]
    beats: list[dict] = []
    for i in range(n):
        beats.append({
            "index": i,
            "function": arch[i % len(arch)],
            "pov_may_learn": None,      # Director điền, qua POV firewall
            "state_delta_expected": None,
            "clue_slot": None,          # scheduler gán
        })
    quiet = [b for b in beats if b["function"] in QUIET_BEATS]
    for d, b in zip(contract_seed.get("plant_directives", []), quiet or beats):
        b["clue_slot"] = d
    return beats
