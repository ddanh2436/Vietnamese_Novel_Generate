"""Bảng điều khiển chất lượng (§13.1, §13.3) — chạy M1–M12 trên canon đã ghi.

Mỗi metric có MỤC TIÊU và NGƯỠNG BÁO ĐỘNG riêng; bảng ở §13.1 cũng ghi nguyên
nhân thường gặp, nên báo động chỉ ra luôn chỗ nên tìm.

`value = None` (chưa đo được) KHÔNG phải báo động: thiếu dữ liệu khác hẳn kém
chất lượng. Trộn hai thứ đó lại là cách chắc chắn để người dùng tắt bảng này đi
(P2, §0.1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from novel_engine.eval import metrics as M
from novel_engine.eval.metrics import ChapterData

OUT_CHAPTERS = Path("output/chapters")
OUT_REPORTS = Path("output/reports")

# (mô tả mục tiêu, hàm báo động, nguyên nhân thường gặp) — §13.1 + §13.3
SPECS: dict[str, tuple[str, callable, str]] = {
    "M13_scene_repetition": ("≤ 0,085", lambda v: v > 0.085,
                             "cảnh sau dựng lại cảnh trước — trạng thái "
                             "không đổi giữa hai cảnh"),
    "M14_tic_saturation": ("≤ 3 lần/chương", lambda v: v > 3,
                           "cử chỉ nhận dạng thành con dấu đóng lại; "
                           "`somatic_forbidden` chưa tới được Writer"),
    "M15_hygiene_residue": ("0", lambda v: v > 0,
                            "có đường ghi văn xuôi đi vòng qua "
                            "`sanitize_prose`"),
    "M1_entity_consistency": ("≥ 0,98", lambda v: v < 0.95,
                              "Extractor gán sai `epistemic`; canon chưa lock"),
    "M2_clue_payoff_rate": ("≥ 0,90", lambda v: v < 0.80,
                            "`payoff_deadline` quá xa; ATTENTION_BUDGET quá chặt"),
    "M3_voice_distinctiveness": ("≥ 0,28", lambda v: v < 0.20,
                                 "`signature_lexicon` quá chung; thiếu `forbidden_lexicon`"),
    "M4_npc_reuse_ratio": ("≥ 0,45", lambda v: v < 0.30,
                           "nhân vật phụ mới sinh ra liên tục, không ai quay lại"),
    "M5_relationship_pacing": ("0 vi phạm", lambda v: v >= 1,
                               "guard bị bỏ qua; Extractor cộng intimacy quá tay"),
    "M6_blind_attribution": ("≥ 0,70", lambda v: v < 0.55,
                             "giọng đang bị làm phẳng — sửa `VoiceFingerprint`"),
    "M7_scene_necessity": ("≥ 0,85", lambda v: v < 0.70,
                           "`scene_must_change` chưa được cưỡng chế"),
    "M8_tension_mae": ("≤ 0,15", lambda v: v > 0.25,
                       "model bỏ qua `decompress`; cần ràng buộc cứng hơn"),
    "M9_prose_rhythm": ("σ ≥ 8,5", lambda v: v < 7.0,
                        "thiếu style exemplar; Polish chưa nhận `rhythm_audit`"),
    "M10_extraction_fidelity": ("≥ 0,95", lambda v: v < 0.90,
                                "prompt chưa cảnh báo span sẽ bị đối chiếu"),
    "M11_plan_fulfillment": ("≥ 0,85", lambda v: v < 0.70,
                             "`intensity` quá thấp; `carrier` không có trong cảnh"),
    # `likely_cause` cũ đổ cho "News Dispatcher chưa bật" — trên Arc 1 nó ĐANG
    # bật và ghi 3 bản tri thức mỗi chương. Nguyên nhân thật nằm ở outline.
    "M12_irony_gap": ("0,5 – 5,0", lambda v: v == 0 or v > 6,
                      "một POV nắm gần hết cảnh nên độc giả và nhân vật biết "
                      "cùng lúc — xem `zero_gap_ratio`, không chỉ trung bình"),
}
_SCENE_RE = re.compile(r"^## Cảnh (\d+)\s*$", re.M)


def load_chapters(eng, numbers, *, chapters_dir: Path | None = OUT_CHAPTERS,
                  reports_dir: Path | None = OUT_REPORTS) -> list[ChapterData]:
    chapters_dir = Path(chapters_dir or OUT_CHAPTERS)
    reports_dir = Path(reports_dir or OUT_REPORTS)
    out = []
    for n in numbers:
        md = chapters_dir / f"ch{n:03d}.md"
        scenes = []
        if md.exists():
            parts = _SCENE_RE.split(md.read_text(encoding="utf-8"))
            scenes = [{"scene_id": f"CH{n:03d}_S{int(parts[i]):02d}",
                       "prose": parts[i + 1].strip()}
                      for i in range(1, len(parts), 2)]
        deltas = eng.store.get_deltas(n)
        rp = reports_dir / f"ch{n:03d}.json"
        out.append(ChapterData(
            number=n, scenes=scenes,
            delta=deltas[0] if deltas else None,
            report=json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {},
            frames=eng.store.get_frames(n)))
    return out


def evaluate(eng, numbers, *, judge=None, chapters_dir: Path = OUT_CHAPTERS,
             reports_dir: Path = OUT_REPORTS, scene_sample: int | None = 6) -> dict:
    chapters = load_chapters(eng, numbers, chapters_dir=chapters_dir,
                             reports_dir=reports_dir)
    last = max(numbers) if numbers else 0
    results = {
        "M1_entity_consistency": M.M1_entity_consistency(chapters),
        "M2_clue_payoff_rate": M.M2_clue_payoff_rate(eng.graph, chapters, last),
        "M3_voice_distinctiveness": M.M3_voice_distinctiveness(chapters, eng.chars),
        "M4_npc_reuse_ratio": M.M4_npc_reuse_ratio(chapters, eng.chars),
        "M5_relationship_pacing": M.M5_relationship_pacing(eng.graph.relationships),
        "M6_blind_attribution": M.M6_blind_attribution(chapters, eng.chars, judge),
        "M7_scene_necessity": M.M7_scene_necessity(chapters, judge, scene_sample),
        "M8_tension_mae": M.M8_tension_mae(eng.store, chapters, eng.total_chapters),
        "M9_prose_rhythm": M.M9_prose_rhythm(chapters),
        "M10_extraction_fidelity": M.M10_extraction_fidelity(chapters),
        "M11_plan_fulfillment": M.M11_plan_fulfillment(chapters),
        "M12_irony_gap": M.M12_irony_gap(chapters, eng.graph, eng.planner),
        "M13_scene_repetition": M.M13_scene_repetition(chapters),
        "M14_tic_saturation": M.M14_tic_saturation(chapters, eng.chars),
        "M15_hygiene_residue": M.M15_hygiene_residue(chapters),
    }
    alerts, chua_do = [], []
    for name, res in results.items():
        muc_tieu, canh_bao, nguyen_nhan = SPECS[name]
        res["target"] = muc_tieu
        v = res.get("value")
        if v is None:
            chua_do.append(name)
            continue
        if canh_bao(v):
            alerts.append({"metric": name, "value": v, "target": muc_tieu,
                           "likely_cause": nguyen_nhan})
    return {"chapters": list(numbers), "metrics": results, "alerts": alerts,
            "not_measured": chua_do,
            "verdict": "ALERT" if alerts else "OK"}
