"""Auditor tầng thuật toán — gộp mọi kiểm tra bằng code cho một cảnh (§10, §10.3).

Chạy TRƯỚC Auditor LLM (§9.2): rẻ, chắc chắn, và nếu đã có blocker thì bỏ hẳn
lượt gọi LLM.

`pov_leak_scan` nhận TÊN hiển thị của POV, không nhận mã. Mã §10.3 truyền
`contract["pov_character"]` (tức `CHAR_KAELEN`), mà `pov_leak_scan` tìm tên đó
TRONG VĂN BẢN để miễn trừ câu POV tự nhận mình không biết. Mã không bao giờ xuất
hiện trong văn xuôi, nên "Kaelen không biết rằng mình đang bị theo dõi" — câu
hoàn toàn hợp lệ — bị báo là rò rỉ POV mức BLOCKER. NT-8 lần nữa.

Self-plagiarism (§4.3) cần vector store nên chưa có ở đây.
"""
from __future__ import annotations

from novel_engine.audit.prose import prose_audit, sensory_channels
from novel_engine.audit.rhythm import rhythm_audit, rhythm_stats
from novel_engine.character.firewall import pov_leak_scan
from novel_engine.character.voice_check import attribute_dialogue, voice_report
from novel_engine.planner.contract import SceneContract
from novel_engine.planner.tension import tension_directive


def _pov_name(contract: dict, eng) -> str:
    if contract.get("pov_character_name"):
        return contract["pov_character_name"]
    prof = eng.chars.get(contract.get("pov_character", "")) if eng else None
    return prof.name if prof else contract.get("pov_character", "")


def _speaker_names(contract: dict) -> dict[str, str]:
    return {x["name"]: x["id"] for x in contract.get("active_characters", [])
            if x.get("name") and x.get("id")}


def deterministic_audit(prose: str, contract: dict, chapter: int, eng) -> list[dict]:
    out: list[dict] = []
    out += prose_audit(prose, contract)
    out += rhythm_audit(prose, contract, lang="vi")
    for h in pov_leak_scan(prose, _pov_name(contract, eng)):
        out.append({**h, "message": "có thể rò rỉ POV: …"
                    + " ".join(h["span"].split()) + "…"})

    attr = attribute_dialogue(prose, _speaker_names(contract))
    for cid, lines in attr["by_speaker"].items():
        prof = eng.chars.get(cid) if eng else None
        if prof is None:
            continue
        for v in voice_report(lines, prof, attr["inferred"][cid])["violations"]:
            out.append({"severity": v["severity"], "check": "voice",
                        "message": f"{prof.name}: {v['message']}"})
    return out


def audit_stats(prose: str, contract: dict) -> dict:
    """Số đo để HIỆU CHỈNH ngưỡng (§10.3.3), không phải để chấm điểm."""
    attr = attribute_dialogue(prose, _speaker_names(contract))
    return {
        "words": len(prose.split()),
        "rhythm": rhythm_stats(prose),
        "sensory": sensory_channels(prose),
        "dialogue_attributed": {cid: len(v) + len(attr["inferred"][cid])
                                for cid, v in attr["by_speaker"].items()},
        "dialogue_inferred": sum(len(v) for v in attr["inferred"].values()),
        "dialogue_unattributed": len(attr["unattributed"]),
    }


_DEFAULTS = SceneContract.model_fields


def audit_contract(eng, chapter: int, si: int) -> dict:
    """Dựng lại phần hợp đồng mà kiểm tra tất định cần, cho một cảnh ĐÃ VIẾT.

    Contract không được lưu, nhưng mọi trường kiểm tra dùng đều tất định: POV và
    người có mặt từ outline, nhịp từ `tension_directive`, ngân sách từ chính
    default của `SceneContract` — không khai lại con số lần thứ hai (NT-11).
    """
    pov = eng.planner.pov_for(chapter, si)
    present = eng.planner.present_characters(chapter, si)
    return {
        "scene_id": f"CH{chapter:03d}_S{si:02d}",
        "pov_character": pov,
        "pov_character_name": eng.chars[pov].name if pov in eng.chars else pov,
        "active_characters": [{"id": c, "name": eng.chars[c].name}
                              for c in present if c in eng.chars],
        "tension": tension_directive(chapter, eng.total_chapters,
                                     eng.store.measured_tension(chapter - 1)),
        "word_budget": _DEFAULTS["word_budget"].default,
        "sensory_channels_required": _DEFAULTS["sensory_channels_required"].default,
        "max_explicit_goal_statements": _DEFAULTS["max_explicit_goal_statements"].default,
        "forbidden_cliches": [],
    }
