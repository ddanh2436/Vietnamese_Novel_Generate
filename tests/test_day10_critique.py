"""Ngày 10 — Bounded Critique Loop & Polish (§9.2, §9.3, §10.3.2).

Kịch bản chạy bằng FakeLLM có chèn lỗi: Writer nhận thêm một câu vi phạm ở
những lần viết chỉ định, Auditor LLM và Polish trả về thứ test chọn. Mọi quyết
định định tuyến vẫn là quyết định THẬT của đồ thị.
"""
from __future__ import annotations

import json
from collections import Counter

import pytest

from novel_engine.audit.critique import (
    LLM_CHECKS, audit_checklist, clean_polish_output, content_drifted,
    new_serious, parse_llm_findings, polish_notes, routing_severity,
    writer_feedback,
)
from novel_engine.graph.build import recursion_limit_for, run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.graph.routing import MAX_REVISIONS, after_audit
from novel_engine.llm.fake import FakeLLM
from novel_engine.prompts import WRITER_TMPL

LEAK = "Sự thật là Serena đang che giấu điều gì đó."
MAJOR = "— Tôi muốn tìm kẻ đó. Anh phải đến lò phản ứng ngay. Em cần cứu cô ấy."
TRI = " ".join([
    "Khi đèn pha quét qua mặt nước, Kaelen lùi vào bóng tối, trong khi Vhal vẫn cúi đầu ghi sổ.",
    "Khi cần trục dừng lại, gió đổi hướng về phía cảng, và bụi quặng bay ngang mặt đường.",
    "Khi còi tàu hú lên lần nữa, đám công nhân tản ra, còn người gác cổng vẫn đứng yên.",
])
EVID = "Kaelen đặt tấm thẻ kim loại lên mặt bàn gỗ."


class KichBan(FakeLLM):
    """FakeLLM có kịch bản theo CẢNH và theo LẦN GỌI trong cảnh đó.

    Cảnh hiện tại = số lượt `scene_digest` đã gọi — mỗi ranh giới cảnh gọi đúng
    một lần, nên đếm được mà không cần đọc prompt.
    """

    def __init__(self, writer=None, auditor=None, polish=None):
        super().__init__()
        self.writer_script = writer or {}
        self.auditor_script = auditor or {}
        self.polish_fn = polish
        self.scene = 0
        self.tries: Counter = Counter()

    def invoke(self, prompt, *, role=""):
        out = super().invoke(prompt, role=role)
        si = self.scene
        if role == "scene_digest":
            self.scene += 1
        elif role in ("writer", "auditor"):
            k = self.tries[(role, si)]
            self.tries[(role, si)] += 1
            script = (self.writer_script if role == "writer" else self.auditor_script).get(si) or []
            v = script[k] if k < len(script) else None
            if isinstance(v, Exception):
                raise v
            if v and role == "writer":
                out = out + "\n\n" + v
            elif v is not None and role == "auditor":
                out = v
        elif role == "polish" and self.polish_fn:
            out = self.polish_fn(out)
        return out


def _chay(llm, **flags):
    eng = build_engines(llm, db_path=":memory:")
    # Văn xuôi FakeLLM dựng từ cùng một kho câu nhỏ nên các cảnh CÙNG CHỖ
    # trùng nhau thật (§4.3). Tắt dò lặp ở fixture tổng hợp; test Ngày 21
    # kiểm cơ chế đó bằng một LLM lặp có chủ đích.
    eng.repetition_check = False
    for k, v in flags.items():
        setattr(eng, k, v)
    return eng, run_chapter(eng, 1)


def _vai(llm):
    return Counter(c["role"] for c in llm.calls)


def _prompts(llm, role):
    return [c["prompt"] for c in llm.calls if c["role"] == role]


# ═══════════════ VÒNG KIỂM TOÁN ═══════════════

def test_ban_nhap_sach_khong_bi_viet_lai():
    llm = KichBan()
    eng, out = _chay(llm)
    assert not out.get("escalated"), out.get("escalation_reason")
    assert _vai(llm)["writer"] == 6 and _vai(llm)["auditor"] == 6
    assert [x["draft"] for x in out["audit_log"]] == [0] * 6
    eng.store.close()


def test_blocker_mot_lan_duoc_viet_lai_va_phan_hoi_den_writer():
    llm = KichBan(writer={0: [LEAK]})
    eng, out = _chay(llm)
    assert not out.get("escalated")
    assert _vai(llm)["writer"] == 7
    lan_hai = _prompts(llm, "writer")[1]
    assert "PHẢN HỒI" in lan_hai and "Sự thật là Serena" in lan_hai
    assert "PHẢN HỒI" not in _prompts(llm, "writer")[0]

    s0 = out["scene_outputs"][0]
    assert s0["audit"]["revisions"] == 1 and LEAK not in s0["prose"]
    log0 = [x for x in out["audit_log"] if x["scene_id"] == "CH001_S00"]
    assert log0[0]["max_severity"] == "blocker"
    # Blocker từ tầng thuật toán → bỏ lượt gọi LLM cho bản nháp chắc chắn bỏ.
    assert log0[0]["llm_called"] is False and log0[1]["llm_called"] is True
    assert _vai(llm)["auditor"] == 6
    eng.store.close()


def test_blocker_dai_dang_escalate_sau_dung_hai_lan_viet_lai():
    llm = KichBan(writer={2: [LEAK] * 5})
    eng, out = _chay(llm)
    assert out["escalated"] is True
    assert _vai(llm)["writer"] == 1 + 1 + (MAX_REVISIONS + 1)
    assert "CH001_S02" in out["escalation_reason"]
    assert f"sau {MAX_REVISIONS} lần viết lại" in out["escalation_reason"]
    assert "extractor_diff" not in _vai(llm)        # không trích xuất chương dở

    blocked = out["escalated_scene"]
    assert blocked["scene_index"] == 2 and LEAK in blocked["prose"]
    assert any(f["check"] == "pov_leak" for f in blocked["findings"])
    assert len(out["scene_outputs"]) == 2           # văn xuôi đã viết vẫn còn
    eng.store.close()


def test_escalate_giua_vong_canh_tra_canon_store_ve_truoc_chuong():
    """Frame của hai cảnh đầu đã vào store. Để lại thì chương sau đọc mốc thời
    gian của nửa chương không bao giờ được xuất bản."""
    llm = KichBan(writer={2: [LEAK] * 5})
    eng, out = _chay(llm)
    assert out["rolled_back"] is True
    assert eng.store.get_frames(1) == []
    assert not eng.store.has_chapter(1)
    eng.store.close()


def test_major_chi_duoc_viet_lai_mot_lan_roi_di_polish():
    llm = KichBan(writer={0: [MAJOR] * 5})
    eng, out = _chay(llm)
    assert not out.get("escalated")
    assert _vai(llm)["writer"] == 6 + 1
    s0 = out["scene_outputs"][0]["audit"]
    assert s0["revisions"] == 1
    assert any(f["check"] == "on_the_nose" and f["severity"] == "major"
               for f in s0["residual"])            # còn lại được GHI, không mất
    eng.store.close()


def test_chi_so_nhip_major_KHONG_quay_lai_writer():
    """§10.3.2 quy tắc 2 — mâu thuẫn giữa §10.3.1 (major) và §9.3 (major → revise)."""
    llm = KichBan(writer={0: [TRI]})
    eng, out = _chay(llm)
    assert _vai(llm)["writer"] == 6
    assert not any("khuôn" in p for p in _prompts(llm, "writer"))
    assert out["scene_outputs"][0]["audit"]["polish"]["called"] is True
    assert any("Khi A, B, trong khi C" in p for p in _prompts(llm, "polish"))
    eng.store.close()


def test_revision_count_reset_o_moi_canh():
    """C6: không reset thì cảnh 1 escalate ngay vì cảnh 0 đã tiêu hết lượt."""
    llm = KichBan(writer={0: [LEAK, LEAK], 1: [LEAK, LEAK]})
    eng, out = _chay(llm)
    assert not out.get("escalated"), out.get("escalation_reason")
    assert [s["audit"]["revisions"] for s in out["scene_outputs"][:2]] == [2, 2]
    eng.store.close()


def test_truong_hop_xau_nhat_khong_cham_tran_recursion():
    """Mọi cảnh tiêu hết lượt viết lại. Trần suy từ MAX_REVISIONS, không đoán."""
    llm = KichBan(writer={i: [LEAK, LEAK] for i in range(6)})
    eng, out = _chay(llm)
    assert not out.get("escalated"), out.get("escalation_reason")
    assert _vai(llm)["writer"] == 6 * (MAX_REVISIONS + 1)
    assert recursion_limit_for(6) >= 6 * (2 * (MAX_REVISIONS + 1) + 2)
    eng.store.close()


def test_router_uu_tien_co_escalated():
    assert after_audit({"escalated": True, "max_severity": "note"}) == "escalate"


# ═══════════════ AUDITOR LLM ═══════════════

def test_loi_giai_thich_cua_auditor_llm_KHONG_vao_prompt_writer():
    """Auditor đọc hợp đồng đầy đủ, gồm bí mật NPC. Đưa `message` của nó cho
    Writer là tự xuyên thủng POV Firewall."""
    finding = json.dumps({"findings": [{
        "severity": "major", "check": "hidden_secret_exposed",
        "message": "Cảnh để lộ BI_MAT_CUA_SERENA", "evidence": EVID}]},
        ensure_ascii=False)
    llm = KichBan(writer={0: [EVID]}, auditor={0: [finding]})
    eng, out = _chay(llm)
    lan_hai = _prompts(llm, "writer")[1]
    assert "tấm thẻ kim loại" in lan_hai
    assert LLM_CHECKS["pov_knowledge"] in lan_hai
    assert "BI_MAT_CUA_SERENA" not in lan_hai
    eng.store.close()


def test_auditor_llm_hong_khong_dung_chuong():
    llm = KichBan(auditor={0: [RuntimeError("hết quota")]})
    eng, out = _chay(llm)
    assert not out.get("escalated")
    assert out["audit_log"][0]["issues"][0]["reason"] == "llm_error"
    eng.store.close()


def test_tat_tang_llm_thi_khong_goi_auditor():
    llm = KichBan()
    eng, out = _chay(llm, llm_audit=False)
    assert "auditor" not in _vai(llm) and not out.get("escalated")
    eng.store.close()


PROSE = "Kaelen đặt tấm thẻ kim loại lên mặt bàn gỗ. Vhal không nhìn lên."


def test_finding_khong_trich_duoc_nguyen_van_bi_loai():
    raw = json.dumps({"findings": [
        {"severity": "major", "check": "subtext", "message": "x",
         "evidence": "một câu không hề có trong văn bản"},
        {"severity": "major", "check": "subtext", "message": "y"}]})
    fs, issues = parse_llm_findings(raw, PROSE)
    assert fs == []
    assert [i["reason"] for i in issues] == ["span_not_found", "no_evidence"]


def test_chi_pov_knowledge_duoc_la_blocker():
    raw = json.dumps({"findings": [
        {"severity": "blocker", "check": "subtext", "message": "", "evidence": EVID},
        {"severity": "critical", "check": "pov", "message": "", "evidence": EVID}]})
    fs, issues = parse_llm_findings(raw, PROSE)
    assert [(f["check"], f["severity"]) for f in fs] == [
        ("subtext", "major"), ("pov_knowledge", "blocker")]
    assert issues[0]["reason"] == "blocker_capped"


def test_json_hong_hoac_sai_hinh_khong_nem_loi():
    assert parse_llm_findings("không phải JSON", PROSE)[1][0]["reason"] == "unparseable"
    fs, _ = parse_llm_findings(json.dumps([{"severity": "low", "check": "?",
                                            "message": "", "evidence": EVID}]), PROSE)
    assert fs[0]["severity"] == "minor" and fs[0]["check"] == "other"


def test_checklist_dua_bi_mat_npc_cho_auditor_khong_dua_cua_pov():
    c = {"pov_character": "CHAR_KAELEN", "pov_character_name": "Kaelen",
         "pov_knowledge_boundary": ["Mã phê duyệt"],
         "active_characters": [
             {"id": "CHAR_KAELEN", "name": "Kaelen", "hidden_action": "POV_TU_BIET",
              "must_not_reveal": []},
             {"id": "CHAR_SERENA", "name": "Serena", "hidden_action": "giấu mã",
              "must_not_reveal": ["đã ký lệnh"]}],
         "scene_must_change": "Kaelen mất đường về", "plant_directives": []}
    t = audit_checklist(c)
    assert "giấu mã" in t and "đã ký lệnh" in t and "Mã phê duyệt" in t
    assert "POV_TU_BIET" not in t and "plant_visibility" not in t


# ═══════════════ QUYẾT ĐỊNH THUẦN ═══════════════

def test_routing_severity_bo_qua_chi_so_nhip():
    fs = [{"check": "triclause_template", "severity": "major"},
          {"check": "length", "severity": "minor"}]
    assert routing_severity(fs) == "minor"
    assert writer_feedback(fs) == ""
    assert routing_severity([]) == "note"


def test_polish_chi_nhan_ghi_chu_sua_duoc_bang_cau_chu():
    fs = [{"check": "length", "severity": "minor", "message": "ngắn"},
          {"check": "cliche", "severity": "minor", "message": "sáo ngữ: “nín thở”"},
          {"check": "subtext", "severity": "minor", "message": "m", "source": "llm"},
          {"check": "flat_run", "severity": "minor", "message": "6 câu", "evidence": "Gió."}]
    assert polish_notes(fs) == ["sáo ngữ: “nín thở”", "6 câu — ví dụ: “Gió.”"]


@pytest.mark.parametrize("draft, polished, ly_do", [
    ("Kaelen đứng dậy rồi bước ra khỏi phòng chờ lạnh.",
     "Kaelen đứng dậy, bước ra khỏi phòng chờ lạnh.", None),
    ("Kaelen đứng dậy rồi bước ra khỏi phòng chờ lạnh.",
     "Anh đứng dậy rồi bước ra khỏi phòng chờ lạnh.", "tên riêng"),
    ("Cần trục kêu ba tiếng rồi im, bụi quặng lắng xuống vai áo 12 người.",
     "Cần trục kêu ba tiếng rồi im, bụi quặng lắng xuống vai áo 13 người.", "con số"),
    ("Kaelen đứng dậy rồi bước ra khỏi phòng chờ lạnh.",
     "Serena mở cửa tàu, gió biển thổi mạnh vào khoang hàng tối om.", "giống"),
    ("Kaelen đứng dậy.", "   ", "rỗng"),
])
def test_content_drifted(draft, polished, ly_do):
    r = content_drifted(draft, polished, names=["Kaelen", "Serena"])
    assert (r is None) if ly_do is None else (ly_do in r)


def test_lam_sach_dau_ra_polish():
    assert clean_polish_output("```\nVăn xuôi.\n```") == "Văn xuôi."
    assert clean_polish_output("## VĂN XUÔI\nVăn xuôi.") == "Văn xuôi."
    assert clean_polish_output(
        "Văn xuôi.\n\nChỉ xuất văn xuôi đã trau chuốt, không giải thích.") == "Văn xuôi."


def test_mang_json_tran_khong_bi_cat_con_mot_phan_tu():
    raw = json.dumps([{"severity": "minor", "check": "subtext", "message": "a",
                       "evidence": EVID},
                      {"severity": "minor", "check": "subtext", "message": "b",
                       "evidence": "Vhal không nhìn lên."}], ensure_ascii=False)
    fs, _ = parse_llm_findings(raw, PROSE)
    assert [f["message"] for f in fs] == ["a", "b"]


def test_new_serious_chi_dem_loi_moi():
    truoc = [{"check": "voice", "severity": "major"}]
    sau = [{"check": "voice", "severity": "major"},
           {"check": "pov_leak", "severity": "blocker"},
           {"check": "flat_run", "severity": "minor"}]
    assert new_serious(truoc, sau) == ["pov_leak:blocker"]


# ═══════════════ POLISH TRONG ĐỒ THỊ ═══════════════

def test_polish_doi_con_so_bi_tu_choi_giu_ban_nhap():
    llm = KichBan(writer={0: [TRI]}, polish=lambda p: p + " Cần trục kêu 47 tiếng.")
    eng, out = _chay(llm)
    s0 = out["scene_outputs"][0]
    assert s0["audit"]["polish"]["accepted"] is False
    assert "con số" in s0["audit"]["polish"]["reason"]
    assert "47" not in s0["prose"]
    eng.store.close()


def test_polish_sinh_ro_ri_pov_bi_tu_choi():
    """§9.2 đưa bản Polish thẳng tới scene_boundary. Không kiểm lại thì Polish
    là lối vòng qua toàn bộ vòng kiểm toán."""
    llm = KichBan(writer={0: [TRI]},
                  polish=lambda p: p + "\n\nTrong lòng: một thứ gì đó vỡ ra.")
    eng, out = _chay(llm)
    pr = out["scene_outputs"][0]["audit"]["polish"]
    assert pr["accepted"] is False and "pov_leak" in pr["reason"]
    eng.store.close()


def test_polish_llm_loi_giu_ban_nhap_chuong_van_xong():
    def hong(_p):
        raise RuntimeError("503")
    llm = KichBan(writer={0: [TRI]}, polish=hong)
    eng, out = _chay(llm)
    assert not out.get("escalated")
    assert "lỗi LLM" in out["scene_outputs"][0]["audit"]["polish"]["reason"]
    eng.store.close()


def test_polish_duoc_nhan_khi_chi_sua_cau_chu():
    llm = KichBan(writer={0: [TRI]},
                  polish=lambda p: p.replace("vẫn cúi đầu ghi sổ", "cúi đầu ghi sổ"))
    eng, out = _chay(llm)
    s0 = out["scene_outputs"][0]
    assert s0["audit"]["polish"]["accepted"] is True
    assert "Vhal cúi đầu ghi sổ" in s0["prose"]
    eng.store.close()


# ═══════════════ PROMPT & CLI ═══════════════

def test_writer_chi_nhan_rang_buoc_menh_de_phu_khong_nhan_nguong_so():
    """§10.3.2: ngoại lệ DUY NHẤT được vào prompt Writer."""
    assert '"Khi", "Trong khi" hoặc' in WRITER_TMPL
    assert "σ" not in WRITER_TMPL and "8.5" not in WRITER_TMPL


def test_cli_write_escalate_giu_ban_nhap_bi_chan(tmp_path, monkeypatch, capsys):
    import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_llm",
                        lambda _name, *a, **kw: KichBan(writer={1: [LEAK] * 5}))
    db = str(tmp_path / "t.db")
    assert cli.main(["write", "--chapter", "1", "--db", db]) == 1

    md = (tmp_path / "output" / "chapters" / "ch001.escalated.md").read_text(encoding="utf-8")
    assert "## Cảnh 0" in md and "BỊ CHẶN sau 2 lần viết lại" in md and LEAK in md
    rep = json.loads((tmp_path / "output" / "reports" / "ch001.escalated.json")
                     .read_text(encoding="utf-8"))
    assert rep["rolled_back"] is True and rep["scenes_done"] == ["CH001_S00"]
    err = capsys.readouterr().err
    assert "pov_leak" in err and "--force" not in err


# ═══════════════ BÀI HỌC TỪ LƯỢT GEMINI ĐẦU TIÊN ═══════════════

def test_audit_log_ghi_NOI_DUNG_loi_nghiem_trong():
    """Lượt Gemini Chương 2: CH002_S04 bị viết lại vì `pov_leak`, nhưng log chỉ
    có tên loại — không biết là rò rỉ thật hay báo động giả."""
    llm = KichBan(writer={0: [LEAK]})
    eng, out = _chay(llm)
    first = out["audit_log"][0]
    assert first["serious"][0]["check"] == "pov_leak"
    assert "Sự thật là Serena" in first["serious"][0]["message"]
    assert out["audit_log"][1]["serious"] == []
    eng.store.close()


def test_polish_bi_tu_choi_van_giu_ban_da_viet():
    llm = KichBan(writer={0: [TRI]}, polish=lambda p: p + " Cần trục kêu 47 tiếng.")
    eng, out = _chay(llm)
    pr = out["scene_outputs"][0]["audit"]["polish"]
    assert pr["accepted"] is False and "47 tiếng" in pr["rejected_text"]
    eng.store.close()


def test_writer_khong_duoc_rai_lexicon_dac_trung_vao_loi_ke():
    """Lượt Gemini Chương 2, CH002_S03 (POV Serena): lời kể nhồi "Theo thẩm
    quyền…", "Hồ sơ cho thấy…" — từ đặc trưng của THOẠI. Prompt cũ đòi chúng
    "phải xuất hiện" mà không nói ở đâu, nên lần viết lại không sửa được."""
    assert "mang cách nói ấy vào lời kể" in WRITER_TMPL
    # Và nay còn đi xa hơn: Writer không còn nhìn thấy danh sách khẩu ngữ nào
    # để mà rải. Xem `test_writer_khong_con_nhan_thuoc_do_giong`.
    assert "signature_lexicon" not in WRITER_TMPL


def _ch(cid, name):
    return {"id": cid, "name": name, "voice_reminder": {
        "register": "formal", "syntactic_tic": "",
        "voice_exemplars": ["Tôi hiểu, tuy nhiên hồ sơ đã được lưu."],
        "forbidden_lexicon": ["vinh quang"]}}


def test_canh_mot_minh_khong_dua_lexicon_dac_trung_cho_writer():
    """Lượt Gemini thứ hai vẫn rải lexicon vào lời kể (60 → 66): 5/6 cảnh Chương 2
    chỉ có một nhân vật, nên "chỉ dùng trong thoại" là ràng buộc không thể thoả."""
    from novel_engine.graph.nodes import _characters_brief
    solo = _characters_brief([_ch("CHAR_SERENA", "Serena")])
    assert "tuy nhiên hồ sơ" not in solo and "MỘT MÌNH" in solo
    assert "vinh quang" in solo                     # từ CẤM vẫn giữ

    doi = _characters_brief([_ch("CHAR_SERENA", "Serena"), _ch("CHAR_KAELEN", "Kaelen")])
    assert "tuy nhiên hồ sơ" in doi and "không phải câu để chép lại" in doi


def test_writer_khong_con_nhan_thuoc_do_giong():
    """`signature_lexicon` là THƯỚC ĐO, không phải nguyên liệu.

    M3 chấm giọng bằng phân bố của chính danh sách ấy, và ngân sách tật ngôn
    ngữ đếm theo nó. Suốt Arc 1 ta vừa đo bằng thước vừa đưa thước cho người
    bị đo — rồi ngạc nhiên vì Serena dùng cả năm cụm, mỗi cụm ba lần. Nay
    Writer chỉ nhận CÂU MẪU: mẫu dạy cách nói, danh sách dạy dùng lại từ nào.
    """
    from novel_engine.graph.nodes import _characters_brief

    c = _ch("CHAR_SERENA", "Serena")
    c["voice_reminder"]["signature_lexicon"] = ["theo thẩm quyền", "hồ sơ cho thấy"]
    ra = _characters_brief([c, _ch("CHAR_KAELEN", "Kaelen")])
    assert "theo thẩm quyền" not in ra
