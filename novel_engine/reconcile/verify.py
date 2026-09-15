"""Xác minh span và đối chiếu kế hoạch (§10.5.2, §10.5.3).

`extractor_node` là cầu nối DUY NHẤT đưa văn xuôi trở lại Canon, và cũng là
điểm nghẽn nguy hiểm nhất của kiến trúc event-sourced: lỗi ở đây không crash,
không cảnh báo — nó chỉ âm thầm ghi sai một sự thật, và hai mươi chương sau
mới thấy hậu quả mà không truy được nguồn.

NT-5: mỗi mệnh đề ghi vào canon phải có bằng chứng khớp được BẰNG CHUỖI.
Không có `span` xác minh thì không có thay đổi trạng thái, bất kể LLM khai gì.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

from novel_engine.canon.models import StateDelta
from novel_engine.reconcile.lenient import parse_delta_lenient

MIN_SPAN_LEN = 12
FUZZY_RATIO = 0.90


def _norm(s: str) -> str:
    """Chuẩn hoá trước khi so khớp.

    NFC là bắt buộc với tiếng Việt: "ế" viết được bằng một code point (U+1EBF)
    hoặc bằng "e" + hai dấu tổ hợp. Hai chuỗi hiện ra giống hệt nhau trên màn
    hình nhưng `in` trả False — đúng loại lỗi âm thầm mà NT-5 sinh ra để chặn,
    và nó sẽ loại đúng những span mà model chép lại CHÍNH XÁC nhất.
    """
    s = unicodedata.normalize("NFC", s or "").lower()
    s = (s.replace("“", '"').replace("”", '"')
          .replace("‘", "'").replace("’", "'")
          .replace("—", "-").replace("–", "-"))
    return re.sub(r"\s+", " ", s).strip()


def _word_starts(s: str) -> list[int]:
    """Vị trí bắt đầu của mỗi từ. Đây là tập ứng viên để trượt cửa sổ."""
    out = [0]
    out += [i + 1 for i, ch in enumerate(s) if ch == " " and i + 1 < len(s)]
    return out


def _fuzzy_present(needle: str, haystack: str, ratio: float = FUZZY_RATIO) -> bool:
    """Dự phòng cho trường hợp Polish Agent đã sửa nhẹ câu văn sau khi
    Extractor trích (ở GĐ2, Polish chạy SAU Writer nhưng TRƯỚC Extractor).

    ═══ VÌ SAO KHÔNG DÙNG `step = n // 4` NHƯ §10.5.2 ═══════════════════════

    Bước nhảy cố định CÓ THỂ NHẢY QUA đúng vị trí khớp, và khi đó nó hỏng IM
    LẶNG — loại một mệnh đề hoàn toàn hợp lệ. Đo thực trên câu 38 ký tự, chỉ
    khác một dấu phẩy:

        vị trí đúng  i=104  →  ratio 0.974   ✓ vượt ngưỡng
        step ghé     i= 99  →  ratio 0.868   ✗
        step ghé     i=108  →  ratio 0.868   ✗

    `step = 38//4 = 9` không bao giờ ghé 104. Xác suất trúng khoảng 1/9, nên
    lớp dự phòng mà §10.5.2 tuyên bố là có thực ra gần như không hoạt động.

    Thay bằng NEO VÀO ĐẦU TỪ: một bản sửa văn phong hầu như luôn giữ nguyên
    ranh giới từ, nên vị trí khớp thật gần như chắc chắn rơi vào đầu một từ.
    Số ứng viên cũng ÍT hơn bước nhảy dày (≈1 trên mỗi 6 ký tự tiếng Việt),
    nên vừa chính xác hơn vừa không chậm hơn.
    """
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq1(needle)
    for i in _word_starts(haystack):
        if i + n > len(haystack):
            break
        sm.set_seq2(haystack[i:i + n])
        # `real_quick_ratio`/`quick_ratio` là chặn trên rẻ tiền — bỏ sớm phần
        # lớn ứng viên mà không phải chạy thuật toán khớp đầy đủ.
        if (sm.real_quick_ratio() >= ratio and sm.quick_ratio() >= ratio
                and sm.ratio() >= ratio):
            return True
    return False


def _found(span: str, hay: str, min_len: int) -> tuple[bool, str]:
    nd = _norm(span)
    if len(nd) < min_len:
        # Ngưỡng này loại các span rỗng nghĩa kiểu "anh nói" vốn khớp được ở
        # mọi nơi — chúng "xác minh" thành công mà không chứng minh gì cả.
        return False, "span_too_short"
    if nd in hay or _fuzzy_present(nd, hay):
        return True, ""
    return False, "span_not_found"


def verify_spans(delta: StateDelta, prose: str,
                 min_len: int = MIN_SPAN_LEN) -> tuple[StateDelta, list[dict]]:
    """Lượt 3 — chi phí 0, hiệu quả cao nhất.

    Mọi `Assertion` mang `span` là trích dẫn nguyên văn, nên kiểm chứng thành
    khớp chuỗi thuần tuý: câu trích không có thật thì mệnh đề bị loại. Không
    cần LLM, không cần phán đoán.

    Đây là thứ chống **ảo giác xác nhận** — lỗi mà chính lượt 1 tạo ra. Khi
    được hỏi "manh mối CLUE_042 đã cài chưa?", model có xu hướng mạnh trả lời
    *rồi* và bịa một span nghe hợp lý.
    """
    hay = _norm(prose)
    kept: list = []
    rejected: list[dict] = []

    for a in delta.assertions:
        ok, why = _found(a.span, hay, min_len)
        if ok:
            kept.append(a)
        else:
            rejected.append({"reason": why, "subject": a.subject,
                             "predicate": a.predicate, "span": a.span})
    delta.assertions = kept

    # Bằng chứng cài manh mối đi qua ĐÚNG cơ chế xác minh đó. `verified` chỉ
    # được đặt Ở ĐÂY — không bao giờ do LLM tự khai (NT-5).
    for pe in delta.plant_evidence:
        ok, why = _found(pe.span, hay, min_len)
        pe.verified = ok
        if not ok:
            rejected.append({"reason": f"plant_{why}", "clue_id": pe.clue_id,
                             "span": pe.span})
    return delta, rejected


def plan_coverage(contracts: list[dict], delta: StateDelta,
                  known_clues: set[str] | None = None) -> dict:
    """Đối chiếu những gì contract HỨA với những gì đã XÁC MINH được.

    B1/NT-8: bản trước đối chiếu `d["clue_id"]` với
    `{a.subject for a in delta.assertions}`. Hai không gian định danh này
    không bao giờ giao nhau — `subject` là thực thể trong câu văn ("con dấu",
    "Serena"), `clue_id` là mã hệ thống. Phép so luôn cho False: mọi manh mối
    bị tính là thất bại và mọi `clue_transitions` bị xoá sạch. Nay dùng
    `delta.plant_evidence`, một kênh riêng có cùng không gian định danh.
    """
    promised: list[str] = []
    plan_by_clue: dict[str, dict] = {}
    fulfilled: list[str] = []
    missed: list[dict] = []
    verified = {e.clue_id for e in delta.plant_evidence if e.verified}

    for c in contracts:
        for d in c.get("plant_directives", []):
            cid = d.get("clue_id")
            if not cid:
                continue
            promised.append(cid)
            plan_by_clue.setdefault(cid, d)
            if cid in verified:
                fulfilled.append(cid)
            else:
                missed.append({"clue_id": cid, "scene": c.get("scene_id", "?"),
                               "mode": d.get("mode", "?")})

    # ── Bằng chứng cho manh mối KHÔNG được hứa ─────────────────────────
    # Lượt chạy Gemini thật: chương không hứa manh mối nào, vậy mà model xuất 6
    # mục `plant_evidence` với mã tự đặt kiểu `CH001_S00_01`. Span có thật nên
    # 5/6 qua lượt 3 với `verified=True`. `reconcile_node` §11 gọi
    # `clues.touch()` cho MỌI bằng chứng đã xác minh — tức chạm vào những manh
    # mối không tồn tại.
    #
    # Không loại hết thứ không được hứa: Writer có thể tình cờ cài một manh mối
    # CÓ THẬT mà Scheduler không giao, và đó là bằng chứng hợp lệ
    # (`incidental`). Chỉ loại mã không phải manh mối nào trong canon. Không
    # truyền `known_clues` thì chế độ nghiêm: mọi bằng chứng ngoài lời hứa đều bị
    # loại.
    promised_set = set(promised)
    incidental: list[dict] = []
    phantom: list[dict] = []
    giu = []
    for e in delta.plant_evidence:
        if e.clue_id in promised_set:
            giu.append(e)
        elif known_clues is not None and e.clue_id in known_clues:
            giu.append(e)
            incidental.append({"clue_id": e.clue_id, "scene_id": e.scene_id,
                               "verified": e.verified})
        else:
            phantom.append({"clue_id": e.clue_id, "scene_id": e.scene_id,
                            "verified": e.verified})
    delta.plant_evidence = giu

    # Ghi phần kế hoạch mà commit cần, NGAY BÂY GIỜ — contract không được
    # lưu, nên lúc `cli.py commit` chạy thì nó đã mất (NT-13: code đặt).
    for cid, d in plan_by_clue.items():
        if d.get("intensity") is not None:
            delta.plant_intensity[cid] = float(d["intensity"])
        if d.get("mode"):
            delta.plant_modes[cid] = str(d["mode"])
    for e in delta.plant_evidence:
        d = plan_by_clue.get(e.clue_id)
        if d and d.get("surface_form") and not e.surface_form:
            e.surface_form = d["surface_form"]
    verified = {e.clue_id for e in delta.plant_evidence if e.verified}

    # ── QUY TẮC CỨNG (NT-5) ────────────────────────────────────────────
    # Manh mối không có bằng chứng XÁC MINH thì KHÔNG được chuyển trạng thái,
    # dù Extractor có khai gì đi nữa.
    #
    # Bản §10.5.3 chỉ xoá transition của manh mối ĐÃ HỨA mà trượt. Nhưng một
    # transition mà model tự bịa cho manh mối CHƯA TỪNG được hứa thì không nằm
    # trong `missed_ids`, nên nó đi thẳng vào canon — đúng con đường ô nhiễm
    # mà chính mục này tuyên bố là đã đóng. Ở đây gác theo bằng chứng, không
    # theo lời hứa: có `verified` thì qua, không thì loại.
    bo_qua: list[dict] = []
    for cid in list(delta.clue_transitions):
        if cid not in verified:
            bo_qua.append({"clue_id": cid,
                           "status": delta.clue_transitions[cid].value,
                           "promised": cid in promised})
            del delta.clue_transitions[cid]

    findings = [{"severity": "major", "check": "unfulfilled_plant",
                 "message": (f"{m['clue_id']} được lên kế hoạch cài ở "
                             f"{m['scene']} nhưng không tìm thấy trong văn bản")}
                for m in missed]
    findings += [{"severity": "major", "check": "unverified_transition",
                  "message": (f"{b['clue_id']} được khai chuyển sang "
                              f"'{b['status']}' mà không có bằng chứng xác "
                              f"minh — đã loại")}
                 for b in bo_qua if not b["promised"]]

    findings += [{"severity": "note", "check": "phantom_plant_evidence",
                  "message": (f"bằng chứng cài '{p['clue_id']}' ở {p['scene_id']} "
                              f"không trỏ tới manh mối nào trong canon — đã loại")}
                 for p in phantom]

    return {
        "plan_fulfillment_rate": round(len(fulfilled) / max(len(promised), 1), 3),
        "promised_plants": promised,
        "fulfilled_plants": fulfilled,
        "missed_plants": missed,
        "incidental_plants": incidental,
        "phantom_evidence": phantom,
        "dropped_transitions": bo_qua,
        "findings": findings,
    }


def merge_extractions(audited_json: str, emergent_json: str, chapter: int,
                      repair_llm=None,
                      dropped: list[dict] | None = None) -> StateDelta:
    """Gộp hai lượt. Lượt 1 (có hệ quy chiếu) LUÔN thắng khi trùng.

    B2: bản trước chỉ gộp `assertions` và `new_entities`. Mọi `new_relations`,
    `retracted_relations`, `clue_transitions` và `relationship_updates` mà
    lượt 2 tìm được đều bị vứt IM LẶNG — nghĩa là một mối thù hay một liên
    minh phát sinh ngoài kế hoạch không bao giờ tới được `reconcile_node`.

    Parse KHOAN DUNG từng mục (`lenient.py`): một thực thể mang `kind` lạ không
    được làm mất cả chương. Mục bị ép kiểu hoặc bị loại được nối vào `dropped`.
    """
    a, issues_a = parse_delta_lenient(audited_json, repair_llm, source="luot_1")
    e, issues_e = parse_delta_lenient(emergent_json, repair_llm, source="luot_2")
    if dropped is not None:
        dropped.extend(issues_a + issues_e)

    seen = {(x.subject, x.predicate) for x in a.assertions}
    a.assertions += [x for x in e.assertions
                     if (x.subject, x.predicate) not in seen]

    known = {x.id for x in a.new_entities}
    a.new_entities += [x for x in e.new_entities if x.id not in known]

    rel_seen = {(r.src, r.dst, r.type) for r in a.new_relations}
    a.new_relations += [r for r in e.new_relations
                        if (r.src, r.dst, r.type) not in rel_seen]

    ret_seen = {(r.src, r.dst, r.type) for r in a.retracted_relations}
    a.retracted_relations += [r for r in e.retracted_relations
                              if (r.src, r.dst, r.type) not in ret_seen]

    # `setdefault`, KHÔNG `update`: `update` để lượt 2 ghi đè lượt 1, đảo
    # ngược quy tắc ưu tiên đã công bố. Lượt 2 không thấy contract nên phán
    # đoán chuyển trạng thái manh mối của nó kém tin cậy hơn hẳn.
    for k, v in e.clue_transitions.items():
        a.clue_transitions.setdefault(k, v)

    # Quan hệ phải GỘP THEO CẶP, không nối đuôi. Nối đuôi khiến
    # `apply_scene_effects` (§7.3) chạy hai lần trên cùng một cặp và cộng dồn
    # intimacy — biểu hiện ra ngoài là M5 báo nhảy cóc giai đoạn mà nhìn vào
    # guard thì guard đúng.
    by_pair: dict[tuple[str, str], object] = {
        (u.a, u.b): u for u in a.relationship_updates}
    for u in e.relationship_updates:
        if (u.a, u.b) not in by_pair and (u.b, u.a) not in by_pair:
            by_pair[(u.a, u.b)] = u
    a.relationship_updates = list(by_pair.values())

    # `plant_evidence` chỉ lượt 1 sinh ra — lượt 2 không biết kế hoạch là gì.
    return a.stamp(chapter)


def _scene_index(scene_id: str) -> int | None:
    try:
        return int(scene_id.rsplit("_S", 1)[1])
    except (IndexError, ValueError):
        return None


def assign_scenes(delta: StateDelta, scenes: list[dict],
                  min_len: int = MIN_SPAN_LEN) -> list[dict]:
    """Gán cảnh cho mệnh đề và bằng chứng theo VỊ TRÍ span trong văn xuôi.

    NT-13: chỉ số cảnh là metadata của hệ thống, không phải thứ model khai.

    Lượt Gemini thật ở Chương 2: `CHAR_KAELEN.knows_secret` ("Chính tay Kaelen
    đã khóa chặt van điều áp số bốn...") được model khai `scene=0` — cảnh hiện
    tại ở tick 20.028. Hai hậu quả, cả hai đều âm thầm:
    - sự thật VĨNH VIỄN được chứng thực sai mốc, làm hỏng cửa sổ khởi phát mà
      Forward-Reachability Audit (§3.6.4) dựa vào;
    - `flashback_findings` lọc theo `scene`, nên nếu span thuộc cảnh hồi ức thì
      mệnh đề thoát khỏi kiểm tra hồi ức hoàn toàn.

    Span xuất hiện ở NHIỀU cảnh (câu lặp lại) thì không đoán — giữ số model
    khai và ghi chú `scene_ambiguous`. Không tìm thấy ở cảnh nào (thường là
    span vắt qua ranh giới hai cảnh) thì giữ nguyên, không ghi chú.
    """
    hays = [(i, s["scene_id"], _norm(s.get("prose", "")))
            for s in scenes
            if (i := _scene_index(s.get("scene_id", ""))) is not None]

    def locate(span: str):
        nd = _norm(span)
        if len(nd) < min_len:
            return None
        exact = [h for h in hays if nd in h[2]]
        if len(exact) == 1:
            return exact[0]
        if exact:
            return "ambiguous"
        fuzzy = [h for h in hays if _fuzzy_present(nd, h[2])]
        if len(fuzzy) == 1:
            return fuzzy[0]
        return "ambiguous" if fuzzy else None

    notes: list[dict] = []
    for a in delta.assertions:
        hit = locate(a.span)
        if hit == "ambiguous":
            notes.append({"reason": "scene_ambiguous", "subject": a.subject,
                          "predicate": a.predicate, "span": a.span[:120]})
        elif hit is not None and a.scene != hit[0]:
            notes.append({"reason": "scene_reassigned", "subject": a.subject,
                          "predicate": a.predicate, "from": a.scene,
                          "to": hit[0], "span": a.span[:120]})
            a.scene = hit[0]
    for pe in delta.plant_evidence:
        hit = locate(pe.span)
        if isinstance(hit, tuple) and pe.scene_id != hit[1]:
            notes.append({"reason": "plant_scene_reassigned",
                          "clue_id": pe.clue_id, "from": pe.scene_id,
                          "to": hit[1]})
            pe.scene_id = hit[1]
    return notes
