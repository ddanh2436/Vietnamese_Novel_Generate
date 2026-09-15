"""Drift Reconciliation — phân loại delta (§11).

Writer viết ra một chi tiết hay nhưng không có trong kế hoạch. Ép viết lại thì
mất chi tiết hay; thả nổi thì cấu trúc sụp. Giải pháp của §11: phân mỗi mục
trong delta vào ĐÚNG MỘT trong ba hạng rồi xử lý khác nhau.

    contradiction  phá canon do tác giả chốt       → escalate, dừng ghi
    improvement    đổi kế hoạch các chương sau     → ghi + replan (GĐ3)
    enrichment     bổ sung, không đụng kế hoạch    → ghi

═══ HẠNG THỨ TƯ: CÁCH LY ════════════════════════════════════════════════

Lượt chạy Gemini thật đầu tiên (Chương 1–2) cho thấy ba hạng không đủ. Delta
thật chứa những mục KHÔNG phải mâu thuẫn, nhưng ghi vào canon thì hỏng:

- 19 trên 20 thực thể mới là ĐỊA ĐIỂM CON (`LOC_PIER_3`, `LOC_BAY_9`...).
  Ghi vào graph thì chúng thành node cô lập trong route graph, `travel_ticks`
  trả `None`, và mọi luật liên tục đổ vỡ. Địa lý là của tác giả: Extractor
  không thêm được tuyến đường, nên không được thêm địa điểm.
- `CHAR_KAELEN at LOC_REACTOR_3` với span "...tính từ thời điểm niêm phong Lò
  Phản ứng Số 3." Span CÓ THẬT nên qua được lượt 3, nhưng nó chỉ NHẮC tới Lò 3.
  Frame nói Kaelen ở Cảng Quặng.
- `CHAR_KAELEN bước qua cánh cửa thép` — lời kể sự kiện, không phải khoá sự
  thật. Ghi vào `_truth[(subject, predicate)]` là tạo khoá rác.

Những mục này đi vào `quarantine` kèm lý do, KHÔNG vào `classification`.
`StateDelta.classification` vẫn giữ đúng ba giá trị; `reconcile_node` (GĐ3)
chỉ commit thứ có trong `classification`, nên thứ bị cách ly không bao giờ
lọt vào canon — nhưng cũng không biến mất: tác giả thấy nó ở CP-2.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from novel_engine.canon.flashback import PERMANENT_PREDICATES
from novel_engine.canon.models import (
    Assertion, Entity, Relation, StateDelta, item_key,
)
from novel_engine.canon.timeline import ContinuityFrame
from novel_engine.canon.vocabulary import check_assertion

__all__ = [
    "LOCATION_PREDICATES", "affects_downstream_plan", "classify_delta",
    "conflicts_with_locked_canon", "is_objective", "item_key", "retract_key",
]

# Vị từ chỉ VỊ TRÍ. Vị trí nhân vật do `ContinuityFrame` sở hữu (NT-12) — nó
# thay đổi theo thời gian, nên không bao giờ được ghi thành sự thật tĩnh
# `_truth[(subject, predicate)]`, vốn chỉ giữ MỘT giá trị và sẽ ghi đè lịch sử.
LOCATION_PREDICATES = frozenset({
    "at", "is_at", "located_at", "located_in", "location", "position",
    "ở", "tại", "đang_ở",
})

# Khoá sự thật hợp lệ: snake_case ASCII. "bước qua" là lời kể sự kiện.
_CANON_PREDICATE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_ENTITY_ID = re.compile(r"\b(?:CHAR|FACT|LOC|OBJ|EV|DOC|CLUE)_[A-Z0-9_]+\b")


def _nfc_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


def _pred(p: str) -> str:
    return re.sub(r"\s+", "_", _nfc_lower(p).strip())


def is_objective(a: Assertion) -> bool:
    """Cổng epistemic — NT-11: MỘT hàm, mọi nơi cùng gọi.

    C5: §10.2 lọc `epistemic` đúng, §11 thì quên — hai hàm, một hàm kiểm, một
    hàm không, và kết quả là KHÔNG NHÂN VẬT NÀO ĐƯỢC PHÉP NÓI DỐI. Kiểm tra
    mâu thuẫn ở §10.2 (GĐ2) phải gọi đúng hàm này, không viết lại điều kiện.
    """
    return a.epistemic == "objective"


def retract_key(r: Relation) -> str:
    """Khoá cho quan hệ bị THU HỒI. Không dùng `item_key`: một delta vừa thêm
    vừa thu hồi cùng một bộ ba sẽ sinh hai khoá trùng nhau, và
    `StateDelta.item()` — vốn tra `new_relations` trước — trả nhầm phần tử."""
    return f"X:{r.src}|{r.type}|{r.dst}"


# ═════════════════════ hai hàm §11 gọi mà chưa từng định nghĩa ═════════════════════

def conflicts_with_locked_canon(item: Entity | Relation | Assertion, graph) -> bool:
    """Mục này có phá canon mà tác giả đã chốt không.

    §11 gọi hàm này nhưng tài liệu không định nghĩa nó ở đâu cả (NT-12).

    - Assertion: chỉ mệnh đề KHÁCH QUAN mới có thể mâu thuẫn. Lời nói và niềm
      tin không bao giờ phá canon — chúng là sự kiện tâm lý (C5).
    - Relation: loại trừ lẫn nhau (BETRAYED/PROTECTS...) với một quan hệ còn
      hiệu lực DO TÁC GIẢ ĐẶT. Loại trừ với quan hệ do Extractor ghi ở chương
      trước thì không phải mâu thuẫn canon — đó là diễn biến.
    """
    if isinstance(item, Assertion):
        return is_objective(item) and graph.conflicts_with_truth(item)
    if isinstance(item, Relation):
        if not graph.exists(item.src):
            return False
        return any(c.get("provenance") == "author"
                   for c in graph.conflicting_relations(item.src, item.dst,
                                                        item.type))
    return False


def affects_downstream_plan(item: Entity | Relation | Assertion, planner,
                            chapter: int) -> bool:
    """Mục này có đụng tới dàn ý của các chương SAU không.

    §11 gọi hàm này nhưng tài liệu không định nghĩa nó. Định nghĩa ở đây cố ý
    HẸP, vì hạng `improvement` kích hoạt replan — rộng quá thì mỗi chương đều
    replan, và tác giả sẽ bỏ qua cảnh báo.

    - Thực thể mới: dàn ý chương sau nhắc tới nó (bằng mã hoặc bằng tên).
    - Quan hệ mới: CẢ HAI đầu mút xuất hiện ở chương sau.
    - Mệnh đề: khách quan, vị từ VĨNH VIỄN (`has_scar`, `is_dead`...), và chủ
      thể xuất hiện ở chương sau. Đây là ví dụ "Kaelen có một người em gái"
      của §11: một sự thật vĩnh viễn về nhân vật còn xuất hiện có thể vô hiệu
      hoá kế hoạch đã viết sẵn.
    """
    return bool(downstream_chapters(item, planner, chapter))


def downstream_chapters(item: Entity | Relation | Assertion, planner,
                        chapter: int) -> list[int]:
    """CÁC chương sau mà mục này đụng tới.

    NT-11: `affects_downstream_plan` chỉ hỏi "có không" và gọi hàm này; PlanPatch
    (§11) cần biết "chương nào". Một luật, một hàm — hai bản sao của cùng điều
    kiện sẽ lệch nhau ở lần sửa đầu tiên.

    Quan hệ: cả hai đầu mút phải cùng xuất hiện trong MỘT chương, không phải rải
    rác ở hai chương khác nhau.
    """
    chapters = getattr(planner, "chapters", None) or {}
    out: list[int] = []
    for n in sorted(k for k in chapters if k > chapter):
        raw = json.dumps(chapters[n], ensure_ascii=False)
        ids = set(_ENTITY_ID.findall(raw))
        blob = _nfc_lower(raw)
        if isinstance(item, Entity):
            name = _nfc_lower(item.name).strip()
            hit = item.id in ids or (len(name) >= 3 and name in blob)
        elif isinstance(item, Relation):
            hit = item.src in ids and item.dst in ids
        elif isinstance(item, Assertion):
            hit = (is_objective(item)
                   and _pred(item.predicate) in PERMANENT_PREDICATES
                   and item.subject in ids)
        else:
            hit = False
        if hit:
            out.append(n)
    return out


# ═════════════════════════════ classify_delta ═════════════════════════════

def classify_delta(delta: StateDelta, graph, planner=None,
                   frames: list[ContinuityFrame | dict] | None = None,
                   vocabulary=None) -> dict:
    """Phân loại mọi mục của delta. Đặt `delta.classification`.

    Mỗi mục kết thúc ở ĐÚNG MỘT chỗ: `classification`, `retractions`, hoặc
    `quarantine`. Không mục nào bị bỏ im lặng — thứ bị bỏ im lặng là thứ tác
    giả không bao giờ biết đã từng tồn tại.

    Thứ tự kiểm tra với mệnh đề là QUAN TRỌNG: cổng epistemic đứng ĐẦU. Nếu
    kiểm vị trí trước, "Serena nói Kaelen đang ở cầu tàu số ba" sẽ bị cách ly
    vì `LOC_PIER_3` không tồn tại — tức là nhân vật lại không được phép nói
    dối, chỉ là theo một đường khác (C5).
    """
    ch = delta.chapter
    prefix = f"CH{ch:03d}_"
    frames = [f if isinstance(f, ContinuityFrame)
              else ContinuityFrame.model_validate(f) for f in (frames or [])]
    frames = [f for f in frames if f.scene_id.startswith(prefix)]
    known_locs = set(graph.location_ids())
    index = {e["id"]: e for e in graph.entity_index()}

    cls: dict[str, str] = {}
    retractions: dict[str, str] = {}
    quarantine: list[dict] = []
    irony: list[dict] = []
    findings: list[dict] = []
    seen: set[str] = set()

    def hold(key: str, reason: str, message: str, severity: str = "note") -> None:
        seen.add(key)
        quarantine.append({"key": key, "reason": reason,
                           "severity": severity, "message": message})
        findings.append({"severity": severity, "check": reason,
                         "message": f"{key}: {message}"})

    def decide(key: str, item: Any) -> None:
        seen.add(key)
        if conflicts_with_locked_canon(item, graph):
            cls[key] = "contradiction"
            findings.append({"severity": "blocker", "check": "contradiction",
                             "message": f"{key}: mâu thuẫn với canon do tác giả chốt"})
        elif affects_downstream_plan(item, planner, ch):
            cls[key] = "improvement"
        else:
            cls[key] = "enrichment"

    # ── 1. THỰC THỂ ─────────────────────────────────────────────────────
    accepted_new: dict[str, Entity] = {}
    for e in delta.new_entities:
        k = item_key(e)
        if k in seen:
            continue
        if e.id in index:
            if index[e.id].get("kind") != e.kind:
                seen.add(k)
                cls[k] = "contradiction"
                findings.append({
                    "severity": "blocker", "check": "entity_kind_conflict",
                    "message": (f"{k}: canon ghi '{e.id}' là "
                                f"{index[e.id].get('kind')}, delta ghi {e.kind}")})
            else:
                hold(k, "already_known", f"'{e.id}' đã có trong canon")
            continue
        if e.kind == "location":
            hold(k, "location_without_route",
                 (f"'{e.name}' là địa điểm mới nhưng không có tuyến đường. Địa "
                  f"lý thuộc Unified Route Graph do tác giả sở hữu: quy nó về "
                  f"một địa điểm có sẵn, hoặc khai trong bible/world.yaml kèm "
                  f"tuyến đường"), "major")
            continue
        accepted_new[e.id] = e
        decide(k, e)

    quarantined_new = {e.id for e in delta.new_entities
                       if e.id not in accepted_new and e.id not in index}

    def known(eid: str) -> bool:
        return eid in index or eid in accepted_new

    def names(eid: str) -> list[str]:
        if eid in accepted_new:
            e = accepted_new[eid]
            return [e.id, e.name, *e.aliases]
        return graph.names_of(eid)

    # ── 2. QUAN HỆ ──────────────────────────────────────────────────────
    added = {(r.src, r.type, r.dst) for r in delta.new_relations}
    removed = {(r.src, r.type, r.dst) for r in delta.retracted_relations}

    for r in delta.new_relations:
        k = item_key(r)
        if k in seen:
            continue
        if (r.src, r.type, r.dst) in removed:
            hold(k, "added_and_retracted",
                 "delta vừa thêm vừa thu hồi cùng một quan hệ", "major")
            continue
        if r.type == "LOCATED_IN" and r.src.startswith("CHAR_"):
            hold(k, "owned_by_frames",
                 "vị trí nhân vật do ContinuityFrame sở hữu (NT-12)")
            continue
        missing = [x for x in (r.src, r.dst) if not known(x)]
        if missing:
            hold(k, "dangling_endpoint",
                 f"đầu mút không có trong canon: {missing}", "major")
            continue
        decide(k, r)

    # F7: `classify_delta` ở §11 duyệt new_entities + new_relations + assertions
    # và bỏ sót `retracted_relations` hoàn toàn.
    for r in delta.retracted_relations:
        xk = retract_key(r)
        if xk in seen:
            continue
        if (r.src, r.type, r.dst) in added:
            hold(xk, "added_and_retracted",
                 "delta vừa thêm vừa thu hồi cùng một quan hệ", "major")
            continue
        if not graph.active_relation(r.src, r.dst, r.type):
            hold(xk, "retract_missing",
                 "thu hồi một quan hệ không còn hiệu lực trong canon")
            continue
        seen.add(xk)
        if graph.is_locked(r.src, r.dst, r.type):
            retractions[xk] = "contradiction"
            findings.append({"severity": "blocker", "check": "retract_locked",
                             "message": f"{xk}: thu hồi quan hệ do tác giả đặt"})
        else:
            retractions[xk] = "enrichment"

    # ── 3. MỆNH ĐỀ ──────────────────────────────────────────────────────
    for a in delta.assertions:
        k = item_key(a)
        if k in seen:
            continue

        # (a) CỔNG EPISTEMIC — đứng đầu, trước MỌI kiểm tra khác.
        if not is_objective(a):
            if not a.holder:
                # §11 `reconcile_node` gặp mục này sẽ `continue` im lặng.
                hold(k, "belief_without_holder",
                     f"mệnh đề {a.epistemic} không có người nói/tin", "major")
                continue
            seen.add(k)
            cls[k] = "enrichment"
            if graph.conflicts_with_truth(a):
                # Chênh lệch giữa điều nhân vật nói/tin và sự thật là nguyên
                # liệu của mỉa mai kịch tính (M12). Ghi nhận, đừng chặn.
                irony.append({
                    "holder": a.holder, "subject": a.subject,
                    "predicate": a.predicate, "claim": a.object,
                    "span": a.span, "severity": "note",
                    "note": ("độc giả biết điều này sai — Director có thể "
                             "khai thác ở chương sau")})
            continue

        # (b) chủ thể phải tồn tại
        if a.subject in quarantined_new:
            hold(k, "subject_quarantined",
                 f"chủ thể '{a.subject}' đang bị cách ly")
            continue
        if not known(a.subject):
            hold(k, "unknown_subject",
                 f"chủ thể '{a.subject}' không có trong canon lẫn delta")
            continue

        # (c) vị trí — sở hữu bởi frame
        p = _pred(a.predicate)
        is_char = (a.subject.startswith("CHAR_")
                   or index.get(a.subject, {}).get("kind") == "character")
        if p in LOCATION_PREDICATES:
            obj = str(a.object)
            if obj not in known_locs:
                hold(k, "unknown_location",
                     f"'{obj}' không có trong route graph", "major")
                continue
            if is_char:
                noi = [f.locations[a.subject] for f in frames
                       if a.subject in f.locations]
                if not frames:
                    hold(k, "no_frames_to_check",
                         "không có frame của chương để đối chiếu vị trí")
                elif obj in noi:
                    hold(k, "owned_by_frames",
                         "khớp frame — vị trí do ContinuityFrame sở hữu")
                else:
                    hold(k, "contradicts_frames",
                         (f"văn bản được trích đặt {a.subject} ở {obj}, frame "
                          f"của chương ghi {sorted(set(noi)) or 'không có mặt'}"),
                         "major")
                continue
        elif vocabulary is not None:
            # Tập đóng do tác giả khai (bible/predicates.yaml). Không truyền
            # từ vựng thì quay về luật snake_case bên dưới.
            loi = check_assertion(a, vocabulary, index, accepted_new)
            if loi is not None:
                hold(k, loi[0], loi[1],
                     "note" if loi[0] == "predicate_not_in_vocabulary" else "major")
                continue
        elif not _CANON_PREDICATE.match(p):
            hold(k, "non_canonical_predicate",
                 (f"vị từ '{a.predicate}' là lời kể sự kiện, không phải khoá "
                  f"sự thật (cần snake_case ASCII)"))
            continue

        # (d) span phải nhắc tới chủ thể. Lượt 3 chỉ chứng minh span CÓ THẬT,
        # không chứng minh span ĐỠ ĐƯỢC mệnh đề.
        ns = [_nfc_lower(n) for n in names(a.subject) if n and len(n) >= 2]
        if ns and not any(n in _nfc_lower(a.span) for n in ns):
            hold(k, "span_does_not_mention_subject",
                 (f"span không nhắc tới {a.subject} bằng tên hay bí danh — có "
                  f"thể chỉ là câu tình cờ chứa từ khoá"))
            continue

        decide(k, a)

    delta.classification = cls
    n_contra = (sum(1 for v in cls.values() if v == "contradiction")
                + sum(1 for v in retractions.values() if v == "contradiction"))
    return {
        "classification": cls,
        "retractions": retractions,
        "quarantine": quarantine,
        "irony_seeds": irony,
        "findings": findings,
        "summary": {
            "enrichment": sum(1 for v in cls.values() if v == "enrichment"),
            "improvement": sum(1 for v in cls.values() if v == "improvement"),
            "contradictions": n_contra,
            "quarantined": len(quarantine),
            "irony_seeds": len(irony),
        },
    }
