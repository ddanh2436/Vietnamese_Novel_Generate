"""Kiểm tra liên tục trên trục EPOCH (§3.6.1).

Luật `_time_monotonic` của bản 2.0 là SAI THIẾT KẾ, không chỉ thiếu tính năng.
Nó giả định ngầm rằng thứ tự đọc trùng thứ tự xảy ra. Giả định đó vỡ ngay khi
có hồi ức, và vỡ nghiêm trọng hơn với chương song song: chương 14 kể Serena
tại Toà Thánh, chương 15 kể Kaelen tại Lò Năng Lượng trong CÙNG khoảng thời
gian — luật cũ dựng cờ `blocker` giả ở mọi chương như vậy.

Điểm mấu chốt: mọi kiểm tra chạy trên CHUỖI FRAME CỦA TỪNG NHÂN VẬT, sắp theo
`epoch_tick` — không phải trên hai chương đọc kề nhau.
"""
from __future__ import annotations

from novel_engine.canon.timeline import ContinuityFrame


def character_track(cid: str,
                    frames: list[ContinuityFrame]) -> list[ContinuityFrame]:
    """Dòng đời của MỘT nhân vật, sắp theo thời gian thế giới.

    `cid` là MÃ ĐỊNH DANH (CHAR_*). `f.locations` phải được khoá bằng cùng
    không gian định danh đó — khoá bằng tên hiển thị ("Kaelen") làm hàm này
    trả rỗng mãi mãi mà không ném lỗi, và mọi luật dưới đây thành mã chết
    (NT-8). Đó là lý do `SCENE_DIGEST_TMPL` đòi khoá dạng CHAR_* tường minh.
    """
    return sorted((f for f in frames if cid in f.locations),
                  key=lambda f: (f.time.epoch_tick, f.time.narrative_order))


def _no_teleport_epoch(prev: ContinuityFrame, cur: ContinuityFrame,
                       cid: str, graph) -> list[str]:
    a, b = prev.locations[cid], cur.locations[cid]
    if a == b:
        return []
    gap = cur.time.epoch_tick - prev.time.end_tick
    # Tra trên ROUTE graph — KHÔNG phải khoảng cách euclid: núi, chốt kiểm
    # soát và biển làm hỏng mọi tính toán theo bán kính.
    need = graph.travel_ticks(a, b)
    if need is None:
        return [f"{cid}: không tồn tại tuyến đường {a} → {b}"]
    if gap < need:
        return [f"{cid}: {a} → {b} cần {need} tick, chỉ có {gap} tick "
                f"(cảnh {prev.scene_id} → {cur.scene_id})"]
    return []


def no_teleport(frames: list[ContinuityFrame], graph) -> list[str]:
    bad: list[str] = []
    cids = {cid for f in frames for cid in f.locations}
    for cid in sorted(cids):
        track = character_track(cid, frames)
        for prev, cur in zip(track, track[1:]):
            bad += _no_teleport_epoch(prev, cur, cid, graph)
    return bad


def no_bilocation(frames: list[ContinuityFrame]) -> list[str]:
    """Một nhân vật không thể ở hai nơi trong hai khoảng epoch chồng lấn.
    Đây là lỗi mà chương song song sinh ra, và luật cũ KHÔNG bắt được."""
    bad: list[str] = []
    by_char: dict[str, list] = {}
    for f in frames:
        for cid, loc in f.locations.items():
            by_char.setdefault(cid, []).append((f, loc))
    for cid, items in by_char.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (fa, la), (fb, lb) = items[i], items[j]
                if la != lb and fa.time.overlaps(fb.time):
                    bad.append(f"{cid} đồng thời ở {la} ({fa.scene_id}) và "
                               f"{lb} ({fb.scene_id}) — epoch chồng lấn")
    return bad


def mode_valid(f: ContinuityFrame, anchors: dict) -> list[str]:
    """Ba tình huống KHÁC NHAU, và bản đầu gộp hai cái đầu làm một:

    1. hồi ức KHÔNG KHAI `anchor_scene`      → lỗi của Director (blocker)
    2. khai rồi nhưng anchor KHÔNG CÓ trong tập frame đang xét → không kết
       luận được. Đây gần như luôn là lỗi PHẠM VI của người gọi: neo trỏ sang
       chương khác, mà `check_continuity` lại chỉ được đưa frame của một
       chương. Báo blocker ở đây là báo động giả, và P2 (§0.1) nói báo động
       giả dẫn tới việc tắt luật.
    3. khai đúng, anchor có mặt, nhưng mốc sai → blocker thật.
    """
    if f.time.mode in ("flashback", "vision"):
        if not f.time.anchor_scene:
            return [f"{f.scene_id}: hồi ức không khai anchor_scene"]
        a = anchors.get(f.time.anchor_scene)
        if a is None:
            return [f"{f.scene_id}: không tra được cảnh neo "
                    f"'{f.time.anchor_scene}' trong tập frame đang xét — "
                    f"luật này cần frame của CẢ các chương trước"]
        if f.time.end_tick > a.time.epoch_tick:
            return [f"{f.scene_id}: hồi ức tại tick {f.time.epoch_tick} kết "
                    f"thúc ở {f.time.end_tick}, SAU mốc neo "
                    f"{a.scene_id} (tick {a.time.epoch_tick}) — không phải hồi ức"]
    if f.time.mode == "concurrent":
        a = anchors.get(f.time.anchor_scene) if f.time.anchor_scene else None
        if a is not None and not f.time.overlaps(a.time):
            return [f"{f.scene_id}: khai là song song nhưng không chồng lấn "
                    f"epoch với {a.scene_id}"]
    return []


def narrative_monotonic(frames: list[ContinuityFrame]) -> list[str]:
    """Trục ĐỌC phải đánh số sạch: mỗi cảnh một số, không trùng.

    Bản gốc §3.6.1 so `seen == sorted(seen)` trên danh sách frame nhận vào —
    tức nó giả định danh sách ĐÃ theo thứ tự đọc. Nhưng `get_frames()` trả
    theo trục EPOCH, vì mọi luật khác chạy trên dòng đời nhân vật (B4, NT-6).
    Với một chương có hồi ức, hồi ức đứng đầu danh sách (tick nhỏ nhất) mang
    `narrative_order` lớn — và luật báo "không đơn điệu" cho một chương hoàn
    toàn hợp lệ.

    Đúng NT-8 một lần nữa: hai TRẬT TỰ cùng tồn tại, và ở đây chúng bị đem so
    trực tiếp. Cái luật này THỰC SỰ cần kiểm là lỗi đánh số — số trùng nhau —
    và điều đó không phụ thuộc thứ tự danh sách đầu vào.
    """
    orders = [f.time.narrative_order for f in frames]
    trung = {o for o in orders if orders.count(o) > 1}
    if trung:
        ids = sorted(f.scene_id for f in frames
                     if f.time.narrative_order in trung)
        return [f"narrative_order trùng nhau {sorted(trung)} ở {ids} — "
                f"lỗi đánh số cảnh"]
    return []


def check_continuity(frames: list[ContinuityFrame], graph) -> list[dict]:
    """Chạy toàn bộ luật, trả về findings theo dạng chung của Auditor."""
    anchors = {f.scene_id: f for f in frames}
    msgs: list[tuple[str, str]] = []
    msgs += [("no_teleport", m) for m in no_teleport(frames, graph)]
    msgs += [("no_bilocation", m) for m in no_bilocation(frames)]
    msgs += [("narrative_monotonic", m) for m in narrative_monotonic(frames)]
    for f in frames:
        msgs += [("mode_valid", m) for m in mode_valid(f, anchors)]
    return [{"severity": "blocker", "check": chk, "message": m}
            for chk, m in msgs]
