"""Cấp phát thời gian cho từng cảnh (§8.2.1).

NT-12: `SceneContract.time` không tự sinh ra. Chủ sở hữu trường đó là hàm này,
chạy trong Director. Đây là CODE THUẦN (NT-1) — không hỏi LLM giờ giấc, vì
giờ giấc là số học, và model sẽ đoán sai một cách tự tin.
"""
from __future__ import annotations

from typing import Any

from novel_engine.canon.sqlite_store import NON_ADVANCING_MODES
from novel_engine.canon.timeline import StoryTime

# Ước lượng độ dài theo chức năng beat. Đây là mặc định để hệ thống chạy được;
# tác giả ghi đè ở outline khi cần.
BEAT_DURATION_TICKS: dict[str, int] = {
    "nhịp thở": 2, "thăm dò": 3, "va chạm": 1, "leo thang": 2,
    "mất kiểm soát": 1, "hậu quả": 4, "phát hiện": 2, "hiểu lầm": 2,
    "rẽ hướng": 3, "hậu chấn": 6, "sinh hoạt": 8, "hồi ức": 0,
    "gắn kết": 4, "mầm mống mới": 2, "đặt cược": 3,
}
DEFAULT_SCENE_TICKS = 3
INTER_CHAPTER_GAP = 6      # khoảng nghỉ mặc định giữa hai chương


def allocate_scene_times(chapter: int, beats: list[dict], store,
                         overrides: dict[int, dict] | None = None, *,
                         locations: list[str] | None = None,
                         graph=None, epoch_start: int = 0) -> list[StoryTime]:
    """Nối tiếp thời gian từ cảnh cuối của chương trước.

    Ba trường hợp, và điểm mấu chốt nằm ở chỗ con trỏ nhích hay không:

    1. `present`    — bắt đầu ở `cursor`, con trỏ nhích thêm `duration`.
    2. `flashback` / `vision` — bắt đầu ở mốc TÁC GIẢ chỉ định; con trỏ dòng
       chính ĐỨNG YÊN. "Mười năm trước" là một quyết định sáng tác, hệ thống
       không có cơ sở để đoán.
    3. `concurrent` — bắt đầu ở mốc của cảnh neo; con trỏ cũng ĐỨNG YÊN.
       Nhờ lấy mốc từ anchor, `_mode_valid` (§3.6.1) chắc chắn đi qua: ràng
       buộc được thoả mãn theo thiết kế chứ không nhờ may mắn.

    Nếu hồi ức cũng nhích con trỏ, một chương có hai cảnh hồi ức sẽ đẩy dòng
    thời gian hiện tại lùi hoặc nhảy vô lý.

    `narrative_order` thì NGƯỢC LẠI: tịnh tiến đơn điệu qua MỌI cảnh, kể cả
    hồi ức — độc giả vẫn đọc chúng theo thứ tự. Đó chính là chỗ hai trục tách
    nhau (NT-6), và `_narrative_monotonic` (§3.6.1) dựa vào tính đơn điệu này.

    ═══ THỜI GIAN ĐI ĐƯỜNG ═══════════════════════════════════════════════

    `locations` + `graph` là bổ sung so với §8.2.1, và nó sửa một mâu thuẫn
    giữa hai mục của chính tài liệu: §8.2.1 cấp phát thời gian mà KHÔNG biết
    địa lý, rồi §3.6.1 kiểm tra thời gian ĐỐI CHIẾU với địa lý. Không có gì
    nối hai cơ chế, nên mọi cảnh đổi địa điểm đều sinh blocker:

        S00 ở Cảng Quặng kết thúc tick 8
        S01 ở Trạm Chốt bắt đầu tick 8   ← cần 4 tick đi đường, có 0

    Lượt chạy thật đầu tiên dựng đúng cờ này. Và nó là loại lỗi nguy hiểm
    nhất theo P2 (§0.1): sai theo hướng GÂY PHIỀN. Tác giả gặp báo động giả ở
    mọi chương sẽ tắt luật đi — tắt xong thì mất luôn phần bảo vệ đúng đắn.

    Bắt tác giả tự điền `gap_after` cho từng lần đổi địa điểm cũng không phải
    lời giải: đó chính là thứ §5.6 nói "tác giả sẽ quên". Hệ thống biết địa
    lý, hệ thống tự cộng.
    """
    overrides = overrides or {}
    # `epoch_start` là GỐC trục epoch, chỉ dùng khi store còn rỗng (chương đầu
    # tiên). KHÔNG để 0: hồi ức cần chỗ trong quá khứ. Sự cố hai năm trước là
    # 17.520 tick, nên nếu hiện tại bắt đầu ở 0 thì mốc hồi ức phải là số ÂM —
    # `StoryTime` không cấm, nhưng mọi log sẽ đọc như lỗi và `_mode_valid` báo
    # "hồi ức kết thúc SAU mốc neo" cho một hồi ức hoàn toàn hợp lệ.
    prev_end = store.last_epoch_tick(chapter - 1)
    cursor = (prev_end + INTER_CHAPTER_GAP) if prev_end else epoch_start
    base_order = store.last_narrative_order(chapter - 1) + 1
    out: list[StoryTime] = []
    prev_loc: str | None = None

    for i, beat in enumerate(beats):
        ov: dict[str, Any] = overrides.get(i, {})
        dur = ov.get("duration_ticks",
                     BEAT_DURATION_TICKS.get(beat.get("function"),
                                             DEFAULT_SCENE_TICKS))
        mode = ov.get("mode", "present")

        if mode in NON_ADVANCING_MODES:          # flashback | vision
            anchor = ov.get("anchor_scene")
            if anchor is None:
                raise ValueError(
                    f"cảnh {i} chương {chapter} khai mode='{mode}' nhưng thiếu "
                    f"`anchor_scene` — hồi ức không neo được thì "
                    f"`_mode_valid` (§3.6.1) sẽ dựng blocker ở mọi lần chạy")
            if "epoch_tick" not in ov:
                raise ValueError(
                    f"cảnh {i} chương {chapter} khai mode='{mode}' nhưng thiếu "
                    f"`epoch_tick` — mốc quá khứ là quyết định sáng tác, hệ "
                    f"thống không suy ra được")
            start = ov["epoch_tick"]
        elif mode == "concurrent":
            anchor = ov.get("anchor_scene")
            if anchor is None:
                raise ValueError(
                    f"cảnh {i} chương {chapter} khai mode='concurrent' nhưng "
                    f"thiếu `anchor_scene`")
            start = store.epoch_tick_of(anchor)   # ném KeyError nếu anchor lạ
        else:
            anchor = None
            # Cộng thời gian đi đường TRƯỚC khi cảnh bắt đầu. Chỉ áp cho dòng
            # chính: hồi ức và cảnh song song neo vào chỗ khác nên không có
            # "đi từ đâu tới".
            loc = locations[i] if locations and i < len(locations) else None
            if graph is not None and prev_loc and loc and loc != prev_loc:
                need = graph.travel_ticks(prev_loc, loc)
                if need:
                    cursor += need
            start = cursor
            cursor += dur + ov.get("gap_after", 0)
            if loc:
                prev_loc = loc

        out.append(StoryTime(epoch_tick=start, duration_ticks=dur,
                             narrative_order=base_order + i,
                             mode=mode, anchor_scene=anchor))
    return out
