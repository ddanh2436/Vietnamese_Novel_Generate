"""`StorePort` trên SQLite — thư viện chuẩn, 0 dependency thêm.

`:memory:` cho pytest (hàng chục lượt trong vài mili-giây, không để lại file
rác), file thật cho CLI. CLI chạy ĐA TIẾN TRÌNH: `cli.py write --chapter 1`
hôm nay và `--chapter 2` hôm sau là hai tiến trình khác nhau, nên in-memory
làm Chương 2 mất sạch `last_epoch_tick`, L1 digest và L2 summary của Chương 1.

═══ HAI CỘT THÊM SO VỚI SCHEMA ĐỀ XUẤT, VÀ VÌ SAO ═══════════════════════════

`continuity_frames` có thêm `end_tick` và `mode`. Cả hai đều sửa lỗi chồng lấn
thời gian, không phải trang trí:

1. `end_tick` — `allocate_scene_times` tính `cursor = last_epoch_tick(ch-1) +
   INTER_CHAPTER_GAP`. Nếu `last_epoch_tick` trả MAX(`epoch_tick`) thuần thì
   nó BỎ QUA thời lượng của cảnh cuối. Cảnh cuối Chương 1 bắt đầu ở tick 100
   kéo dài 8 tick ("ba tuần lênh đênh trên biển" — §3.6.3 khuyến khích
   `duration_ticks` lớn) kết thúc ở 108, nhưng Chương 2 lại bắt đầu ở
   100 + 6 = 106. Hai chương chồng lấn 2 tick, và `_no_bilocation` (§3.6.1)
   sẽ dựng cờ blocker ở mọi nhân vật xuất hiện ở cả hai — một báo động thật,
   sinh ra bởi chính bộ cấp phát.

2. `mode` — một cảnh `vision` nhìn về tick 90.000 hoặc một `flashback` ở tick
   100 không được phép kéo con trỏ dòng chính theo. `last_epoch_tick` vì vậy
   chỉ đếm cảnh `present`/`concurrent`. Lọc trong SQL rẻ hơn nhiều so với
   giải JSON từng frame rồi lọc bằng Python.

`arc_summaries` là bảng thứ năm: `StorePort.arc_summaries()` (L3, §4.1) cần
chỗ để đọc, và bốn bảng kia không có chỗ nào hợp lý để nhét nó vào.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novel_engine.canon.models import StateDelta
from novel_engine.canon.timeline import ContinuityFrame

# Cảnh KHÔNG tiêu thụ thời gian của dòng chính (§8.2.1). Giữ ở đây và ở
# `timeline_alloc` bằng cùng một hằng số — NT-11: một quy tắc, một nơi.
NON_ADVANCING_MODES = ("flashback", "vision")

SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_digests (
    chapter   INTEGER,
    scene_idx INTEGER,
    scene_id  TEXT,
    digest    TEXT,
    PRIMARY KEY (chapter, scene_idx)
);

CREATE TABLE IF NOT EXISTS chapter_summaries (
    chapter          INTEGER PRIMARY KEY,
    summary          TEXT,
    measured_tension REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS arc_summaries (
    arc_id        TEXT PRIMARY KEY,
    closed_after  INTEGER,
    summary       TEXT
);

CREATE TABLE IF NOT EXISTS continuity_frames (
    scene_id        TEXT PRIMARY KEY,
    chapter         INTEGER,
    epoch_tick      INTEGER,
    end_tick        INTEGER,
    narrative_order INTEGER,
    mode            TEXT,
    frame_json      TEXT
);

CREATE TABLE IF NOT EXISTS delta_log (
    delta_id     TEXT PRIMARY KEY,
    chapter      INTEGER,
    delta_json   TEXT,
    committed_at TEXT
);

CREATE TABLE IF NOT EXISTS plan_patches (
    patch_id   TEXT PRIMARY KEY,
    chapter    INTEGER,
    patch_json TEXT
);

CREATE INDEX IF NOT EXISTS ix_frames_chapter ON continuity_frames(chapter);
CREATE INDEX IF NOT EXISTS ix_frames_epoch   ON continuity_frames(epoch_tick);
CREATE INDEX IF NOT EXISTS ix_frames_order   ON continuity_frames(narrative_order);
CREATE INDEX IF NOT EXISTS ix_digests_chapter ON scene_digests(chapter, scene_idx);
CREATE INDEX IF NOT EXISTS ix_delta_chapter  ON delta_log(chapter);
"""


class SqliteStore:
    """Triển khai `StorePort`. Mặc định `:memory:` — production gọi
    `SqliteStore.from_file(...)`."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            # WAL: `cli.py write --chapter 1` hôm nay và `--chapter 2` hôm sau
            # là hai TIẾN TRÌNH. Một tiến trình bị Ctrl-C giữa chương để lại
            # file `-wal`/`-shm`; WAL khôi phục sạch ở lần mở sau, còn journal
            # mặc định (DELETE) thì có thể để lại lock chết.
            # `:memory:` không dùng được WAL và cũng không cần — nó chết theo
            # tiến trình.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @classmethod
    def from_file(cls, path: str | Path = "novel_storage.db") -> "SqliteStore":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return cls(str(p))

    def close(self) -> None:
        self.conn.close()

    # ═════════════ IDEMPOTENCY (§16.1 CP-2) ═════════════

    def has_chapter(self, chapter: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM continuity_frames WHERE chapter = ? LIMIT 1",
            (chapter,)).fetchone()
        return row is not None

    def clear_chapter(self, chapter: int) -> None:
        """Xoá SẠCH mọi dấu vết của một chương trước khi viết lại.

        `INSERT OR REPLACE` một mình KHÔNG đủ: nếu lần chạy trước có 6 cảnh và
        lần này có 4, hai cảnh cũ vẫn nằm lại trong `continuity_frames` với
        `narrative_order` cao hơn. `last_narrative_order` sẽ trả số của cảnh
        ma, và `_narrative_monotonic` (§3.6.1) thấy trục đọc đứt quãng — một
        lỗi âm thầm chỉ lộ ra ở chương sau nữa.

        Đây cũng là nền cho rollback ở §3.1: "bỏ chương 18, viết lại".
        """
        for tbl in ("continuity_frames", "scene_digests", "delta_log",
                    "plan_patches"):
            self.conn.execute(f"DELETE FROM {tbl} WHERE chapter = ?", (chapter,))
        self.conn.execute("DELETE FROM chapter_summaries WHERE chapter = ?",
                          (chapter,))
        self.conn.commit()

    def discard_scene_records(self, chapter: int) -> None:
        """Bỏ digest và frame của một chương CHƯA hoàn tất — rollback khi đồ thị
        escalate giữa vòng cảnh (§9.3).

        Khác `clear_chapter`: KHÔNG đụng `delta_log` và `plan_patches`. Escalate
        ở `reconcile` giữ delta lại cho tác giả duyệt; xoá nó cùng lúc là mất
        đúng thứ đang chờ quyết định.
        """
        for tbl in ("continuity_frames", "scene_digests"):
            self.conn.execute(f"DELETE FROM {tbl} WHERE chapter = ?", (chapter,))
        self.conn.commit()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ═════════════ 1. CON TRỎ THỜI GIAN (§8.2.1) ═════════════

    def last_epoch_tick(self, chapter: int) -> int:
        """Mốc thời gian dòng chính KẾT THÚC sau `chapter`.

        Trả `end_tick` (= epoch_tick + duration), không phải `epoch_tick`
        thuần — xem docstring module. Chỉ đếm cảnh `present`/`concurrent`:
        hồi ức và ảo ảnh neo vào chỗ khác, không đẩy dòng chính.
        """
        if chapter <= 0:
            return 0
        placeholders = ",".join("?" * len(NON_ADVANCING_MODES))
        row = self.conn.execute(
            f"SELECT MAX(end_tick) AS t FROM continuity_frames "
            f"WHERE chapter <= ? AND mode NOT IN ({placeholders})",
            (chapter, *NON_ADVANCING_MODES)).fetchone()
        return row["t"] or 0

    def last_narrative_order(self, chapter: int) -> int:
        """Trục ĐỌC — mọi cảnh đều tính, kể cả hồi ức, vì độc giả vẫn đọc chúng
        theo thứ tự. Đây chính là chỗ hai trục tách nhau (NT-6)."""
        if chapter <= 0:
            return 0
        row = self.conn.execute(
            "SELECT MAX(narrative_order) AS n FROM continuity_frames "
            "WHERE chapter <= ?", (chapter,)).fetchone()
        return row["n"] or 0

    def epoch_tick_of(self, scene_id: str) -> int:
        """Mốc của cảnh neo. Ném `KeyError` khi không có — cảnh hồi ức trỏ vào
        một anchor không tồn tại là lỗi thật, và `_mode_valid` (§3.6.1) cần
        thấy nó chứ không phải nhận một số 0 im lặng."""
        row = self.conn.execute(
            "SELECT epoch_tick FROM continuity_frames WHERE scene_id = ?",
            (scene_id,)).fetchone()
        if row is None:
            raise KeyError(f"không có cảnh neo '{scene_id}' trong store")
        return row["epoch_tick"]

    def tick_of_chapter(self, chapter: int) -> int:
        """Mốc BẮT ĐẦU của một chương (§11) — cảnh dòng chính sớm nhất."""
        placeholders = ",".join("?" * len(NON_ADVANCING_MODES))
        row = self.conn.execute(
            f"SELECT MIN(epoch_tick) AS t FROM continuity_frames "
            f"WHERE chapter = ? AND mode NOT IN ({placeholders})",
            (chapter, *NON_ADVANCING_MODES)).fetchone()
        return row["t"] or 0

    # ═════════════ 2. MEMORY HIERARCHY L1–L3 (§4.1) ═════════════

    def put_scene_digest(self, chapter: int, scene_idx: int, digest: str,
                         scene_id: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO scene_digests "
            "(chapter, scene_idx, scene_id, digest) VALUES (?,?,?,?)",
            (chapter, scene_idx, scene_id or f"CH{chapter:03d}_S{scene_idx:02d}",
             digest))
        self.conn.commit()

    def recent_scene_digests(self, chapter: int, scene_idx: int,
                             k: int = 2) -> list[str]:
        """`k` cảnh liền TRƯỚC (chapter, scene_idx), vượt được ranh giới chương.

        Trả theo thứ tự CŨ → MỚI. Thứ tự này là hợp đồng với
        `truncate_to(..., recency_weighted=True)`, vốn chấm điểm `(i+1)/n` —
        đảo thứ tự ở đây làm nó giữ lại đúng những cảnh cũ nhất (F1).
        """
        rows = self.conn.execute(
            "SELECT digest FROM scene_digests "
            "WHERE (chapter < ?) OR (chapter = ? AND scene_idx < ?) "
            "ORDER BY chapter DESC, scene_idx DESC LIMIT ?",
            (chapter, chapter, scene_idx, k)).fetchall()
        return [r["digest"] for r in reversed(rows)]

    def scene_digests_with_time(self) -> list[dict]:
        """Mọi digest kèm `epoch_tick`, cho tầng nhớ-lại-theo-nội-dung (L5).

        Nối với `continuity_frames` chứ không lưu thêm cột: mốc epoch của một
        cảnh đã có một chủ sở hữu duy nhất ở đó (NT-12). Cảnh chưa có frame thì
        không có mốc, và không mốc thì không lọc được theo NT-6 — bỏ, chứ không
        đoán bằng số chương.
        """
        rows = self.conn.execute(
            "SELECT d.chapter, d.scene_idx, d.scene_id, d.digest, f.epoch_tick "
            "FROM scene_digests d JOIN continuity_frames f "
            "ON f.scene_id = d.scene_id "
            "ORDER BY d.chapter, d.scene_idx").fetchall()
        return [dict(r) for r in rows]

    def put_chapter_summary(self, chapter: int, summary: str,
                            measured_tension: float = 0.5) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO chapter_summaries "
            "(chapter, summary, measured_tension) VALUES (?,?,?)",
            (chapter, summary, measured_tension))
        self.conn.commit()

    def chapter_summaries(self, from_ch: int, to_ch: int) -> list[str]:
        """Khoảng ĐÓNG [from_ch, to_ch], thứ tự CŨ → MỚI (xem
        `recent_scene_digests` về vì sao thứ tự là hợp đồng)."""
        if to_ch < from_ch:
            return []
        rows = self.conn.execute(
            "SELECT summary FROM chapter_summaries "
            "WHERE chapter BETWEEN ? AND ? ORDER BY chapter ASC",
            (from_ch, to_ch)).fetchall()
        return [r["summary"] for r in rows]

    def put_arc_summary(self, arc_id: str, closed_after: int,
                        summary: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO arc_summaries "
            "(arc_id, closed_after, summary) VALUES (?,?,?)",
            (arc_id, closed_after, summary))
        self.conn.commit()

    def arc_summaries(self, before_chapter: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT summary FROM arc_summaries WHERE closed_after <= ? "
            "ORDER BY closed_after ASC", (before_chapter,)).fetchall()
        return [r["summary"] for r in rows]

    def measured_tension(self, chapter: int) -> float:
        """`director_node` gọi `measured_tension(ch - 1)`; ở Chương 1 điều đó
        là `measured_tension(0)`, chưa có gì. Trả 0.5 (trung tính) thay vì ném
        — chương đầu tiên không có tiền sử là chuyện bình thường, không phải
        lỗi."""
        row = self.conn.execute(
            "SELECT measured_tension AS t FROM chapter_summaries WHERE chapter = ?",
            (chapter,)).fetchone()
        return 0.5 if row is None else float(row["t"])

    # ═════════════ 3. FRAME & EVENT SOURCING ═════════════

    def put_frame(self, frame: ContinuityFrame) -> None:
        chapter = _chapter_of_scene(frame.scene_id)
        self.conn.execute(
            "INSERT OR REPLACE INTO continuity_frames "
            "(scene_id, chapter, epoch_tick, end_tick, narrative_order, mode, "
            " frame_json) VALUES (?,?,?,?,?,?,?)",
            (frame.scene_id, chapter, frame.time.epoch_tick, frame.time.end_tick,
             frame.time.narrative_order, frame.time.mode,
             frame.model_dump_json()))
        self.conn.commit()

    def get_frames(self, chapter: int | None = None) -> list[ContinuityFrame]:
        """Sắp theo trục EPOCH (NT-6, B4). Các luật liên tục ở §3.6.1 chạy trên
        DÒNG ĐỜI của nhân vật, không trên thứ tự đọc — trả theo
        `narrative_order` là tái tạo đúng lỗi mà §3.6 sinh ra để chặn."""
        if chapter is None:
            rows = self.conn.execute(
                "SELECT frame_json FROM continuity_frames "
                "ORDER BY epoch_tick ASC, narrative_order ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT frame_json FROM continuity_frames WHERE chapter = ? "
                "ORDER BY epoch_tick ASC, narrative_order ASC",
                (chapter,)).fetchall()
        return [ContinuityFrame.model_validate_json(r["frame_json"]) for r in rows]

    def append_delta(self, delta: StateDelta) -> None:
        """Log CHỈ GHI THÊM, theo thứ tự chương (lịch sử sáng tác, bất biến,
        dùng để rollback). World state thì fold theo `epoch_tick` — một log,
        hai trật tự đọc (§3.6.4)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO delta_log "
            "(delta_id, chapter, delta_json, committed_at) VALUES (?,?,?,?)",
            (delta.delta_id or f"d_ch{delta.chapter:03d}", delta.chapter,
             delta.model_dump_json(),
             datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_deltas(self, chapter: int | None = None, *,
                   committed_only: bool = False) -> list[StateDelta]:
        if chapter is None:
            rows = self.conn.execute(
                "SELECT delta_json FROM delta_log ORDER BY chapter ASC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT delta_json FROM delta_log WHERE chapter = ?",
                (chapter,)).fetchall()
        out = [StateDelta.model_validate_json(r["delta_json"]) for r in rows]
        return [d for d in out if d.committed] if committed_only else out

    # ═════════════ PLAN PATCH (§11) ═════════════

    def put_plan_patch(self, chapter: int, patch: dict) -> None:
        """PlanPatch là QUYẾT ĐỊNH của tác giả (CP-4), không phải thao tác của
        hệ thống: agent không bao giờ tự sửa `bible/outline.yaml`."""
        self.conn.execute(
            "INSERT OR REPLACE INTO plan_patches (patch_id, chapter, patch_json) "
            "VALUES (?,?,?)",
            (patch["patch_id"], chapter, json.dumps(patch, ensure_ascii=False)))
        self.conn.commit()

    def get_plan_patches(self, chapter: int | None = None) -> list[dict]:
        if chapter is None:
            rows = self.conn.execute(
                "SELECT patch_json FROM plan_patches ORDER BY chapter").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT patch_json FROM plan_patches WHERE chapter = ?",
                (chapter,)).fetchall()
        return [json.loads(r["patch_json"]) for r in rows]


def _chapter_of_scene(scene_id: str) -> int:
    """`CH001_S02` → 1. Trả 0 khi không đọc được, để một scene_id lạ không
    làm sập cả chương."""
    try:
        return int(scene_id.split("_")[0].removeprefix("CH"))
    except (ValueError, IndexError):
        return 0
