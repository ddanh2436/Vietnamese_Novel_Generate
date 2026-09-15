"""Context Assembler — nạp ngữ cảnh trong ngân sách token (§4.2)."""
from __future__ import annotations

import functools

from novel_engine.memory.hierarchy import MemoryBudget


@functools.lru_cache(maxsize=1)
def _encoder():
    """Nạp tiktoken LƯỜI. Lần đầu tiên `get_encoding` phải tải bảng mã về
    (~5 giây, cần mạng); nạp ở module level làm mọi lần `import` của mọi test
    phải trả giá đó, và làm cả test suite phụ thuộc mạng.

    Không có tiktoken thì vẫn chạy được: ước lượng thô vẫn tốt hơn là sập.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def ntok(s: str) -> int:
    enc = _encoder()
    if enc is None:
        # Ước lượng dự phòng cho tiếng Việt: ~1,6 token mỗi âm tiết.
        return int(len(s.split()) * 1.6) + 1
    return len(enc.encode(s))


def truncate_to(items: list[str] | list[tuple[float, str]], budget: int,
                recency_weighted: bool = False) -> list[str]:
    """Cắt theo ngân sách token.

    F1: hàm khai nhận `list[tuple[float, str]]` nhưng `build()` truyền thẳng
    `list[str]` từ store. `-x[0]` trên chuỗi ném
    `TypeError: bad operand type for unary -: 'str'`.

    Nhưng chỉ nhận thêm `list[str]` rồi gán điểm 1.0 cho tất cả thì SAI theo
    một cách TỆ HƠN CẢ CRASH: `sorted` ổn định nên giữ nguyên thứ tự, và khi
    cắt, thứ bị bỏ là phần tử CUỐI — mà với tóm tắt chương xếp theo thời gian,
    phần tử cuối chính là chương GẦN NHẤT. Bạn sẽ lặng lẽ vứt đúng thứ quan
    trọng nhất. Vì vậy có `recency_weighted`.
    """
    scored: list[tuple[float, str]] = []
    n = len(items)
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            scored.append(item)
        elif recency_weighted:
            scored.append(((i + 1) / n, str(item)))   # càng cuối càng mới
        else:
            scored.append((1.0, str(item)))

    out: list[str] = []
    used = 0
    for _, text in sorted(scored, key=lambda x: -x[0]):
        c = ntok(text)
        if used + c > budget:
            continue          # bỏ qua, thử mục nhỏ hơn — KHÔNG break
        out.append(text)
        used += c
    return out


class ContextAssembler:
    def __init__(self, graph, store, budget: MemoryBudget | None = None):
        self.g = graph
        self.store = store
        self.b = budget or MemoryBudget.for_vietnamese()

    def build(self, chapter: int, scene_idx: int, pov_id: str,
              present: list[str], location_id: str, epoch_tick: int) -> dict:
        # --- L1: 2 cảnh liền trước, nguyên văn digest ---
        l1 = self.store.recent_scene_digests(chapter, scene_idx, k=2)

        # --- L2: 5 chương gần nhất ---
        l2 = self.store.chapter_summaries(chapter - 5, chapter - 1)

        # --- L3: các arc đã đóng ---
        l3 = self.store.arc_summaries(before_chapter=chapter - 5)

        # --- L4: fact liên quan, ĐÃ LỌC QUA POV FIREWALL ---
        # E5/NT-6: tri thức nhân vật khoá theo epoch_tick. Truyền `chapter` vào
        # đây (như bản trước) khiến truy vấn chạy `since_tick <= 14` trong khi
        # thế giới đã ở tick 2400 — POV mù nhận thức hoàn toàn.
        epistemic = self.g.known_by(pov_id, epoch_tick)
        tensions = self.g.faction_tensions(location_id, chapter)
        facts = self._score_facts(epistemic, tensions, present)

        # `known_facts` giữ dạng list[dict] có `id`, KHÔNG phải list[str].
        #
        # LỖI ĐÃ SỬA (họ hàng của F1, cùng cơ chế): §4.2 cho `known_facts` đi
        # qua `truncate_to` — trả về list[str] — rồi §5.4.1 `filter_memory`
        # lại chạy `f.get("id")` trên từng phần tử. `str` không có `.get`, nên
        # writer_node ném AttributeError ngay ở Cảnh 0. Hai mục của cùng tài
        # liệu thoả thuận hai kiểu dữ liệu khác nhau cho cùng một khoá.
        #
        # Cắt theo ngân sách TRÊN PHẦN VĂN BẢN, nhưng trả về bản ghi đầy đủ để
        # tầng lọc hạ nguồn còn `id` mà đối chiếu. Việc dẹp thành chuỗi cho
        # prompt là của `render_facts()`, ở cuối đường ống.
        kept = set(truncate_to([(f["score"], f["text"]) for f in facts],
                               self.b.l4_facts))

        return {
            # recency_weighted=True cho L1–L3: khi phải cắt, cắt cái CŨ trước
            "recent_scenes": truncate_to(l1, self.b.l1_recent_scenes, True),
            "recent_chapters": truncate_to(l2, self.b.l2_recent_chapters, True),
            "arc_history": truncate_to(l3, self.b.l3_arcs, True),
            "known_facts": [f for f in facts if f["text"] in kept],
            "pov_blindspots": epistemic["suspected"],   # điều POV chỉ NGHI
        }

    @staticmethod
    def _score_facts(epistemic: dict, tensions: list[dict],
                     present: list[str]) -> list[dict]:
        scored: list[dict] = []
        for f in epistemic["known"]:
            # fact liên quan tới nhân vật có mặt được ưu tiên
            rel = 1.0 if f["id"] in present else 0.45
            scored.append({"id": f["id"], "score": rel,
                           "text": f"[BIẾT] {f['name']}"})
        for s in epistemic["suspected"]:
            scored.append({"id": s["id"], "score": 0.8,
                           "text": f"[NGHI, {s['conf']:.0%}] {s['name']}"})
        for t in tensions:
            scored.append({"id": t.get("faction_id"), "score": 0.6,
                           "text": f"[THẾ LỰC] {t['faction']}: {t['feuds']}"})
        return scored


def render_facts(known_facts: list[dict]) -> list[str]:
    """Dẹp `known_facts` thành chuỗi cho prompt. Bước CUỐI CÙNG, sau mọi tầng
    lọc — dẹp sớm là vứt mất `id` mà firewall cần."""
    return [f["text"] for f in known_facts]
