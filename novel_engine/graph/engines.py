"""Bó động cơ mà mọi node dùng chung.

§9.2 gọi `cfg["configurable"]["engines"]` rồi truy `eng.store`, `eng.graph`,
`eng.chars`, `eng.firewall`, `eng.assembler`, `eng.planner`, `eng.llm` — nhưng
không mục nào trong tài liệu định nghĩa `eng`. Đây là nó.

Gom lại một chỗ thay vì truyền bảy tham số qua mỗi node có một lý do cụ thể:
LangGraph tuần tự hoá STATE vào checkpoint, nhưng KHÔNG tuần tự hoá `config`.
Một kết nối SQLite hay một client LLM mà lọt vào state sẽ làm `SqliteSaver`
ném lỗi pickle ở ranh giới cảnh đầu tiên. Động cơ đi qua config; chỉ dữ liệu
thuần đi qua state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from novel_engine.canon.bible import load_bible
from novel_engine.canon.networkx_graph import NetworkXGraph
from novel_engine.canon.sqlite_store import SqliteStore
from novel_engine.character.firewall import POVFirewall
from novel_engine.memory.assembler import ContextAssembler
from novel_engine.memory.hierarchy import MemoryBudget
from novel_engine.planner.outline_planner import OutlinePlanner
from novel_engine.reconcile.commit import replay_committed
from novel_engine.canon.vocabulary import Vocabulary, load_vocabulary


@dataclass
class Engines:
    graph: NetworkXGraph
    store: SqliteStore
    chars: dict[str, Any]
    planner: OutlinePlanner
    firewall: POVFirewall
    assembler: ContextAssembler
    llm: Any
    meta: dict = field(default_factory=dict)
    bible_dir: Path | None = None
    budget: MemoryBudget | None = None
    replayed: int = 0                 # số delta đã fold lúc dựng canon
    vocabulary: Vocabulary | None = None   # bible/predicates.yaml

    @property
    def total_chapters(self) -> int:
        return int(self.meta.get("total_chapters", 5))

    @property
    def epoch_start(self) -> int:
        """Gốc trục epoch, từ `bible/world.yaml`. Xem `allocate_scene_times`
        về vì sao nó không được là 0."""
        return int(self.meta.get("epoch_start", 0))

    def rebuild_canon(self) -> int:
        """Nạp lại bible rồi fold mọi delta ĐÃ GHI (§3.1).

        Gọi sau khi log thay đổi ngoài luồng ghi bình thường — cụ thể là
        `write --force` xoá delta của một chương. Không dựng lại thì graph
        trong bộ nhớ vẫn chứa sự thật của bản cũ, và chương viết lại sẽ đọc
        ngữ cảnh từ chính văn bản mà nó đang thay thế.

        Firewall và Assembler giữ tham chiếu tới graph cũ, nên phải dựng lại cả
        hai.
        """
        graph, chars, meta = (load_bible(self.bible_dir) if self.bible_dir
                              else load_bible())
        self.graph, self.chars, self.meta = graph, chars, meta
        self.vocabulary = load_vocabulary(self.bible_dir)
        self.firewall = POVFirewall(graph)
        self.assembler = ContextAssembler(
            graph, self.store, self.budget or MemoryBudget.for_vietnamese())
        self.replayed = replay_committed(graph, chars, self.store)
        return self.replayed


def build_engines(llm, *, bible_dir: Path | str | None = None,
                  db_path: str = ":memory:",
                  budget: MemoryBudget | None = None,
                  replay: bool = True) -> Engines:
    """Dựng toàn bộ động cơ từ bible. `db_path=":memory:"` cho test,
    `novel_storage.db` cho CLI."""
    graph, chars, meta = load_bible(bible_dir) if bible_dir else load_bible()
    store = SqliteStore(db_path)
    budget = budget or MemoryBudget.for_vietnamese()
    eng = Engines(
        graph=graph, store=store, chars=chars,
        planner=OutlinePlanner(),
        firewall=POVFirewall(graph),
        assembler=ContextAssembler(graph, store, budget),
        llm=llm, meta=meta,
        bible_dir=Path(bible_dir) if bible_dir else None, budget=budget,
        vocabulary=load_vocabulary(bible_dir),
    )
    # §3.1 — canon là FOLD của các delta đã ghi, không phải trạng thái trong bộ
    # nhớ. CLI chạy đa tiến trình: không fold lại thì mọi thứ đã ghi ở tiến
    # trình trước biến mất.
    if replay:
        eng.replayed = replay_committed(graph, chars, store)
    return eng
