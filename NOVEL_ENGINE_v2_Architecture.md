# NOVEL ENGINE v2 — Kiến trúc Multi-Agent cho tiểu thuyết dài kỳ

> Tài liệu thiết kế & triển khai kỹ thuật
> Stack: Python 3.11+ · LangGraph · Neo4j · Qdrant · Pydantic v2
> Phiên bản: 2.5 — mở rộng từ bản thiết kế v1, đã qua sáu vòng rà soát

---

## Mục lục

| # | Phần | Nội dung |
|---|------|----------|
| 0 | [Tóm tắt điều hành](#0-tóm-tắt-điều-hành) | v2 khác v1 ở đâu |
| 1 | [Chẩn đoán](#1-chẩn-đoán-3-giới-hạn-gốc--7-lỗ-hổng-còn-lại-của-v1) | 3 giới hạn gốc + 7 lỗ hổng còn lại |
| 2 | [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể-v2) | Sơ đồ 5 tầng + luồng dữ liệu |
| 3 | [Canon Store](#3-canon-store--world-state-theo-event-sourcing) | Event sourcing, schema, Neo4j, Timeline Engine |
| 4 | [Narrative Memory Hierarchy](#4-narrative-memory-hierarchy--chống-context-drift) | L0–L4, Context Assembler |
| 5 | [Character Engine](#5-character-engine--chống-stereotype-flattening) | BDI, utility, Voice Fingerprint, News Dispatcher |
| 6 | [Foreshadowing Engine](#6-foreshadowing-engine--thuật-toán-rải-manh-mối) | Clue DAG, scheduler, payoff |
| 7 | [Relationship Machine](#7-relationship--romance-state-machine) | Intimacy × Friction, guards |
| 8 | [Narrative Planner](#8-narrative-planner--tension-curve--beat-sheet) | Tension curve, Beat Sheet |
| 9 | [Execution Loop](#9-execution-loop--langgraph) | LangGraph đầy đủ |
| 10 | [Auditor](#10-auditor--kiểm-tra-bằng-thuật-toán-không-chỉ-bằng-llm) | Continuity Ledger, nhịp văn, diff-driven extraction |
| 11 | [Drift Reconciliation](#11-drift-reconciliation--khi-writer-đi-chệch-kế-hoạch) | Xử lý lệch kế hoạch |
| 12 | [Prompt Library](#12-prompt-library) | Template thật |
| 13 | [Eval Harness](#13-eval-harness--đo-hệ-thống-có-tốt-lên-không) | Metrics tự động |
| 14 | [Cost & Performance](#14-cost--performance) | Caching, ngân sách |
| 15 | [Repo & Lộ trình](#15-repo-layout--lộ-trình-triển-khai) | 6 giai đoạn |
| 16 | [Phụ lục](#16-phụ-lục) | Thuật ngữ, checklist |

---

## 0. Tóm tắt điều hành

Bản v1 của bạn đã đúng ở **tầng kiến trúc**: tách World / Character / Planner / Execution, dùng Knowledge Graph thay vì vector DB thuần, dùng multi-agent thay vì một prompt khổng lồ. Đó là ba quyết định nền tảng và chúng không cần sửa.

Vấn đề của v1 nằm ở **tầng cơ chế**: nó mô tả *cái gì tồn tại* nhưng chưa mô tả *cái gì chạy*. Cụ thể, v1 chưa trả lời được 7 câu hỏi vận hành mà bất kỳ hệ thống nào viết đến chương 40 cũng sẽ vấp phải:

1. Khi Writer viết ra một sự thật mới không có trong graph (nhân vật nhắc tên một con sông chưa từng tồn tại), sự thật đó đi đâu?
2. Chương 37 cần biết gì từ chương 3, và làm sao nạp được mà không vỡ context window?
3. Critic bảo "chưa đạt" ba lần liên tiếp thì sao? Lặp vô hạn? Ai cắt?
4. Hai nhân vật cùng là "kiếm sĩ lạnh lùng" — cơ chế nào ép giọng văn của họ khác nhau ở *tầng câu chữ*, chứ không phải ở tầng mô tả hồ sơ?
5. Chương này nên cài mấy manh mối? 1 hay 5? Theo tiêu chí gì?
6. Writer viết hay nhưng đi chệch outline — ép viết lại, hay cập nhật outline?
7. Làm sao biết hệ thống sau khi sửa đã tốt lên, chứ không phải "cảm giác tốt hơn"?

v2 bổ sung chính xác 7 cơ chế để trả lời 7 câu đó:

| # | Cơ chế mới | Giải quyết |
|---|-----------|-----------|
| A | **Canon Store event-sourced** — world state là kết quả fold các `StateDelta`, không phải bảng bị ghi đè | Q1, khả năng rollback/diff |
| B | **Narrative Memory Hierarchy (L0–L4)** + Context Assembler có ngân sách token | Q2, Context Drift |
| C | **Bounded Critique Loop** — severity gating, tối đa 2 vòng, escalate lên người | Q3, chi phí |
| D | **Voice Fingerprint** — ràng buộc định lượng ở tầng câu chữ + blind attribution test | Q4, Stereotype Flattening |
| E | **Clue Scheduler** — thuật toán chọn manh mối theo topological order + salience decay + tension curve | Q5, Chekhov's Gun |
| F | **Drift Reconciliation** — phân loại lệch thành `CONTRADICTION` / `ENRICHMENT` / `IMPROVEMENT`, xử lý khác nhau | Q6 |
| G | **Eval Harness** — 8 metric tự động chạy trên mỗi bản build | Q7 |

Ngoài ra v2 sửa ba chi tiết thiết kế của v1 có rủi ro thực tế:

- **Sơ đồ v1 vẽ luồng một chiều** (World → Character → Planner → Execution). Thực tế luồng phải **hai chiều**: Execution ghi ngược vào World. Thiếu chiều ngược này chính là nguyên nhân gốc của Context Drift, chứ không phải context window nhỏ.
- **`Chekhov_Registry` bắt buộc mọi NPC phải có vai trò** là quá cứng. Một thế giới mà *mọi* người bán rượu đều là đầu mối manh mối sẽ đọc như phim trinh thám rẻ tiền, mất hoàn toàn cảm giác thế giới thật. v2 thêm hạng `AMBIENT` có hạn ngạch (quota) và có cơ chế **thăng hạng hồi tố** (retroactive promotion).
- **`Ideological_Friction` là một chỉ số đơn** không đủ. Xung đột "tôi không đồng ý với cách làm của anh" và xung đột "nếu anh thành công thì gia tộc tôi bị diệt" khác nhau về bản chất. v2 tách thành `friction_value` (bất đồng giá trị, có thể hoà giải) và `stake_conflict` (xung đột lợi ích cấu trúc, không thể hoà giải bằng đối thoại).

### 0.1 Cập nhật v2.1 — bốn sửa đổi

Bốn mục dưới đây đến từ vòng rà soát thứ hai. Ba trong số đó sửa **lỗi thiết kế** trong bản 2.0, không phải bổ sung tính năng.

| # | Vấn đề | Tính chất | Mục |
|---|--------|-----------|-----|
| **P1** | `extractor_node` một lượt là điểm nghẽn của toàn bộ event sourcing: lỗi ở đây không crash, chỉ âm thầm ghi sai canon | **Lỗi kiến trúc** | §10.5 |
| **P2** | `_time_monotonic` giả định thứ tự đọc trùng thứ tự xảy ra — dựng cờ blocker giả ở mọi hồi ức và mọi chương song song | **Lỗi thiết kế** | §3.6 |
| **P3** | Không có cơ chế lan truyền thông tin: `KNOWS_ABOUT` phải thêm tay, và tác giả sẽ quên | **Thiếu cơ chế** | §5.6 |
| **P4** | `VoiceFingerprint` chỉ kiểm soát thoại; tầng trần thuật không có ràng buộc nào | **Lỗ hổng** | §10.3.1 |

Điểm chung của P1 và P2: cả hai đều là lỗi **âm thầm**. Chúng không làm hệ thống dừng lại, chúng chỉ làm nó sai dần. P2 còn tệ hơn ở chỗ nó sai theo hướng gây phiền (báo động giả), khiến người dùng có xu hướng tắt luật đi — và tắt xong thì mất luôn phần bảo vệ đúng đắn.

Ba nguyên tắc mới được rút ra từ vòng này, bổ sung cho bốn nguyên tắc ở §2.1:

**NT-5 — Mỗi mệnh đề ghi vào canon phải có bằng chứng khớp được bằng chuỗi.** Không có `span` xác minh thì không có thay đổi trạng thái, bất kể LLM khai gì.

**NT-6 — Tri thức nhân vật khoá theo `epoch_tick`; tri thức độc giả khoá theo `narrative_order`.** Đừng bao giờ trộn hai trục.

**NT-7 — Chỉ số văn phong là công cụ chẩn đoán, không phải mục tiêu đưa cho Writer.** Đưa ngưỡng thống kê vào prompt sẽ sinh ra văn xuôi thoả mãn con số và đọc như máy (§10.3.2).

### 0.2 Cập nhật v2.2 — năm lỗi runtime

Vòng rà soát thứ ba soi bản 2.1 ở góc độ **code chạy thật**, và tìm ra năm lỗi. Bốn trong số đó thuộc loại nguy hiểm nhất: chúng không ném ngoại lệ, chỉ âm thầm cho kết quả sai.

| # | Lỗi | Biểu hiện | Mục |
|---|-----|-----------|-----|
| **B1** | `plan_coverage` so `clue_id` với `Assertion.subject` — hai không gian định danh khác nhau, phép so luôn cho `False` | Tỉ lệ cài manh mối báo 0%, mọi `clue_transitions` bị xoá | §3.4, §10.5.3 |
| **B2** | `merge_extractions` chỉ gộp 2/6 trường của `StateDelta` | Quan hệ và chuyển trạng thái phát sinh ngoài kế hoạch biến mất im lặng | §10.5.3 |
| **B3** | `propagate` dùng điều kiện Dijkstra đơn mục tiêu | Tin đồn nhanh vĩnh viễn chặn tin chính thống chậm; nhân vật không bao giờ được đính chính | §5.6.3 |
| **B4** | Fold theo `narrative_order` trong khi thời gian khoá theo `epoch_tick` | Hồi ức ghi đè quá khứ, vô hiệu hoá manh mối đã cài mà không luật nào phát hiện | §3.6.4 |
| **B5** | `model_validate_json` gọi thẳng trên `.content` | `JSONDecodeError` làm sập LangGraph ở bước cuối, sau khi đã trả tiền cả chương | §9.5 |

B1 và B4 đáng chú ý vì cùng một nguyên nhân gốc: **hai không gian định danh bị đem so sánh trực tiếp**. B1 trộn ID hệ thống với thực thể ngôn ngữ; B4 trộn trục thời gian đọc với trục thời gian thế giới. Bài học chung thành nguyên tắc thứ tám:

**NT-8 — Khi hai không gian định danh cùng tồn tại trong hệ thống, mọi phép so sánh giữa chúng phải đi qua một hàm ánh xạ có tên.** Không bao giờ so trực tiếp bằng `in` hay `==`, vì lỗi loại này không ném ngoại lệ — nó chỉ trả về `False` mãi mãi.

Ba mở rộng đi kèm, vượt ra ngoài phần sửa lỗi:

- **B1 →** giải pháp không phải thêm `related_clue_id` vào `Assertion`, mà là một kênh riêng `PlantEvidence`. Lý do ở §3.4: phần lớn manh mối được cài **không sinh ra assertion nào cả**, nên đo độ phủ qua assertion vẫn đếm thiếu — và đếm thiếu đúng ở những manh mối tinh tế nhất.
- **B3 →** sau khi sửa Pareto, thêm `accept_correction` (§5.6.6): tin đúng đến nơi không có nghĩa nhân vật tin nó. Nhân vật từ chối đính chính là khắc hoạ tính cách, không phải lỗi — nhưng phải được ghi lại.
- **B5 →** thêm `safe_node` (§9.5): trong một lần chạy 40 chương, luôn có node ném lỗi. Lỗi phải biến thành escalation, không thành stack trace.

### 0.3 Cập nhật v2.3 — năm lỗi thực thi

Vòng rà soát thứ tư soi bản 2.2 ở cấp **runtime của LangGraph** và **luồng dữ liệu giữa các node**. Hai trong năm lỗi khiến hệ thống không bao giờ chạy xong chương đầu tiên.

| # | Lỗi | Loại | Mục |
|---|-----|------|-----|
| **C1** | Không node nào tăng `scene_index`; router LangGraph là hàm thuần, không đổi được state | **Vòng lặp vô hạn** ở Cảnh 0 | §9.1, §9.2, §9.4 |
| **C2** | `extractor_node` dùng `state["polished"]` (một cảnh) để xác minh span của cả 6 cảnh | **Mất dữ liệu**: manh mối 5 cảnh đầu bị xoá | §9.2 |
| **C3** | `_score` trả `-1.0` cho manh mối quá hạn, trong khi `debt_load` lại đếm chính chúng là nợ và chặn Planner | **Deadlock** toàn hệ thống | §6.3, §6.3.1 |
| **C4** | `filter_contract` được gọi trên `ctx` chứ không trên contract; contract vào prompt không qua lọc | **Thủng POV Firewall** | §5.4.1 |
| **C5** | `classify_delta` không xét `epistemic` | **Báo động giả**: nhân vật không được phép nói dối | §11 |

Ba lỗi nữa phát hiện trong cùng vòng, không nằm trong danh sách trên:

| # | Lỗi | Hậu quả |
|---|-----|---------|
| **C6** | `revision_count` không reset giữa các cảnh | Cảnh 1 bị escalate ở lỗi `major` đầu tiên vì đã "tiêu" hết lượt của Cảnh 0 |
| **C7** | `st.chapters_in_stage += 1` nằm trong `apply_scene_effects`, vốn chạy **mỗi cảnh** | Bộ đếm "số chương" chạy nhanh gấp 6 lần; mọi guard `MIN_CHAPTERS_IN_STAGE` mở sớm gấp 6 — đây chính là nguyên nhân M5 báo nhảy cóc mà nhìn vào guard thì guard đúng |
| **C8** | Không hàm nào gán `st.stage = decision["to"]` | Quan hệ kẹt ở `STRANGERS` vĩnh viễn dù intimacy đạt 100 |

Cùng với `StateDelta.item()` và `item_key()` được gọi nhưng chưa từng được định nghĩa, tổng cộng **mười** khiếm khuyết đã được vá trong vòng này.

Ba nguyên tắc rút ra:

**NT-9 — Router của LangGraph không đổi được state.** Mọi biến điều khiển vòng lặp phải được cập nhật trong một **node**. Hệ quả thực hành: mỗi vòng lặp trong đồ thị cần một node "chốt sổ" riêng, và node đó là nơi duy nhất được đụng vào biến đếm. Ở đây là `scene_boundary_node` (§9.2) — nó cũng gánh luôn hai việc §4.1 và §10.1 khai là có mà chưa node nào làm: sinh scene digest L1 và sinh ContinuityFrame.

**NT-10 — Mọi ngưỡng chặn phải có đường thoát.** `ATTENTION_BUDGET`, `debt_load`, `MAX_REVISIONS`, `payoff_deadline` đều là van; van không có đường xả sớm muộn thành khoá. Khi rà soát, tìm mọi `continue` và `return -1` rồi hỏi: nếu điều kiện này đúng mãi thì sao? (§6.3.1)

**NT-11 — Một quy tắc được kiểm ở hai nơi thì sẽ có một nơi quên.** `epistemic` được lọc đúng ở §10.2 và bị bỏ quên ở §11. Quy tắc phải nằm trong **một** hàm mà cả hai chỗ cùng gọi, không phải hai bản sao của cùng một điều kiện.

### 0.4 Cập nhật v2.4 — tám lỗi thực thi cuối cùng

Vòng rà soát thứ năm soi từng **biến, chữ ký hàm và giá trị trả về**. Bốn lỗi ném ngoại lệ ngay, bốn lỗi làm lệch hệ thống âm thầm. Ba trong số đó (E1, E2, E7) là lỗi **do chính bản vá v2.3 tạo ra** — bằng chứng thực tế cho một điều đáng nhớ: mỗi bản vá là một lượt thay đổi mới, và cần được rà soát như mọi thay đổi khác.

| # | Lỗi | Loại | Mục |
|---|-----|------|-----|
| **E1** | `epoch_tick` dùng trong `writer_node` mà không tồn tại; `SceneContract` không có trường thời gian | `NameError` ở Cảnh 0 | §8.2.1, §9.2 |
| **E2** | `SceneClose` và `SCENE_DIGEST_TMPL` được gọi nhưng chưa từng định nghĩa | `NameError` sau cảnh đầu | §12.4 |
| **E3** | `StateDelta` có 3 trường bắt buộc mà LLM không bao giờ sinh | `ValidationError` ở lượt 2 | §3.4 |
| **E4** | Không chỗ nào cập nhật `last_touched_chapter`/`salience` | `decay()` trả 0.0 vĩnh viễn → ép re-plant mọi chương, nuốt sạch ngân sách chú ý | §11 |
| **E5** | `ContextAssembler` truyền `chapter` vào chỗ cần `epoch_tick` | POV mù nhận thức hoàn toàn (`since_tick <= 14` trong khi thế giới ở tick 2400) | §4.2 |
| **E6** | `reconcile_node` replan theo `improvement` nhưng không commit chúng | Kế hoạch xoay quanh một nhân vật không tồn tại trong canon | §11 |
| **E7** | Test gọi `after_polish` sau khi hàm này đã đổi thành luôn trả `"scene_boundary"` | Test suite đỏ dù logic đúng | §15.3 |
| **E8** | `schedule()` đổi thành trả tuple ở §6.3.1, nhưng `director_node` vẫn dùng như list | `AttributeError: 'list' object has no attribute '__dict__'` | §9.2 |

E8 do tôi tìm ra khi kiểm chứng bảy lỗi trên, và nó minh hoạ đúng cái bẫy của E1/E2: **đổi chữ ký hàm mà không truy hết các chỗ gọi**. Ba lỗi cùng một cơ chế trong một vòng vá.

Ba nguyên tắc cuối:

**NT-12 — Mỗi trường mới trong schema phải có một chủ sở hữu được nêu tên.** `SceneContract.time` không tự sinh ra; phải chỉ rõ ai điền (`allocate_scene_times` trong Director) và bằng dữ liệu gì. Thêm trường mà không chỉ định chủ sở hữu là cách tạo ra `NameError` ở hạ nguồn.

**NT-13 — Metadata hệ thống không bao giờ nằm trong schema mà LLM phải điền.** `delta_id`, `created_at`, `chapter` là việc của code. Để chúng bắt buộc trong `StateDelta` là ép LLM đoán, và nó sẽ đoán sai hoặc bỏ trống. Cho `default_factory`, rồi code ghi đè sau khi parse.

**NT-14 — Văn xuôi đã viết ra là sự thật; canon phải khớp với trang giấy.** Đây là lý do `improvement` phải được commit chứ không chỉ dùng để replan. Từ chối ghi một sự thật đã in ra không giữ được canon sạch — nó tạo ra phân ly canon–văn bản, và phân ly thì không có cách nào phát hiện tự động.

### 0.5 Cập nhật v2.5 — chín lỗi, và một kết luận về chính quy trình rà soát

Vòng thứ sáu tìm ra chín khiếm khuyết: bốn ném ngoại lệ, năm sai âm thầm.

| # | Lỗi | Loại | Mục |
|---|-----|------|-----|
| **F1** | `truncate_to` khai nhận `list[tuple]` nhưng `build()` truyền `list[str]` | `TypeError` ở Cảnh 0 | §4.2 |
| **F2** | `StoryTime.narrative_order` bắt buộc, LLM không biết mà sinh | `ValidationError` ở ranh giới cảnh | §12.4 |
| **F3** | `writer_node` ở §9.2 không tăng `revision_count` (chỉ có ghi chú ở §9.4) | **Vòng lặp vô hạn** writer↔auditor | §9.2 |
| **F4** | `schedule()` gọi trong vòng lặp cảnh → 6 lần/chương, cùng 3 manh mối vào cả 6 cảnh | Vỡ `ATTENTION_BUDGET` gấp 6 | §9.2 |
| **F5** | `clue_escalations` hứng rồi bỏ | Manh mối treo, tác giả không biết | §9.2 |
| **F6** | Assertion phi khách quan xếp `enrichment` rồi bị `commit` thẳng vào graph | **Ô nhiễm canon**: lời nói dối thành sự thật lịch sử | §11 |
| **F7** | `retracted_relations` không được duyệt ở `classify_delta` lẫn `reconcile_node` | Quan hệ cũ không bao giờ đóng | §11 |
| **F8** | `eng.canon` không tồn tại; `GraphPort` thiếu ba method cho flashback audit | `AttributeError` | §3.7, §11 |
| **F9** | `item_key` dùng `hash()` — ngẫu nhiên hoá theo `PYTHONHASHSEED` | `KeyError` khi khôi phục từ checkpoint ở tiến trình mới | §11 |

**F6 là lỗi đáng nghiên cứu nhất**, vì không bản vá nào đơn lẻ gây ra nó. v2.3 sửa `classify_delta` để lời nói dối không bị coi là mâu thuẫn — xếp chúng vào `enrichment`. v2.4 sửa `reconcile_node` để commit cả `enrichment` lẫn `improvement`. Mỗi bản vá đúng trong phạm vi của nó. **Hợp lại, chúng tái tạo chính xác cái lỗi mà `epistemic` được thiết kế ra để chặn.**

#### Đánh giá trung thực về quy trình rà soát tĩnh

Số khiếm khuyết tìm được qua sáu vòng: **5 → 5 → 8 → 8 → 9**. Không giảm.

Đáng chú ý hơn con số: trong chín lỗi vòng này, **ba lỗi do chính các bản vá vòng trước tạo ra** (F2 và F5 sinh từ v2.4; F6 sinh từ tổ hợp v2.3+v2.4). Tỉ lệ tiêm lỗi khoảng 30% mỗi vòng vá.

Điều đó dẫn tới một kết luận nên nói thẳng: **tài liệu này chưa "production-ready", và rà soát tĩnh thêm một vòng nữa cũng sẽ không làm nó thành production-ready.** Không phải vì các bản vá sai, mà vì:

1. Rà soát tĩnh **không hội tụ** ở quy mô này. Mỗi vòng vá là một thay đổi mới, có tỉ lệ lỗi riêng của nó, và tỉ lệ đó không nhỏ hơn tỉ lệ lỗi của mã gốc.
2. Những lỗi còn lại thuộc loại **chỉ lộ ra khi chạy**: kiểu dữ liệu thực của `store.*`, hành vi thật của reducer trong LangGraph, JSON model thật sự trả về, thời gian thực của một chương.
3. Sáu vòng đã tìm ra **42 khiếm khuyết**. Một bộ `pytest` chạy trong hai giây sẽ tìm ra F1, F2, F3, F8, F9 ngay ở lần chạy đầu tiên — nhanh hơn và chắc chắn hơn mọi vòng đọc mã.

Khuyến nghị: dừng rà soát, dựng walking skeleton (GĐ 1, §15.2) với 3 nhân vật và 5 manh mối. Viết `tests/` trước `novel_engine/` cho Tầng 2, vì tầng đó deterministic nên test rẻ và bắt được gần hết. Để `pytest` và một chương thật tìm vòng lỗi tiếp theo.

Ba nguyên tắc cuối:

**NT-15 — Hai bản vá đúng có thể hợp thành một lỗi.** F6 là ví dụ mẫu. Khi sửa hai hàm ở hai mục khác nhau trong cùng một luồng dữ liệu, phải kiểm luồng **từ đầu tới cuối**, không chỉ kiểm từng hàm.

**NT-16 — Không bao giờ dùng `hash()` cho khoá tồn tại lâu hơn một tiến trình.** Python ngẫu nhiên hoá seed mỗi lần khởi động. Mọi khoá đi vào checkpoint, database hay file phải dùng `hashlib`.

**NT-17 — Escalation không đồng nghĩa với dừng.** Nợ manh mối quá hạn 8 chương là quyết định *kế hoạch* (CP-4), không phải lỗi chặn (CP-3). Dừng cả chương vì nó là phản ứng quá tay: manh mối đã treo 8 chương rồi, thêm một chương nữa không làm hỏng gì, trong khi chặn sản xuất thì làm hỏng.

---

## 1. Chẩn đoán: 3 giới hạn gốc + 7 lỗ hổng còn lại của v1

### 1.1 Ba giới hạn gốc — chẩn đoán lại nguyên nhân

Bạn đã gọi tên đúng triệu chứng. Nhưng nguyên nhân gốc thì khác với cách thường được mô tả, và điều này quyết định cách chữa.

**Context Drift** — thường bị quy cho "context window nhỏ". Sai. Ngay cả với cửa sổ 1M token, nhồi toàn bộ 300.000 từ đã viết vào prompt vẫn hỏng, vì hai lý do: (a) *lost in the middle* — độ chính xác truy hồi sụt mạnh ở vùng giữa prompt; (b) văn xuôi là **dữ liệu có nhiễu cao** — một câu "Aeryn nhíu mày" có thể là dấu hiệu nghi ngờ, có thể chỉ là nhịp văn. Nguyên nhân thật là: *sự thật của câu chuyện chưa từng được trích xuất ra khỏi văn xuôi*. Chữa bằng cách **cải đạo văn xuôi thành fact có cấu trúc ngay sau khi viết** (mục 3.4 và 4.1), không phải bằng cách nhét thêm văn xuôi vào prompt.

**Stereotype Flattening** — thường bị quy cho "model nghèo nàn". Sai. Nguyên nhân là **prompt mô tả tính cách bằng tính từ**. "Kiêu ngạo, thông minh, lạnh lùng" là ba tính từ, và mọi nhân vật kiêu-ngạo-thông-minh-lạnh-lùng trong dữ liệu huấn luyện đều nói giống nhau — model đang đi vào đúng trung tâm phân phối. Chữa bằng cách thay tính từ bằng **ràng buộc hành vi có thể kiểm chứng**: "khi bị chất vấn, nhân vật này không phản bác mà đặt ngược câu hỏi"; "không bao giờ dùng từ chỉ cảm xúc về bản thân"; "câu dài trung bình dưới 9 từ khi tức giận". Tính từ là mô tả, ràng buộc là mã.

**Vi phạm Chekhov's Gun** — thường bị quy cho "quên". Đúng một nửa. Nửa còn lại: kể cả có registry, hệ thống vẫn hỏng vì **không có áp lực lịch trình**. Một manh mối cài ở chương 5 với `payoff_threshold: chapter_22` sẽ nằm im, và ở chương 21 chưa ai đụng tới, rồi chương 22 nó được "trả bài" một cách đột ngột và vô duyên. Chữa bằng **salience decay** (độ hiện diện của manh mối tụt dần theo số chương không được nhắc, buộc hệ thống phải nhắc lại — *re-planting*) và **payoff pressure** (áp lực tăng dần khi tiến gần ngưỡng).

### 1.2 Bảy lỗ hổng v1 chưa đề cập

| # | Lỗ hổng | Hậu quả thực tế | Mục xử lý |
|---|---------|-----------------|-----------|
| L1 | **Không có đường ghi ngược từ Writer về World** | Writer bịa ra tên riêng, địa danh, tập tục; những thứ này không vào graph; chương sau mâu thuẫn | §3.4 |
| L2 | **Không có Continuity Ledger** | Nhân vật bị thương vai trái chương 12, chương 13 vung kiếm bằng tay trái; trời đang đêm, hai đoạn sau là hoàng hôn | §10.1 |
| L3 | **Critique loop không có điểm dừng** | Vòng lặp Writer↔Critic không hội tụ; chi phí một chương phình 8–10× | §9.3 |
| L4 | **Không có POV firewall được cưỡng chế** | Writer biết toàn bộ world state nên vô thức để nhân vật POV "biết" điều chưa được biết — lỗi rò rỉ tri thức, phá nát mọi suspense | §5.4 |
| L5 | **Không có chiến lược xử lý lệch kế hoạch** | Hoặc ép Writer viết lại (mất đoạn hay), hoặc thả nổi (mất cấu trúc) | §11 |
| L6 | **Không có eval** | Mỗi lần chỉnh prompt là một canh bạc; không biết đang tốt lên hay xấu đi | §13 |
| L7 | **Không có human-in-the-loop checkpoint** | Hệ thống chạy 40 chương rồi tác giả mới phát hiện arc sai từ chương 6 | §16 |

---

## 2. Kiến trúc tổng thể v2

Thay đổi quan trọng nhất so với sơ đồ v1: **thêm luồng ghi ngược** (đường `▲`) và tách **Canon Store** thành một tầng độc lập mà mọi tầng khác đều đọc/ghi qua đó.

```
                        ┌──────────────────────────────────┐
                        │        AUTHOR CONSOLE            │
                        │  (Bible, outline, checkpoints,   │
                        │   veto, nhận diff & approve)     │
                        └───────────────┬──────────────────┘
                                        │
╔═══════════════════════════════════════▼══════════════════════════════════╗
║                     TẦNG 1 — CANON STORE (Single Source of Truth)        ║
║  ┌────────────┐  ┌─────────────┐  ┌───────────┐  ┌────────────────────┐  ║
║  │  Neo4j     │  │  Postgres   │  │  Qdrant   │  │  Object Store      │  ║
║  │  World     │  │  Delta Log  │  │  Prose    │  │  Prose thô (L0)    │  ║
║  │  Graph     │  │ (event src) │  │  Vectors  │  │                    │  ║
║  └────────────┘  └─────────────┘  └───────────┘  └────────────────────┘  ║
╚══▲═══════════════════════════════════════════════════════════════════▲═══╝
   │ đọc                                                     ghi delta  │
┌──┴────────────────────────────────────────────────────────────────────┴──┐
│                  TẦNG 2 — DOMAIN ENGINES (thuần tuý, không LLM)          │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ World/Lore   │ │ Character    │ │ Foreshadow   │ │ Relationship     │ │
│ │ GraphRAG     │ │ BDI + Voice  │ │ Clue DAG     │ │ State Machine    │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘ │
│  → Tất cả deterministic, unit-test được, KHÔNG gọi LLM                   │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ cung cấp facts
┌────────────────────────────────▼─────────────────────────────────────────┐
│              TẦNG 3 — CONTEXT ASSEMBLER (ngân sách token)                │
│  Nhận: chapter_id, pov, scene_index                                      │
│  Trả:  SceneContract (JSON) đã lọc theo POV firewall, ≤ N token           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                  TẦNG 4 — AGENT LOOP (LangGraph, có LLM)                 │
│                                                                          │
│   ┌──────────┐   ┌────────┐   ┌─────────┐   ┌────────┐   ┌───────────┐   │
│   │ DIRECTOR │──▶│ WRITER │──▶│ AUDITOR │──▶│ POLISH │──▶│ EXTRACTOR │   │
│   │  (beat)  │   │ (prose)│   │(gate)   │   │(style) │   │ (→ delta) │   │
│   └──────────┘   └───▲────┘   └────┬────┘   └────────┘   └─────┬─────┘   │
│                      │             │ BLOCKER/MAJOR             │         │
│                      └─────────────┘ (tối đa 2 vòng)           │         │
└────────────────────────────────────────────────────────────────┼─────────┘
                                                                 │
┌────────────────────────────────────────────────────────────────▼─────────┐
│                TẦNG 5 — RECONCILIATION & COMMIT                          │
│  Phân loại delta: CONTRADICTION → chặn | ENRICHMENT → ghi canon           │
│  IMPROVEMENT → cập nhật outline & Event DAG hạ nguồn                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Nguyên tắc kiến trúc (bốn điều bất di bất dịch)

**NT-1 — Tầng 2 không được gọi LLM.** Mọi logic thế giới (ai biết gì, manh mối nào đến hạn, quan hệ đang ở trạng thái nào) phải là code thuần. Lý do: nó cần *deterministic* để test được, và cần *rẻ* để chạy hàng nghìn lần. LLM chỉ được dùng ở Tầng 4 để sinh và thẩm định văn xuôi.

**NT-2 — Không agent nào được ghi trực tiếp vào Canon.** Mọi thay đổi đi qua `StateDelta` và phải qua Tầng 5. Điều này cho phép rollback, diff và audit.

**NT-3 — Writer chỉ thấy `SceneContract`.** Writer không có quyền truy vấn graph. Nếu Writer truy vấn được, POV firewall vô nghĩa vì model sẽ vô thức dùng thông tin nó vừa đọc.

**NT-4 — Mọi ràng buộc nghệ thuật phải quy được về một hàm kiểm tra trả `bool` hoặc `float`.** "Đối thoại phải có subtext" là ước nguyện. "Tỉ lệ câu thoại nói thẳng ra mục tiêu của mình ≤ 15%" là ràng buộc. Chỉ cái thứ hai mới vào được hệ thống.
---

## 3. Canon Store — World State theo Event Sourcing

### 3.1 Vì sao không dùng bảng bị ghi đè

Cách làm ngây thơ: giữ một bảng `characters` với cột `location`, mỗi chương thì `UPDATE`. Cách này hỏng vì:

- Không trả lời được "ở chương 14, Kaelen đang ở đâu?" — chỉ biết trạng thái *hiện tại*.
- Không rollback được khi tác giả bảo "bỏ chương 18, viết lại".
- Không diff được: không biết chương 18 đã thay đổi những gì trong thế giới.
- Auditor không có gì để so sánh: nó không biết cái gì *mới*.

Event sourcing giải quyết cả bốn. Canon là hàm fold:

```
WorldState(n) = fold(apply, WorldState(0), [delta_1, delta_2, ..., delta_n])
```

Trong đó `delta_k` là tập thay đổi do chương `k` sinh ra. Neo4j giữ **materialized view** của state hiện tại (để query nhanh), Postgres giữ **delta log** (nguồn sự thật thực sự).

> **Đính chính quan trọng (v2.2):** công thức trên gấp delta theo *thứ tự chương*. Kể từ khi §3.6 tách `epoch_tick` khỏi `narrative_order`, thứ tự chương không còn là thứ tự thời gian thế giới. Log vẫn **ghi thêm** theo thứ tự chương, nhưng world state phải được **fold theo `epoch_tick`** — xem §3.6.4 để biết vì sao và điều đó ràng buộc cảnh hồi ức thế nào.

### 3.2 Schema cốt lõi (Pydantic v2)

```python
# novel_engine/canon/models.py
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Any
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────  ENUMS  ─────────────────────────────

class ClueStatus(str, Enum):
    DRAFTED         = "drafted"          # đã thiết kế, chưa xuất hiện
    PLANTED         = "planted"          # đã cài, độc giả đã thấy
    REINFORCED      = "reinforced"       # đã nhắc lại ≥1 lần
    PARTIALLY_READ  = "partially_read"   # có nhân vật hiểu một phần
    PAID_OFF        = "paid_off"         # đã trả bài
    RETIRED         = "retired"          # bỏ, không dùng nữa (phải ghi lý do)


class NPCRole(str, Enum):
    NARRATIVE_CATALYST  = "narrative_catalyst"   # giữ một mảnh manh mối
    THEMATIC_MIRROR     = "thematic_mirror"      # phản chiếu góc khuất của main
    FUTURE_CONSEQUENCE  = "future_consequence"   # hành động với NPC dội ngược sau
    AMBIENT             = "ambient"              # chỉ tạo texture — CÓ HẠN NGẠCH


class RelationStage(str, Enum):
    STRANGERS        = "strangers"
    FRICTION         = "friction"              # GĐ1: va chạm giá trị
    VULNERABILITY    = "vulnerability"         # GĐ2: bộc lộ tổn thương
    TRIAL            = "trial"                 # GĐ3: thử thách lòng tin
    CATHARSIS        = "catharsis"             # GĐ4: gắn kết mang tính trả giá
    RUPTURE          = "rupture"               # đổ vỡ (có thể quay lại TRIAL)
    SEVERED          = "severed"               # chấm dứt vĩnh viễn


class Severity(str, Enum):
    BLOCKER = "blocker"   # mâu thuẫn canon / rò rỉ POV → BẮT BUỘC viết lại
    MAJOR   = "major"     # lệch tính cách / manh mối hỏng → nên viết lại
    MINOR   = "minor"     # vấn đề văn phong → để Polish xử lý
    NOTE    = "note"      # ghi nhận, không hành động


# ─────────────────────────────  WORLD  ─────────────────────────────

class Entity(BaseModel):
    """Node nền cho mọi thực thể trong graph."""
    id: str                                   # CHAR_KAELEN, FACT_CHURCH, LOC_FORGE
    kind: Literal["character", "faction", "location", "object",
                  "event", "doctrine", "resource", "clue"]
    name: str
    aliases: list[str] = Field(default_factory=list)
    first_appearance: int | None = None        # số chương
    canon_locked: bool = False                 # True = tác giả chốt, agent cấm sửa
    attributes: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """Cạnh có hướng, có trọng số và có thời hiệu."""
    src: str
    dst: str
    type: Literal[
        "HAS_FEUD_WITH", "DEPENDS_ON", "PRACTICES_HERESY", "ORIGINATED_FROM",
        "CONTROLS", "MEMBER_OF", "OWES_DEBT_TO", "BETRAYED", "PROTECTS",
        "KNOWS_ABOUT", "SUSPECTS", "LOCATED_IN", "EVIDENCE_FOR",
    ]
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    since_chapter: int = 0
    until_chapter: int | None = None            # None = còn hiệu lực
    # Quan hệ này được ai biết? Rỗng = sự thật khách quan chưa ai biết.
    known_by: list[str] = Field(default_factory=list)
    provenance: str = "author"                  # author | extracted_ch12 | inferred
```

### 3.3 Clue — nâng cấp từ schema v1 của bạn

Schema v1 của bạn thiếu bốn trường quan trọng. So sánh:

```python
class Clue(BaseModel):
    clue_id: str
    macro_event_target: str                  # node đích trong Event DAG
    description: str                         # mô tả nội bộ (KHÔNG đưa cho Writer)

    # --- v1 đã có ---
    status: ClueStatus = ClueStatus.DRAFTED
    planted_in_chapter: int | None = None
    payoff_threshold: int                     # chương SỚM NHẤT được trả bài
    understood_by_characters: list[str] = Field(default_factory=list)

    # --- v2 bổ sung ---
    payoff_deadline: int                      # chương MUỘN NHẤT — thiếu trường
                                              # này thì manh mối treo vĩnh viễn
    prerequisites: list[str] = Field(default_factory=list)
                                              # clue_id phải PLANTED trước
    surface_forms: list[str] = Field(default_factory=list)
                                              # 3–5 cách hiện hình khác nhau, để
                                              # re-plant không lặp từ
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
                                              # độ "còn trong trí nhớ độc giả"
    last_touched_chapter: int | None = None
    subtlety_target: float = Field(default=0.6, ge=0.0, le=1.0)
                                              # 0 = đập vào mặt, 1 = gần như ẩn
    retire_reason: str | None = None

    @field_validator("payoff_deadline")
    @classmethod
    def _deadline_after_threshold(cls, v: int, info):
        thr = info.data.get("payoff_threshold")
        if thr is not None and v < thr:
            raise ValueError("payoff_deadline phải ≥ payoff_threshold")
        return v
```

Bốn trường mới giải quyết bốn lỗi cụ thể:

- `payoff_deadline` — nếu thiếu, không có áp lực trả bài, manh mối treo mãi (đây là lỗi Chekhov phổ biến nhất ở hệ sinh tự động).
- `prerequisites` — đảm bảo thứ tự tiết lộ hợp lý: không thể hiểu "con dấu bị ăn mòn bởi axit công nghiệp" trước khi độc giả biết "giáo hội cấm công nghiệp hoá".
- `surface_forms` — nếu không có, mỗi lần re-plant hệ thống sẽ viết lại gần y hệt câu cũ, độc giả thấy lộ liễu.
- `salience` + `last_touched_chapter` — cơ sở cho thuật toán decay ở §6.4.

`description` **không bao giờ** được đưa nguyên văn cho Writer Agent. Writer chỉ nhận một `surface_form` cụ thể kèm chỉ thị cường độ. Đưa description thẳng vào prompt là cách chắc chắn nhất để có một câu văn kiểu "anh chú ý thấy con dấu có vết ăn mòn, điều đó thật đáng ngờ".

### 3.4 StateDelta — luồng ghi ngược (lỗ hổng L1)

Đây là thứ v1 thiếu hoàn toàn. Sau khi chương được viết, một **Extractor Agent** đọc văn xuôi và xuất ra các mệnh đề có cấu trúc.

```python
class Assertion(BaseModel):
    """Một mệnh đề được trích từ văn xuôi vừa viết."""
    subject: str
    predicate: str
    object: str | int | float | bool
    chapter: int
    scene: int
    span: str                      # trích dẫn nguyên văn làm bằng chứng
    confidence: float = Field(ge=0.0, le=1.0)
    # Mệnh đề này là sự thật khách quan, hay chỉ là điều nhân vật TIN?
    epistemic: Literal["objective", "believed_by", "claimed_by"] = "objective"
    holder: str | None = None      # ai tin/ai nói, nếu không phải objective


class PlantEvidence(BaseModel):
    """Bằng chứng một plant_directive đã thực sự xuất hiện trên trang giấy.

    Đây là kênh RIÊNG, không dùng chung với `Assertion`. Hai lý do:
    (1) không gian định danh khác nhau — `Assertion.subject` là thực thể
        trong câu văn, `clue_id` là mã hệ thống; trộn chúng là nguồn của
        một lớp lỗi so khớp luôn-sai;
    (2) quan trọng hơn: PHẦN LỚN manh mối được cài KHÔNG sinh ra assertion
        nào cả. Một con dấu hoen gỉ tả trong bối cảnh không khẳng định sự
        thật mới nào về thế giới, nên Extractor sẽ không xuất assertion cho
        nó — nhưng nó ĐÃ được cài. Nếu đo độ phủ qua assertion, bạn sẽ đếm
        thiếu một cách có hệ thống, và đếm thiếu đúng ở những manh mối tinh
        tế nhất, tức là những cái quan trọng nhất.
    """
    clue_id: str
    scene_id: str
    span: str                       # trích dẫn nguyên văn, bắt buộc
    carrier_used: str               # object | dialogue | setting | behavior
    verified: bool = False          # do verify_spans() đặt, KHÔNG do LLM
    reacted_by: list[str] = Field(default_factory=list)
    concluded_by: list[str] = Field(default_factory=list)
                                    # ai đã RÚT RA KẾT LUẬN — khác với ai
                                    # nhìn thấy; dùng để chặn lộ bài sớm


class StateDelta(BaseModel):
    # NT-13: metadata hệ thống PHẢI có default. LLM không bao giờ sinh
    # `delta_id` hay `created_at`, nên để chúng bắt buộc là đảm bảo
    # `parse_model` ném ValidationError ở lượt trích xuất thứ hai.
    # `delta_id` dẫn xuất tất định (không dùng timestamp) để bộ hồi quy
    # §13.2 chạy cùng seed cho ra cùng id.
    delta_id: str = ""             # code gán sau khi parse: f"d_ch{chapter:03d}"
    chapter: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    new_entities: list[Entity] = Field(default_factory=list)
    new_relations: list[Relation] = Field(default_factory=list)
    retracted_relations: list[Relation] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    clue_transitions: dict[str, ClueStatus] = Field(default_factory=dict)
    relationship_updates: list["RelationshipState"] = Field(default_factory=list)
    plant_evidence: list[PlantEvidence] = Field(default_factory=list)

    def item(self, key: str):
        """Tra một phần tử theo khoá do item_key() sinh (§11). Bản trước gọi
        `delta.item(k)` trong reconcile_node mà không định nghĩa hàm này —
        AttributeError ngay cuối chương đầu tiên."""
        for coll in (self.new_entities, self.new_relations,
                     self.retracted_relations, self.assertions):
            for obj in coll:
                if item_key(obj) == key:
                    return obj
        raise KeyError(key)
    # Kết quả phân loại ở Tầng 5
    classification: dict[str, Literal["contradiction", "enrichment",
                                      "improvement"]] = Field(default_factory=dict)
    committed: bool = False
```

Trường `epistemic` là chi tiết nhỏ nhưng cực kỳ quan trọng. Khi nhân vật Serena nói "Hạm Đội Số 3 đã bị giải tán", hệ thống **không** được ghi vào canon rằng hạm đội đã giải tán. Nó ghi: `claimed_by=Serena`. Nếu Serena đang nói dối, sự thật khách quan vẫn nguyên vẹn, và mâu thuẫn sau này là *chủ ý* chứ không phải lỗi. Thiếu trường này, mọi lời nói dối của nhân vật đều làm ô nhiễm canon — đây là một trong những lỗi ngấm ngầm nhất của hệ thống sinh truyện tự động.

### 3.5 Neo4j — mô hình và truy vấn

```cypher
// ── Khởi tạo constraint ───────────────────────────────────────────
CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE INDEX entity_kind IF NOT EXISTS
  FOR (e:Entity) ON (e.kind);
CREATE INDEX clue_status IF NOT EXISTS
  FOR (c:Clue) ON (c.status);
```

```cypher
// ── Q1: Bối cảnh chính trị quanh một địa điểm, sâu 2 tầng ─────────
//      Dùng cho Director khi dựng beat sheet.
MATCH (loc:Entity {id: $location_id})
MATCH path = (loc)<-[:CONTROLS|LOCATED_IN*1..2]-(f:Entity {kind:'faction'})
OPTIONAL MATCH (f)-[feud:HAS_FEUD_WITH]->(other:Entity)
WHERE feud.since_chapter <= $chapter
  AND (feud.until_chapter IS NULL OR feud.until_chapter > $chapter)
RETURN f.name AS faction,
       collect(DISTINCT {
         with: other.name, intensity: feud.weight
       })[0..3] AS feuds
LIMIT 5;
```

```cypher
// ── Q2: POV FIREWALL — nhân vật POV thực sự biết những gì? ────────
//      Đây là truy vấn quan trọng nhất của toàn hệ thống (§5.4).
MATCH (pov:Entity {id: $pov_id})
// (a) Sự thật nhân vật trực tiếp biết
OPTIONAL MATCH (pov)-[k:KNOWS_ABOUT]->(fact:Entity)
WHERE k.since_chapter <= $chapter
// (b) Sự thật nhân vật nghi ngờ (biết một phần)
OPTIONAL MATCH (pov)-[s:SUSPECTS]->(susp:Entity)
WHERE s.since_chapter <= $chapter
// (c) Manh mối nhân vật đã HIỂU (khác với đã nhìn thấy)
OPTIONAL MATCH (c:Clue)
WHERE $pov_id IN c.understood_by_characters
  AND c.planted_in_chapter <= $chapter
RETURN
  collect(DISTINCT {id: fact.id, name: fact.name,
                    certainty: 'known'})      AS known,
  collect(DISTINCT {id: susp.id, name: susp.name,
                    certainty: 'suspected', conf: s.weight}) AS suspected,
  collect(DISTINCT {id: c.clue_id, form: c.surface_forms[0]}) AS clues_held;
```

```cypher
// ── Q3: Manh mối đến hạn ở chương N ───────────────────────────────
MATCH (c:Clue)
WHERE c.status IN ['planted', 'reinforced', 'partially_read']
  AND c.payoff_deadline <= $chapter + $lookahead
  AND c.payoff_threshold <= $chapter
RETURN c.clue_id, c.payoff_deadline, c.salience,
       (c.payoff_deadline - $chapter) AS slack
ORDER BY slack ASC, c.salience DESC;
```

```cypher
// ── Q4: Kiểm tra mâu thuẫn — quan hệ mới có phá quan hệ cũ không? ─
//      Chạy trong Reconciliation (§11).
MATCH (a:Entity {id: $src})-[r]->(b:Entity {id: $dst})
WHERE r.until_chapter IS NULL
  AND type(r) IN $mutually_exclusive_with   // vd: BETRAYED vs PROTECTS
RETURN type(r) AS existing, r.since_chapter AS since, r.provenance;
```

### 3.6 Timeline Engine — thời gian phi tuyến và Multi-POV song song

Luật `_time_monotonic` ở §10.1 của bản 2.0 là **sai thiết kế**, không chỉ thiếu tính năng. Nó giả định ngầm rằng thứ tự đọc trùng với thứ tự xảy ra. Giả định đó vỡ ngay khi có hồi ức, và vỡ nghiêm trọng hơn với chương song song: chương 14 kể Serena tại Toà Thánh, chương 15 kể Kaelen tại Lò Năng Lượng trong **cùng khoảng thời gian** — luật cũ sẽ dựng cờ `blocker` giả ở mọi chương như vậy.

Nguyên nhân gốc: hệ thống đang dùng **một trục** cho hai đại lượng độc lập.

| Trục | Trả lời câu hỏi | Khoá theo |
|------|-----------------|-----------|
| **Story time** | Sự việc xảy ra khi nào trong thế giới? | `epoch_tick` |
| **Narrative time** | Độc giả biết điều đó ở thời điểm nào? | `narrative_order` |

Tách hai trục không chỉ sửa lỗi — nó mở ra một công cụ tự sự: **khoảng cách giữa hai trục chính là mỉa mai kịch tính (dramatic irony)**, và một khi đo được thì điều khiển được (xem `M12` ở §13).

```python
# novel_engine/canon/timeline.py
from pydantic import BaseModel, Field
from typing import Literal

TICKS_PER_HOUR = 1     # 1 tick = 1 giờ truyện. Đổi granularity DUY NHẤT ở đây;
                       # đừng rải hằng số thời gian ra khắp codebase.

class StoryTime(BaseModel):
    epoch_tick: int                    # trục TUYỆT ĐỐI của thế giới
    duration_ticks: int = 1
    narrative_order: int               # thứ tự độc giả đọc (đơn điệu tăng)
    mode: Literal["present", "flashback", "concurrent", "vision"] = "present"
    anchor_scene: str | None = None    # flashback/concurrent neo vào cảnh nào

    @property
    def end_tick(self) -> int:
        return self.epoch_tick + self.duration_ticks

    def overlaps(self, other: "StoryTime") -> bool:
        return self.epoch_tick < other.end_tick and other.epoch_tick < self.end_tick


class ContinuityFrame(BaseModel):
    """Trạng thái vật lý tại cuối mỗi cảnh — khoá theo epoch, KHÔNG theo chương."""
    scene_id: str
    time: StoryTime
    locations: dict[str, str]              # char_id → location_id
    injuries: dict[str, list[str]]
    possessions: dict[str, list[str]]
    weather: str | None = None
```

### 3.6.1 Kiểm tra liên tục trên trục epoch

Điểm mấu chốt: mọi kiểm tra liên tục phải chạy trên **chuỗi frame của từng nhân vật, sắp theo `epoch_tick`**, chứ không phải trên hai chương đọc kề nhau.

```python
# novel_engine/audit/timeline_rules.py

def character_track(cid: str, frames: list[ContinuityFrame]) -> list[ContinuityFrame]:
    """Dòng đời của MỘT nhân vật, sắp theo thời gian thế giới."""
    return sorted((f for f in frames if cid in f.locations),
                  key=lambda f: (f.time.epoch_tick, f.time.narrative_order))


def _no_teleport_epoch(prev, cur, cid, graph) -> list[str]:
    a, b = prev.locations[cid], cur.locations[cid]
    if a == b:
        return []
    gap  = cur.time.epoch_tick - prev.time.end_tick
    need = graph.travel_ticks(a, b)      # tra trên ROUTE graph — KHÔNG phải
                                         # khoảng cách euclid: núi, chốt kiểm
                                         # soát và biển làm hỏng mọi tính toán
                                         # theo bán kính
    if need is None:
        return [f"{cid}: không tồn tại tuyến đường {a} → {b}"]
    if gap < need:
        return [f"{cid}: {a} → {b} cần {need} tick, chỉ có {gap} tick "
                f"(cảnh {prev.scene_id} → {cur.scene_id})"]
    return []


def _no_bilocation(frames: list[ContinuityFrame]) -> list[str]:
    """Một nhân vật không thể ở hai nơi trong hai khoảng epoch chồng lấn.
    Đây là lỗi mà chương song song sinh ra, và luật cũ KHÔNG bắt được."""
    bad = []
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


def _mode_valid(f: ContinuityFrame, anchors: dict) -> list[str]:
    if f.time.mode == "flashback":
        a = anchors.get(f.time.anchor_scene)
        if a is None:
            return [f"{f.scene_id}: flashback không có anchor_scene"]
        if f.time.end_tick > a.time.epoch_tick:
            return [f"{f.scene_id}: hồi ức kết thúc SAU mốc neo — không phải hồi ức"]
    if f.time.mode == "concurrent":
        a = anchors.get(f.time.anchor_scene)
        if a is not None and not f.time.overlaps(a.time):
            return [f"{f.scene_id}: khai là song song nhưng không chồng lấn "
                    f"epoch với {a.scene_id}"]
    return []


def _narrative_monotonic(frames) -> list[str]:
    """Trục ĐỌC vẫn phải đơn điệu — đây là luật duy nhất còn giữ dạng cũ."""
    seen = [f.time.narrative_order for f in frames]
    return ([] if seen == sorted(seen)
            else ["narrative_order không đơn điệu — lỗi đánh số cảnh"])
```

### 3.6.2 Hệ quả bắt buộc: tri thức khoá theo epoch, không theo chương

Đây là hệ quả dễ bỏ sót nhất, và nếu bỏ sót thì mọi cảnh hồi ức đều rò rỉ. Truy vấn POV firewall ở §3.5 (Q2) và mọi chữ ký hàm `known_by(pov_id, chapter)` phải đổi sang `known_by(pov_id, epoch_tick)`.

Lý do rất cụ thể: một cảnh hồi ức ở chương 30 kể về sự kiện tại tick 400, trong khi hiện tại truyện đang ở tick 9.000. Nếu firewall lọc theo `chapter=30`, nhân vật trong hồi ức sẽ "biết" mọi thứ đã học được suốt 8.600 tick sau đó. Bản thân model sẽ không phát hiện ra điều này, vì trong nội bộ cảnh mọi thứ đọc vẫn hợp lý.

```cypher
// Q2′ — thay thế Q2 ở §3.5. Khác biệt duy nhất nhưng quyết định:
//       lọc theo since_tick thay vì since_chapter.
MATCH (pov:Entity {id: $pov_id})
OPTIONAL MATCH (pov)-[k:KNOWS_ABOUT]->(fact:Entity)
WHERE k.since_tick <= $epoch_tick
OPTIONAL MATCH (pov)-[s:SUSPECTS]->(susp:Entity)
WHERE s.since_tick <= $epoch_tick
RETURN collect(DISTINCT {id: fact.id, name: fact.name}) AS known,
       collect(DISTINCT {id: susp.id, name: susp.name, conf: s.weight}) AS suspected;
```

Quy tắc vận hành gọn lại thành một câu: **tri thức nhân vật khoá theo `epoch_tick`; tri thức độc giả khoá theo `narrative_order`; đừng bao giờ trộn hai thứ.**

### 3.6.3 Chọn độ hạt cho tick

1 tick = 1 giờ là điểm khởi đầu tốt cho tiểu thuyết có nhịp ngày-đêm. Ba lưu ý thực tế:

- Đặt tick quá mịn (1 phút) làm mọi `travel_ticks` phải ước lượng chính xác giả tạo, và tác giả sẽ phải điền những con số họ không quan tâm.
- Đặt tick quá thô (1 ngày) làm hỏng các cảnh song song trong cùng một buổi — đúng loại cảnh mà cơ chế này sinh ra để phục vụ.
- Cho phép `duration_ticks` lớn với cảnh kiểu "ba tuần lênh đênh trên biển" thay vì cố chia nhỏ. Khoảng thời gian dài là một khoảng, không phải một chuỗi.

### 3.6.4 Nghịch lý nhân quả hồi ức — và trật tự fold

Tách `epoch_tick` khỏi `narrative_order` ở §3.6 mở ra một mâu thuẫn với §3.1 mà bản 2.1 chưa xử lý. Công thức fold ở §3.1 gấp delta theo **thứ tự chương**, tức theo `narrative_order`. Nhưng trạng thái thế giới là hàm của **thời gian thế giới**. Hai thứ này giờ đã khác nhau.

Kịch bản hỏng cụ thể:

- Chương 10 (`narrative_order=10`, `epoch_tick=5000`): văn bản khẳng định Kaelen có vết sẹo chéo trên trán.
- Chương 15 (`narrative_order=15`) là hồi ức về Kaelen năm 10 tuổi (`epoch_tick=500`). Writer vô tình tả cậu bé *đã có* vết sẹo ấy.
- Fold theo `narrative_order` thì delta của chương 15 được áp *sau*, ghi đè trạng thái tại tick 500. Canon giờ nói vết sẹo có từ tick 500. Không luật nào ở §10.1 phát hiện, vì trong nội bộ mỗi chương mọi thứ đều hợp lý.

Hậu quả không chỉ là một chi tiết sai. Nếu vết sẹo ấy là kết quả của một trận đánh ở tick 3000 mà cốt truyện đã cài làm manh mối, thì manh mối đó vừa bị vô hiệu hoá lặng lẽ.

**Sửa ở hai tầng.**

**Tầng 1 — hai chỉ mục trên cùng một log.** Delta log vẫn *ghi thêm* theo `narrative_order` (đó là lịch sử sáng tác, bất biến, dùng để rollback). Nhưng world state được *fold* theo `epoch_tick`:

$$\text{WorldState}(\tau) = \text{fold}\big(\text{apply},\ \text{WorldState}(0),\ \text{sort}_{\text{epoch}}[\delta_1 \dots \delta_n]\big)$$

Một log, hai trật tự đọc. Chi phí gần như bằng không, vì delta log vốn đã được lưu đầy đủ.

**Tầng 2 — hồi ức chỉ được phép CỘNG THÊM.** Fold theo epoch nghĩa là một delta hồi ức được chèn vào *giữa* lịch sử đã có, nên mọi trạng thái sau nó đều có thể bị thay đổi. Đó chính là mối nguy. Vì vậy delta thuộc cảnh hồi ức chịu ràng buộc đơn điệu: được bổ sung sự thật về quá khứ, **không** được phủ định sự thật đã ghi.

```python
# novel_engine/canon/flashback.py
from pydantic import BaseModel, Field

# Thuộc tính VĨNH VIỄN — một khi có thì không mất, nên thời điểm khởi phát
# của chúng là kiểm chứng được.
PERMANENT_PREDICATES = {
    "has_scar", "is_dead", "lost_limb", "bears_brand", "holds_title",
    "sworn_oath", "is_exiled", "knows_secret",
}

class AttributeWindow(BaseModel):
    """Cửa sổ thời gian mà một thuộc tính vĩnh viễn có thể đã khởi phát."""
    entity: str
    attribute: str
    attested_present: list[int] = Field(default_factory=list)  # tick CÓ bằng chứng
    attested_absent:  list[int] = Field(default_factory=list)  # tick bằng chứng CHƯA có

    @property
    def onset_bounds(self) -> tuple[int | None, int | None]:
        lo = max(self.attested_absent)  if self.attested_absent  else None
        hi = min(self.attested_present) if self.attested_present else None
        return lo, hi

    @property
    def consistent(self) -> bool:
        lo, hi = self.onset_bounds
        return lo is None or hi is None or lo < hi


def flashback_admissible(delta: StateDelta, frame: ContinuityFrame,
                         canon) -> list[dict]:
    """Forward-Reachability Audit: delta hồi ức có phá vỡ tương lai không?"""
    if frame.time.mode not in ("flashback", "vision"):
        return []
    out, tick = [], frame.time.epoch_tick

    # (a) Thuộc tính vĩnh viễn: kiểm tra cửa sổ khởi phát còn hợp lệ không
    for a in delta.assertions:
        if a.predicate not in PERMANENT_PREDICATES:
            continue
        w = canon.attribute_window(a.subject, a.predicate)
        w.attested_present.append(tick)
        if not w.consistent:
            lo, hi = w.onset_bounds
            out.append({
                "severity": "blocker", "check": "flashback_causality",
                "message": (f"hồi ức tại tick {tick} khẳng định "
                            f"{a.subject}.{a.predicate}, nhưng canon đã ghi "
                            f"thuộc tính này CHƯA có tại tick {lo} "
                            f"(muộn hơn) — mâu thuẫn nhân quả"),
                "evidence": a.span,
            })

    # (b) Hồi ức KHÔNG được huỷ quan hệ — huỷ là thao tác của hiện tại
    if delta.retracted_relations:
        out.append({
            "severity": "blocker", "check": "flashback_retraction",
            "message": ("cảnh hồi ức cố huỷ quan hệ "
                        f"{[(r.src, r.type, r.dst) for r in delta.retracted_relations]}"
                        " — hồi ức chỉ được bổ sung, không được phủ định"),
        })

    # (c) Hồi ức không được giết ai còn sống ở hiện tại, và ngược lại
    for a in delta.assertions:
        if a.predicate == "is_dead" and canon.alive_after(a.subject, tick):
            out.append({
                "severity": "blocker", "check": "flashback_mortality",
                "message": f"{a.subject} chết ở tick {tick} nhưng còn sống sau đó",
            })

    # (d) Thực thể mới do hồi ức tạo ra phải khai báo số phận ở hiện tại
    for e in delta.new_entities:
        if e.kind == "character" and not canon.exists(e.id):
            out.append({
                "severity": "note", "check": "flashback_orphan",
                "message": (f"hồi ức giới thiệu nhân vật mới '{e.name}' — "
                            f"cần quyết định người này hiện ở đâu, hoặc "
                            f"đăng ký vào Chekhov Registry (§5.5)"),
            })
    return out
```

Mục (d) đáng chú ý vì nó biến một lỗi tiềm tàng thành một cơ hội: nhân vật xuất hiện trong hồi ức mà không có tung tích ở hiện tại là một khoảng trống độc giả sẽ để ý. Buộc hệ thống quyết định ngay — chết, mất tích, hay vẫn đang ở đâu đó — thường sinh ra tuyến truyện tốt.

**Quy tắc vận hành:** `flashback_admissible` chạy trong `reconcile_node` (§11), **trước** khi ghi canon, và mọi `blocker` ở đây đều đi thẳng tới escalation cho tác giả. Không tự động sửa — hai phiên bản quá khứ mâu thuẫn nhau là quyết định sáng tác, không phải lỗi kỹ thuật.

### 3.7 Khi nào KHÔNG dùng Neo4j

Neo4j hợp lý khi đồ thị đủ lớn (>2.000 node, >8.000 cạnh) hoặc bạn cần truy vấn đường đi biến độ sâu. Cho một tiểu thuyết 60 chương với ~150 thực thể, **SQLite + NetworkX cho kết quả tương đương với 0 chi phí vận hành**. Lời khuyên thực tế: bọc tầng truy cập sau một interface để hoán đổi được.

```python
# novel_engine/canon/graph_port.py
from typing import Protocol

class GraphPort(Protocol):
    def upsert_entity(self, e: Entity) -> None: ...
    def upsert_relation(self, r: Relation) -> None: ...
    def close_relation(self, src: str, dst: str, rel_type: str,
                       until_chapter: int) -> None: ...             # F7
    def commit_belief(self, holder: str, subject: str, predicate: str,
                      object, kind: str, since_tick: int) -> None: ...  # F6
    def known_by(self, pov_id: str, epoch_tick: int) -> dict: ...   # NT-6
    # Forward-Reachability Audit của flashback (§3.6.4) — thiếu ba method này
    # thì `flashback_admissible` không chạy được:
    def exists(self, entity_id: str) -> bool: ...
    def alive_after(self, entity_id: str, tick: int) -> bool: ...
    def attribute_window(self, entity_id: str,
                         attribute: str) -> "AttributeWindow": ...
    def is_locked(self, src: str, dst: str, rel_type: str) -> bool: ...
    def faction_tensions(self, location_id: str, chapter: int) -> list[dict]: ...
    def due_clues(self, chapter: int, lookahead: int = 3) -> list[dict]: ...
```

Bắt đầu bằng `NetworkXGraph`, chuyển sang `Neo4jGraph` khi thật sự chạm trần. Đừng dựng Neo4j ở tuần đầu tiên — nó sẽ ngốn ba ngày setup mà chưa viết được chữ nào.

---

## 4. Narrative Memory Hierarchy — chống Context Drift

### 4.1 Năm tầng nén

Chống drift không phải bằng cách nhớ nhiều hơn, mà bằng cách **nhớ ở đúng độ phân giải**. Chương 37 cần biết *chi tiết* chương 36, cần biết *tóm tắt* chương 30–35, và chỉ cần biết *sự kiện nào đã xảy ra* ở chương 1–10.

| Tầng | Nội dung | Kích thước | Sinh ra khi nào | Dùng cho |
|------|----------|-----------|-----------------|----------|
| **L0** | Văn xuôi thô | ~3.000 từ/chương | Writer xuất bản | Xuất bản, tra cứu nguyên văn |
| **L1** | Scene digest — 5–8 câu/cảnh: ai, ở đâu, làm gì, đổi gì | ~200 từ/chương | Ngay sau Writer | Chương kế tiếp (chi tiết cao) |
| **L2** | Chapter summary — 1 đoạn + danh sách thay đổi trạng thái | ~120 từ/chương | Sau Auditor | Cửa sổ 5 chương gần nhất |
| **L3** | Arc summary — gộp 8–12 chương | ~300 từ/arc | Khi đóng arc | Toàn bộ quá khứ xa |
| **L4** | Standing facts — mệnh đề bất biến đã vào canon | Graph | Extractor | Luôn luôn, qua truy vấn |

Điểm mấu chốt: **L4 không phải văn bản, nó là graph**. Nghĩa là nó không tiêu tốn context theo tuyến tính — Context Assembler chỉ kéo ra đúng những fact liên quan tới cảnh sắp viết.

```python
# novel_engine/memory/hierarchy.py
from dataclasses import dataclass

@dataclass
class MemoryBudget:
    """Ngân sách token cho từng tầng. Tổng phải ≤ ngưỡng an toàn."""
    l1_recent_scenes: int = 1200      # 2 cảnh liền trước, nguyên digest
    l2_recent_chapters: int = 900     # 5 chương gần nhất
    l3_arcs: int = 700                # toàn bộ arc đã đóng
    l4_facts: int = 1500              # fact truy vấn theo cảnh
    character_sheets: int = 1800      # hồ sơ nhân vật có mặt
    clue_directives: int = 400
    style_exemplars: int = 800        # trích đoạn văn phong mẫu (từ Qdrant)

    @property
    def total(self) -> int:
        return sum(v for k, v in self.__dict__.items() if isinstance(v, int))
    # ≈ 7.300 token context + ~2.000 token output → vùng an toàn cho mọi model
```

Ngân sách này là **cứng**. Nếu L4 trả về 3.000 token fact, Context Assembler phải cắt xuống 1.500 theo điểm liên quan, không được nới ngân sách. Nới ngân sách là con đường quay lại đúng vấn đề ban đầu.

### 4.2 Context Assembler

```python
# novel_engine/memory/assembler.py
import tiktoken
from novel_engine.canon.graph_port import GraphPort
from novel_engine.canon.models import Clue

_enc = tiktoken.get_encoding("cl100k_base")

def ntok(s: str) -> int:
    return len(_enc.encode(s))


def truncate_to(items: list[str] | list[tuple[float, str]], budget: int,
                recency_weighted: bool = False) -> list[str]:
    """Cắt theo ngân sách token.

    LỖI ĐÃ SỬA (F1): hàm khai nhận `list[tuple[float, str]]` nhưng `build()`
    truyền thẳng `list[str]` từ store. `-x[0]` trên chuỗi ném
    `TypeError: bad operand type for unary -: 'str'`.

    Nhưng chỉ nhận thêm `list[str]` rồi gán điểm 1.0 cho tất cả thì SAI theo
    một cách tệ hơn cả crash: `sorted` ổn định nên giữ nguyên thứ tự, và khi
    cắt, thứ bị bỏ là phần TỬ CUỐI — mà với tóm tắt chương xếp theo thời gian,
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

    out, used = [], 0
    for _, text in sorted(scored, key=lambda x: -x[0]):
        c = ntok(text)
        if used + c > budget:
            continue          # bỏ qua, thử mục nhỏ hơn — KHÔNG break
        out.append(text)
        used += c
    return out


class ContextAssembler:
    def __init__(self, graph: GraphPort, store, budget: MemoryBudget):
        self.g, self.store, self.b = graph, store, budget

    def build(self, chapter: int, scene_idx: int, pov_id: str,
              present: list[str], location_id: str, epoch_tick: int) -> dict:
        # --- L1: 2 cảnh liền trước, nguyên văn digest ---
        l1 = self.store.recent_scene_digests(chapter, scene_idx, k=2)

        # --- L2: 5 chương gần nhất ---
        l2 = self.store.chapter_summaries(chapter - 5, chapter - 1)

        # --- L3: các arc đã đóng ---
        l3 = self.store.arc_summaries(before_chapter=chapter - 5)

        # --- L4: fact liên quan, ĐÃ LỌC QUA POV FIREWALL ---
        # NT-6: tri thức nhân vật khoá theo epoch_tick. Truyền `chapter` vào
        # đây (như bản trước) khiến Cypher chạy `since_tick <= 14` trong khi
        # thế giới đã ở tick 2400 — POV mù nhận thức hoàn toàn.
        epistemic = self.g.known_by(pov_id, epoch_tick)
        tensions  = self.g.faction_tensions(location_id, chapter)
        facts = self._score_facts(epistemic, tensions, present)

        return {
            # recency_weighted=True cho L1–L3: khi phải cắt, cắt cái CŨ trước
            "recent_scenes":    truncate_to(l1, self.b.l1_recent_scenes, True),
            "recent_chapters":  truncate_to(l2, self.b.l2_recent_chapters, True),
            "arc_history":      truncate_to(l3, self.b.l3_arcs, True),
            "known_facts":      truncate_to(facts, self.b.l4_facts),
            "pov_blindspots":   epistemic["suspected"],   # điều POV chỉ NGHI
        }

    @staticmethod
    def _score_facts(epistemic, tensions, present) -> list[tuple[float, str]]:
        scored = []
        for f in epistemic["known"]:
            # fact liên quan tới nhân vật có mặt được ưu tiên
            rel = 1.0 if f["id"] in present else 0.45
            scored.append((rel, f"[BIẾT] {f['name']}"))
        for s in epistemic["suspected"]:
            scored.append((0.8, f"[NGHI, {s['conf']:.0%}] {s['name']}"))
        for t in tensions:
            scored.append((0.6, f"[THẾ LỰC] {t['faction']}: {t['feuds']}"))
        return scored
```

Chi tiết dễ bỏ sót trong `truncate_to`: dùng `continue` chứ không `break`. Nếu `break`, một fact dài ở giữa danh sách sẽ chặn mọi fact ngắn có điểm thấp hơn — mà những fact ngắn đó có thể rất quan trọng.

### 4.3 Vai trò thực của Vector Store

Bản v1 nói dùng Qdrant cho "ngữ cảnh cục bộ từng chương". Đây là chỗ dễ dùng sai. Vector search trên chính văn xuôi của mình cho kết quả kém, vì mọi chương đều nói về cùng những nhân vật đó nên embedding gần như đồng nhất — truy hồi sẽ trả về các chương ngẫu nhiên.

Vector store nên dùng cho đúng ba việc:

1. **Style exemplars** — kho trích đoạn văn hay (của tác giả khác hoặc của chính bạn đã duyệt), truy hồi theo loại cảnh: "cảnh đối đầu trong không gian hẹp", "cảnh tang lễ". Đây là nguồn chống sáo rỗng hiệu quả nhất.
2. **Chống lặp** — trước khi chốt một đoạn, tìm các đoạn đã viết có similarity > 0.86. Nếu có, model đang tự lặp lại mình (một dạng drift rất hay gặp ở chương 25+).
3. **Tra cứu ngữ nghĩa cho tác giả** — "đoạn nào tôi tả cái lò năng lượng?"

Truy hồi ngữ cảnh cho việc viết thì dùng **graph**, không dùng vector.

```python
# novel_engine/memory/repetition.py
def self_plagiarism_check(new_prose: str, store, threshold: float = 0.86):
    """Trả về các đoạn cũ mà đoạn mới đang lặp lại."""
    hits = []
    for para in split_paragraphs(new_prose):
        if len(para.split()) < 25:      # bỏ qua đoạn quá ngắn, nhiễu cao
            continue
        for match in store.similar(para, top_k=3):
            if match.score >= threshold and match.chapter != CURRENT:
                hits.append({
                    "new": para[:120],
                    "old_chapter": match.chapter,
                    "old": match.text[:120],
                    "score": round(match.score, 3),
                })
    return hits
```
---

## 5. Character Engine — chống Stereotype Flattening

### 5.1 Hồ sơ nhân vật: từ tính từ sang ràng buộc

Năm trường của v1 (Want/Need, Fatal Flaw, Private Agenda, Skill & Blindspot, Private KB) là đúng hướng nhưng chưa đủ để *cưỡng chế*. v2 giữ nguyên năm trường đó và thêm tầng có thể kiểm chứng bằng máy.

```python
# novel_engine/character/models.py
from pydantic import BaseModel, Field
from typing import Literal

class VoiceFingerprint(BaseModel):
    """Ràng buộc ĐỊNH LƯỢNG ở tầng câu chữ. Đây là thứ chống flattening,
    không phải mô tả tính cách."""
    mean_sentence_len: tuple[int, int]          # (min, max) từ/câu
    max_sentence_len: int
    register: Literal["formal", "clipped", "ornate", "vernacular", "clinical"]
    signature_lexicon: list[str]                # 8–15 từ/cụm nhân vật hay dùng
    forbidden_lexicon: list[str]                # từ nhân vật KHÔNG BAO GIỜ dùng
    syntactic_tic: str                          # vd "hay bỏ lửng câu bằng '—'"
    question_ratio: tuple[float, float]         # tỉ lệ câu hỏi trong thoại
    self_reference_rate: float                  # tần suất nói về bản thân
    # Dưới áp lực thì giọng đổi thế nào — đây là chỗ nhân vật trở nên "người"
    under_stress_shift: str                     # vd "câu ngắn lại, chuyển sang
                                                #     tiếng mẹ đẻ, xưng hô đổi"

class Flaw(BaseModel):
    name: str                                   # "kiêu ngạo trí thức"
    # KHÔNG mô tả bằng tính từ — mô tả bằng luật biến dạng quyết định
    distortion_rule: str                        # "khi nhận lời khuyên từ người
                                                #  ít học hơn, luôn tìm lý do
                                                #  kỹ thuật để bác bỏ"
    trigger_conditions: list[str]
    cost_already_paid: list[str] = Field(default_factory=list)
                                                # khuyết điểm này ĐÃ gây hại gì
    arc_direction: Literal["deepens", "heals", "transmutes", "static"]

class BeliefState(BaseModel):
    """Belief trong BDI — tách BIẾT khỏi TIN."""
    proposition: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str                                 # ai/cái gì khiến tin điều này
    is_actually_true: bool | None = None        # None = hệ thống chưa quyết
    # Chênh lệch giữa confidence cao và is_actually_true=False
    # chính là nguồn hiểu lầm TỰ NHIÊN, không cần tình tiết gượng ép.

class Desire(BaseModel):
    label: str
    layer: Literal["want", "need"]              # bề nổi vs nội tâm
    urgency: float = Field(ge=0.0, le=1.0)
    # want và need phải mâu thuẫn — validator kiểm tra ở CharacterProfile
    satisfied_by: list[str] = Field(default_factory=list)
    threatened_by: list[str] = Field(default_factory=list)

class CharacterProfile(BaseModel):
    id: str
    name: str
    role_tier: Literal["protagonist", "major", "supporting", "npc"]

    # ── BDI ──
    beliefs: list[BeliefState]
    desires: list[Desire]
    intentions: list[str]                       # kế hoạch đang theo đuổi

    # ── Năm trường của v1 ──
    want: str
    need: str
    fatal_flaw: Flaw
    private_agenda: str | None                  # toan tính riêng khi đồng hành
    skills: dict[str, float]                    # skill → độ thành thạo 0..1
    blindspots: list[str]                       # vấn đề luôn phán đoán sai
    private_knowledge: list[BeliefState]

    # ── v2 bổ sung ──
    voice: VoiceFingerprint
    somatic_signature: list[str]                # phản ứng cơ thể đặc trưng:
                                                # "siết khớp ngón tay", KHÔNG
                                                # dùng "tim đập nhanh" (sáo)
    moral_line: str                             # điều tuyệt đối không làm
    moral_line_breached: bool = False           # đã vượt lằn ranh chưa
    leverage_over: dict[str, str] = Field(default_factory=dict)
                                                # char_id → thứ mình nắm thóp
    debt_to: dict[str, str] = Field(default_factory=dict)
```

Hai trường có sức nặng bất ngờ:

- **`moral_line`** — mỗi nhân vật có một điều tuyệt đối không làm. Kịch tính thật sự sinh ra khi cốt truyện dồn nhân vật tới chỗ phải chọn giữa lằn ranh đó và thứ họ yêu. Khi `moral_line_breached=True`, Auditor tự động nâng chuẩn: từ chương đó trở đi nhân vật **không được** hành xử như chưa có gì xảy ra.
- **`somatic_signature`** — danh sách phản ứng cơ thể *đặc thù*, kèm danh sách cấm những biểu hiện sáo rỗng dùng chung ("tim đập thình thịch", "mồ hôi lạnh", "nuốt khan"). Đây là đòn bẩy hiệu quả nhất để hai nhân vật cùng kiểu tính cách vẫn đọc ra khác nhau.

### 5.2 Utility-based action selection — làm BDI thật sự chạy

v1 nói "hoạt động dựa trên mô hình BDI" nhưng không nói BDI *sinh ra* cái gì. Nếu không có bước này, BDI chỉ là văn bản trang trí trong prompt và model sẽ bỏ qua nó.

Trước mỗi cảnh, Character Engine **tính toán** (không gọi LLM) lựa chọn mà mỗi nhân vật sẽ nghiêng về, rồi đưa kết quả đó cho Director như một ràng buộc.

```python
# novel_engine/character/deliberation.py
from dataclasses import dataclass

@dataclass
class ActionOption:
    label: str
    serves: list[str]          # desire labels được phục vụ
    threatens: list[str]       # desire labels bị đe doạ
    requires_skill: str | None
    exposes_secret: str | None  # bí mật bị lộ nếu làm
    moral_cost: float = 0.0     # 0..1, 1 = vượt moral_line


def utility(char: CharacterProfile, opt: ActionOption,
            scene_pressure: float) -> tuple[float, str]:
    """Trả (điểm, lý do). Deterministic — unit-test được."""
    u, trace = 0.0, []

    # 1. Want được phục vụ → cộng theo urgency
    for d in char.desires:
        if d.label in opt.serves:
            w = d.urgency * (1.0 if d.layer == "want" else 0.55)
            u += w
            trace.append(f"+{w:.2f} phục vụ {d.layer} '{d.label}'")
        if d.label in opt.threatens:
            w = d.urgency * (1.0 if d.layer == "want" else 0.75)
            u -= w
            trace.append(f"-{w:.2f} đe doạ {d.layer} '{d.label}'")

    # 2. Kỹ năng — thiếu kỹ năng thì giảm sức hấp dẫn
    if opt.requires_skill:
        lvl = char.skills.get(opt.requires_skill, 0.0)
        u += (lvl - 0.5) * 0.6
        trace.append(f"{(lvl-0.5)*0.6:+.2f} kỹ năng {opt.requires_skill}={lvl}")

    # 3. Private agenda — trọng số cao, đây là thứ khiến nhân vật phụ có đời sống riêng
    if char.private_agenda and char.private_agenda in opt.serves:
        u += 0.9
        trace.append("+0.90 phục vụ private agenda")

    # 4. Lộ bí mật → phạt nặng, và phạt nặng hơn khi áp lực cảnh thấp
    #    (dưới áp lực cao người ta liều hơn — đây là chỗ sinh ra hành động bột phát)
    if opt.exposes_secret:
        pen = 1.2 * (1.0 - 0.5 * scene_pressure)
        u -= pen
        trace.append(f"-{pen:.2f} nguy cơ lộ '{opt.exposes_secret}'")

    # 5. Moral line — rào chắn phi tuyến, không phải phạt tuyến tính
    if opt.moral_cost > 0.8 and not char.moral_line_breached:
        u -= 3.0
        trace.append("-3.00 vượt moral_line (chưa từng vượt)")
    elif opt.moral_cost > 0.8:
        u -= 0.8   # đã vượt một lần thì lần sau dễ hơn — đúng tâm lý
        trace.append("-0.80 vượt moral_line (đã từng vượt)")

    # 6. FATAL FLAW — bóp méo quyết định. Đây là điều kiện tiên quyết
    #    của "nhân vật không phải robot".
    u, t = apply_flaw_distortion(char, opt, u)
    trace += t
    return u, " | ".join(trace)


def apply_flaw_distortion(char, opt, u: float) -> tuple[float, list[str]]:
    """Khuyết điểm chí mạng làm lệch utility một cách CÓ HỆ THỐNG."""
    f, trace = char.fatal_flaw, []
    if any(trig in opt.label.lower() for trig in f.trigger_conditions):
        if f.name in ("kiêu ngạo trí thức", "kiêu ngạo"):
            # phương án đòi hỏi nhờ vả/nghe lời người khác bị trừ điểm
            if "nhờ" in opt.label or "nghe theo" in opt.label:
                u -= 1.1
                trace.append("-1.10 flaw: kiêu ngạo, từ chối nhờ vả")
        elif f.name == "sợ bị phản bội":
            if "tin tưởng" in opt.label or "giao" in opt.label:
                u -= 1.3
                trace.append("-1.30 flaw: sợ phản bội, từ chối trao niềm tin")
        elif f.name == "đố kỵ":
            if opt.serves and "nâng đỡ người khác" in opt.serves:
                u -= 1.0
                trace.append("-1.00 flaw: đố kỵ")
    return u, trace


def deliberate(char, options: list[ActionOption], pressure: float) -> dict:
    scored = [(*utility(char, o, pressure), o) for o in options]
    scored.sort(key=lambda x: -x[0])
    best_u, best_trace, best_opt = scored[0]
    runner_u, _, runner_opt = scored[1] if len(scored) > 1 else (None, None, None)
    return {
        "chosen": best_opt.label,
        "why": best_trace,
        # Khoảng cách nhỏ giữa hai lựa chọn = nhân vật đang giằng xé.
        # Director sẽ yêu cầu Writer thể hiện sự do dự này.
        "conflicted": runner_u is not None and (best_u - runner_u) < 0.4,
        "rejected": runner_opt.label if runner_opt else None,
    }
```

Giá trị thực của đoạn code này: nó buộc hệ thống phải **biết trước** nhân vật chọn gì và vì sao, thay vì để model tuỳ hứng. Trường `conflicted` đặc biệt hữu ích — khi hai phương án chênh nhau dưới 0.4, Director chèn chỉ thị "nhân vật gần như chọn X, hãy để dấu vết của lưỡng lự đó trong hành động/thoại".

### 5.3 Cưỡng chế Voice Fingerprint

Có hồ sơ giọng là chưa đủ; phải kiểm tra sau khi viết. Đây là kiểm tra thuần code, rẻ, chạy mọi lần.

```python
# novel_engine/character/voice_check.py
import re, statistics
from collections import Counter

DIALOGUE_RE = re.compile(r'[—"“](.+?)["”\n]')

def extract_dialogue(prose: str, speaker: str) -> list[str]:
    """Trích thoại gán cho một nhân vật. Thực tế nên để Extractor Agent
    gán speaker rồi lưu có cấu trúc — regex chỉ là fallback."""
    lines = []
    for para in prose.split("\n"):
        if speaker.lower() in para.lower() or para.strip().startswith(("—", "“")):
            lines += DIALOGUE_RE.findall(para)
    return lines


def voice_report(prose: str, char: CharacterProfile) -> dict:
    d = extract_dialogue(prose, char.name)
    if not d:
        return {"ok": True, "reason": "không có thoại"}

    lens = [len(s.split()) for s in d]
    mean_len = statistics.mean(lens)
    lo, hi = char.voice.mean_sentence_len
    text = " ".join(d).lower()

    violations = []
    if not (lo <= mean_len <= hi):
        violations.append(
            f"độ dài câu trung bình {mean_len:.1f} ngoài khoảng [{lo},{hi}]")
    if max(lens) > char.voice.max_sentence_len:
        violations.append(f"có câu {max(lens)} từ > trần {char.voice.max_sentence_len}")

    used_forbidden = [w for w in char.voice.forbidden_lexicon if w.lower() in text]
    if used_forbidden:
        violations.append(f"dùng từ cấm: {used_forbidden}")

    sig_hits = sum(1 for w in char.voice.signature_lexicon if w.lower() in text)
    if len(d) >= 6 and sig_hits == 0:
        violations.append("không có dấu vết lexicon đặc trưng trong ≥6 lượt thoại")

    q = sum(1 for s in d if s.strip().endswith("?")) / len(d)
    qlo, qhi = char.voice.question_ratio
    if not (qlo <= q <= qhi):
        violations.append(f"tỉ lệ câu hỏi {q:.2f} ngoài [{qlo},{qhi}]")

    return {
        "ok": not violations,
        "violations": violations,
        "stats": {"mean_len": round(mean_len, 1), "n_lines": len(d),
                  "q_ratio": round(q, 2), "sig_hits": sig_hits},
    }
```

### 5.4 POV Firewall — lỗ hổng nghiêm trọng nhất (L4)

Nếu Writer Agent thấy toàn bộ world state, nó **sẽ** rò rỉ. Không phải vì model kém, mà vì bất cứ thông tin nào trong context đều làm tăng xác suất xuất hiện ở output. Một câu vô hại như "Kaelen không biết rằng phía sau bức tường là xưởng đúc bị niêm phong" đã phá hỏng toàn bộ suspense — vì nó vừa nói ra điều đáng lẽ phải ẩn.

Cưỡng chế bằng ba lớp:

Cưỡng chế bằng ba lớp, cài đặt ở §5.4.1 (bản ở đây trong v2.0–v2.2 có một
lỗi gọi hàm làm lớp 3 thành mã chết, nên đã được thay hoàn toàn):

| Lớp | Lọc gì | Áp lên |
|-----|--------|--------|
| 1 | Bỏ mọi fact POV không biết | ngữ cảnh bộ nhớ (L1–L4) |
| 2 | Điều POV chỉ NGHI → gắn nhãn "phỏng đoán" | ngữ cảnh bộ nhớ |
| 3 | Xoá nội tâm của nhân vật không phải POV | `SceneContract` |

Lớp 1–2 thuộc `filter_memory`, lớp 3 thuộc `filter_scene_contract` — hai hàm
khác nhau vì chúng lọc hai đối tượng khác nhau. Gộp làm một chính là nguyên
nhân của lỗi C4 (§0.3).


Và một bộ kiểm tra sau khi viết, vì lớp lọc prompt vẫn có thể bị model vượt qua:

```python
LEAK_PATTERNS = [
    r"(?:anh|cô|hắn|nàng) không (?:hề )?biết rằng",
    r"(?:điều|thứ) mà .{1,25} (?:không|chưa) (?:hề )?(?:biết|hay)",
    r"trong (?:thâm tâm|đầu|lòng) (?!của (?:mình|tôi|anh ấy)\b)",   # nội tâm người khác
    r"(?:thực ra|sự thật là) .{1,40} đang (?:nói dối|che giấu)",
]

def pov_leak_scan(prose: str, pov_name: str) -> list[dict]:
    hits = []
    for pat in LEAK_PATTERNS:
        for m in re.finditer(pat, prose, flags=re.IGNORECASE):
            ctx = prose[max(0, m.start()-60): m.end()+60]
            if pov_name.lower() in ctx.lower() and "không biết" in ctx:
                continue      # POV tự nhận mình không biết → hợp lệ
            hits.append({"pattern": pat, "span": ctx, "severity": "blocker"})
    return hits
```

Mẫu regex là heuristic, không hoàn hảo, nhưng bắt được phần lớn ca kinh điển với chi phí bằng không. Ca tinh vi thì để Auditor Agent xử lý (§10).

### 5.4.1 Hai bộ lọc, không phải một

Bản trước có một lỗi gọi hàm làm vô hiệu hoá toàn bộ cơ chế này:

```text
ctx = eng.assembler.build(...)                                   # ngữ cảnh L1–L4
ctx = eng.firewall.filter_contract(ctx, c["pov_character"], ...)   ← SAI
prose = STYLIST.invoke(WRITER_TMPL.format(contract=c, context=ctx, ...))
                                          ↑ contract THÔ, chưa qua bộ lọc nào
```

`filter_contract` nhận `ctx` — một dict không có khoá `active_characters` — nên lớp 3 của nó luôn duyệt danh sách rỗng và trở thành mã chết. Còn `SceneContract` (`c`) đi thẳng vào prompt **không qua bộ lọc nào**, mang theo `deliberation`, `secret_fear` và `must_not_reveal` của mọi nhân vật.

Nguyên nhân là một hàm làm hai việc khác nhau. Tách đôi:

```python
# novel_engine/character/firewall.py

class POVFirewall:
    def __init__(self, graph: GraphPort):
        self.g = graph

    def filter_memory(self, ctx: dict, pov_id: str, epoch_tick: int) -> dict:
        """Lọc NGỮ CẢNH BỘ NHỚ (L1–L4). Khoá theo epoch_tick, không theo
        chương (NT-6, §3.6.2)."""
        ep = self.g.known_by(pov_id, epoch_tick)
        allowed = {f["id"] for f in ep["known"]}
        out = dict(ctx)
        out["known_facts"] = [f for f in ctx["known_facts"]
                              if f.get("id") in allowed]
        out["hypotheses"] = [
            {"text": s["name"], "certainty": s["conf"],
             "must_render_as": "phỏng đoán, không phải sự thật"}
            for s in ep["suspected"]]
        return out

    def filter_scene_contract(self, contract: dict, pov_id: str) -> dict:
        """Lọc HỢP ĐỒNG CẢNH. Việc khác hẳn: ở đây ta không lọc sự thật về
        thế giới mà lọc NỘI TÂM của các nhân vật không phải POV."""
        out = dict(contract)
        cleaned = []
        for ch in contract.get("active_characters", []):
            if ch["id"] == pov_id:
                cleaned.append(ch)                  # POV giữ nguyên mọi thứ
                continue

            safe = {k: v for k, v in ch.items() if k in (
                "id", "name", "immediate_goal", "skill_deployed",
                "voice_reminder", "somatic_allowed", "somatic_forbidden")}

            # BỎ HẲN: đây là nội tâm, và là thứ model sẽ tường thuật thẳng
            # nếu nhìn thấy. `deliberation.why` đặc biệt nguy hiểm vì nó là
            # chuỗi giải thích nội bộ kiểu "-1.30 flaw: sợ phản bội" — đưa
            # vào prompt là mời model viết luận về tâm lý nhân vật.
            for k in ("secret_fear", "internal_conflict", "deliberation",
                      "must_not_reveal", "knows_in_this_scene"):
                safe.pop(k, None)

            # GIỮ LẠI có kiểm soát: hành động ngầm. Writer CẦN biết nó,
            # vì dấu vết quan sát được phải là dấu vết CỦA CHÍNH hành động
            # đó. Thay bằng một câu chung chung ("có vẻ đáng ngờ") sẽ cho ra
            # văn xuôi chung chung. Nhưng phải đóng khung rõ ràng:
            if ch.get("hidden_action"):
                safe["director_only"] = {
                    "hidden_action": ch["hidden_action"],
                    "_hard_constraint": (
                        "POV KHÔNG biết và KHÔNG suy ra được điều này. Chỉ "
                        "được gieo MỘT dấu vết vật lý mà POV nhìn thấy nhưng "
                        "hiểu sai hoặc bỏ qua. Cấm mọi câu tường thuật nội "
                        "tâm, ý định hay cảm xúc của nhân vật này."),
                }
            # Chỉ hành động do POV quan sát được mới ở dạng trần
            safe["observable_behavior"] = ch.get("observable_behavior", "")
            cleaned.append(safe)

        out["active_characters"] = cleaned
        return out
```

Và chỗ gọi được sửa thành:

```python
def writer_node(state: ChapterState, cfg) -> dict:
    ...
    ctx    = eng.firewall.filter_memory(ctx, c["pov_character"], epoch_tick)
    c_safe = eng.firewall.filter_scene_contract(c, c["pov_character"])
    prose  = STYLIST.invoke(WRITER_TMPL.format(
        contract=c_safe, context=ctx, feedback=feedback,
        exemplars=eng.style.exemplars_for(c["dramatic_question"], k=2),
    )).content
```

Một điểm tôi giữ khác với cách sửa trực giác: **`hidden_action` vẫn phải đi vào prompt.** Xoá nó và thay bằng "có hành vi đáng ngờ" nghe an toàn hơn nhưng sẽ làm hỏng chính cơ chế mà nó bảo vệ — Writer không thể gieo một dấu vết *cụ thể* của một hành động mà nó không biết là hành động gì. Thứ phải xoá là `deliberation` và `secret_fear`: chúng không tạo ra dấu vết nào, chúng chỉ mời model viết thẳng nội tâm ra.

Kiểm chứng bằng test, không bằng niềm tin:

```python
def test_contract_khong_ro_ri_noi_tam():
    c = {"pov_character": "CHAR_A", "active_characters": [
        {"id": "CHAR_A", "name": "A", "deliberation": {"why": "..."}},
        {"id": "CHAR_B", "name": "B", "secret_fear": "sợ bị lộ",
         "deliberation": {"why": "-1.30 flaw"}, "hidden_action": "gắn thiết bị"},
    ]}
    out = POVFirewall(FakeGraph()).filter_scene_contract(c, "CHAR_A")
    b = next(x for x in out["active_characters"] if x["id"] == "CHAR_B")
    assert "secret_fear" not in b and "deliberation" not in b
    assert b["director_only"]["hidden_action"] == "gắn thiết bị"
    a = next(x for x in out["active_characters"] if x["id"] == "CHAR_A")
    assert "deliberation" in a          # POV giữ nguyên
```

### 5.5 Chekhov's Registry — sửa lại cho mềm hơn

Quy tắc v1 ("mọi NPC phải có 1 trong 3 vai trò, nếu không thì cấm tạo") sẽ tạo ra một thế giới nơi mọi người bán rượu đều là đầu mối. Thế giới đó không có cảm giác thật, vì thế giới thật đầy người không liên quan gì cả.

v2 thêm hạng `AMBIENT` **có hạn ngạch** và cơ chế thăng hạng hồi tố.

```python
# novel_engine/character/registry.py

AMBIENT_QUOTA_PER_CHAPTER = 2      # tối đa 2 NPC vô danh mới mỗi chương
REUSE_PRESSURE_THRESHOLD  = 12     # >12 NPC đang sống thì ép tái sử dụng

class ChekhovRegistry:
    def __init__(self, graph: GraphPort):
        self.g = graph

    def request_npc(self, chapter: int, proposed_role: NPCRole,
                    justification: str) -> dict:
        existing = self.g.living_npcs()
        ambient_this_ch = self.g.count_ambient_created(chapter)

        # (a) Ép tái sử dụng khi dân số NPC đã đông
        if len(existing) > REUSE_PRESSURE_THRESHOLD:
            cands = self.g.reusable_npcs(chapter, limit=5)
            if cands:
                return {"granted": False, "reason": "reuse_pressure",
                        "must_use_one_of": cands}

        # (b) AMBIENT bị giới hạn hạn ngạch
        if proposed_role == NPCRole.AMBIENT:
            if ambient_this_ch >= AMBIENT_QUOTA_PER_CHAPTER:
                return {"granted": False, "reason": "ambient_quota_exceeded"}
            return {"granted": True, "role": NPCRole.AMBIENT,
                    "constraint": ("KHÔNG đặt tên riêng. Không quá 2 lượt thoại. "
                                   "Không nắm giữ thông tin nào.")}

        # (c) Ba vai trò có trọng lượng: bắt buộc khai báo ràng buộc hạ nguồn
        if not justification or len(justification) < 30:
            return {"granted": False, "reason": "insufficient_justification"}
        return {"granted": True, "role": proposed_role,
                "obligation": self._obligation_for(proposed_role, chapter)}

    @staticmethod
    def _obligation_for(role: NPCRole, chapter: int) -> dict:
        """Tạo NPC có vai trò = tạo một MÓN NỢ TỰ SỰ phải trả."""
        return {
            NPCRole.NARRATIVE_CATALYST: {
                "must_resurface_by": chapter + 8,
                "kind": "clue_delivery"},
            NPCRole.THEMATIC_MIRROR: {
                "must_resurface_by": chapter + 12,
                "kind": "contrast_echo"},
            NPCRole.FUTURE_CONSEQUENCE: {
                "must_resurface_by": chapter + 15,
                "kind": "consequence_return"},
        }[role]

    def retroactive_promotion(self, chapter: int) -> list[dict]:
        """Nâng hạng một NPC AMBIENT cũ thay vì tạo nhân vật mới.
        Đây là cơ chế tự sự mạnh nhất mà v1 bỏ sót: người bán rượu vô danh
        ở chương 4 hoá ra là nhân chứng của vụ thảm sát — độc giả cảm thấy
        thế giới có chiều sâu vì họ ĐÃ thấy người đó trước khi biết."""
        return self.g.ambient_npcs_with_reuse_potential(before=chapter - 5)
```

`retroactive_promotion` đáng được ưu tiên trong Director: trước khi cho phép tạo `NARRATIVE_CATALYST` mới, luôn hỏi xem có NPC ambient cũ nào dùng được không. Hiệu ứng lên độc giả mạnh hơn nhiều so với việc giới thiệu người mới.

### 5.6 News Dispatcher — lan truyền thông tin & Fog of War

`POVFirewall` ở §5.4 trả lời chiều *cấm*: nhân vật **không** được biết gì. Nó không trả lời chiều *cho phép*: nhân vật biết một sự kiện ở nơi khác **từ lúc nào**. Thiếu vế thứ hai, mọi cạnh `KNOWS_ABOUT` đều phải do tác giả thêm tay — và trong thực tế tác giả sẽ quên, dẫn tới hai lỗi ngược nhau, cả hai đều chết người: nhân vật biết tin quá sớm (phá suspense), hoặc mãi không biết dù tin đã lan khắp nơi (độc giả thấy nhân vật ngu ngốc).

### 5.6.1 Kênh truyền và đồ thị tuyến đường

Mô hình theo **bán kính** không dùng được: một dãy núi, một chốt kiểm soát của giáo hội, hay một eo biển làm hỏng mọi tính toán khoảng cách thẳng. Dùng đồ thị tuyến đường có độ trễ trên từng cạnh.

```python
# novel_engine/world/news.py
from pydantic import BaseModel, Field
from typing import Literal
import heapq, random

class Channel(BaseModel):
    kind: Literal["courier", "signal", "rumor", "ritual", "trade_caravan"]
    latency_multiplier: float          # nhân với latency_ticks của cạnh
    reliability: float                 # xác suất tin tới nơi ở mỗi chặng
    distortion_rate: float             # xác suất nội dung biến dạng mỗi chặng
    interceptable: bool
    controlled_by: str | None = None   # phe kiểm soát kênh — có quyền bịt tin


CHANNELS = {
    "courier":       Channel(kind="courier", latency_multiplier=1.0,
                             reliability=0.92, distortion_rate=0.04,
                             interceptable=True),
    "signal":        Channel(kind="signal", latency_multiplier=0.02,
                             reliability=0.75, distortion_rate=0.01,
                             interceptable=True, controlled_by="FACT_CHURCH"),
    "rumor":         Channel(kind="rumor", latency_multiplier=2.4,
                             reliability=0.99, distortion_rate=0.38,
                             interceptable=False),
    "trade_caravan": Channel(kind="trade_caravan", latency_multiplier=3.0,
                             reliability=0.88, distortion_rate=0.12,
                             interceptable=True),
}


class NewsItem(BaseModel):
    news_id: str
    origin_location: str
    origin_tick: int
    truth: str                                  # nội dung THẬT
    subject_entities: list[str]
    channels: list[str] = Field(default_factory=lambda: ["courier", "rumor"])
    suppressed_by: list[str] = Field(default_factory=list)
    # phiên bản đã biến dạng theo từng nơi — nguồn của hiểu lầm TỰ NHIÊN
    variants: dict[str, dict] = Field(default_factory=dict)
```

### 5.6.2 Biến dạng quan trọng hơn độ trễ

Đây là chỗ tôi mở rộng đề xuất của bạn. Giảm `confidence` theo khoảng cách cho ra "Kaelen tin 40% rằng mình bị truy nã" — một trạng thái nhạt, khó viết thành cảnh. Cái thực sự sinh ra kịch tính là **nội dung bị bóp méo**: Kaelen nghe tin mình *đã chết*, trong khi lệnh thật là *truy nã sống*. Từ đó mọi hành động sai lầm của anh ta đều hợp lý, và độc giả — vốn đã đọc chương 8 — ngồi trên đống lửa.

Vì vậy kênh mang theo `distortion_rate`, và biến dạng được thực hiện bằng **toán tử có tên**, không phải bằng nhiễu ngẫu nhiên.

```python
DISTORTION_OPERATORS = {
    # tên              → mô tả cấu trúc (code chọn, Writer diễn đạt)
    "exaggerate":       "nhân số lượng/quy mô lên 2–5 lần",
    "substitute_agent": "đổi người gây ra sang một phe/nhân vật khác",
    "invert_outcome":   "đảo kết cục: bị bắt ↔ đã chết ↔ trốn thoát",
    "drop_qualifier":   "bỏ điều kiện: 'nếu không đầu hàng sẽ bị truy nã' "
                        "→ 'đã bị truy nã'",
    "merge_with_prior": "trộn với một tin cũ về cùng nhân vật",
    "attribute_motive": "gán động cơ không có trong tin gốc",
}


def distort(payload: dict, rng: random.Random) -> dict:
    op = rng.choice(list(DISTORTION_OPERATORS))
    return {
        "content_ref": payload["content_ref"],
        "applied_ops": payload.get("applied_ops", []) + [op],
        "fidelity": round(payload["fidelity"] * 0.72, 3),
    }
```

Lưu ý về nguyên tắc NT-1 (§2.1): hàm `distort` **không gọi LLM**. Nó chỉ chọn toán tử và ghi lại chuỗi toán tử đã áp dụng. Việc diễn đạt phiên bản méo mó thành lời thoại xảy ra ở Writer, khi `applied_ops` được đưa vào Scene Contract như một chỉ thị. Giữ được ranh giới này thì propagation vẫn deterministic và test được.

### 5.6.3 Thuật toán lan truyền

```python
MAX_FRONTIER = 4          # số phiên bản tối đa một nơi giữ lại cho một tin
CORRECTION_GAIN = 0.25    # mức tăng fidelity tối thiểu để coi là "đính chính"


def propagate(news: NewsItem, graph, upto_tick: int,
              rng: random.Random) -> list[dict]:
    """Lan truyền ĐA MỤC TIÊU. Một nơi có thể nhận NHIỀU phiên bản của cùng
    một tin, ở các thời điểm khác nhau, với độ tin cậy khác nhau.

    LỖI ĐÃ SỬA: bản trước dùng điều kiện Dijkstra tiêu chuẩn
    `if loc in best and best[loc] <= tick: continue`, tức chỉ giữ bản ĐẾN
    SỚM NHẤT. Hậu quả: tin đồn đi nhanh (rumor, fidelity 0,3) tới nơi ở
    tick 100 sẽ vĩnh viễn chặn đoàn sứ giả chính thức (courier, fidelity
    0,95) tới ở tick 150. Nhân vật ở đó không bao giờ được đính chính.

    Đây không phải bài toán đường đi ngắn nhất mà là bài toán đường đi
    PARETO hai mục tiêu: sớm và chính xác. Một bản chỉ bị loại khi có bản
    khác vừa đến sớm hơn (hoặc bằng) VỪA chính xác hơn (hoặc bằng).
    """
    frontier: dict[str, list[tuple[int, float]]] = {}
    arrivals: list[dict] = []
    counter = 0

    def dominated(loc: str, tick: int, fid: float) -> bool:
        return any(t <= tick and f >= fid for t, f in frontier.get(loc, []))

    def admit(loc: str, tick: int, fid: float) -> None:
        keep = [(t, f) for t, f in frontier.get(loc, [])
                if not (tick <= t and fid >= f)]      # bỏ bản bị bản mới trội
        keep.append((tick, fid))
        keep.sort(key=lambda x: (x[0], -x[1]))
        frontier[loc] = keep[:MAX_FRONTIER]           # chặn phình tổ hợp

    for cname in news.channels:
        ch = CHANNELS[cname]
        if ch.controlled_by and ch.controlled_by in news.suppressed_by:
            continue

        pq = [(news.origin_tick, counter, news.origin_location,
               {"content_ref": news.news_id, "applied_ops": [], "fidelity": 1.0})]
        counter += 1

        while pq:
            tick, _, loc, payload = heapq.heappop(pq)
            fid = payload["fidelity"]
            if tick > upto_tick or dominated(loc, tick, fid):
                continue
            admit(loc, tick, fid)

            prior = [a for a in arrivals if a["location"] == loc and a["tick"] < tick]
            best_prior = max((a["payload"]["fidelity"] for a in prior), default=0.0)
            arrivals.append({
                "location": loc, "tick": tick, "channel": cname,
                "payload": payload,
                # Đính chính là một BEAT tự sự, không chỉ là cập nhật dữ liệu.
                # Cảnh "hoá ra anh ta vẫn còn sống" tự nó là một cảnh.
                "is_correction": bool(prior) and fid - best_prior >= CORRECTION_GAIN,
            })

            for edge in graph.routes_from(loc):
                if edge["blocked_by"] and set(edge["blocked_by"]) & set(news.suppressed_by):
                    continue
                if rng.random() > ch.reliability:
                    continue
                nxt = distort(payload, rng) if rng.random() < ch.distortion_rate \
                      else payload
                counter += 1
                heapq.heappush(
                    pq,
                    (tick + int(edge["latency_ticks"] * ch.latency_multiplier),
                     counter, edge["to"], nxt))

    return sorted(arrivals, key=lambda a: a["tick"])
```

### 5.6.4 Bộ lọc quan tâm — tránh phình đồ thị

Cảnh báo quan trọng: chạy thẳng `propagate` cho mọi tin sẽ sinh ra hàng nghìn cạnh `KNOWS_ABOUT` vô dụng và làm chậm mọi truy vấn firewall. Người thật cũng không ghi nhớ mọi tin họ nghe được. Thêm một cổng lọc trước khi ghi canon:

```python
def interest_score(char: CharacterProfile, news: NewsItem, graph) -> float:
    s = 0.0
    for e in news.subject_entities:
        if e == char.id:                                   s += 1.0
        if e in char.leverage_over or e in char.debt_to:   s += 0.5
        if graph.same_faction(char.id, e):                 s += 0.4
        if any(e in d.threatened_by for d in char.desires): s += 0.6
        if e in [b.proposition for b in char.beliefs]:     s += 0.3
    return s


INTEREST_THRESHOLD = 0.5

def commit_arrivals(news, arrivals, graph, characters) -> list[dict]:
    """Chỉ nhân vật ĐỦ QUAN TÂM mới giữ lại tin. Người khác có nghe,
    nhưng không lưu — đúng như đời thật, và giữ đồ thị nhỏ."""
    edges = []
    for a in arrivals:
        for cid in graph.characters_at(a["location"], a["tick"]):
            ch = characters[cid]
            if interest_score(ch, news, graph) < INTEREST_THRESHOLD:
                continue
            edges.append({
                "src": cid, "dst": news.news_id, "type": "KNOWS_ABOUT",
                "since_tick": a["tick"],
                "weight": a["payload"]["fidelity"],
                "version": a["payload"]["applied_ops"],   # rỗng = nghe bản thật
                "provenance": f"propagated:{a['channel']}",
            })
    return edges
```

### 5.6.5 Ba thứ cơ chế này tặng thêm miễn phí

**Beat sheet tự sinh.** `arrivals` cho Director biết chính xác chương nào một tin quan trọng chạm tới nhân vật nào. "Tin đến" là một beat hoàn chỉnh, và nó xuất hiện mà không ai phải nghĩ ra.

```python
def news_beats(chapter_tick_range: tuple[int, int], graph) -> list[dict]:
    lo, hi = chapter_tick_range
    return [{"beat": "news_arrival", "news_id": e["dst"], "character": e["src"],
             "tick": e["since_tick"], "distorted": bool(e["version"])}
            for e in graph.knowledge_edges_between(lo, hi)]
```

**Manh mối biết đi.** Một `NewsItem` có thể đồng thời là `surface_form` của một `Clue`. Khi đó Foreshadow Scheduler (§6) không cần bịa ra cớ để nhắc lại manh mối — tin đồn tự mang nó đi, và mỗi chặng lại méo thêm một chút. Đây là cách cài manh mối tự nhiên nhất mà hệ thống có.

**Thông tin thành tài nguyên chiến lược.** `suppressed_by` và `interceptable` biến việc kiểm soát kênh truyền thành một mục tiêu mà các phe phái tranh giành. Một phe nắm `signal` là một phe có lợi thế cấu trúc — và điều đó cho Director những xung đột không cần bịa.

**Về tính tái lập:** truyền `rng = random.Random(seed)` với seed dẫn xuất từ `news_id`, không phải seed toàn cục. Nếu không, thêm một tin mới ở chương 12 sẽ làm đổi toàn bộ đường lan của các tin cũ, và bộ hồi quy ở §13.2 mất giá trị.

### 5.6.6 Đính chính không tự động được chấp nhận

Sửa được lỗi Pareto ở trên mới chỉ đảm bảo bản tin chính xác **đến nơi**. Cho nhân vật tự động tin nó là một sai lầm khác, và là sai lầm về nhân vật chứ không về thuật toán: người ta không cập nhật niềm tin chỉ vì có thông tin mới, nhất là khi thông tin cũ đã phù hợp với nỗi sợ sẵn có của họ.

```python
CHANNEL_TRUST = {"courier": 0.85, "signal": 0.75, "trade_caravan": 0.55,
                 "rumor": 0.25, "ritual": 0.70}

FLAW_RESISTANCE = {           # khuyết điểm nào làm khó tiếp nhận đính chính
    "sợ bị phản bội": 0.30,   # tin xấu về người khác thì tin ngay, tin tốt thì không
    "kiêu ngạo trí thức": 0.25,  # đã kết luận rồi thì khó rút lại
    "hoài nghi": 0.20,
}

def accept_correction(char: CharacterProfile, old_fid: float,
                      new_fid: float, channel: str) -> tuple[bool, str]:
    trust = CHANNEL_TRUST.get(channel, 0.5)
    resist = FLAW_RESISTANCE.get(char.fatal_flaw.name, 0.0)
    gain = (new_fid - old_fid) * trust
    if gain > 0.15 + resist:
        return True, f"chấp nhận: gain {gain:.2f} > ngưỡng {0.15 + resist:.2f}"
    return False, (f"GIỮ NGUYÊN niềm tin cũ: {char.fatal_flaw.name} làm tăng "
                   f"ngưỡng lên {0.15 + resist:.2f}, tin qua '{channel}' "
                   f"chỉ đạt {gain:.2f}")
```

Nhân vật **từ chối** đính chính không phải lỗi — đó là khắc hoạ tính cách, và thường là nguồn bi kịch tốt nhất mà hệ thống sinh ra được. Nhưng nó phải được **ghi lại**, không phải xảy ra âm thầm: kết quả `accept_correction` đi vào Scene Contract dưới dạng chỉ thị cho Writer ("nhân vật nghe tin cải chính nhưng không tin; hãy để sự cứng đầu đó lộ ra qua hành động, không qua độc thoại giải thích").

Hệ quả cho `BeliefState`: khi nhân vật giữ niềm tin sai dù đã nghe bản đúng, `confidence` của niềm tin cũ **không** giảm về 0 mà chỉ giảm nhẹ, và `is_actually_true=False` vẫn nguyên. Đúng theo §5.1, chênh lệch giữa `confidence` cao và `is_actually_true=False` chính là nguồn hiểu lầm tự nhiên — News Dispatcher giờ sinh ra chênh lệch đó một cách có hệ thống thay vì trông chờ tác giả nghĩ ra.

---

## 6. Foreshadowing Engine — thuật toán rải manh mối

### 6.1 Hình thức hoá bài toán

Cho DAG $G = (C \cup E, A)$ với $C$ là tập manh mối, $E$ là tập sự kiện lớn, $A$ là cạnh phụ thuộc. Mỗi chương $n$, cần chọn tập con $S_n \subseteq C$ để cài, thoả:

1. **Ràng buộc thứ tự** — $\forall c \in S_n$: mọi `prerequisite` của $c$ đã ở trạng thái ≥ `PLANTED`.
2. **Ràng buộc ngưỡng** — không trả bài $c$ trước `payoff_threshold`.
3. **Ràng buộc hạn chót** — mọi $c$ phải `PAID_OFF` hoặc `RETIRED` trước `payoff_deadline`.
4. **Ràng buộc mật độ** — $|S_n| \le 3$ và tổng "trọng lượng chú ý" ≤ ngân sách chương.
5. **Ràng buộc hiện diện** — clue chỉ cài được nếu cảnh có vật mang phù hợp (đồ vật, lời thoại, bối cảnh).
6. **Ràng buộc tri thức** — POV phải *có thể* quan sát được biểu hiện của clue.
7. **Ràng buộc độ hiện diện** — không để `salience` của clue đang sống tụt dưới `SALIENCE_FLOOR`.

Đây là bài toán lập lịch có ràng buộc. Không cần giải tối ưu — một greedy có điểm số là quá đủ và chạy trong micro giây.

### 6.2 Salience decay

Độ "còn trong trí nhớ độc giả" tụt theo số chương không được nhắc:

$$\text{salience}(c, n) = s_0 \cdot e^{-\lambda (n - t_{\text{last}})}$$

với $\lambda \approx 0{,}18$ (nửa đời ≈ 3,8 chương — khớp với nhịp đọc thực tế của tiểu thuyết dài kỳ). Khi salience tụt dưới ngưỡng mà chưa tới hạn trả bài, hệ thống **bắt buộc re-plant** bằng một `surface_form` khác.

### 6.3 Scheduler — code đầy đủ

```python
# novel_engine/foreshadow/scheduler.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
from novel_engine.canon.models import Clue, ClueStatus

DECAY_LAMBDA        = 0.18
SALIENCE_FLOOR      = 0.25    # dưới mức này độc giả đã quên
MAX_PLANTS_PER_CH   = 3
ATTENTION_BUDGET    = 1.8     # tổng "trọng lượng chú ý" một chương chịu được
URGENT_SLACK        = 3       # còn ≤3 chương tới deadline = khẩn cấp


def decay(clue: Clue, chapter: int) -> float:
    if clue.last_touched_chapter is None:
        return 0.0
    gap = chapter - clue.last_touched_chapter
    return clue.salience * math.exp(-DECAY_LAMBDA * gap)


@dataclass
class PlantDirective:
    """Chỉ thị đưa cho Writer. CHÚ Ý: không chứa `description` của clue."""
    clue_id: str
    surface_form: str
    intensity: float            # 0..1 — 0.2 = thoáng qua, 0.9 = nhấn mạnh
    carrier: str                # "object" | "dialogue" | "setting" | "behavior"
    mode: str                   # "plant" | "reinforce" | "payoff"
    instruction: str
    weight: float = 0.6         # trọng lượng chú ý tiêu tốn


class ForeshadowScheduler:
    def __init__(self, graph, clues: dict[str, Clue]):
        self.g, self.clues = graph, clues

    # ── điểm ưu tiên ────────────────────────────────────────────────
    def _score(self, c: Clue, chapter: int) -> float:
        if c.status in (ClueStatus.PAID_OFF, ClueStatus.RETIRED):
            return -1.0

        slack = c.payoff_deadline - chapter

        # LỖI ĐÃ SỬA — DEADLOCK: bản trước trả -1.0 khi slack <= 0, tức là
        # loại vĩnh viễn mọi manh mối đã quá hạn khỏi lịch. Trong khi đó
        # `narrative_debt_report` lại đếm đúng những manh mối ấy là nợ và
        # chặn Planner khi debt_load vượt ngưỡng. Kết quả: hệ thống không
        # được phép tạo cốt truyện mới, mà cũng không có cách nào trả nợ cũ.
        # Khoá cứng hoàn toàn. Quá hạn phải là ưu tiên CAO NHẤT, không phải
        # điều kiện loại trừ.
        if slack < 0:
            return 10.0 + abs(slack) * 2.0

        deadline_pressure = 1.0 / max(slack, 1) ** 0.7

        cur = decay(c, chapter)
        forget_pressure = 0.0
        if c.status in (ClueStatus.PLANTED, ClueStatus.REINFORCED,
                        ClueStatus.PARTIALLY_READ) and cur < SALIENCE_FLOOR:
            forget_pressure = (SALIENCE_FLOOR - cur) * 3.2

        ready = all(
            (self.clues.get(p).status if self.clues.get(p) else
             ClueStatus.DRAFTED) not in (ClueStatus.DRAFTED, ClueStatus.RETIRED)
            for p in c.prerequisites
        )
        if not ready:
            return -1.0

        virgin_urgency = 1.4 if (c.status == ClueStatus.DRAFTED
                                 and slack <= URGENT_SLACK * 2) else 0.0
        return deadline_pressure + forget_pressure + virgin_urgency

    # ── chọn mode ───────────────────────────────────────────────────
    @staticmethod
    def _mode(c: Clue, chapter: int) -> str:
        if c.status == ClueStatus.DRAFTED:
            return "plant"
        if chapter >= c.payoff_threshold and \
           (c.payoff_deadline - chapter) <= URGENT_SLACK:
            return "payoff"
        return "reinforce"

    # ── chọn surface form chưa dùng ─────────────────────────────────
    def _pick_form(self, c: Clue) -> str:
        used = self.g.used_surface_forms(c.clue_id)
        unused = [f for f in c.surface_forms if f not in used]
        return unused[0] if unused else c.surface_forms[-1]

    # ── cường độ: hàm của mode và subtlety_target ───────────────────
    @staticmethod
    def _intensity(c: Clue, mode: str, slack: int) -> float:
        if mode == "payoff":
            return 0.95
        base = 1.0 - c.subtlety_target          # subtlety cao → intensity thấp
        # càng gần deadline càng phải rõ dần — đường cong "làm nóng"
        ramp = max(0.0, (8 - slack) / 8) * 0.35
        return min(0.9, base + ramp)

    # ── API chính ───────────────────────────────────────────────────
    def schedule(self, chapter: int, scene_affordances: dict[str, list[str]],
                 pov_id: str) -> list[PlantDirective]:
        """scene_affordances: carrier -> danh sách vật mang khả dụng trong cảnh,
        vd {"object": ["con dấu", "đèn dầu"], "dialogue": ["Serena"]}"""
        ranked = sorted(
            ((self._score(c, chapter), c) for c in self.clues.values()),
            key=lambda x: -x[0],
        )

        out: list[PlantDirective] = []
        spent = 0.0
        for score, c in ranked:
            if score < 0 or len(out) >= MAX_PLANTS_PER_CH:
                break

            mode = self._mode(c, chapter)
            if mode == "payoff" and chapter < c.payoff_threshold:
                continue                                  # ràng buộc 2

            carrier = self._match_carrier(c, scene_affordances)
            if carrier is None:
                continue                                  # ràng buộc 5

            if not self.g.pov_can_observe(pov_id, c.clue_id, chapter):
                continue                                  # ràng buộc 6

            slack = c.payoff_deadline - chapter
            inten = self._intensity(c, mode, slack)
            w = 0.35 + inten * 0.7
            if spent + w > ATTENTION_BUDGET:               # ràng buộc 4
                continue

            out.append(PlantDirective(
                clue_id=c.clue_id,
                surface_form=self._pick_form(c),
                intensity=round(inten, 2),
                carrier=carrier,
                mode=mode,
                weight=round(w, 2),
                instruction=self._instruction(mode, inten, carrier),
            ))
            spent += w
        return out

    @staticmethod
    def _match_carrier(c: Clue, aff: dict[str, list[str]]) -> str | None:
        for carrier in ("object", "setting", "behavior", "dialogue"):
            if aff.get(carrier):
                return carrier
        return None

    @staticmethod
    def _instruction(mode: str, intensity: float, carrier: str) -> str:
        if mode == "payoff":
            return ("TRẢ BÀI: để nhân vật tự ghép nối, KHÔNG giải thích lại "
                    "chuỗi manh mối cho độc giả. Độc giả phải hiểu trước "
                    "nhân vật khoảng nửa nhịp.")
        if intensity < 0.35:
            return (f"Cài qua {carrier}. Đặt ở VỊ TRÍ KHÔNG NHẤN — giữa một "
                    f"đoạn đang nói chuyện khác. Không có câu nào bình luận "
                    f"về nó. Không có nhân vật nào phản ứng.")
        if intensity < 0.65:
            return (f"Cài qua {carrier}. Một nhân vật để ý nhưng hiểu SAI ý "
                    f"nghĩa, hoặc bị cắt ngang trước khi kịp hỏi.")
        return (f"Cài qua {carrier} và cho nhân vật phản ứng rõ. Được phép "
                f"dừng lại một nhịp, nhưng KHÔNG được kết luận.")
```

### 6.3.1 Van thoát cho manh mối quá hạn

Nâng điểm ưu tiên lên 10.0 mới chỉ mở được một nửa khoá. Một manh mối quá hạn vẫn có thể bị chặn tiếp bởi ba bộ lọc phía sau trong `schedule`: không có `carrier` phù hợp trong cảnh, POV không quan sát được, hoặc `ATTENTION_BUDGET` đã cạn vì một manh mối quá hạn khác. Nếu cả ba tiếp tục chặn qua vài chương, bạn chỉ dời cái khoá đi một bước.

Cần một van thoát có ba nấc, và nấc cuối phải là con người.

```python
FORCE_RESOLVE_AFTER = 3      # quá hạn quá 3 chương thì bỏ qua mọi bộ lọc mềm
ABANDON_AFTER       = 8      # quá 8 chương thì buộc tác giả quyết định

def schedule(self, chapter: int, scene_affordances: dict[str, list[str]],
             pov_id: str) -> tuple[list[PlantDirective], list[dict]]:
    ranked = sorted(((self._score(c, chapter), c) for c in self.clues.values()),
                    key=lambda x: -x[0])
    out, spent, escalations = [], 0.0, []

    for score, c in ranked:
        if score < 0:
            continue                      # `continue`, KHÔNG `break`: một
                                          # clue chưa đủ prerequisite không
                                          # được chặn các clue xếp sau nó
        if len(out) >= MAX_PLANTS_PER_CH:
            break

        overdue = max(0, chapter - c.payoff_deadline)

        # Nấc 3 — bỏ quá lâu: không tự quyết, đẩy lên tác giả (§16.1 CP-3)
        if overdue >= ABANDON_AFTER:
            escalations.append({
                "clue_id": c.clue_id, "overdue_by": overdue,
                "question": "Gia hạn payoff_deadline, hay RETIRE manh mối này?",
                "retire_cost": self._retire_impact(c),
            })
            continue

        mode = self._mode(c, chapter)
        if mode == "payoff" and chapter < c.payoff_threshold:
            continue

        # Nấc 2 — quá hạn vừa: bỏ qua ràng buộc carrier và POV.
        # Manh mối vẫn trả được bài qua lời kể của nhân vật khác, qua thư từ,
        # qua tin đồn đến nơi (§5.6) — không cần POV tận mắt thấy.
        forcing = overdue >= FORCE_RESOLVE_AFTER
        carrier = self._match_carrier(c, scene_affordances) or \
                  ("dialogue" if forcing else None)
        if carrier is None:
            continue
        if not forcing and not self.g.pov_can_observe(pov_id, c.clue_id, chapter):
            continue

        slack = c.payoff_deadline - chapter
        inten = 0.95 if forcing else self._intensity(c, mode, slack)
        w = 0.35 + inten * 0.7

        # Nấc 1 — quá hạn thì được vượt ngân sách chú ý. Một chương hơi nặng
        # vẫn tốt hơn một manh mối chết.
        if spent + w > ATTENTION_BUDGET and not forcing:
            continue

        out.append(PlantDirective(
            clue_id=c.clue_id, surface_form=self._pick_form(c),
            intensity=round(inten, 2), carrier=carrier,
            mode="payoff" if forcing else mode, weight=round(w, 2),
            instruction=self._instruction("payoff" if forcing else mode,
                                          inten, carrier)))
        spent += w

    return out, escalations
```

`_retire_impact` cho tác giả biết cái giá của việc bỏ manh mối: những manh mối nào khác phụ thuộc vào nó qua `prerequisites`, và sự kiện lớn nào mất một chân.

```python
def _retire_impact(self, c: Clue) -> dict:
    dependents = [x.clue_id for x in self.clues.values()
                  if c.clue_id in x.prerequisites]
    return {"orphaned_clues": dependents,
            "macro_event_at_risk": c.macro_event_target,
            "chapters_since_planted": c.planted_in_chapter}
```

Quy tắc chung rút ra: **mọi ngưỡng chặn trong hệ thống phải có đường thoát.** `ATTENTION_BUDGET`, `MAX_PLANTS_PER_CH`, `debt_load`, `MAX_REVISIONS` — mỗi cái đều là một cái van, và một cái van không có đường xả thì sớm muộn cũng thành cái khoá. Khi rà soát, hãy tìm mọi chỗ có `continue` hoặc `return -1` và hỏi: nếu điều kiện này đúng mãi thì sao?

### 6.4 Phát hiện nợ tự sự quá hạn

Một vòng kiểm tra chạy sau mỗi chương, độc lập với Writer:

```python
# novel_engine/foreshadow/debt.py
from novel_engine.foreshadow.scheduler import decay, SALIENCE_FLOOR, URGENT_SLACK
from novel_engine.canon.models import Clue, ClueStatus

def narrative_debt_report(clues: dict[str, Clue], registry_obligations: list,
                          chapter: int) -> dict:
    overdue, at_risk, forgotten = [], [], []

    for c in clues.values():
        if c.status in (ClueStatus.PAID_OFF, ClueStatus.RETIRED):
            continue
        slack = c.payoff_deadline - chapter
        if slack < 0:
            overdue.append({"clue": c.clue_id, "overdue_by": -slack,
                            "severity": "blocker"})
        elif slack <= URGENT_SLACK:
            at_risk.append({"clue": c.clue_id, "slack": slack})
        if decay(c, chapter) < SALIENCE_FLOOR and c.status != ClueStatus.DRAFTED:
            forgotten.append({"clue": c.clue_id,
                              "salience": round(decay(c, chapter), 3),
                              "silent_for": chapter - (c.last_touched_chapter or 0)})

    for ob in registry_obligations:
        if ob["must_resurface_by"] < chapter and not ob.get("fulfilled"):
            overdue.append({"npc": ob["npc_id"], "kind": ob["kind"],
                            "overdue_by": chapter - ob["must_resurface_by"],
                            "severity": "major"})

    return {"overdue": overdue, "at_risk": at_risk, "forgotten": forgotten,
            "debt_load": len(overdue) * 2 + len(at_risk)}
```

Khi `debt_load` vượt ngưỡng (gợi ý: 8), Planner **bị chặn** không được giới thiệu cốt truyện phụ mới cho tới khi trả bớt nợ. Đây chính là cơ chế ngăn hiện tượng tiểu thuyết "phình ra rồi không đóng lại được" — lỗi phổ biến nhất của truyện dài kỳ, cả do người lẫn do máy viết.

---

## 7. Relationship & Romance State Machine

### 7.1 Không gian trạng thái ba chiều

Bản v1 dùng hai chỉ số. v2 tách `Ideological_Friction` thành hai vì chúng hành xử khác nhau:

| Chỉ số | Ý nghĩa | Đặc tính |
|--------|---------|----------|
| `intimacy` | Độ thấu hiểu, gắn kết | Tăng chậm, giảm nhanh khi bị phản bội |
| `friction` | Bất đồng giá trị/cách làm | **Có thể hoà giải bằng đối thoại** |
| `stake_conflict` | Xung đột lợi ích cấu trúc | **Không thể hoà giải bằng đối thoại** — chỉ đổi khi hoàn cảnh bên ngoài đổi |

Phân biệt này quan trọng: cãi nhau về phương pháp thì nói chuyện là xong. Còn "nếu anh lật đổ giáo hội thì em mất tất cả" thì có nói bao nhiêu cũng không xong — chỉ có hy sinh hoặc thảm kịch. Gộp hai thứ này vào một chỉ số sẽ dẫn tới những cảnh hoà giải rẻ tiền.

Vùng kịch tính cao nhất: `intimacy` cao **và** `stake_conflict` cao. Hai người hiểu nhau nhất nhưng buộc phải chĩa vũ khí vào nhau — đúng như bạn đã nhận định trong v1.

```python
# novel_engine/relationship/models.py
from pydantic import BaseModel, Field

class RelationshipState(BaseModel):
    a: str
    b: str
    stage: RelationStage = RelationStage.STRANGERS
    intimacy: float = Field(default=0.0, ge=0.0, le=100.0)
    friction: float = Field(default=0.0, ge=0.0, le=100.0)
    stake_conflict: float = Field(default=0.0, ge=0.0, le=100.0)

    # Bất đối xứng — A có thể yêu B nhiều hơn B yêu A. Thiếu điều này,
    # mọi quan hệ đều là quan hệ song phương hoàn hảo, cực kỳ giả.
    intimacy_asym: float = Field(default=0.0, ge=-50.0, le=50.0)

    shared_ordeals: list[str] = Field(default_factory=list)
    unresolved_debts: list[str] = Field(default_factory=list)
    scars: list[str] = Field(default_factory=list)   # tổn thương KHÔNG lành
    stage_entered_chapter: int = 0
    chapters_in_stage: int = 0
    last_counted_chapter: int | None = None   # chống đếm theo cảnh (C7)
    history: list[dict] = Field(default_factory=list)
```

### 7.2 Transition guards — cấm nhảy cóc

```python
# novel_engine/relationship/machine.py

MIN_CHAPTERS_IN_STAGE = {
    RelationStage.STRANGERS:     1,
    RelationStage.FRICTION:      3,   # phải va chạm đủ lâu
    RelationStage.VULNERABILITY: 2,
    RelationStage.TRIAL:         2,
    RelationStage.CATHARSIS:     0,
}

def can_advance(st: RelationshipState, chapter: int) -> tuple[bool, str]:
    """Guard cứng. Đây là thứ ngăn 'người lạ → yêu say đắm' trong 2 chương."""
    if st.chapters_in_stage < MIN_CHAPTERS_IN_STAGE[st.stage]:
        return False, (f"mới ở {st.stage.value} {st.chapters_in_stage} chương, "
                       f"cần ≥{MIN_CHAPTERS_IN_STAGE[st.stage]}")

    if st.stage == RelationStage.STRANGERS:
        # Cần một va chạm thật, không phải "gặp nhau thấy hợp"
        if st.friction < 15:
            return False, "chưa có va chạm giá trị nào đáng kể"
        return True, "đã có bất đồng đủ để bắt đầu"

    if st.stage == RelationStage.FRICTION:
        # GĐ2 đòi hỏi NGHỊCH CẢNH CHUNG, không phải tâm sự tự nguyện.
        # Đây là điểm mà mọi hệ thống sinh truyện đều làm sai: nhân vật
        # ngồi xuống và kể về tuổi thơ. Không ai làm thế.
        if not st.shared_ordeals:
            return False, "chưa cùng trải qua nghịch cảnh nào"
        if st.intimacy < 20:
            return False, f"intimacy {st.intimacy:.0f} < 20"
        return True, "có nghịch cảnh chung + đủ gắn kết để lộ điểm yếu"

    if st.stage == RelationStage.VULNERABILITY:
        # GĐ3 chỉ kích hoạt khi lợi ích thực sự đối đầu
        if st.stake_conflict < 40:
            return False, (f"stake_conflict {st.stake_conflict:.0f} < 40 — "
                           f"chưa có gì thật sự phải đánh đổi")
        if st.intimacy < 45:
            return False, "chưa đủ gắn kết để sự phản bội có sức nặng"
        return True, "lợi ích đối đầu trực diện"

    if st.stage == RelationStage.TRIAL:
        # GĐ4 yêu cầu đã có HY SINH THẬT, và để lại SẸO
        if not st.scars:
            return False, "chưa ai trả giá gì — catharsis sẽ rỗng"
        if st.stake_conflict > 55:
            return False, ("xung đột lợi ích còn quá cao; catharsis lúc này "
                           "là kết thúc cổ tích giả tạo")
        return True, "đã trả giá, xung đột đã hạ xuống mức sống chung được"

    return False, "đã ở trạng thái cuối"


def advance_or_hold(st: RelationshipState, chapter: int) -> dict:
    ok, reason = can_advance(st, chapter)
    order = [RelationStage.STRANGERS, RelationStage.FRICTION,
             RelationStage.VULNERABILITY, RelationStage.TRIAL,
             RelationStage.CATHARSIS]
    if ok and st.stage in order[:-1]:
        nxt = order[order.index(st.stage) + 1]
        return {"action": "advance", "to": nxt, "reason": reason,
                "scene_requirement": SCENE_REQUIREMENT[nxt]}
    return {"action": "hold", "reason": reason,
            "scene_requirement": UNBLOCK_HINT.get(st.stage, "")}


SCENE_REQUIREMENT = {
    RelationStage.FRICTION: (
        "Cảnh phải chứa một BẤT ĐỒNG VỀ PHƯƠNG PHÁP mà cả hai đều có lý. "
        "Không được để một bên rõ ràng đúng."),
    RelationStage.VULNERABILITY: (
        "Điểm yếu phải LỘ RA DO HOÀN CẢNH, không do tự nguyện kể. "
        "Nhân vật phải cố che giấu và thất bại."),
    RelationStage.TRIAL: (
        "Phải có một lựa chọn mà MỌI phương án đều mất mát. "
        "Cấm phương án 'vừa cứu được người vừa giữ được mục tiêu'."),
    RelationStage.CATHARSIS: (
        "Chấp nhận nhau KÈM vết sẹo. Phải có ít nhất một điều vĩnh viễn "
        "không thể lấy lại được, và cả hai đều biết điều đó."),
}

UNBLOCK_HINT = {
    RelationStage.STRANGERS: "cần một cảnh hai người phải hợp tác dù bất đồng",
    RelationStage.FRICTION: "cần một nghịch cảnh buộc hai người phụ thuộc nhau",
    RelationStage.VULNERABILITY: "cần nâng stake_conflict: đưa vào một sự kiện "
                                 "khiến mục tiêu của hai người loại trừ nhau",
    RelationStage.TRIAL: "cần một hy sinh có thật, để lại hậu quả không đảo ngược",
}
```

### 7.3 Cập nhật chỉ số và cơ chế thoái lui

```python
# novel_engine/relationship/dynamics.py

def apply_scene_effects(st: RelationshipState, ev: dict, chapter: int):
    """ev do Extractor sinh sau khi đọc chương."""
    # Gắn kết tăng chậm — và chỉ tăng qua HÀNH ĐỘNG, không qua lời nói
    if ev.get("acted_against_own_interest_for_other"):
        st.intimacy = min(100, st.intimacy + 9)
    if ev.get("shared_silence_or_ordeal"):
        st.intimacy = min(100, st.intimacy + 4)
    if ev.get("verbal_affection_only"):
        st.intimacy = min(100, st.intimacy + 1)   # lời nói gần như vô giá trị

    # Phản bội — giảm mạnh và để lại SẸO vĩnh viễn
    if ev.get("betrayal"):
        st.intimacy = max(0, st.intimacy - 28)
        st.scars.append(f"ch{chapter}: {ev['betrayal']}")
        st.stage = RelationStage.RUPTURE
        st.stage_entered_chapter = chapter
        st.chapters_in_stage = 0

    st.friction       = clamp(st.friction + ev.get("d_friction", 0))
    st.stake_conflict = clamp(st.stake_conflict + ev.get("d_stake", 0))
    st.intimacy_asym += ev.get("d_asym", 0)

    # SẸO KHÔNG LÀNH: mỗi vết sẹo đặt trần cho intimacy tương lai.
    # Đây là cơ chế cốt lõi của "gắn kết mang tính trả giá" ở GĐ4.
    if st.scars:
        cap = 100 - 7 * len(st.scars)
        st.intimacy = min(st.intimacy, cap)

    # LỖI ĐÃ SỬA: bản trước viết `st.chapters_in_stage += 1` ở đây. Hàm này
    # chạy MỘT LẦN MỖI CẢNH, nên với 6 cảnh/chương thì bộ đếm "số chương"
    # chạy nhanh gấp 6 lần thực tế, và mọi guard MIN_CHAPTERS_IN_STAGE ở
    # §7.2 mở sớm gấp 6 lần. Đây chính là nguyên nhân M5 báo nhảy cóc giai
    # đoạn mà nhìn vào guard thì thấy guard hoàn toàn đúng.
    if st.last_counted_chapter != chapter:
        st.chapters_in_stage += 1
        st.last_counted_chapter = chapter


def clamp(v: float, lo=0.0, hi=100.0) -> float:
    return max(lo, min(hi, v))


def apply_relationship_advancement(st: RelationshipState, chapter: int) -> dict:
    """LỖI ĐÃ SỬA: `advance_or_hold` ở §7.2 chỉ TRẢ VỀ một quyết định;
    không hàm nào từng gán `st.stage = nxt`. Hệ quả: một cặp nhân vật tích
    luỹ đủ 100 điểm intimacy, có sẹo, có nghịch cảnh chung — và vẫn mang
    nhãn STRANGERS đến hết truyện.

    Lưu ý về THỜI ĐIỂM gọi: Director cũng gọi `advance_or_hold` khi dựng
    contract, nhưng đó là một CHỈ THỊ (cảnh này nên đạt được gì), tính trên
    trạng thái TRƯỚC khi viết. Hàm này là PHÁN QUYẾT (cảnh đã đạt được gì),
    tính trên trạng thái SAU khi Extractor đã cập nhật chỉ số. Hai lần gọi,
    hai vai trò — đừng gộp.
    """
    decision = advance_or_hold(st, chapter)
    if decision["action"] != "advance":
        return decision

    st.stage = decision["to"]
    st.stage_entered_chapter = chapter
    st.chapters_in_stage = 0
    st.last_counted_chapter = chapter
    st.history.append({"stage": st.stage, "chapter": chapter,
                       "intimacy": st.intimacy, "scars": len(st.scars)})
    return decision


def advance_from_rupture(st: RelationshipState, chapter: int) -> None:
    """RUPTURE có đúng hai lối ra, và không lối nào dẫn thẳng tới CATHARSIS."""
    if st.stage != RelationStage.RUPTURE:
        return
    if st.chapters_in_stage >= 3 and st.intimacy >= 25:
        st.stage = RelationStage.TRIAL          # hàn gắn: quay lại thử thách
    elif st.chapters_in_stage >= 6:
        st.stage = RelationStage.SEVERED
    else:
        return
    st.stage_entered_chapter = chapter
    st.chapters_in_stage = 0
```

Ba lựa chọn thiết kế ở đây đáng chú ý:

- `verbal_affection_only` chỉ cộng 1 điểm, trong khi `acted_against_own_interest_for_other` cộng 9. Điều này ép hệ thống thể hiện tình cảm bằng **hành động có giá**, chứ không bằng những đoạn thoại tỏ tình. Đây là chống sáo rỗng ở tầng cơ chế, mạnh hơn mọi chỉ thị trong prompt.
- Sẹo đặt trần vĩnh viễn cho intimacy. Một quan hệ từng đổ vỡ không bao giờ về được mức nguyên vẹn — đúng như "không phải một kết thúc cổ tích đơn giản" mà v1 nêu ra.
- `RUPTURE` là trạng thái quay lui hợp lệ, và từ RUPTURE chỉ có hai đường: về `TRIAL` (hàn gắn, khó) hoặc sang `SEVERED` (chấm dứt). Không có đường tắt về `CATHARSIS`.
---

## 8. Narrative Planner — Tension Curve & Beat Sheet

### 8.1 Tension Curve Controller

Nhịp truyện không thể phó mặc cho model. Nếu không kiểm soát, LLM có xu hướng đẩy mọi chương lên cao trào — đọc 10 chương là mệt, đọc 40 chương là tê liệt. Cần một **đường cong mục tiêu** và một bộ điều khiển kéo chương về đường cong đó.

```python
# novel_engine/planner/tension.py
import math

def target_tension(chapter: int, total: int, acts: int = 3) -> float:
    """Đường cong mục tiêu 0..1. Dạng răng cưa đi lên:
    mỗi act có cao trào riêng, cao trào sau cao hơn cao trào trước,
    và SAU mỗi cao trào có một vùng trũng để độc giả thở."""
    p = chapter / total
    macro = 0.25 + 0.62 * (p ** 1.35)              # nền đi lên phi tuyến
    act_len = 1.0 / acts
    within = (p % act_len) / act_len               # vị trí trong act, 0..1
    micro = 0.22 * math.sin(math.pi * within ** 1.5)
    # Vùng trũng bắt buộc ngay sau mỗi cao trào act
    if within < 0.12 and chapter > total / acts * 0.5:
        micro -= 0.18
    # Cao trào cuối
    if p > 0.90:
        macro = 0.92 + 0.08 * (p - 0.90) / 0.10
    return max(0.05, min(1.0, macro + micro))


def tension_directive(chapter: int, total: int, measured_prev: float) -> dict:
    tgt = target_tension(chapter, total)
    prev_tgt = target_tension(chapter - 1, total) if chapter > 1 else 0.2
    delta = tgt - measured_prev

    if delta > 0.18:
        mode, note = "escalate", "Chương trước hạ nhiệt hơn kế hoạch — cần leo thang rõ."
    elif delta < -0.18:
        mode, note = "decompress", ("Chương trước quá căng. Chương này PHẢI có "
                                    "vùng lặng: sinh hoạt, hồi ức, hoặc một cảnh "
                                    "hoàn toàn không có nguy hiểm vật lý.")
    else:
        mode, note = "sustain", "Giữ nhịp, đổi LOẠI áp lực thay vì tăng cường độ."

    return {
        "target": round(tgt, 3), "previous_measured": round(measured_prev, 3),
        "mode": mode, "note": note,
        "rising": tgt > prev_tgt,
        # Đổi LOẠI áp lực là kỹ thuật chống bào mòn quan trọng nhất
        "pressure_type": ["physical", "social", "moral", "epistemic",
                          "temporal"][chapter % 5],
    }
```

`pressure_type` xoay vòng là chi tiết nhỏ nhưng hiệu quả lớn: nếu mọi chương đều là nguy hiểm thể chất, độc giả chai lì. Xen kẽ áp lực đạo đức (phải chọn điều sai), áp lực nhận thức (biết quá ít / biết quá muộn) và áp lực thời gian tạo cảm giác căng thẳng đa dạng mà không cần tăng cường độ.

### 8.2 Scene Contract — nâng cấp từ mẫu JSON của bạn

Mẫu JSON v1 của bạn đã rất tốt. v2 bổ sung các trường mà Auditor cần để chấm điểm, và các trường cưỡng chế mà thiếu chúng thì Writer sẽ trôi tự do.

```python
# novel_engine/planner/contract.py
from pydantic import BaseModel, Field

class CharacterInScene(BaseModel):
    id: str
    name: str
    immediate_goal: str
    # v1 đã có
    secret_fear: str | None = None
    internal_conflict: str | None = None
    hidden_action: str | None = None
    skill_deployed: str | None = None
    # v2 bổ sung
    deliberation: dict                    # kết quả từ deliberate() §5.2
    voice_reminder: dict                  # trích từ VoiceFingerprint
    somatic_allowed: list[str]            # phản ứng cơ thể được dùng
    somatic_forbidden: list[str]          # sáo ngữ bị cấm trong cảnh này
    knows_in_this_scene: list[str]        # đã qua POV firewall
    must_not_reveal: list[str]            # tuyệt đối không nói ra

class SceneContract(BaseModel):
    scene_id: str
    chapter: int
    scene_index: int
    time: StoryTime                       # §3.6 — bắt buộc; xem §8.2.1 về
                                          # ai cấp phát trường này (NT-12)
    location: str
    pov_character: str
    pov_knowledge_boundary: list[str]     # POV KHÔNG biết gì — để Writer né

    active_characters: list[CharacterInScene]

    # v2: mục tiêu tự sự đo được
    dramatic_question: str                # câu hỏi cảnh này đặt ra
    scene_must_change: str                # cảnh KHÔNG được kết thúc nguyên trạng
    entry_state: str
    exit_state: str

    tension: dict                         # từ tension_directive()
    relationship_directives: list[dict]   # từ advance_or_hold()
    plant_directives: list[dict]          # từ ForeshadowScheduler
    lore_integration: dict

    # v2: ràng buộc văn phong đo được
    word_budget: tuple[int, int] = (900, 1600)
    forbidden_cliches: list[str] = Field(default_factory=list)
    subtext_requirement: str
    max_explicit_goal_statements: int = 1  # tối đa 1 lần nhân vật nói thẳng
                                           # mục tiêu của mình
    sensory_channels_required: int = 3     # ≥3 giác quan, không chỉ thị giác
```

Ba trường đáng nhấn mạnh:

- **`scene_must_change`** — cảnh nào không thay đổi trạng thái thì là cảnh thừa. Ràng buộc này một mình đã loại bỏ phần lớn các cảnh "đi lại nói chuyện" vô nghĩa mà LLM rất hay sinh ra.
- **`max_explicit_goal_statements`** — chống bệnh "nhân vật tự thuyết minh động cơ". Đặt là 1 hoặc 0.
- **`pov_knowledge_boundary`** — liệt kê rõ điều POV *không* biết, kèm chỉ thị "nếu cần nhắc, chỉ được nhắc qua sự hiểu sai của POV".

### 8.2.1 Cấp phát thời gian cho từng cảnh

`SceneContract` phải mang mốc thời gian, nếu không `writer_node` không có gì để truyền cho POV firewall (NT-6) và `ContinuityFrame` không có `StoryTime` để chốt sổ. Thêm một trường bắt buộc:

```python
class SceneContract(BaseModel):
    scene_id: str
    chapter: int
    scene_index: int
    time: StoryTime                       # ← bắt buộc, §3.6
    location: str
    pov_character: str
    ...
```

Nhưng thêm trường thì phải có ai đó điền nó. Cấp phát thời gian là việc của Director, và là **code thuần** (NT-1) — không hỏi LLM giờ giấc.

```python
# novel_engine/planner/timeline_alloc.py
from novel_engine.canon.timeline import StoryTime

# Ước lượng độ dài theo chức năng beat. Đây là mặc định để hệ thống chạy
# được; tác giả ghi đè ở outline khi cần.
BEAT_DURATION_TICKS = {
    "nhịp thở": 2, "thăm dò": 3, "va chạm": 1, "leo thang": 2,
    "mất kiểm soát": 1, "hậu quả": 4, "phát hiện": 2, "hiểu lầm": 2,
    "rẽ hướng": 3, "hậu chấn": 6, "sinh hoạt": 8, "hồi ức": 0,
    "gắn kết": 4, "mầm mống mới": 2, "đặt cược": 3,
}
DEFAULT_SCENE_TICKS = 3
INTER_CHAPTER_GAP   = 6      # khoảng nghỉ mặc định giữa hai chương


def allocate_scene_times(chapter: int, beats: list[dict], store,
                         overrides: dict[int, dict] | None = None
                         ) -> list[StoryTime]:
    """Nối tiếp thời gian từ cảnh cuối của chương trước. Cảnh hồi ức và cảnh
    song song KHÔNG tiêu thụ thời gian của dòng chính — chúng neo vào chỗ
    khác, nên con trỏ `cursor` không nhích khi gặp chúng."""
    overrides = overrides or {}
    cursor = store.last_epoch_tick(chapter - 1) + INTER_CHAPTER_GAP
    base_order = store.last_narrative_order(chapter - 1) + 1
    out = []

    for i, beat in enumerate(beats):
        ov  = overrides.get(i, {})
        dur = ov.get("duration_ticks",
                     BEAT_DURATION_TICKS.get(beat["function"], DEFAULT_SCENE_TICKS))
        mode = ov.get("mode", "present")

        if mode in ("flashback", "vision"):
            anchor = ov["anchor_scene"]
            start  = ov["epoch_tick"]          # tác giả phải chỉ định mốc quá khứ
        elif mode == "concurrent":
            anchor = ov["anchor_scene"]
            start  = store.epoch_tick_of(anchor)
        else:
            anchor, start = None, cursor
            cursor += dur + ov.get("gap_after", 0)

        out.append(StoryTime(epoch_tick=start, duration_ticks=dur,
                             narrative_order=base_order + i,
                             mode=mode, anchor_scene=anchor))
    return out
```

Ba chi tiết đáng lưu ý:

- **Hồi ức và cảnh song song không nhích con trỏ.** Nếu nhích, một chương có hai cảnh hồi ức sẽ đẩy dòng thời gian hiện tại lùi hoặc nhảy vô lý.
- **`epoch_tick` của hồi ức do tác giả chỉ định**, không suy ra được. "Mười năm trước" là một quyết định sáng tác; hệ thống không có cơ sở để đoán.
- **Cảnh song song lấy mốc từ cảnh neo**, nên `_mode_valid` (§3.6.1) chắc chắn đi qua — ràng buộc được thoả mãn theo thiết kế chứ không nhờ may mắn.

Và `writer_node` lấy mốc từ chính hợp đồng, không phải từ biến trôi nổi:

```python
def writer_node(state: ChapterState, cfg) -> dict:
    c   = state["contracts"][state["scene_index"]]
    eng = cfg["configurable"]["engines"]
    epoch_tick = c["time"]["epoch_tick"]        # ← nguồn duy nhất
    ...
    ctx    = eng.firewall.filter_memory(ctx, c["pov_character"], epoch_tick)
    c_safe = eng.firewall.filter_scene_contract(c, c["pov_character"])
```

### 8.3 Beat Sheet generation

```python
# novel_engine/planner/beats.py
BEAT_ARCHETYPES = {
    "escalate":   ["đặt cược", "va chạm", "leo thang", "mất kiểm soát", "hậu quả"],
    "sustain":    ["nhịp thở", "thăm dò", "phát hiện", "hiểu lầm", "rẽ hướng"],
    "decompress": ["hậu chấn", "sinh hoạt", "hồi ức", "gắn kết", "mầm mống mới"],
}

def build_beats(contract_seed: dict, n: int = 6) -> list[dict]:
    mode = contract_seed["tension"]["mode"]
    arch = BEAT_ARCHETYPES[mode]
    beats = []
    for i in range(n):
        beats.append({
            "index": i,
            "function": arch[i % len(arch)],
            "pov_may_learn": None,      # Director điền, qua POV firewall
            "state_delta_expected": None,
            "clue_slot": None,          # scheduler gán
        })
    # Phân bổ plant directive vào các beat KHÔNG phải cao trào —
    # manh mối cài trong lúc cao trào sẽ bị nuốt mất, độc giả không ghi nhận
    quiet = [b for b in beats if b["function"] in
             ("nhịp thở", "thăm dò", "sinh hoạt", "hậu chấn", "đặt cược")]
    for d, b in zip(contract_seed.get("plant_directives", []), quiet or beats):
        b["clue_slot"] = d
    return beats
```

Nguyên tắc "cài manh mối ở beat yên tĩnh" là kinh nghiệm thực tế quan trọng: trong cảnh hành động dồn dập, độc giả lướt qua chi tiết. Manh mối phải được đặt ở chỗ độc giả đang đọc chậm.

---

## 9. Execution Loop — LangGraph

### 9.1 State

```python
# novel_engine/graph/state.py
from typing import TypedDict, Annotated
import operator

class ChapterState(TypedDict):
    # đầu vào
    chapter: int
    total_chapters: int
    outline_beat: str

    # do Director sinh
    contracts: list[dict]          # SceneContract cho từng cảnh
    beats: list[dict]

    # do Writer sinh
    scene_index: int
    current_draft: str
    # Một mục MỖI CẢNH: {scene_id, prose, digest}. Thay cho `drafts` là
    # list[str] trần — Extractor cần biết span thuộc cảnh nào (C2).
    scene_outputs: Annotated[list[dict], operator.add]
    frames: Annotated[list[dict], operator.add]      # ContinuityFrame (§10.1)

    # do Auditor sinh
    findings: list[dict]
    max_severity: str
    revision_count: int

    # kết quả
    polished: str                  # CHỈ cảnh hiện tại — đừng dùng cho cả chương
    delta: dict
    extraction_report: dict
    clue_escalations: list[dict]   # F5: quyết định CP-4, không chặn sản xuất
    unresolved: Annotated[list[str], operator.add]   # chỉ mục treo, từ SceneClose
    irony_seeds: list[dict]
    escalated: bool
    escalation_reason: str
```

### 9.2 Các node

```python
# novel_engine/graph/nodes.py
from langchain_anthropic import ChatAnthropic
from novel_engine.prompts import (DIRECTOR_TMPL, WRITER_TMPL, AUDITOR_TMPL,
                                 POLISH_TMPL, EXTRACT_DIFF_TMPL,
                                 EXTRACT_EMERGENT_TMPL)

REASONER = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3,
                         max_tokens=4000)
STYLIST  = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.85,
                         max_tokens=4000)


def director_node(state: ChapterState, cfg) -> dict:
    """KHÔNG dùng LLM để quyết định logic. LLM chỉ diễn giải thành beat sheet."""
    eng = cfg["configurable"]["engines"]
    ch  = state["chapter"]

    tension = tension_directive(ch, state["total_chapters"],
                                eng.store.measured_tension(ch - 1))
    rel_dirs = [advance_or_hold(st, ch) for st in eng.rel.active_pairs(ch)]
    debt = narrative_debt_report(eng.clues.all(), eng.registry.obligations(), ch)

    # ── F4: LẬP LỊCH MANH MỐI MỘT LẦN CHO CẢ CHƯƠNG ──────────────────
    # Bản trước gọi schedule() bên trong vòng lặp cảnh. Vì trạng thái manh
    # mối chỉ đổi ở reconcile_node (cuối chương), cả 6 lần gọi trả về CÙNG
    # 3 manh mối, và cả 6 cảnh nhận lệnh cài cùng 3 thứ đó — tổng trọng
    # lượng chú ý 6 × 1,8 = 10,8 so với trần 1,8. Ngoài ra nó vô hiệu hoá
    # luôn cơ chế phân bổ vào "beat yên tĩnh" ở §8.3, vốn là lý do
    # build_beats() nhận plant_directives ngay từ đầu.
    chapter_plants, clue_escalations = eng.foreshadow.schedule(
        ch, eng.planner.chapter_affordances(ch), eng.planner.pov_for(ch, 0))

    beats = build_beats({"tension": tension,
                         "plant_directives": chapter_plants}, n=6)
    times = allocate_scene_times(ch, beats, eng.store,
                                 overrides=eng.planner.time_overrides(ch))

    # F5: escalation KHÔNG dừng chương (NT-17). Nợ manh mối quá 8 chương là
    # quyết định kế hoạch (CP-4), không phải lỗi chặn (CP-3). Nó đi vào báo
    # cáo chương và vào Author Console; sản xuất vẫn chạy tiếp.
    contracts = []
    for si, beat in enumerate(beats):
        pov = eng.planner.pov_for(ch, si)
        present = eng.planner.present_characters(ch, si)

        # Chỉ manh mối được phân RIÊNG cho cảnh này, và kiểm quan sát được
        # theo POV CỦA CHÍNH CẢNH ĐÓ — không phải POV dẫn dắt của chương.
        scene_plants = [d for d in [beat.get("clue_slot")] if d
                        and eng.graph.pov_can_observe(pov, d["clue_id"], ch)]

        chars = []
        for cid in present:
            prof = eng.chars.get(cid)
            opts = eng.planner.action_options(cid, ch, si)
            chars.append(CharacterInScene(
                id=cid, name=prof.name,
                immediate_goal=eng.planner.goal_of(cid, ch, si),
                secret_fear=prof.private_knowledge[0].proposition
                            if prof.private_knowledge else None,
                hidden_action=eng.planner.hidden_action(cid, ch, si),
                deliberation=deliberate(prof, opts, tension["target"]),
                voice_reminder=prof.voice.model_dump(
                    include={"register", "signature_lexicon",
                             "forbidden_lexicon", "syntactic_tic",
                             "mean_sentence_len", "under_stress_shift"}),
                somatic_allowed=prof.somatic_signature,
                somatic_forbidden=CLICHE_SOMATICS,
                knows_in_this_scene=eng.firewall.known_ids(
                    cid, times[si].epoch_tick),
                must_not_reveal=eng.planner.secrets_of(cid, ch),
            ).model_dump())

        raw = SceneContract(
            scene_id=f"CH{ch:03d}_S{si:02d}", chapter=ch, scene_index=si,
            time=times[si],
            location=eng.planner.location(ch, si), pov_character=pov,
            pov_knowledge_boundary=eng.firewall.boundary(
                pov, times[si].epoch_tick),
            active_characters=chars,
            dramatic_question=beat["function"],
            scene_must_change="", entry_state="", exit_state="",
            tension=tension,
            relationship_directives=rel_dirs,
            plant_directives=scene_plants,
            lore_integration=eng.graph.faction_tensions(
                eng.planner.location_id(ch, si), ch),
            subtext_requirement=beat["function"],
            forbidden_cliches=eng.style.recent_cliches(ch, window=3),
        ).model_dump()

        filled = REASONER.invoke(DIRECTOR_TMPL.format(
            contract=raw, beat=beat, debt=debt,
            outline=state["outline_beat"])).content
        contracts.append(merge_json(raw, filled))

    return {"contracts": contracts, "scene_index": 0, "revision_count": 0,
            "clue_escalations": clue_escalations}


def writer_node(state: ChapterState, cfg) -> dict:
    c   = state["contracts"][state["scene_index"]]
    eng = cfg["configurable"]["engines"]
    epoch_tick = c["time"]["epoch_tick"]        # E1: nguồn DUY NHẤT của mốc
                                                # thời gian cảnh này

    ctx = eng.assembler.build(
        chapter=state["chapter"], scene_idx=state["scene_index"],
        pov_id=c["pov_character"],
        present=[x["id"] for x in c["active_characters"]],
        location_id=c["location"], epoch_tick=epoch_tick)
    ctx    = eng.firewall.filter_memory(ctx, c["pov_character"], epoch_tick)
    c_safe = eng.firewall.filter_scene_contract(c, c["pov_character"])  # §5.4.1

    feedback = ""
    if state.get("findings"):
        feedback = render_findings(
            [f for f in state["findings"]
             if f["severity"] in ("blocker", "major")])

    prose = STYLIST.invoke(WRITER_TMPL.format(
        contract=c_safe, context=ctx, feedback=feedback,
        exemplars=eng.style.exemplars_for(c["dramatic_question"], k=2),
    )).content
    # F3: §9.4 có ghi chú về việc này nhưng thân hàm ở đây thì không làm —
    # và thiếu nó thì `after_audit` không bao giờ chạm ngưỡng MAX_REVISIONS,
    # vòng writer↔auditor chạy vô tận. Đúng đắn được nhờ scene_boundary_node
    # reset `findings` về [] ở mỗi ranh giới cảnh.
    has_feedback = bool(state.get("findings"))
    return {"current_draft": prose,
            "revision_count": state.get("revision_count", 0)
                              + (1 if has_feedback else 0)}


def auditor_node(state: ChapterState, cfg) -> dict:
    """Kiểm tra HAI TẦNG: thuật toán trước (rẻ, chắc chắn), LLM sau (đắt, tinh)."""
    eng = cfg["configurable"]["engines"]
    c = state["contracts"][state["scene_index"]]
    prose = state["current_draft"]

    findings = []
    findings += eng.audit.deterministic(prose, c, state["chapter"])  # §10

    # Chỉ gọi LLM nếu tầng thuật toán chưa tìm thấy BLOCKER — tiết kiệm tiền
    if not any(f["severity"] == "blocker" for f in findings):
        llm_out = REASONER.invoke(AUDITOR_TMPL.format(
            prose=prose, contract=c,
            checklist=eng.audit.llm_checklist(c))).content
        findings += parse_findings(llm_out)

    sev_rank = {"blocker": 3, "major": 2, "minor": 1, "note": 0}
    top = max((f["severity"] for f in findings),
              key=lambda s: sev_rank[s], default="note")
    return {"findings": findings, "max_severity": top}


def polish_node(state: ChapterState, cfg) -> dict:
    """CHỈ trau chuốt văn phong. Không đụng vào biến điều khiển vòng lặp —
    việc đó thuộc scene_boundary_node."""
    eng = cfg["configurable"]["engines"]
    minors = [f for f in state["findings"] if f["severity"] in ("minor", "note")]
    out = STYLIST.invoke(POLISH_TMPL.format(
        prose=state["current_draft"], notes=minors,
        voice_sheets=eng.chars.voice_sheets_for(state["contracts"]
                                                [state["scene_index"]]),
        banned=eng.style.global_banlist(),
    )).content
    # Kiểm tra lại sau polish: Polish Agent hay vô tình sửa nội dung
    if eng.audit.content_drifted(state["current_draft"], out, threshold=0.25):
        out = state["current_draft"]     # từ chối bản polish, giữ bản gốc
    return {"polished": out}


def scene_boundary_node(state: ChapterState, cfg) -> dict:
    """Chốt sổ một cảnh. Node này tồn tại vì BỐN lý do, và cả bốn đều là
    những thứ bản trước đánh rơi:

    1. TĂNG `scene_index`. Router của LangGraph là hàm điều kiện thuần —
       nó KHÔNG thay đổi state được. Bản trước không node nào tăng biến này,
       nên đồ thị viết lại Cảnh 0 vô hạn cho tới khi cạn quota.
    2. RESET `revision_count`. Không reset thì Cảnh 0 tốn 2 lượt sửa sẽ
       khiến Cảnh 1 bị escalate ngay ở lỗi `major` đầu tiên.
    3. Sinh SCENE DIGEST (L1, §4.1). §4.1 khai là có, nhưng bản trước không
       node nào sinh ra nó — Context Assembler đọc vào một kho rỗng.
    4. Sinh CONTINUITY FRAME (§10.1). Cũng vậy: các luật liên tục không có
       dữ liệu để chạy.
    """
    eng = cfg["configurable"]["engines"]
    idx = state["scene_index"]
    c   = state["contracts"][idx]
    prose = state["polished"]

    digest = REASONER.invoke(SCENE_DIGEST_TMPL.format(      # §12.4
        scene_id=c["scene_id"], epoch_tick=c["time"]["epoch_tick"],
        planned_duration=c["time"]["duration_ticks"],
        present=[x["name"] for x in c["active_characters"]],
        prose=prose)).content
    close = parse_model(digest, SceneClose)          # §9.5

    # F2: StoryTime do CODE lắp, không do LLM sinh. Model chỉ đóng góp
    # `actual_duration_ticks` — thứ duy nhất phải đọc văn bản mới biết.
    frame = ContinuityFrame(
        scene_id=c["scene_id"],
        time=StoryTime.model_validate(c["time"]).model_copy(
            update={"duration_ticks": close.actual_duration_ticks}),
        **close.continuity.model_dump())

    eng.store.put_scene_digest(state["chapter"], idx, close.digest)
    eng.store.put_frame(frame)

    return {
        "scene_index": idx + 1,
        "revision_count": 0,
        "findings": [],
        "max_severity": "note",
        "scene_outputs": [{"scene_id": c["scene_id"], "prose": prose,
                           "digest": close.digest}],
        "frames": [frame.model_dump()],
        "unresolved": close.unresolved,
    }


def extractor_node(state: ChapterState, cfg) -> dict:
    """Luồng ghi ngược. BA LƯỢT, không phải một — xem §10.5 để biết lý do.

    LỖI ĐÃ SỬA: bản trước dùng `state["polished"]`, vốn chỉ chứa văn xuôi
    của CẢNH VỪA XONG, trong khi `contracts` chứa cả 6 cảnh. `verify_spans`
    đi tìm span của cảnh 1–5 trong văn bản cảnh 6, không thấy, và
    `plan_coverage` xoá sạch manh mối của 5 cảnh đầu. Toàn bộ chương chỉ
    còn lại manh mối của cảnh cuối.
    """
    eng = cfg["configurable"]["engines"]
    scenes = state["scene_outputs"]                  # có scene_id đi kèm
    full_prose = "\n\n".join(s["prose"] for s in scenes)
    contracts  = state["contracts"]

    # LƯỢT 1 — KIỂM TOÁN VI SAI: có SceneContract làm hệ quy chiếu.
    #          Truyền văn xuôi ĐÃ GẮN NHÃN CẢNH, để `PlantEvidence.scene_id`
    #          là dữ liệu chứ không phải phỏng đoán của model.
    labelled = "\n\n".join(f"[{s['scene_id']}]\n{s['prose']}" for s in scenes)
    audited = REASONER.invoke(EXTRACT_DIFF_TMPL.format(
        prose=labelled, contracts=contracts, chapter=state["chapter"])).content

    # LƯỢT 2 — QUÉT PHÁT SINH: KHÔNG đưa contract vào.
    emergent = REASONER.invoke(EXTRACT_EMERGENT_TMPL.format(
        prose=full_prose, chapter=state["chapter"],
        known_entities=eng.graph.entity_index())).content

    delta = merge_extractions(audited, emergent, chapter=state["chapter"])

    # LƯỢT 3 — XÁC MINH SPAN trên TOÀN BỘ văn xuôi của chương.
    delta, rejected = verify_spans(delta, full_prose)

    coverage = plan_coverage(contracts, delta)
    return {"delta": delta.model_dump(),
            "extraction_report": {"rejected_spans": rejected,
                                  "assertions_kept": len(delta.assertions),
                                  **coverage}}
```

### 9.3 Bounded Critique Loop — lỗ hổng L3

```python
# novel_engine/graph/routing.py

MAX_REVISIONS = 2

def after_audit(state: ChapterState) -> str:
    sev = state["max_severity"]
    n   = state["revision_count"]

    if sev == "blocker":
        if n >= MAX_REVISIONS:
            return "escalate"        # KHÔNG lặp vô hạn — đưa lên cho người
        return "revise"

    if sev == "major":
        if n >= 1:
            return "polish"          # major chỉ được một lần viết lại
        return "revise"

    return "polish"                  # minor/note → Polish xử lý


def after_polish(state: ChapterState) -> str:
    """Luôn đi qua scene_boundary — đó là node DUY NHẤT được tăng scene_index
    (NT-9). Router không đổi được state."""
    return "scene_boundary"


def after_scene_boundary(state: ChapterState) -> str:
    # scene_index ĐÃ được tăng trong node, nên so sánh không cộng thêm 1
    if state["scene_index"] < len(state["contracts"]):
        return "next_scene"
    return "extract"
```

Ba quy tắc quan trọng ở đây:

1. **BLOCKER được viết lại tối đa 2 lần** rồi escalate. Nếu model không sửa được sau 2 lần, lần thứ 3 cũng sẽ không sửa được — lặp thêm chỉ đốt tiền.
2. **MAJOR chỉ được viết lại 1 lần.** Lệch tính cách nhẹ tốt hơn là một đoạn văn đã bị mài mòn qua bốn vòng viết lại — văn xuôi qua nhiều vòng revision thường trở nên an toàn và nhạt.
3. **MINOR không bao giờ gây viết lại.** Chúng đi thẳng vào Polish.

### 9.4 Lắp graph

```python
# novel_engine/graph/build.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_chapter_graph():
    g = StateGraph(ChapterState)

    g.add_node("director",  director_node)
    g.add_node("writer",    writer_node)
    g.add_node("auditor",   auditor_node)
    g.add_node("polish",    polish_node)
    g.add_node("scene_boundary", scene_boundary_node)
    g.add_node("extract",   extractor_node)
    g.add_node("reconcile", reconcile_node)
    g.add_node("escalate",  escalate_node)

    g.add_edge(START, "director")
    g.add_edge("director", "writer")
    g.add_edge("writer", "auditor")

    g.add_conditional_edges(
        "auditor", after_audit,
        {"revise": "writer", "polish": "polish", "escalate": "escalate"},
    )
    g.add_edge("polish", "scene_boundary")
    g.add_conditional_edges(
        "scene_boundary", after_scene_boundary,
        {"next_scene": "writer", "extract": "extract"},
    )
    g.add_edge("extract", "reconcile")
    g.add_edge("reconcile", END)
    g.add_edge("escalate", END)

    # Checkpointer bắt buộc: một chương có thể chạy 3–8 phút, và bạn
    # sẽ muốn dừng/tiếp/phát lại. Cũng là cơ sở cho human-in-the-loop.
    return g.compile(
        checkpointer=SqliteSaver.from_conn_string("novel_checkpoints.db"),
        interrupt_before=["reconcile"],     # tác giả duyệt delta trước khi ghi canon
    )
```

Lưu ý `revision_count` phải được tăng trong `writer_node` khi có `feedback`, nếu không vòng lặp `auditor → writer → auditor` sẽ không bao giờ đạt ngưỡng và chạy mãi. Thêm vào cuối `writer_node`:

```python
def writer_node(state: ChapterState, cfg) -> dict:
    ...
    return {"current_draft": prose,
            "revision_count": state["revision_count"] + (1 if feedback else 0)}
```

`interrupt_before=["reconcile"]` là điểm human-in-the-loop quan trọng nhất (lỗ hổng L7): trước khi bất cứ thứ gì được ghi vào canon vĩnh viễn, tác giả thấy diff và có quyền phủ quyết.

### 9.5 Ranh giới JSON — chỗ pipeline gãy sớm nhất

`StateDelta.model_validate_json(raw)` gọi thẳng trên `.content` của LLM là dòng code sẽ hỏng đầu tiên khi bạn chạy thật. Dù prompt có ghi "Chỉ xuất JSON", model vẫn thường bọc đầu ra trong ` ```json … ``` `, đôi khi kèm một câu mở đầu lịch sự. `json.decoder.JSONDecodeError` ném ra giữa `extractor_node` sẽ làm sập cả luồng LangGraph — và sập ở bước cuối cùng, sau khi đã trả tiền cho toàn bộ 6 cảnh.

Ba lớp phòng thủ, theo thứ tự nên áp dụng.

**Lớp 1 — dùng structured output nếu có.** Đây mới là cách sửa thật. Buộc model trả về theo schema (tool use / JSON mode) loại bỏ hẳn lớp lỗi này thay vì dọn dẹp sau. Hai lớp dưới là phòng thủ chiều sâu cho những chỗ không dùng được.

```python
# novel_engine/llm/json_io.py
import re, json
from pydantic import BaseModel, ValidationError

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")

def strip_fences(raw: str) -> str:
    """Lột markdown fence và lời dẫn quanh JSON."""
    raw = (raw or "").strip()
    blocks = FENCE_RE.findall(raw)
    if blocks:
        # lấy khối DÀI NHẤT, không phải khối đầu tiên: model hay mở đầu bằng
        # một khối ví dụ ngắn rồi mới tới kết quả thật
        return max(blocks, key=len).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        return raw[i:j + 1]
    return raw


def parse_model(raw: str, model_cls: type[BaseModel],
                repair_llm=None, max_repair: int = 1) -> BaseModel:
    """Lớp 2 (lột fence) + lớp 3 (một lượt tự sửa)."""
    text = strip_fences(raw)
    last_err = None
    for attempt in range(max_repair + 1):
        try:
            return model_cls.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            if repair_llm is None or attempt == max_repair:
                break
            # Một lượt sửa rẻ hơn nhiều so với viết lại cả chương.
            text = strip_fences(repair_llm.invoke(JSON_REPAIR_TMPL.format(
                schema=json.dumps(model_cls.model_json_schema())[:4000],
                broken=text[:6000], error=str(e)[:1200])).content)
    raise ValueError(f"không parse được {model_cls.__name__}: {last_err}")


JSON_REPAIR_TMPL = """JSON dưới đây không hợp lệ so với schema. Sửa và chỉ
xuất JSON đã sửa, không giải thích, không bọc trong dấu ```.

SCHEMA: {schema}
LỖI: {error}
JSON HỎNG:
{broken}
"""
```

**Lớp 4 — đừng để một ngoại lệ giết cả chương.** Trong một lần chạy 40 chương, sẽ có node ném lỗi vì lý do nào đó. Bọc mọi node để lỗi biến thành escalation thay vì stack trace:

```python
# novel_engine/graph/safety.py
import functools, traceback

def safe_node(fn):
    @functools.wraps(fn)
    def wrapper(state, cfg):
        try:
            return fn(state, cfg)
        except Exception as e:
            return {"escalated": True,
                    "escalation_reason": f"{fn.__name__} lỗi: {type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-2000:]}
    return wrapper
```

Áp dụng cho toàn bộ node ở §9.2, và thêm một cạnh điều kiện kiểm `state.get("escalated")` ngay sau mỗi node. Với checkpointer đã bật (§9.4), một node hỏng chỉ mất công đoạn đó chứ không mất cả chương — bạn sửa rồi chạy tiếp từ checkpoint.

Một lưu ý về `strip_fences`: nó lấy khối dài nhất chứ không phải khối đầu tiên. Điều này quan trọng hơn vẻ ngoài, vì prompt của Extractor có chứa ví dụ JSON, và model thỉnh thoảng nhắc lại ví dụ ấy trước khi xuất kết quả thật. Lấy khối đầu tiên sẽ nuốt phải ví dụ, parse thành công, và ghi một delta rỗng vào canon — đúng kiểu lỗi âm thầm mà NT-5 sinh ra để chặn.

---

## 10. Auditor — kiểm tra bằng thuật toán, không chỉ bằng LLM

Sai lầm thường gặp: giao toàn bộ việc thẩm định cho một LLM Critic. Vấn đề là LLM critic **không đáng tin ở những việc mà code làm hoàn hảo** (đếm, so khớp, tra bảng), và nó tốn tiền. Chia việc theo đúng thế mạnh.

| Kiểm tra | Ai làm | Chi phí | Độ tin cậy |
|----------|--------|---------|-----------|
| Mâu thuẫn canon (vị trí, thời gian, vật phẩm, vết thương) | Code | ~0 | Cao |
| Rò rỉ POV (mẫu hiển nhiên) | Code (regex) | ~0 | Trung bình-cao |
| Voice fingerprint | Code | ~0 | Cao |
| Tự lặp lại | Code (vector) | Thấp | Cao |
| Sáo ngữ trong danh sách cấm | Code | ~0 | Cao |
| Nhịp trần thuật (σ, mệnh đề phụ, burstiness) | Code | ~0 | Cao |
| Xác minh span của mệnh đề trích xuất | Code | ~0 | Rất cao |
| Ngân sách từ, số giác quan, số lần nói thẳng mục tiêu | Code | ~0 | Cao |
| Manh mối quá lộ / quá mờ | LLM | Cao | Trung bình |
| Subtext trong thoại | LLM | Cao | Trung bình |
| Nhân vật có "quá dễ dãi" không | LLM | Cao | Trung bình |
| Cảnh có thay đổi trạng thái không | LLM | Cao | Cao |

### 10.1 Continuity Ledger — lỗ hổng L2

```python
# novel_engine/audit/ledger.py
from novel_engine.canon.timeline import StoryTime, ContinuityFrame
from novel_engine.audit.timeline_rules import (
    character_track, _no_teleport_epoch, _no_bilocation,
    _mode_valid, _narrative_monotonic,
)

# ContinuityFrame nay khoá theo epoch_tick, không theo (chapter, scene) —
# xem §3.6. Các luật dưới đây chạy trên DÒNG ĐỜI của từng nhân vật,
# không phải trên hai chương đọc kề nhau.

def _injury_respected(prev: ContinuityFrame, cur: ContinuityFrame,
                      cid: str) -> list[str]:
    """Vết thương lành theo TICK, không theo số chương. Một hồi ức xen giữa
    không làm vết thương lành thêm chút nào."""
    HEAL_TICKS = {"trầy xước": 24, "vết cắt": 240, "gãy xương": 1440}
    gap = cur.time.epoch_tick - prev.time.end_tick
    bad = []
    for w in prev.injuries.get(cid, []):
        if w in cur.injuries.get(cid, []):
            continue
        need = next((v for k, v in HEAL_TICKS.items() if k in w), 240)
        if gap < need:
            bad.append(f"{cid}: '{w}' biến mất sau {gap} tick, cần ≥{need}")
    return bad


def _items_conserved(prev, cur, cid) -> list[str]:
    lost = set(prev.possessions.get(cid, [])) - set(cur.possessions.get(cid, []))
    return [f"{cid}: mất '{i}' không có sự kiện giải thích" for i in lost]


def check_continuity(frames: list[ContinuityFrame], graph) -> list[dict]:
    """Chạy một lần trên TOÀN BỘ frame đã có, không phải theo cặp kề nhau."""
    out = []
    anchors = {f.scene_id: f for f in frames}

    for msg in _narrative_monotonic(frames):
        out.append({"check": "narrative_order", "severity": "blocker",
                    "message": msg})
    for msg in _no_bilocation(frames):
        out.append({"check": "bilocation", "severity": "blocker",
                    "message": msg})
    for f in frames:
        for msg in _mode_valid(f, anchors):
            out.append({"check": "time_mode", "severity": "blocker",
                        "message": msg})

    cids = {c for f in frames for c in f.locations}
    for cid in cids:
        track = character_track(cid, frames)
        for prev, cur in zip(track, track[1:]):
            for msg in _no_teleport_epoch(prev, cur, cid, graph):
                out.append({"check": "teleport", "severity": "blocker",
                            "message": msg})
            for msg in _injury_respected(prev, cur, cid):
                out.append({"check": "injury", "severity": "major",
                            "message": msg})
            for msg in _items_conserved(prev, cur, cid):
                out.append({"check": "item", "severity": "major",
                            "message": msg})
    return out
```

Ledger này là lý do hệ thống có thể viết đến chương 50 mà nhân vật không "mọc lại tay". Nó rẻ, chạy tức thì, và bắt được đúng loại lỗi mà LLM critic bỏ sót nhiều nhất — vì LLM critic đọc một chương và thấy nó hợp lý *nội tại*, nó không nhớ chương 12.

### 10.2 Contradiction Detection trên graph

```python
# novel_engine/audit/contradiction.py

MUTEX = {
    "BETRAYED":  ["PROTECTS", "TRUSTS"],
    "PROTECTS":  ["BETRAYED"],
    "MEMBER_OF": [],        # có thể thuộc nhiều phe — không mutex
}

def detect_contradictions(delta: StateDelta, graph: GraphPort,
                          chapter: int) -> list[dict]:
    out = []
    for r in delta.new_relations:
        for existing in graph.active_relations(r.src, r.dst, chapter):
            if existing["type"] in MUTEX.get(r.type, []):
                out.append({
                    "severity": "blocker", "check": "relation_mutex",
                    "message": (f"{r.src} -[{r.type}]-> {r.dst} mâu thuẫn với "
                                f"{existing['type']} có từ chương "
                                f"{existing['since_chapter']}"),
                    "resolution_hint": ("nếu đây là bước ngoặt CHỦ Ý, hãy đóng "
                                        "quan hệ cũ bằng until_chapter thay vì "
                                        "thêm quan hệ mới"),
                })

    for a in delta.assertions:
        if a.epistemic != "objective":
            continue        # lời nhân vật nói không đụng vào canon
        clash = graph.conflicting_assertion(a.subject, a.predicate, a.object)
        if clash:
            out.append({
                "severity": "blocker", "check": "fact_conflict",
                "message": (f"'{a.subject} {a.predicate} {a.object}' mâu thuẫn "
                            f"với canon chương {clash['chapter']}: "
                            f"'{clash['object']}'"),
                "evidence_new": a.span, "evidence_old": clash["span"],
            })

    # Thực thể mới chưa từng có — không phải lỗi, nhưng phải được duyệt
    for e in delta.new_entities:
        if not graph.exists(e.id):
            out.append({
                "severity": "note", "check": "new_entity",
                "message": f"Writer tạo thực thể mới: {e.name} ({e.kind})",
                "action": "cần phân loại ENRICHMENT hay CONTRADICTION",
            })
    return out
```

### 10.3 Kiểm tra văn phong bằng code

```python
# novel_engine/audit/prose.py
import re

CLICHE_SOMATICS = [
    "tim đập thình thịch", "tim đập nhanh", "mồ hôi lạnh", "nuốt khan",
    "máu chảy rần rật", "sống lưng lạnh toát", "thở phào nhẹ nhõm",
    "nín thở", "tim như ngừng đập", "hơi thở dồn dập",
]
CLICHE_PHRASES = [
    "không khí trở nên nặng nề", "thời gian như ngừng lại",
    "một nụ cười nhếch mép", "ánh mắt sắc như dao", "im lặng đến đáng sợ",
    "một cảm giác khó tả", "không thể tin vào mắt mình",
]

def prose_audit(prose: str, c: dict) -> list[dict]:
    f = []
    words = len(prose.split())
    lo, hi = c["word_budget"]
    if not (lo <= words <= hi):
        f.append({"severity": "minor", "check": "length",
                  "message": f"{words} từ, ngoài khoảng [{lo},{hi}]"})

    low = prose.lower()
    for cl in CLICHE_SOMATICS + CLICHE_PHRASES + c.get("forbidden_cliches", []):
        if cl in low:
            f.append({"severity": "minor", "check": "cliche",
                      "message": f"sáo ngữ: “{cl}”"})

    # Giác quan — đếm kênh cảm giác được dùng
    channels = {
        "thị giác": r"(nhìn|thấy|ánh|màu|sáng|tối|bóng)",
        "thính giác": r"(nghe|tiếng|âm|vang|im lặng|rít)",
        "khứu giác": r"(mùi|thơm|hôi|khét|tanh)",
        "xúc giác": r"(chạm|lạnh|nóng|rát|nhám|ẩm|sắc)",
        "vị giác": r"(vị|đắng|mặn|chua|ngọt|tanh)",
    }
    used = sum(1 for _, pat in channels.items()
               if re.search(pat, low))
    if used < c.get("sensory_channels_required", 3):
        f.append({"severity": "minor", "check": "sensory",
                  "message": f"chỉ {used} kênh giác quan, cần ≥"
                             f"{c['sensory_channels_required']}"})

    # Nhân vật nói thẳng mục tiêu — bệnh thuyết minh
    explicit = len(re.findall(
        r"(?:tôi|ta|anh|em) (?:muốn|cần|phải) (?:tìm|giết|cứu|lấy|đến)", low))
    if explicit > c.get("max_explicit_goal_statements", 1):
        f.append({"severity": "major", "check": "on_the_nose",
                  "message": f"{explicit} lần nói thẳng mục tiêu, trần là "
                             f"{c['max_explicit_goal_statements']}"})

    # Tỉ lệ thoại — quá ít thoại = chương tường thuật khô
    dial = len(re.findall(r"^\s*[—–]", prose, flags=re.M))
    if dial == 0 and words > 700:
        f.append({"severity": "minor", "check": "no_dialogue",
                  "message": "cảnh dài không có thoại"})
    return f


def deterministic_audit(prose, contract, chapter, eng) -> list[dict]:
    out = []
    out += prose_audit(prose, contract)
    out += rhythm_audit(prose, contract, lang="vi")      # §10.3.1
    out += pov_leak_scan(prose, contract["pov_character"])
    for ch in contract["active_characters"]:
        r = voice_report(prose, eng.chars.get(ch["id"]))
        if not r["ok"]:
            out += [{"severity": "major", "check": "voice",
                     "message": f"{ch['name']}: {v}"} for v in r["violations"]]
    out += [{"severity": "minor", "check": "self_repetition",
             "message": f"lặp chương {h['old_chapter']} (sim={h['score']})"}
            for h in self_plagiarism_check(prose, eng.store)]
    return out
```

### 10.3.1 Nhịp điệu trần thuật — chống văn xuôi đều đều

`VoiceFingerprint` ở §5.3 chỉ kiểm soát **tầng thoại**. Tầng trần thuật hoàn toàn không bị ràng buộc, và đây là nơi LLM bộc lộ điểm yếu dai dẳng nhất: cấu trúc câu ghép an toàn lặp đi lặp lại — *"Khi [mệnh đề A], anh [hành động B], trong khi [mệnh đề C]"*. Từng câu một thì không sai gì cả. Đọc liên tục năm cảnh thì ru ngủ.

Bốn chỉ số dưới đây bắt đúng hiện tượng đó, và đều là code thuần.

```python
# novel_engine/audit/rhythm.py
import re, statistics
from collections import Counter

# Ngưỡng theo NGÔN NGỮ. Tiếng Việt viết rời âm tiết ("nghiên cứu" = 2 token
# khi tách theo khoảng trắng), nên mọi thống kê đếm-từ đều bị thổi lên
# khoảng 1,3–1,5× so với tiếng Anh. Dùng thẳng ngưỡng tiếng Anh cho văn
# tiếng Việt sẽ cho kết quả sai một cách có hệ thống.
THRESH = {
    "en": {"sd_floor": 6.5, "short_max": 4,  "short_ratio_min": 0.10},
    "vi": {"sd_floor": 8.5, "short_max": 5,  "short_ratio_min": 0.08},
}

SUBORDINATE_OPENERS = [
    "Khi", "Sau khi", "Trước khi", "Trong khi", "Trong lúc", "Ngay khi",
    "Cho đến khi", "Dù", "Mặc dù", "Nếu", "Bởi vì", "Vì", "Để", "Kể từ khi",
]

# Đúng khuôn mẫu ba mệnh đề mà LLM lặp nhiều nhất
TRICLAUSE_RE = re.compile(
    r"^(Khi|Trong khi|Sau khi|Ngay khi)\s.{5,70},\s.{5,70},\s(trong khi|còn|và|thì)\s",
    flags=re.M)


def narrative_sentences(prose: str) -> list[str]:
    """Chỉ lấy câu TRẦN THUẬT — bỏ thoại, vì thoại đã có VoiceFingerprint lo."""
    body = "\n".join(l for l in prose.split("\n")
                     if not l.strip().startswith(("—", "–", "“", '"')))
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", body) if s.strip()]


def rhythm_audit(prose: str, contract: dict, lang: str = "vi") -> list[dict]:
    T = THRESH[lang]
    sents = narrative_sentences(prose)
    if len(sents) < 8:
        return []
    lens = [len(s.split()) for s in sents]
    out = []

    # (1) Độ lệch chuẩn độ dài câu
    sd = statistics.pstdev(lens)
    if sd < T["sd_floor"]:
        out.append({"severity": "minor", "check": "rhythm_variance",
                    "message": f"σ độ dài câu = {sd:.1f} < {T['sd_floor']} — "
                               f"nhịp trần thuật đang đều đều"})

    # (2) Thiếu câu dứt khoát. σ một mình GAMEABLE: một phân phối hai cực
    #     (toàn câu 5 từ và toàn câu 40 từ) cho σ rất đẹp mà đọc như máy.
    #     Phải kiểm cả sự CÓ MẶT của câu ngắn.
    short_ratio = sum(1 for n in lens if n <= T["short_max"]) / len(lens)
    if short_ratio < T["short_ratio_min"]:
        out.append({"severity": "minor", "check": "no_staccato",
                    "message": f"chỉ {short_ratio:.0%} câu ≤{T['short_max']} từ — "
                               f"thiếu nhịp dứt"})

    # (3) Chuỗi câu dài xấp xỉ nhau — thứ mà σ toàn cục không thấy
    run = max_run = 1
    for a, b in zip(lens, lens[1:]):
        run = run + 1 if abs(a - b) <= 3 else 1
        max_run = max(max_run, run)
    if max_run > 5:
        out.append({"severity": "minor", "check": "flat_run",
                    "message": f"{max_run} câu liên tiếp dài xấp xỉ nhau"})

    # (4) Khuôn mẫu mệnh đề phụ — MAJOR, vì đây là dấu vân tay của LLM
    sub = sum(1 for s in sents
              if any(s.startswith(o + " ") for o in SUBORDINATE_OPENERS))
    sub_ratio = sub / len(sents)
    if sub_ratio > 0.30:
        out.append({"severity": "major", "check": "subordinate_opener",
                    "message": f"{sub_ratio:.0%} câu mở đầu bằng mệnh đề phụ "
                               f"(trần 30%) — cấu trúc lặp khuôn"})
    tri = len(TRICLAUSE_RE.findall(prose))
    if tri >= 3:
        out.append({"severity": "major", "check": "triclause_template",
                    "message": f"{tri} câu theo khuôn 'Khi A, B, trong khi C'"})

    # (5) Đa dạng từ mở đầu câu
    openers = Counter(s.split()[0] for s in sents if s.split())
    if len(openers) / len(sents) < 0.55:
        top = openers.most_common(1)[0]
        out.append({"severity": "minor", "check": "opener_diversity",
                    "message": f"từ mở câu lặp nhiều, '{top[0]}' xuất hiện "
                               f"{top[1]}/{len(sents)} lần"})

    # (6) Burstiness đoạn văn — cảnh leo thang PHẢI có đoạn một câu
    paras = [p for p in prose.split("\n\n") if p.strip()]
    pcount = [len(narrative_sentences(p)) or 1 for p in paras]
    if contract["tension"]["mode"] == "escalate" and min(pcount) > 1:
        out.append({"severity": "minor", "check": "paragraph_burstiness",
                    "message": "cảnh escalate không có đoạn văn một câu — "
                               "mắt độc giả không có chỗ tăng tốc"})
    if len(pcount) >= 5 and statistics.pstdev(pcount) < 1.2:
        out.append({"severity": "minor", "check": "paragraph_uniformity",
                    "message": "mọi đoạn văn dài xấp xỉ nhau"})
    return out
```

### 10.3.2 Cảnh báo Goodhart: đừng đưa ngưỡng này vào prompt

Sáu chỉ số trên là **công cụ chẩn đoán**, không phải mục tiêu giao cho Writer. Nếu bạn viết vào Writer prompt câu "độ lệch chuẩn độ dài câu phải > 8,5", model sẽ đáp ứng bằng cách xen kẽ máy móc một câu rất ngắn sau mỗi câu rất dài. Kết quả đọc như tiếng máy gõ nhịp — tệ hơn hẳn sự đơn điệu ban đầu, vì đơn điệu thì nhàm còn nhịp giả thì phản cảm.

Vì vậy, ba quy tắc vận hành:

1. **Không chỉ số nào trong nhóm này được lên mức `blocker`.** Nhịp văn dở không phải lỗi thế giới quan.
2. **Kết quả đi vào Polish Agent, không quay lại Writer.** Polish nhận ghi chú cụ thể ("đoạn 4 có 6 câu liên tiếp dài 18–22 từ") và sửa cục bộ.
3. **Với Writer, dùng exemplar thay cho ngưỡng.** Truy hồi từ vector store hai đoạn văn có nhịp tốt ở đúng loại cảnh và đưa vào prompt. Model bắt chước nhịp tốt hơn nhiều so với việc thoả mãn một con số.

Ngoại lệ duy nhất đáng đưa vào prompt là chỉ số (4) — dưới dạng ràng buộc phủ định kèm phương án thay thế, theo nguyên tắc §12: *"Không quá 3 câu trong cảnh được mở đầu bằng Khi/Trong khi/Sau khi. Với các câu còn lại, đặt chủ ngữ hoặc hành động lên đầu."* Ràng buộc này cụ thể, dễ tuân thủ, và không bóp méo nhịp tổng thể.

### 10.3.3 Hiệu chỉnh lại `VoiceFingerprint` cho tiếng Việt

Cùng lý do đếm âm tiết ở trên, các khoảng `mean_sentence_len` trong `VoiceFingerprint` (§5.1) cũng cần hiệu chỉnh. Nếu bạn lấy con số quen thuộc từ tài liệu tiếng Anh, mọi nhân vật sẽ bị gắn cờ vi phạm ngay từ chương đầu:

| Kiểu giọng | Khoảng (tiếng Anh) | Khoảng (tiếng Việt) |
|-----------|--------------------|---------------------|
| `clipped` — cộc lốc, quân đội | 4–9 | 6–13 |
| `vernacular` — đời thường | 7–14 | 10–20 |
| `formal` — nghi thức, giáo sĩ | 12–22 | 17–31 |
| `ornate` — hoa mỹ, cổ | 15–28 | 21–39 |
| `clinical` — kỹ thuật, lạnh | 9–18 | 13–25 |

Cách chắc chắn nhất vẫn là tự hiệu chỉnh: lấy 2.000 từ văn xuôi tiếng Việt mà bạn thấy đúng giọng muốn có, chạy `statistics.mean`/`pstdev` trên đó, rồi lấy khoảng ±1,5σ làm ngưỡng. Bảng trên chỉ là điểm khởi đầu.

### 10.4 Blind attribution test — thước đo chống flattening tốt nhất

Bỏ hết thẻ tên khỏi các lượt thoại rồi hỏi một LLM độc lập: ai nói câu này? Nếu độ chính xác dưới 70%, các nhân vật của bạn đang nói giống nhau.

```python
# novel_engine/audit/attribution.py
import random

def blind_attribution_test(prose: str, speakers: list[str],
                           judge, n: int = 12) -> dict:
    lines = extract_labeled_dialogue(prose)     # [(speaker, text)]
    if len(lines) < 4:
        return {"skipped": True}
    sample = random.sample(lines, min(n, len(lines)))

    correct = 0
    confusion = {}
    for truth, text in sample:
        guess = judge.invoke(
            f"Trong các nhân vật {speakers}, ai có khả năng nói câu này nhất? "
            f"Chỉ trả về tên.\n\nCâu: “{text}”").content.strip()
        if guess == truth:
            correct += 1
        else:
            confusion[f"{truth}→{guess}"] = confusion.get(f"{truth}→{guess}", 0) + 1

    acc = correct / len(sample)
    return {
        "accuracy": round(acc, 3),
        "verdict": "ok" if acc >= 0.70 else "voices_collapsing",
        "most_confused": max(confusion, key=confusion.get) if confusion else None,
        "severity": "major" if acc < 0.55 else ("minor" if acc < 0.70 else "note"),
    }
```

Chạy metric này mỗi 5 chương. Khi nó tụt, hai cặp nhân vật cụ thể đang trộn vào nhau — `most_confused` cho biết cặp nào, và bạn chỉ cần nới rộng `VoiceFingerprint` của một trong hai.

### 10.5 Diff-Driven Extraction — chống ô nhiễm Canon âm thầm

`extractor_node` là cầu nối **duy nhất** đưa văn xuôi trở lại Canon. Nó cũng là điểm nghẽn nguy hiểm nhất của toàn kiến trúc event-sourced, vì lỗi ở đây không gây crash, không gây cảnh báo — nó chỉ âm thầm ghi sai một sự thật, và hai mươi chương sau bạn mới thấy hậu quả mà không truy được nguồn.

Yêu cầu một LLM đọc 3.000 từ văn xuôi tự do rồi tự nhớ và liệt kê **mọi** thực thể, quan hệ và chuyển trạng thái manh mối là tác vụ có tỉ lệ sót cao một cách có hệ thống. Cơ chế của lỗi rõ ràng: đây là bài toán *recall* trên không gian mở, không có danh sách kiểm, không có tín hiệu dừng. Model dừng khi thấy "đã đủ", và "đủ" là một phán đoán chủ quan biến thiên theo từng lần chạy.

### 10.5.1 Ba lượt thay cho một

Cách chữa là thu hẹp không gian tìm kiếm: cấp cho Extractor chính `SceneContract` làm hệ quy chiếu, biến tác vụ từ *đọc hiểu tự do* thành *kiểm toán vi sai có định hướng*.

Nhưng điều đó lại mở ra một lỗi ngược, và đây là chỗ phải cẩn thận: **ảo giác xác nhận**. Khi bạn hỏi "manh mối CLUE_042 đã được cài chưa?", model có xu hướng mạnh trả lời *rồi* và bịa ra một `span` nghe hợp lý. Bạn vừa đổi lỗi *sót* lấy lỗi *bịa* — mà lỗi bịa còn tệ hơn, vì nó đánh dấu manh mối là đã cài trong khi độc giả chưa hề thấy gì.

Vì vậy kiến trúc là ba lượt, mỗi lượt chống một lỗi khác nhau:

| Lượt | Đầu vào | Chống lỗi | Chi phí |
|------|---------|-----------|---------|
| **1. Kiểm toán vi sai** | prose + SceneContract | sót mục có kế hoạch | LLM |
| **2. Quét phát sinh** | prose, **không có contract** | sót mục ngoài kế hoạch | LLM |
| **3. Xác minh span** | prose + kết quả lượt 1–2 | ảo giác xác nhận | 0 |

Lượt 2 **bắt buộc không được thấy contract**. Nếu thấy, model bị neo vào kế hoạch và sẽ bỏ qua đúng những thứ nằm ngoài kế hoạch — tức là toàn bộ mục đích của lượt này. Chi phí thêm một lượt gọi là nhỏ so với việc để một thực thể mới lọt lưới.

### 10.5.2 Xác minh span — cơ chế rẻ nhất, hiệu quả nhất

Mọi `Assertion` đều phải mang `span` là trích dẫn nguyên văn. Điều đó biến việc kiểm chứng thành khớp chuỗi thuần tuý: nếu câu trích không có thật trong văn bản, mệnh đề bị loại. Không cần LLM, không cần phán đoán.

```python
# novel_engine/reconcile/verify.py
import re, unicodedata, difflib

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip()


def _fuzzy_present(needle: str, haystack: str, ratio: float = 0.90) -> bool:
    """Dự phòng cho trường hợp Polish Agent đã sửa nhẹ câu văn sau khi
    Extractor trích. Trượt cửa sổ cùng độ dài, so bằng difflib."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    step = max(1, n // 4)
    for i in range(0, len(haystack) - n + 1, step):
        if difflib.SequenceMatcher(None, needle,
                                   haystack[i:i + n]).ratio() >= ratio:
            return True
    return False


def verify_spans(delta: StateDelta, prose: str,
                 min_len: int = 12) -> tuple[StateDelta, list[dict]]:
    hay = _norm(prose)
    kept, rejected = [], []
    for a in delta.assertions:
        nd = _norm(a.span or "")
        if len(nd) < min_len:
            rejected.append({"reason": "span_too_short", "assertion": a.subject,
                             "span": a.span})
            continue
        if nd in hay or _fuzzy_present(nd, hay):
            kept.append(a)
        else:
            rejected.append({"reason": "span_not_found", "assertion": a.subject,
                             "predicate": a.predicate, "span": a.span})
    delta.assertions = kept

    # Bằng chứng cài manh mối đi qua ĐÚNG cơ chế xác minh đó. `verified` chỉ
    # được đặt ở đây — không bao giờ do LLM tự khai (NT-5).
    for pe in delta.plant_evidence:
        np = _norm(pe.span or "")
        pe.verified = len(np) >= min_len and (np in hay or _fuzzy_present(np, hay))
        if not pe.verified:
            rejected.append({"reason": "plant_span_not_found",
                             "clue_id": pe.clue_id, "span": pe.span})
    return delta, rejected
```

`min_len = 12` ký tự loại các span rỗng nghĩa kiểu "anh nói" vốn khớp được ở mọi nơi. Ngưỡng fuzzy 0,90 đủ rộng cho sửa chữ, đủ chặt để không nuốt một câu bịa.

### 10.5.3 Đối chiếu kế hoạch — đóng một đường ô nhiễm âm thầm

Ở bản 2.0, `clue_transitions` đến thẳng từ lời khai của LLM. Nghĩa là một manh mối có thể bị đánh dấu `planted` mà chưa từng xuất hiện trên trang giấy — và Foreshadow Scheduler từ đó trở đi sẽ không bao giờ cài lại nó nữa. Đây đúng là kiểu lỗi âm thầm nguy hiểm nhất mà event sourcing tạo ra.

```python
def plan_coverage(contracts: list[dict], delta: StateDelta) -> dict:
    """Đối chiếu những gì contract HỨA với những gì đã XÁC MINH được.

    LỖI ĐÃ SỬA: bản trước đối chiếu `d["clue_id"]` với
    `{a.subject for a in delta.assertions}`. Hai không gian định danh này
    không bao giờ giao nhau — `subject` là thực thể trong câu văn
    ("con dấu", "Serena"), `clue_id` là mã hệ thống ("CLUE_042_RUSTED_SEAL").
    Phép so sánh luôn cho False: mọi manh mối bị tính là thất bại và mọi
    `clue_transitions` bị xoá sạch. Nay dùng `delta.plant_evidence`.
    """
    promised, fulfilled, missed = [], [], []
    verified = {e.clue_id for e in delta.plant_evidence if e.verified}

    for c in contracts:
        for d in c.get("plant_directives", []):
            promised.append(d["clue_id"])
            if d["clue_id"] in verified:
                fulfilled.append(d["clue_id"])
            else:
                missed.append({"clue_id": d["clue_id"],
                               "scene": c["scene_id"], "mode": d["mode"]})

    # QUY TẮC CỨNG (NT-5): manh mối không có bằng chứng xác minh thì KHÔNG
    # được chuyển trạng thái, dù Extractor có khai gì đi nữa.
    missed_ids = {m["clue_id"] for m in missed}
    for cid in list(delta.clue_transitions):
        if cid in missed_ids:
            del delta.clue_transitions[cid]

    return {
        "plan_fulfillment_rate": round(len(fulfilled) / max(len(promised), 1), 3),
        "missed_plants": missed,
        "findings": [{"severity": "major", "check": "unfulfilled_plant",
                      "message": f"{m['clue_id']} được lên kế hoạch cài ở "
                                 f"{m['scene']} nhưng không tìm thấy trong văn bản"}
                     for m in missed],
    }


def merge_extractions(audited_json: str, emergent_json: str,
                      chapter: int) -> StateDelta:
    """Gộp hai lượt. Lượt 1 (có hệ quy chiếu) LUÔN thắng khi trùng.

    LỖI ĐÃ SỬA: bản trước chỉ gộp `assertions` và `new_entities`. Mọi
    `new_relations`, `retracted_relations`, `clue_transitions` và
    `relationship_updates` mà lượt 2 tìm được đều bị vứt im lặng — nghĩa là
    một mối thù hay một liên minh phát sinh ngoài kế hoạch sẽ không bao giờ
    tới được `reconcile_node`.
    """
    a = parse_model(audited_json,  StateDelta)      # §9.5
    e = parse_model(emergent_json, StateDelta)

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

    # setdefault, KHÔNG update: `update` để lượt 2 ghi đè lượt 1, đảo ngược
    # quy tắc ưu tiên đã công bố. Lượt 2 không thấy contract nên phán đoán
    # chuyển trạng thái manh mối của nó kém tin cậy hơn hẳn.
    for k, v in e.clue_transitions.items():
        a.clue_transitions.setdefault(k, v)

    # Quan hệ phải GỘP THEO CẶP, không nối đuôi. Nối đuôi khiến
    # `apply_scene_effects` (§7.3) chạy hai lần trên cùng một cặp và cộng dồn
    # intimacy — biểu hiện ra ngoài là M5 báo nhảy cóc giai đoạn mà không rõ
    # nguyên nhân.
    by_pair = {(u.a, u.b): u for u in a.relationship_updates}
    for u in e.relationship_updates:
        if (u.a, u.b) not in by_pair and (u.b, u.a) not in by_pair:
            by_pair[(u.a, u.b)] = u
    a.relationship_updates = list(by_pair.values())

    # plant_evidence chỉ lượt 1 sinh ra — lượt 2 không biết kế hoạch là gì
    a.chapter = chapter
    return a
```

`plan_coverage` trả về `findings` ở mức `major`, đưa thẳng vào báo cáo chương. Khi một manh mối được lên kế hoạch mà không cài được, Scheduler sẽ tự động xếp lại nó ở chương sau với `salience` không đổi — thay vì tưởng rằng việc đã xong.

---

## 11. Drift Reconciliation — khi Writer đi chệch kế hoạch

Đây là tình huống sẽ xảy ra thường xuyên, và cách xử lý quyết định chất lượng cuối cùng. Model viết ra một chi tiết hay nhưng không có trong kế hoạch. Ép viết lại thì mất chi tiết hay. Thả nổi thì cấu trúc sụp.

Giải pháp: **phân loại lệch thành ba hạng và xử lý khác nhau**.

```python
# novel_engine/reconcile/classify.py

import hashlib


def item_key(item) -> str:
    """Khoá định danh ổn định cho một phần tử của StateDelta. Bản trước gọi
    hàm này ở hai nơi mà không định nghĩa nó ở đâu cả."""
    if isinstance(item, Entity):
        return f"E:{item.id}"
    if isinstance(item, Relation):
        return f"R:{item.src}|{item.type}|{item.dst}"
    if isinstance(item, Assertion):
        # F9: `hash()` của Python ngẫu nhiên hoá theo PYTHONHASHSEED ở mỗi
        # lần khởi động tiến trình. Một chương dừng ở checkpoint LangGraph rồi
        # chạy tiếp ở tiến trình khác sẽ sinh khoá khác → `delta.item(k)` ném
        # KeyError. Mọi khoá sống lâu hơn một tiến trình phải dùng hashlib.
        # Dùng blake2b thay md5 vì md5 không khả dụng ở chế độ FIPS.
        h = hashlib.blake2b(item.span.encode("utf-8"), digest_size=3).hexdigest()
        return f"A:{item.subject}|{item.predicate}|{h}"
    raise TypeError(f"không biết khoá cho {type(item).__name__}")


def classify_delta(delta: StateDelta, graph, planner) -> dict:
    """LỖI ĐÃ SỬA: bản trước duyệt toàn bộ `delta.assertions` mà không xét
    `epistemic`. Hậu quả trực tiếp: Serena nói dối "Hạm Đội Số 3 đã bị xoá
    sổ" → Extractor gán đúng nhãn `claimed_by` → `conflicts_with_locked_canon`
    vẫn thấy sai lệch với canon → `contradiction` → `reconcile_node` escalate
    và dừng chương. Tức là trong hệ thống cũ, KHÔNG NHÂN VẬT NÀO ĐƯỢC PHÉP
    NÓI DỐI. §10.2 đã lọc đúng; §11 thì quên — hai hàm, một hàm kiểm, một
    hàm không.
    """
    result, irony_seeds = {}, []

    for item in delta.new_entities + delta.new_relations + delta.assertions:
        key = item_key(item)

        if isinstance(item, Assertion) and item.epistemic != "objective":
            # Lời nói dối / niềm tin sai KHÔNG BAO GIỜ là mâu thuẫn canon.
            # Nó là một sự kiện tâm lý, ghi vào BeliefState của người nói,
            # không ghi vào sự thật khách quan.
            result[key] = "enrichment"
            if graph.conflicts_with_truth(item):
                # Và đây là điều đáng giá: chênh lệch giữa điều nhân vật
                # tin/nói và sự thật chính là nguyên liệu của mỉa mai kịch
                # tính (M12, §13.3). Ghi nhận, đừng chặn.
                irony_seeds.append({
                    "holder": item.holder, "claim": item.object,
                    "truth_conflict": True, "span": item.span,
                    "severity": "note",
                    "note": ("độc giả biết điều này sai — Director có thể "
                             "khai thác ở chương sau"),
                })
            continue

        if conflicts_with_locked_canon(item, graph):
            result[key] = "contradiction"
        elif affects_downstream_plan(item, planner):
            result[key] = "improvement"
        else:
            result[key] = "enrichment"

    delta.classification = result
    return {"classification": result, "irony_seeds": irony_seeds}


def reconcile_node(state: ChapterState, cfg) -> dict:
    eng = cfg["configurable"]["engines"]
    delta = StateDelta.model_validate(state["delta"])
    out = classify_delta(delta, eng.graph, eng.planner)
    cls, irony = out["classification"], out["irony_seeds"]

    # Hồi ức có luật riêng, chạy TRƯỚC mọi thao tác ghi (§3.6.4)
    for f in state.get("frames", []):
        frame = ContinuityFrame.model_validate(f)
        fb = flashback_admissible(delta, frame, eng.graph)   # F8
        if any(x["severity"] == "blocker" for x in fb):
            return {"escalated": True,
                    "escalation_reason": f"nghịch lý nhân quả hồi ức: {fb}"}

    contradictions = [k for k, v in cls.items() if v == "contradiction"]
    if contradictions:
        return {"escalated": True,
                "escalation_reason": f"mâu thuẫn canon: {contradictions}"}

    # LỖI ĐÃ SỬA: bản trước chỉ commit `enrichment`. Sự thật diện
    # `improvement` được dùng để REPLAN nhưng không bao giờ được ghi vào
    # graph. Hậu quả cụ thể: Writer viết ra "Kaelen có một người em gái",
    # kế hoạch chương 31 được sửa để xoay quanh cô ấy, nhưng cô ấy không tồn
    # tại trong canon — nên ở chương 31, POV firewall lọc bỏ mọi nhắc tới cô
    # ấy vì "không có trong danh sách nhân vật POV biết".
    #
    # Nguyên tắc: VĂN XUÔI ĐÃ VIẾT RA LÀ SỰ THẬT. Canon phải khớp với trang
    # giấy, không phải ngược lại. Từ chối ghi chỉ tạo ra phân ly canon–văn bản,
    # tệ hơn hẳn so với ghi rồi sửa.
    patch = None
    improvements = [k for k, v in cls.items() if v == "improvement"]
    if improvements:
        patch = eng.planner.replan_downstream(
            from_chapter=delta.chapter + 1,
            new_facts=[delta.item(k) for k in improvements])

    # ── F6: PHÂN LUỒNG GHI — lỗi nghiêm trọng nhất vòng này ──────────
    # Không bản vá đơn lẻ nào gây ra nó. v2.3 xếp assertion phi khách quan
    # vào `enrichment` (để lời nói dối không bị chặn là mâu thuẫn). v2.4 cho
    # commit cả `enrichment` lẫn `improvement`. Mỗi bản vá đúng trong phạm
    # vi của nó; HỢP LẠI chúng ghi thẳng lời nói dối của Serena vào canon
    # như sự thật lịch sử — đúng cái mà trường `epistemic` sinh ra để chặn.
    #
    # Sự thật khách quan → World Graph. Lời nói và niềm tin → hồ sơ nhân vật
    # (và một cạnh CLAIMED/BELIEVES, để M12 và POV firewall vẫn thấy được
    # chúng mà không nhầm chúng là canon).
    for k, v in cls.items():
        if v not in ("enrichment", "improvement"):
            continue
        item = delta.item(k)

        if isinstance(item, Assertion) and item.epistemic != "objective":
            if item.holder:
                eng.chars.update_belief(
                    char_id=item.holder,
                    proposition=f"{item.subject} {item.predicate} {item.object}",
                    confidence=item.confidence,
                    source=f"scene_ch{delta.chapter}")
                eng.graph.commit_belief(
                    holder=item.holder, subject=item.subject,
                    predicate=item.predicate, object=item.object,
                    kind=item.epistemic,          # claimed_by | believed_by
                    since_tick=eng.store.tick_of_chapter(delta.chapter))
            continue

        eng.graph.commit(
            item, provenance=f"extracted_ch{delta.chapter}",
            provisional_patch=(patch.patch_id if v == "improvement" and patch
                               else None))

    # ── F7: đóng quan hệ bị thu hồi ─────────────────────────────────
    # `classify_delta` duyệt new_entities + new_relations + assertions và bỏ
    # sót `retracted_relations` hoàn toàn. Hậu quả: hai nhân vật phản bội
    # nhau, cạnh BETRAYED được thêm, nhưng cạnh PROTECTS cũ vẫn sống song
    # song mãi mãi — và §10.2 sẽ báo mutex mâu thuẫn ở mọi chương sau.
    for r in delta.retracted_relations:
        if eng.graph.is_locked(r.src, r.dst, r.type):
            return {"escalated": True,
                    "escalation_reason": (f"cố thu hồi quan hệ đã khoá: "
                                          f"{r.src}-[{r.type}]->{r.dst}")}
        eng.graph.close_relation(src=r.src, dst=r.dst, rel_type=r.type,
                                 until_chapter=delta.chapter)

    if patch:
        eng.planner.apply_patch(patch)

    # LỖI ĐÃ SỬA: không chỗ nào cập nhật `last_touched_chapter`/`salience`.
    # Hậu quả: `decay()` trả 0.0 mãi mãi vì `last_touched_chapter is None`,
    # `forget_pressure` chạm trần 0.8 ở MỌI chương sau, và hệ thống ép
    # re-plant cùng một manh mối liên tục, nuốt sạch ATTENTION_BUDGET của
    # các manh mối khác. Công thức decay ở §6.2 không bao giờ được kích hoạt.
    for pe in delta.plant_evidence:
        if not pe.verified:
            continue
        inten = 0.6
        for c in state["contracts"]:
            for d in c.get("plant_directives", []):
                if d["clue_id"] == pe.clue_id:
                    inten = d.get("intensity", 0.6)
        # Một lần nhắc thoáng qua không khôi phục trí nhớ độc giả bằng một
        # cảnh nhấn mạnh — salience hồi theo cường độ, không nhảy thẳng về 1.0.
        restored = min(1.0, 0.45 + 0.55 * inten)
        eng.clues.touch(pe.clue_id, chapter=delta.chapter, salience=restored)
        eng.graph.update_clue_touch(pe.clue_id, delta.chapter, restored)

    # Quan hệ: PHÁN QUYẾT sau khi đã có số liệu thực (§7.3)
    for st in delta.relationship_updates:
        apply_relationship_advancement(st, delta.chapter)
        advance_from_rupture(st, delta.chapter)

    delta.committed = True
    eng.store.append_delta(delta)
    return {"delta": delta.model_dump(), "irony_seeds": irony}
```

`replan_downstream` là hàm quan trọng: khi Writer bịa ra rằng Kaelen có một người em gái, hệ thống không xoá chi tiết đó, mà **ghi nhận nó và kiểm tra xem có kế hoạch nào ở chương sau bị vô hiệu không** (ví dụ: chương 30 có nói Kaelen là người cuối cùng của dòng họ). Nếu có, nó tạo một `PlanPatch` và báo cho tác giả — kèm hai lựa chọn: giữ chi tiết mới và sửa chương 30, hoặc gỡ chi tiết mới.

```python
class PlanPatch(BaseModel):
    from_chapter: int
    invalidated_beats: list[str]
    invalidated_clues: list[str]
    suggested_rewrites: list[dict]
    author_decision_required: bool
```

Nguyên tắc: **hệ thống không bao giờ tự ý xoá một chi tiết hay, và cũng không bao giờ tự ý sửa canon đã chốt.** Khi hai điều đó va nhau, con người quyết định.

---

## 12. Prompt Library

Prompt là nơi rất nhiều thiết kế tốt bị đổ vỡ. Bốn nguyên tắc áp dụng cho toàn bộ thư viện:

1. **Không đưa tính từ tính cách.** Đưa ràng buộc hành vi.
2. **Không đưa `clue.description`.** Đưa `surface_form` + cường độ.
3. **Không đưa thông tin ngoài tầm POV** trừ khi kèm nhãn cấm rõ ràng.
4. **Ràng buộc phủ định phải đi kèm phương án thay thế.** "Đừng dùng 'tim đập thình thịch'" một mình sẽ khiến model dùng một sáo ngữ khác. "Đừng dùng X, thay bằng phản ứng cơ thể trong danh sách Y" mới có tác dụng.

### 12.1 Writer Prompt

````python
# novel_engine/prompts.py

WRITER_TMPL = """Bạn viết một cảnh tiểu thuyết. Viết văn xuôi, không viết tóm tắt.

## GÓC NHÌN — RÀNG BUỘC CỨNG
Người kể: {contract[pov_character]}, ngôi thứ ba giới hạn.
Bạn CHỈ được viết những gì nhân vật này tri giác và suy nghĩ.

Nhân vật này KHÔNG biết những điều sau. Tuyệt đối không viết ra chúng,
kể cả dưới dạng "anh không biết rằng...":
{contract[pov_knowledge_boundary]}

Những điều nhân vật này chỉ NGHI NGỜ — phải viết như phỏng đoán, kèm khả năng sai:
{context[hypotheses]}

## NHÂN VẬT TRONG CẢNH
{contract[active_characters]}

Với mỗi nhân vật KHÔNG phải POV: bạn biết `hidden_action` của họ, nhưng POV
thì không. Hãy để hành động ngầm đó tạo ra một DẤU VẾT QUAN SÁT ĐƯỢC mà POV
nhìn thấy nhưng diễn giải sai hoặc bỏ qua.

## GIỌNG NHÂN VẬT — KIỂM TRA TỰ ĐỘNG SAU KHI VIẾT
Mỗi nhân vật có `voice_reminder`. Các ràng buộc này được kiểm tra bằng máy:
- độ dài câu thoại trung bình phải nằm trong khoảng đã cho
- không được dùng từ trong `forbidden_lexicon`
- phải xuất hiện dấu vết của `signature_lexicon`
Dùng phản ứng cơ thể trong `somatic_allowed`. CẤM mọi biểu hiện trong
`somatic_forbidden` — đó là sáo ngữ.

## NHIỆM VỤ TỰ SỰ
Câu hỏi kịch tính: {contract[dramatic_question]}
Cảnh BẮT ĐẦU ở trạng thái: {contract[entry_state]}
Cảnh PHẢI KẾT THÚC ở trạng thái: {contract[exit_state]}
Điều bắt buộc thay đổi: {contract[scene_must_change]}
Một cảnh kết thúc nguyên trạng là một cảnh hỏng.

Nhịp: {contract[tension][mode]} — {contract[tension][note]}
Loại áp lực chủ đạo: {contract[tension][pressure_type]}

## MANH MỐI CẦN CÀI
{contract[plant_directives]}

Với mỗi mục: dùng đúng `surface_form` đã cho, đặt qua `carrier` đã chỉ định,
theo đúng `instruction`. Không thêm câu nào bình luận về ý nghĩa của nó.
Nếu `intensity` < 0.35: chi tiết phải xuất hiện khi sự chú ý của độc giả
đang hướng về chuyện khác.

## QUAN HỆ
{contract[relationship_directives]}
Với mỗi mục có `scene_requirement`: đó là ràng buộc, không phải gợi ý.

## VĂN PHONG
Ngân sách: {contract[word_budget][0]}–{contract[word_budget][1]} từ.
Tối đa {contract[max_explicit_goal_statements]} lần một nhân vật nói thẳng
mục tiêu của mình. Mọi mục tiêu khác phải lộ qua hành động hoặc qua điều
nhân vật TRÁNH nói.
Cần ≥{contract[sensory_channels_required]} kênh giác quan, phải có ít nhất
một kênh không phải thị giác.
Subtext: {contract[subtext_requirement]}
Cụm từ bị cấm trong cảnh này: {contract[forbidden_cliches]}

## BỐI CẢNH ĐÃ XẢY RA
Cảnh liền trước: {context[recent_scenes]}
Các chương gần đây: {context[recent_chapters]}
Bối cảnh xa: {context[arc_history]}
Sự thật POV nắm được: {context[known_facts]}

## MẪU VĂN PHONG THAM KHẢO (học nhịp, không sao chép nội dung)
{exemplars}

{feedback}

Viết cảnh. Chỉ xuất văn xuôi, không tiêu đề, không ghi chú.
"""
````

### 12.2 Auditor Prompt

````python
AUDITOR_TMPL = """Bạn thẩm định một cảnh tiểu thuyết theo hợp đồng cảnh.
Phần kiểm tra máy móc (độ dài, sáo ngữ, giọng nhân vật) đã chạy rồi.
Bạn chỉ đánh giá những điều máy không làm được.

## HỢP ĐỒNG
{contract}

## VĂN BẢN
{prose}

## KIỂM TRA
{checklist}

Với mỗi vấn đề, xuất một dòng JSON:
{{"check": "<tên>", "severity": "blocker|major|minor|note",
  "span": "<trích dẫn ≤25 từ>", "message": "<vấn đề>",
  "fix": "<sửa cụ thể, không phải lời khuyên chung>"}}

Quy tắc phân mức:
- blocker: mâu thuẫn thế giới quan, rò rỉ tri thức POV, vi phạm ràng buộc cứng
- major:   nhân vật hành xử ngoài hồ sơ, cảnh không thay đổi trạng thái,
           manh mối được giải thích thẳng ra, private_agenda biến mất
- minor:   vấn đề nhịp điệu, câu vụng
- note:    nhận xét không cần hành động

Nếu không có vấn đề nào ở mức blocker/major, nói rõ điều đó.
KHÔNG bịa ra vấn đề để tỏ ra hữu ích. Một cảnh đạt chuẩn là chuyện bình thường.
"""

AUDIT_CHECKLIST = """
1. NHÂN VẬT CÓ QUÁ DỄ DÃI KHÔNG?
   Có nhân vật nào đồng ý, giúp đỡ, hoặc nhượng bộ mà không lấy gì về không?
   Mỗi nhân vật phụ trong cảnh có đang theo đuổi việc riêng của họ không,
   hay chỉ tồn tại để phục vụ POV?

2. PRIVATE AGENDA
   Với mỗi nhân vật có `private_agenda`: toan tính đó có để lại dấu vết
   trong cảnh không? (không cần lộ ra, nhưng phải có dấu vết)

3. MANH MỐI
   Với mỗi plant_directive: chi tiết đã xuất hiện chưa? Có bị bình luận,
   nhấn mạnh, hay đặt ở vị trí quá nổi bật so với `intensity` yêu cầu không?
   Có nhân vật nào rút ra kết luận sớm hơn mức cho phép không?

4. THAY ĐỔI TRẠNG THÁI
   So `entry_state` với `exit_state`: cảnh có thực sự đưa câu chuyện tới đó không?

5. SUBTEXT
   Có bao nhiêu lượt thoại nói thẳng điều nhân vật đang nghĩ?
   Có lượt nào nên chuyển thành né tránh, đổi chủ đề, hoặc im lặng không?

6. RÒ RỈ TRI THỨC
   POV có thể hiện hiểu biết nào vượt quá `pov_knowledge_boundary` không?
   Có đoạn nào mô tả nội tâm của nhân vật không phải POV không?
"""
````

### 12.3 Extractor Prompts — hai lượt tách biệt

Theo kiến trúc ba lượt ở §10.5. Hai template dưới đây **không bao giờ được gộp làm một**: lượt 2 mất giá trị ngay khi nó nhìn thấy kế hoạch.

````python
EXTRACT_DIFF_TMPL = """Bạn là kiểm toán viên. Bạn có KẾ HOẠCH của chương và
VĂN BẢN đã viết. Nhiệm vụ: đối chiếu, không phải đọc hiểu tự do.

## KẾ HOẠCH (SceneContract của từng cảnh)
{contracts}

## VĂN BẢN (chương {chapter})
{prose}

Trả lời đúng ba câu hỏi, theo thứ tự:

### CÂU 1 — KẾ HOẠCH CÓ ĐƯỢC THỰC HIỆN KHÔNG?
Với TỪNG mục trong `plant_directives` và `relationship_directives`:
- Nếu CÓ trong văn bản: trích `span` NGUYÊN VĂN (≥12 ký tự, sao chép chính xác
  từng chữ, không diễn giải, không rút gọn).
- Nếu KHÔNG: ghi `"found": false`. Không tìm thấy là câu trả lời BÌNH THƯỜNG
  và hữu ích. TUYỆT ĐỐI KHÔNG bịa span để mục nào đó trông như đã hoàn thành.

Span của bạn sẽ được đối chiếu tự động với văn bản bằng khớp chuỗi. Span không
tồn tại sẽ bị loại và mục đó bị tính là thất bại — bịa không giúp ích gì.

### CÂU 2 — TRẠNG THÁI THỰC TẾ SAU CẢNH
Với từng nhân vật có mặt: vị trí, vết thương, vật mang theo, và `epoch_tick`
ước lượng của thời điểm kết cảnh.

### CÂU 3 — NHÂN VẬT CÓ ĐI CHỆCH HỢP ĐỒNG KHÔNG?
Với từng nhân vật có `deliberation.chosen`: họ có làm điều đó không?
Nếu làm khác, ghi rõ họ đã làm gì và trích `span`.
Với từng nhân vật có `must_not_reveal`: có bí mật nào bị lộ không?

Xuất một JSON đúng schema StateDelta.
- Mọi `assertion` phải có `span`.
- Với MỖI `plant_directive`, xuất một mục trong `plant_evidence` gồm
  `clue_id`, `scene_id`, `span` nguyên văn, `carrier_used`, và `concluded_by`
  (danh sách nhân vật đã RÚT RA KẾT LUẬN từ chi tiết đó — khác với nhân vật
  chỉ nhìn thấy nó). KHÔNG đặt trường `verified`; hệ thống tự đặt.
- Nếu một plant_directive không xuất hiện, BỎ QUA nó trong `plant_evidence`.
  Không tạo mục với span rỗng.
"""


EXTRACT_EMERGENT_TMPL = """Đọc chương và liệt kê những gì TỒN TẠI trong đó.
Bạn là bộ phận ghi chép, không phải biên tập viên. Không đánh giá văn chương.

## VĂN BẢN (chương {chapter})
{prose}

## THỰC THỂ ĐÃ BIẾT TỪ TRƯỚC
{known_entities}

Nhiệm vụ: tìm những thứ KHÔNG có trong danh sách trên.

- `new_entities`: người, nơi chốn, tổ chức, đồ vật có tên riêng, tập tục,
  luật lệ — bất cứ thứ gì được nhắc tới như thể nó có thật trong thế giới
  nhưng chưa có trong danh sách. Kể cả khi chỉ được nhắc thoáng qua một lần.
- `new_relations`: quan hệ mới được thiết lập giữa các thực thể.
- `retracted_relations`: quan hệ bị chấm dứt.
- `assertions`: mệnh đề sự thật, MỖI mệnh đề kèm `span` là trích dẫn nguyên văn.

QUY TẮC EPISTEMIC — quan trọng nhất:
- `epistemic: "objective"` chỉ khi NGƯỜI KỂ khẳng định
- `epistemic: "claimed_by"` khi một NHÂN VẬT nói ra
- `epistemic: "believed_by"` khi văn bản cho thấy nhân vật TIN

Nhân vật có thể nói dối. Lời nhân vật KHÔNG BAO GIỜ là "objective".

Hãy quét kỹ phần bối cảnh và phần thoại phụ — thực thể mới thường nằm ở đó,
không nằm ở tuyến hành động chính.

Chỉ xuất JSON.
"""
````

Một chi tiết nhỏ trong `EXTRACT_DIFF_TMPL` có tác dụng lớn: câu *"Không tìm thấy là câu trả lời BÌNH THƯỜNG và hữu ích"* kèm lời cảnh báo rằng span sẽ bị đối chiếu tự động. Nói rõ cho model biết việc bịa **sẽ bị phát hiện** làm giảm đáng kể tỉ lệ ảo giác xác nhận — hiệu quả hơn hẳn so với chỉ ra lệnh "đừng bịa".

### 12.4 Scene Digest Prompt và schema đóng cảnh

`scene_boundary_node` (§9.2) gọi `SCENE_DIGEST_TMPL` và `SceneClose` — cả hai được bổ sung ở đây.

```python
# novel_engine/canon/timeline.py  (tiếp)

class ContinuityObservation(BaseModel):
    """Phần trạng thái vật lý mà CHỈ ĐỌC VĂN BẢN mới biết. Cố ý KHÔNG có
    trường `time`.

    F2: bản trước cho `SceneClose.continuity` là `ContinuityFrame`, vốn chứa
    `StoryTime` với `narrative_order` bắt buộc. LLM không có cách nào biết
    `narrative_order` là gì, nên Pydantic ném ValidationError ở mọi ranh giới
    cảnh. Sửa bằng cách CẮT TRƯỜNG ĐÓ KHỎI SCHEMA thay vì dặn model trong
    prompt: schema không có thì model không thể quên (NT-13).
    """
    locations: dict[str, str]
    injuries: dict[str, list[str]]
    possessions: dict[str, list[str]]
    weather: str | None = None


class SceneClose(BaseModel):
    """Kết quả chốt sổ một cảnh: bản tóm tắt L1 + trạng thái vật lý."""
    digest: str                           # 5–8 câu, dùng cho L1 (§4.1)
    continuity: ContinuityObservation
    actual_duration_ticks: int            # thời lượng THỰC so với dự kiến
    unresolved: list[str] = Field(default_factory=list)
```

````python
SCENE_DIGEST_TMPL = """Bạn chốt sổ một cảnh vừa viết xong. Bạn là thư ký
trường quay, không phải biên tập viên: ghi lại cái đã xảy ra, không đánh giá.

## HỢP ĐỒNG CẢNH (mốc thời gian đã được hệ thống ấn định, ĐỪNG tự đặt lại)
scene_id: {scene_id}
epoch_tick bắt đầu: {epoch_tick}
duration_ticks dự kiến: {planned_duration}
Nhân vật có mặt: {present}

## VĂN XUÔI
{prose}

Xuất JSON đúng schema SceneClose:

1. `digest` — 5–8 câu: ai làm gì, đổi gì, kết thúc ở trạng thái nào. Viết cho
   một người sẽ đọc nó ở chương sau mà không đọc lại cảnh này. Bỏ mọi chi
   tiết văn phong; giữ mọi chi tiết có hệ quả.

2. `continuity` — trạng thái VẬT LÝ tại thời điểm cảnh kết thúc:
   - `scene_id`: chép đúng giá trị trên
   - `time`: chép đúng `epoch_tick` trên; `duration_ticks` điền theo thời
     lượng THỰC mà văn bản thể hiện
   - `locations`: mỗi nhân vật có mặt → nơi họ đang đứng lúc cảnh kết thúc
   - `injuries`: vết thương CÒN HIỆU LỰC (gồm cả vết thương có từ trước mà
     văn bản không nói là đã lành)
   - `possessions`: vật mang theo có vai trò trong truyện; bỏ qua quần áo
     và vật dụng thông thường
   - `weather`: chỉ điền nếu văn bản có nói

3. `actual_duration_ticks` — thời gian truyện thực sự trôi qua trong cảnh.
   Lệch nhiều so với dự kiến là thông tin hữu ích, không phải lỗi.

4. `unresolved` — những gì cảnh mở ra mà chưa đóng lại (một câu hỏi bị bỏ
   lửng, một người bước vào mà chưa rời đi, một tiếng động chưa ai kiểm tra).

Chỉ xuất JSON.
"""
````

Ba ràng buộc trong prompt này đáng chú ý vì mỗi cái chặn một lỗi cụ thể:

- **"mốc thời gian đã được ấn định, ĐỪNG tự đặt lại"** — nếu để model tự sinh `epoch_tick`, bạn có hai nguồn sự thật cho cùng một đại lượng và chúng sẽ lệch nhau ở chương thứ ba. Model chỉ được điền `duration_ticks` thực tế, vì đó là thứ chỉ đọc văn bản mới biết.
- **"vết thương còn hiệu lực, gồm cả vết thương có từ trước"** — nếu không nói rõ, model chỉ liệt kê vết thương *mới trong cảnh này*, và `_injury_respected` (§10.1) sẽ thấy mọi vết thương cũ biến mất ở mỗi cảnh.
- **`unresolved`** — đây là nguồn cấp cho `dangling thread` mà Director dùng ở chương sau, và nó rẻ hơn nhiều so với để một agent riêng đi tìm.

---

## 13. Eval Harness — đo hệ thống có tốt lên không

Lỗ hổng L6 là lỗ hổng nguy hiểm nhất về lâu dài. Không có eval, mỗi lần bạn chỉnh một câu trong prompt là một canh bạc: chương tiếp theo đọc khác đi, nhưng bạn không biết là tốt hơn hay chỉ là khác.

Tám metric dưới đây đều chạy tự động và đều rẻ (trừ M6, M7 cần LLM).

```python
# novel_engine/eval/metrics.py

def M1_entity_consistency(chapters, graph) -> float:
    """Tỉ lệ mệnh đề objective KHÔNG mâu thuẫn canon. Mục tiêu ≥0.98."""
    total = conflicts = 0
    for ch in chapters:
        for a in ch.delta.assertions:
            if a.epistemic != "objective":
                continue
            total += 1
            if graph.conflicting_assertion(a.subject, a.predicate, a.object):
                conflicts += 1
    return 1 - conflicts / max(total, 1)


def M2_clue_payoff_rate(clues, last_chapter) -> dict:
    """Tỉ lệ manh mối đã cài được trả bài đúng hạn. Mục tiêu ≥0.90."""
    planted = [c for c in clues if c.status != ClueStatus.DRAFTED]
    paid    = [c for c in planted if c.status == ClueStatus.PAID_OFF]
    overdue = [c for c in planted if c.status != ClueStatus.PAID_OFF
               and c.payoff_deadline < last_chapter]
    return {"rate": len(paid) / max(len(planted), 1),
            "overdue": [c.clue_id for c in overdue],
            "dangling": len(planted) - len(paid)}


def M3_voice_distinctiveness(chapters, characters) -> float:
    """Khoảng cách Jensen-Shannon trung bình giữa phân phối từ vựng của
    các nhân vật. Tụt = các giọng đang trộn vào nhau. Mục tiêu ≥0.28."""
    from scipy.spatial.distance import jensenshannon
    import itertools, numpy as np
    dists = {c.id: lexical_distribution(chapters, c.id) for c in characters}
    vocab = sorted(set().union(*[d.keys() for d in dists.values()]))
    vecs = {k: np.array([d.get(w, 0) for w in vocab]) + 1e-9
            for k, d in dists.items()}
    pairs = [jensenshannon(vecs[a], vecs[b])
             for a, b in itertools.combinations(vecs, 2)]
    return float(np.mean(pairs))


def M4_npc_reuse_ratio(graph, window=10) -> float:
    """NPC tái xuất hiện / tổng NPC. Thấp = thế giới đang phình vô tội vạ.
    Mục tiêu ≥0.45."""
    npcs = graph.npcs_in_window(window)
    return sum(1 for n in npcs if n["appearances"] > 1) / max(len(npcs), 1)


def M5_relationship_pacing(rel_states, total_chapters) -> dict:
    """Phát hiện nhảy cóc giai đoạn. Mục tiêu: violations = 0."""
    bad = []
    for st in rel_states:
        for i in range(1, len(st.history)):
            prev, cur = st.history[i-1], st.history[i]
            if stage_index(cur.stage) - stage_index(prev.stage) > 1:
                bad.append(f"{st.a}-{st.b}: nhảy {prev.stage}→{cur.stage} "
                           f"ở chương {cur.chapter}")
            if cur.stage == RelationStage.CATHARSIS and not cur.scars:
                bad.append(f"{st.a}-{st.b}: catharsis không có sẹo")
    return {"violations": bad, "count": len(bad)}


def M6_blind_attribution(chapters, judge) -> float:
    """§10.4. Mục tiêu ≥0.70."""
    scores = [blind_attribution_test(ch.prose, ch.speakers, judge)["accuracy"]
              for ch in chapters if len(ch.speakers) >= 2]
    return sum(scores) / max(len(scores), 1)


def M7_scene_necessity(chapters, judge) -> float:
    """Tỉ lệ cảnh thực sự thay đổi trạng thái. Mục tiêu ≥0.85.
    Hỏi judge: 'nếu xoá cảnh này, chương có còn hiểu được không?'"""
    keep = 0
    for ch in chapters:
        for sc in ch.scenes:
            ans = judge.invoke(
                f"Cảnh sau thay đổi điều gì trong câu chuyện? Nếu không thay "
                f"đổi gì, trả lời chính xác 'KHÔNG'.\n\n{sc.prose[:2500]}"
            ).content
            keep += 0 if ans.strip().upper().startswith("KHÔNG") else 1
    n = sum(len(ch.scenes) for ch in chapters)
    return keep / max(n, 1)


def M8_tension_tracking(chapters) -> float:
    """MAE giữa tension đo được và đường cong mục tiêu. Mục tiêu ≤0.15."""
    errs = [abs(ch.measured_tension - target_tension(ch.number, len(chapters)))
            for ch in chapters]
    return sum(errs) / max(len(errs), 1)
```

### 13.1 Bảng điều khiển chất lượng

| Metric | Mục tiêu | Báo động khi | Nguyên nhân thường gặp |
|--------|---------|--------------|------------------------|
| M1 Entity consistency | ≥ 0,98 | < 0,95 | Extractor gán sai `epistemic`; canon chưa lock |
| M2 Clue payoff rate | ≥ 0,90 | < 0,80 | `payoff_deadline` đặt quá xa; `ATTENTION_BUDGET` quá chặt |
| M3 Voice distinctiveness | ≥ 0,28 | < 0,20 | `signature_lexicon` quá chung; thiếu `forbidden_lexicon` |
| M4 NPC reuse | ≥ 0,45 | < 0,30 | `retroactive_promotion` chưa được Director gọi |
| M5 Relationship pacing | 0 vi phạm | ≥ 1 | Guard bị bỏ qua; Extractor cộng điểm intimacy quá tay |
| M6 Blind attribution | ≥ 0,70 | < 0,55 | Flattening đang xảy ra — sửa `VoiceFingerprint` |
| M7 Scene necessity | ≥ 0,85 | < 0,70 | `scene_must_change` chưa được cưỡng chế |
| M8 Tension MAE | ≤ 0,15 | > 0,25 | Model bỏ qua `decompress`; cần ràng buộc cứng hơn |

### 13.2 Bộ hồi quy (regression set)

Giữ 5 chương "vàng" đã được duyệt thủ công. Mỗi lần thay đổi prompt hoặc tham số, sinh lại 5 chương đó với cùng seed và so 8 metric. Nếu bất kỳ metric nào tụt quá 10%, rollback.

```python
# novel_engine/eval/regression.py
GOLDEN = ["ch007", "ch014", "ch022", "ch031", "ch045"]

def run_regression(engine, baseline: dict) -> dict:
    cur = {ch: engine.regenerate(ch, seed=42) for ch in GOLDEN}
    report, regressed = {}, []
    for name, fn in ALL_METRICS.items():
        v = fn(list(cur.values()))
        b = baseline[name]
        delta = (v - b) / max(abs(b), 1e-6)
        report[name] = {"baseline": b, "current": v, "delta_pct": round(delta*100, 1)}
        if delta < -0.10:
            regressed.append(name)
    return {"report": report, "regressed": regressed,
            "verdict": "ROLLBACK" if regressed else "PASS"}
```

### 13.3 Bốn metric bổ sung của v2.1

```python
# novel_engine/eval/metrics_v21.py

def M9_prose_rhythm(chapters, lang="vi") -> dict:
    """Nhịp trần thuật. σ thấp hoặc tỉ lệ mệnh đề phụ cao = văn đang đều đều."""
    import statistics
    sds, subs = [], []
    for ch in chapters:
        s = narrative_sentences(ch.prose)
        if len(s) < 8:
            continue
        sds.append(statistics.pstdev([len(x.split()) for x in s]))
        subs.append(sum(1 for x in s if any(x.startswith(o + " ")
                        for o in SUBORDINATE_OPENERS)) / len(s))
    return {"sd_mean": round(statistics.mean(sds), 2),
            "subordinate_ratio": round(statistics.mean(subs), 3),
            "ok": statistics.mean(sds) >= THRESH[lang]["sd_floor"]
                  and statistics.mean(subs) <= 0.30}


def M10_extraction_fidelity(reports) -> float:
    """Tỉ lệ span sống sót qua verify_spans. Thấp = Extractor đang bịa.
    Mục tiêu ≥0,95. Đây là metric CẢNH BÁO SỚM cho ô nhiễm canon."""
    kept = rejected = 0
    for r in reports:
        rejected += len(r["rejected_spans"])
        kept     += r["assertions_kept"]
    return kept / max(kept + rejected, 1)


def M11_plan_fulfillment(reports) -> float:
    """Tỉ lệ plant_directive thực sự xuất hiện trên trang giấy.
    Mục tiêu ≥0,85. Thấp = Writer đang bỏ qua chỉ thị, hoặc chỉ thị quá mờ."""
    return sum(r["plan_fulfillment_rate"] for r in reports) / max(len(reports), 1)


def M12_irony_gap(frames, graph) -> dict:
    """Khoảng cách giữa tri thức độc giả (narrative_order) và tri thức nhân vật
    (epoch_tick). Bằng 0 suốt = không có mỉa mai kịch tính nào.
    Quá cao kéo dài = độc giả phát bực vì nhân vật mãi không hiểu ra."""
    gaps = []
    for f in frames:
        pov = f.pov
        reader = graph.facts_known_to_reader(f.time.narrative_order)
        char   = graph.known_by(pov, f.time.epoch_tick)["known"]
        gaps.append(len(set(reader) - {c["id"] for c in char}))
    import statistics
    return {"mean_gap": round(statistics.mean(gaps), 2),
            "max_sustained": max_consecutive(gaps, lambda g: g > 6),
            "ok": 0.5 <= statistics.mean(gaps) <= 5.0}
```

Bổ sung vào bảng điều khiển §13.1:

| Metric | Mục tiêu | Báo động khi | Nguyên nhân thường gặp |
|--------|---------|--------------|------------------------|
| M9 Prose rhythm | σ ≥ 8,5 (vi); mệnh đề phụ ≤ 0,30 | σ < 7 | Thiếu style exemplar; Polish chưa nhận `rhythm_audit` |
| M10 Extraction fidelity | ≥ 0,95 | < 0,90 | Lượt 1 và lượt 2 bị gộp; prompt chưa cảnh báo span sẽ bị đối chiếu |
| M11 Plan fulfillment | ≥ 0,85 | < 0,70 | `intensity` quá thấp; `carrier` không có mặt trong cảnh |
| M12 Irony gap | 0,5 – 5,0 | = 0 kéo dài, hoặc > 6 quá 4 chương | Không dùng chương song song; News Dispatcher chưa bật |

M10 đáng được theo dõi sát nhất trong bốn metric này. Nó là chỉ báo **sớm** của ô nhiễm canon: khi nó tụt, canon vẫn còn sạch, nhưng chỉ vài chương nữa là không. Ba metric kia đo chất lượng đọc; M10 đo sự toàn vẹn của nền móng.

---

## 14. Cost & Performance

### 14.1 Ngân sách một chương

Ước tính cho một chương 3.000 từ, 6 cảnh (giá tham khảo tại thời điểm viết; hãy tự kiểm tra bảng giá hiện hành):

| Bước | Số lần gọi | Input (tok) | Output (tok) | Ghi chú |
|------|-----------|-------------|--------------|---------|
| Director | 6 | 3.000 | 700 | Phần lớn logic do code làm, LLM chỉ điền |
| Writer | 6 + ~2 revise | 7.500 | 1.400 | Chi phí chính |
| Auditor (code) | 6 | 0 | 0 | Miễn phí |
| Auditor (LLM) | ~4 | 5.000 | 800 | Bỏ qua khi code đã bắt blocker |
| Polish | 6 | 4.000 | 1.500 | |
| Extractor | 1 | 12.000 | 2.000 | |

Tổng ≈ 150k input + 25k output mỗi chương. Với prompt caching, phần tĩnh (hồ sơ nhân vật, style exemplars, checklist) chiếm khoảng 60% input và được giảm giá mạnh.

### 14.2 Bốn đòn tiết kiệm hiệu quả nhất

1. **Kiểm tra bằng code trước, LLM sau.** Auditor tầng thuật toán bắt 60–70% vấn đề với chi phí bằng 0, và cho phép bỏ hẳn lượt gọi LLM khi đã có blocker.
2. **Prompt caching cho phần tĩnh.** Sắp xếp prompt sao cho phần bất biến (checklist, voice sheet, exemplars) nằm ở ĐẦU, phần thay đổi (contract, context) nằm ở cuối.
3. **Bounded revision.** Giới hạn 2 vòng cắt phần đuôi chi phí — trong thực tế đây là thứ ngăn một chương "khó" ngốn gấp 8 lần một chương bình thường.
4. **Model nhỏ cho Extractor.** Trích xuất là bài toán có cấu trúc rõ; một model rẻ hơn làm tốt gần bằng với 1/5 chi phí.

```python
# novel_engine/llm/cached.py
from langchain_anthropic import ChatAnthropic

def cached_writer_messages(static_block: str, dynamic_block: str):
    """Phần static đặt trước và đánh dấu cache — tiết kiệm ~55% input cost."""
    return [
        {"role": "system", "content": [
            {"type": "text", "text": static_block,
             "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "user", "content": dynamic_block},
    ]
```

### 14.3 Song song hoá

Các cảnh trong cùng một chương **không** chạy song song được — cảnh 3 phụ thuộc trạng thái cuối cảnh 2. Nhưng ba việc sau chạy song song tốt:

- Auditor tầng code cho nhiều cảnh cùng lúc (thuần CPU).
- Chuẩn bị context cho cảnh $n+1$ trong lúc Writer đang viết cảnh $n$ (prefetch).
- Sinh nhiều **phương án** cho cùng một cảnh khi cần chọn tay (`n=3`, tác giả chọn).

---

## 15. Repo Layout & Lộ trình triển khai

### 15.1 Cấu trúc thư mục

```
novel-engine/
├── novel_engine/
│   ├── canon/
│   │   ├── models.py            # Entity, Relation, Clue, StateDelta
│   │   ├── timeline.py          # StoryTime, ContinuityFrame  (§3.6)
│   │   ├── flashback.py         # Forward-Reachability Audit (§3.6.4)
│   │   ├── graph_port.py        # Protocol
│   │   ├── networkx_graph.py    # triển khai nhẹ (bắt đầu ở đây)
│   │   ├── neo4j_graph.py       # triển khai đầy đủ
│   │   └── delta_log.py         # event store (SQLite/Postgres)
│   ├── memory/
│   │   ├── hierarchy.py         # L0–L4, MemoryBudget
│   │   ├── assembler.py         # ContextAssembler
│   │   └── repetition.py        # self-plagiarism
│   ├── character/
│   │   ├── models.py            # CharacterProfile, VoiceFingerprint, Flaw
│   │   ├── deliberation.py      # utility + flaw distortion
│   │   ├── voice_check.py
│   │   ├── firewall.py          # POV firewall
│   │   └── registry.py          # ChekhovRegistry
│   ├── foreshadow/
│   │   ├── scheduler.py         # ForeshadowScheduler
│   │   └── debt.py              # narrative_debt_report
│   ├── relationship/
│   │   ├── models.py
│   │   ├── machine.py           # guards
│   │   └── dynamics.py
│   ├── planner/
│   │   ├── tension.py
│   │   ├── beats.py
│   │   └── contract.py          # SceneContract
│   ├── world/
│   │   └── news.py              # News Dispatcher, Fog of War (§5.6)
│   ├── audit/
│   │   ├── ledger.py            # luật liên tục trên trục epoch
│   │   ├── timeline_rules.py    # bilocation, flashback, teleport (§3.6.1)
│   │   ├── rhythm.py            # nhịp trần thuật        (§10.3.1)
│   │   ├── contradiction.py
│   │   ├── prose.py             # cliché, giác quan, độ dài
│   │   └── attribution.py
│   ├── reconcile/
│   │   ├── classify.py
│   │   └── verify.py            # verify_spans, plan_coverage (§10.5)
│   ├── llm/
│   │   └── json_io.py           # strip_fences, parse_model (§9.5)
│   ├── graph/
│   │   ├── safety.py            # safe_node                (§9.5)
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── routing.py
│   │   └── build.py             # StateGraph
│   ├── eval/
│   │   ├── metrics.py
│   │   └── regression.py
│   └── prompts.py
├── bible/                        # đầu vào của tác giả — nguồn sự thật gốc
│   ├── world.yaml
│   ├── characters/*.yaml
│   ├── clues.yaml
│   └── outline.yaml
├── output/
│   ├── chapters/*.md
│   ├── deltas/*.json
│   └── reports/*.json
├── tests/
│   ├── test_scheduler.py        # ưu tiên: logic thuần, test dễ và có giá trị cao
│   ├── test_relationship.py
│   ├── test_deliberation.py
│   ├── test_continuity.py
│   ├── test_regressions_v22.py  # 5 lỗi runtime ở §0.2
│   ├── test_regressions_v23.py  # 8 lỗi thực thi ở §0.3
│   └── fixtures/
└── cli.py
```

### 15.2 Lộ trình 6 giai đoạn

Lời khuyên quan trọng nhất: **đừng dựng cả kiến trúc rồi mới viết chương đầu tiên.** Bạn sẽ tối ưu hoá cho những vấn đề tưởng tượng. Thứ tự dưới đây đưa bạn tới chương thật sớm nhất có thể.

**GĐ 1 — Walking skeleton (3–5 ngày)**
Mục tiêu: sinh được 1 chương, dù dở.
- `models.py` với Entity, Clue, CharacterProfile (bản rút gọn)
- `NetworkXGraph` — bỏ qua Neo4j hoàn toàn
- LangGraph 2 node: `writer → auditor`, không loop
- Bible viết tay cho 3 nhân vật, 5 manh mối
- Tiêu chí xong: chạy `python cli.py write --chapter 1` ra file .md

**GĐ 2 — Vòng ghi ngược (4–6 ngày)**
Mục tiêu: chương 2 biết chương 1 đã xảy ra gì.
- `extractor_node` + `StateDelta`
- `delta_log.py`, `ContextAssembler` với L1/L2
- Tiêu chí xong: viết 5 chương liên tiếp, không mâu thuẫn thô

**GĐ 3 — Cưỡng chế (5–7 ngày)**
Mục tiêu: chất lượng ngừng trôi.
- `POVFirewall` + `pov_leak_scan`
- `ContinuityFrame` + rules
- `voice_check` + VoiceFingerprint đầy đủ cho nhân vật chính
- Bounded critique loop
- Tiêu chí xong: M1 ≥ 0,95 trên 10 chương

**GĐ 4 — Tự sự dài hơi (5–7 ngày)**
Mục tiêu: manh mối và quan hệ vận hành đúng.
- `ForeshadowScheduler` + salience decay
- `narrative_debt_report`
- Relationship state machine + guards
- `ChekhovRegistry`
- Tiêu chí xong: M2 ≥ 0,85, M5 = 0 vi phạm trên 20 chương

**GĐ 5 — Đo lường (3–4 ngày)**
- 8 metric + regression set
- Dashboard đơn giản (một HTML tĩnh đọc `output/reports/*.json` là đủ)
- Tiêu chí xong: mỗi thay đổi prompt đều có số trước/sau

**GĐ 6 — Mở rộng (tuỳ nhu cầu)**
- Neo4j nếu graph > 2.000 node
- Nhiều POV, nhiều tuyến truyện song song
- Human-in-the-loop UI
- Vector store cho style exemplars

### 15.3 Viết test cho cái gì

Điểm mạnh của kiến trúc này là Tầng 2 hoàn toàn deterministic, nên test rất có giá trị và rất rẻ.

```python
# tests/test_scheduler.py
def test_khong_tra_bai_truoc_nguong():
    c = Clue(clue_id="C1", macro_event_target="E1", description="x",
             payoff_threshold=20, payoff_deadline=25,
             surface_forms=["a", "b"], status=ClueStatus.PLANTED)
    s = ForeshadowScheduler(FakeGraph(), {"C1": c})
    out = s.schedule(chapter=15, scene_affordances={"object": ["đèn"]},
                     pov_id="P1")
    assert all(d.mode != "payoff" for d in out)

def test_re_plant_khi_salience_tut():
    c = Clue(clue_id="C1", macro_event_target="E1", description="x",
             payoff_threshold=30, payoff_deadline=40,
             surface_forms=["a", "b", "c"], status=ClueStatus.PLANTED,
             salience=0.9, last_touched_chapter=5)
    s = ForeshadowScheduler(FakeGraph(), {"C1": c})
    out = s.schedule(chapter=15, scene_affordances={"object": ["đèn"]},
                     pov_id="P1")
    assert out and out[0].mode == "reinforce"
    assert out[0].surface_form != "a"      # không lặp lại hình thức cũ

def test_khong_nhay_coc_quan_he():
    st = RelationshipState(a="A", b="B", stage=RelationStage.FRICTION,
                           chapters_in_stage=5, intimacy=60,
                           shared_ordeals=[])          # thiếu nghịch cảnh
    ok, reason = can_advance(st, 10)
    assert not ok and "nghịch cảnh" in reason

def test_flaw_lam_lech_quyet_dinh():
    char = make_char(fatal_flaw=Flaw(name="kiêu ngạo",
                                     distortion_rule="...",
                                     trigger_conditions=["nhờ"],
                                     arc_direction="deepens"))
    opt_nho  = ActionOption(label="nhờ thợ máy giúp", serves=["sống sót"],
                            threatens=[], requires_skill=None, exposes_secret=None)
    opt_tu   = ActionOption(label="tự sửa một mình", serves=["sống sót"],
                            threatens=[], requires_skill="cơ khí",
                            exposes_secret=None)
    r = deliberate(char, [opt_nho, opt_tu], pressure=0.5)
    assert r["chosen"] == "tự sửa một mình"
```

Năm test dưới đây bắt đúng năm lỗi ở §0.2. Mỗi lỗi trong số đó đều từng **chạy không ném ngoại lệ** — đó là lý do chúng cần test, chứ không cần try/except.

```python
# tests/test_regressions_v22.py

def test_plan_coverage_dem_dung_manh_moi():
    """B1: clue_id vs Assertion.subject là hai không gian định danh."""
    delta = StateDelta(
        delta_id="d1", chapter=5, created_at=NOW,
        assertions=[Assertion(subject="con dấu", predicate="có", object="vết ăn mòn",
                              chapter=5, scene=1, span="con dấu hoen một vệt nâu",
                              confidence=0.9)],
        plant_evidence=[PlantEvidence(clue_id="CLUE_042", scene_id="CH005_S01",
                                      span="con dấu hoen một vệt nâu",
                                      carrier_used="object", verified=True)],
    )
    contracts = [{"scene_id": "CH005_S01",
                  "plant_directives": [{"clue_id": "CLUE_042", "mode": "plant"}]}]
    r = plan_coverage(contracts, delta)
    assert r["plan_fulfillment_rate"] == 1.0      # bản cũ trả 0.0
    assert r["missed_plants"] == []


def test_merge_giu_quan_he_phat_sinh():
    """B2: lượt 2 tìm ra mối thù ngoài kế hoạch — không được rơi."""
    a = StateDelta(delta_id="d", chapter=7, created_at=NOW).model_dump_json()
    e = StateDelta(delta_id="d", chapter=7, created_at=NOW,
                   new_relations=[Relation(src="CHAR_A", dst="CHAR_B",
                                           type="HAS_FEUD_WITH")]).model_dump_json()
    m = merge_extractions(a, e, chapter=7)
    assert len(m.new_relations) == 1              # bản cũ trả 0


def test_merge_khong_cong_don_quan_he_cap_doi():
    """B2b: nối đuôi relationship_updates làm intimacy bị cộng hai lần."""
    st = RelationshipState(a="CHAR_A", b="CHAR_B", intimacy=30)
    a = StateDelta(delta_id="d", chapter=7, created_at=NOW,
                   relationship_updates=[st]).model_dump_json()
    e = StateDelta(delta_id="d", chapter=7, created_at=NOW,
                   relationship_updates=[st]).model_dump_json()
    m = merge_extractions(a, e, chapter=7)
    assert len(m.relationship_updates) == 1


def test_tin_chinh_thong_khong_bi_tin_don_chan():
    """B3: rumor nhanh không được vĩnh viễn chặn courier chậm mà chính xác."""
    g = FakeRouteGraph(edges=[("LOC_CAP", "LOC_X", 50)])
    news = NewsItem(news_id="N1", origin_location="LOC_CAP", origin_tick=0,
                    truth="Kaelen bị truy nã", subject_entities=["CHAR_KAELEN"],
                    channels=["rumor", "courier"])
    arr = propagate(news, g, upto_tick=500, rng=random.Random(1))
    at_x = [a for a in arr if a["location"] == "LOC_X"]
    assert len(at_x) >= 2                                   # bản cũ chỉ có 1
    assert any(a["is_correction"] for a in at_x)
    assert max(a["payload"]["fidelity"] for a in at_x) > 0.8


def test_hoi_uc_khong_duoc_pha_nhan_qua():
    """B4: hồi ức gán thuộc tính vĩnh viễn sớm hơn thời điểm canon nói chưa có."""
    canon = FakeCanon(windows={("CHAR_KAELEN", "has_scar"):
                               AttributeWindow(entity="CHAR_KAELEN",
                                               attribute="has_scar",
                                               attested_absent=[2000],
                                               attested_present=[5000])})
    frame = ContinuityFrame(scene_id="CH015_S02",
                            time=StoryTime(epoch_tick=500, narrative_order=15,
                                           mode="flashback",
                                           anchor_scene="CH010_S01"),
                            locations={}, injuries={}, possessions={})
    delta = StateDelta(delta_id="d", chapter=15, created_at=NOW,
                       assertions=[Assertion(subject="CHAR_KAELEN",
                                             predicate="has_scar", object=True,
                                             chapter=15, scene=2,
                                             span="vết sẹo chéo trên trán cậu bé",
                                             confidence=0.9)])
    out = flashback_admissible(delta, frame, canon)
    assert any(f["check"] == "flashback_causality" and f["severity"] == "blocker"
               for f in out)


def test_parse_json_boc_trong_fence():
    """B5: và lấy khối DÀI NHẤT, không phải khối đầu tiên."""
    raw = ('Đây là kết quả:\n```json\n{"delta_id":"x"}\n```\n'
           'Chi tiết:\n```json\n{"delta_id":"d1","chapter":3,'
           '"created_at":"2026-01-01T00:00:00"}\n```')
    d = parse_model(raw, StateDelta)
    assert d.delta_id == "d1"        # không phải "x"
```

```python
# tests/test_regressions_v23.py

def test_scene_index_thuc_su_tang():
    """C1: không có test này thì lỗi chỉ lộ ra khi hoá đơn token về."""
    state = {"scene_index": 0, "contracts": [{}, {}, {}],
             "polished": "x", "findings": [], "revision_count": 2}
    out = scene_boundary_node(state, FAKE_CFG)
    assert out["scene_index"] == 1
    assert out["revision_count"] == 0          # C6

def test_vong_lap_canh_ket_thuc():
    """C1: mô phỏng cả vòng, đảm bảo nó DỪNG."""
    s = {"scene_index": 0, "contracts": [{}] * 6}
    for _ in range(20):
        if after_scene_boundary(s) == "extract":   # E7: after_polish nay
                                                  # luôn trả "scene_boundary"
            break
        s["scene_index"] += 1
    else:
        raise AssertionError("vòng lặp cảnh không kết thúc sau 20 bước")
    assert s["scene_index"] == 6

def test_extractor_doc_ca_chuong():
    """C2: span của cảnh đầu phải tìm được."""
    state = {"chapter": 3, "contracts": [{"scene_id": "S0"}, {"scene_id": "S1"}],
             "scene_outputs": [{"scene_id": "S0", "prose": "con dấu hoen vệt nâu"},
                               {"scene_id": "S1", "prose": "biển động dữ dội"}]}
    out = extractor_node(state, FAKE_CFG)
    assert out["extraction_report"]["rejected_spans"] == []

def test_manh_moi_qua_han_van_duoc_xep_lich():
    """C3: deadlock. Đây là test quan trọng nhất trong file."""
    c = Clue(clue_id="C1", macro_event_target="E1", description="x",
             payoff_threshold=10, payoff_deadline=20,
             surface_forms=["a", "b"], status=ClueStatus.PLANTED)
    s = ForeshadowScheduler(FakeGraph(pov_blind=True), {"C1": c})
    out, esc = s.schedule(chapter=25, scene_affordances={}, pov_id="P1")
    assert out and out[0].mode == "payoff"     # bản cũ trả []
    assert out[0].carrier == "dialogue"        # van thoát nấc 2

def test_manh_moi_bo_qua_lau_thi_escalate():
    """C3 nấc 3: không tự quyết, đẩy lên tác giả."""
    c = Clue(clue_id="C1", macro_event_target="E1", description="x",
             payoff_threshold=5, payoff_deadline=10,
             surface_forms=["a"], status=ClueStatus.PLANTED)
    s = ForeshadowScheduler(FakeGraph(), {"C1": c})
    out, esc = s.schedule(chapter=25, scene_affordances={}, pov_id="P1")
    assert esc and esc[0]["clue_id"] == "C1"

def test_nhan_vat_duoc_phep_noi_doi():
    """C5: lời nói dối không được làm dừng chương."""
    a = Assertion(subject="FLEET_3", predicate="trạng_thái", object="đã bị xoá sổ",
                  chapter=8, scene=2, span="Hạm Đội Số 3 đã bị xoá sổ rồi",
                  confidence=0.9, epistemic="claimed_by", holder="CHAR_SERENA")
    delta = StateDelta(delta_id="d", chapter=8, created_at=NOW, assertions=[a])
    r = classify_delta(delta, FakeGraphWithTruth(), FakePlanner())
    assert "contradiction" not in r["classification"].values()
    assert r["irony_seeds"] and r["irony_seeds"][0]["holder"] == "CHAR_SERENA"

def test_quan_he_thuc_su_tien_giai_doan():
    """C8: bản cũ tích đủ điểm nhưng nhãn không bao giờ đổi."""
    st = RelationshipState(a="A", b="B", stage=RelationStage.FRICTION,
                           chapters_in_stage=4, intimacy=35,
                           shared_ordeals=["hầm mỏ sập"])
    d = apply_relationship_advancement(st, chapter=12)
    assert d["action"] == "advance"
    assert st.stage == RelationStage.VULNERABILITY
    assert st.chapters_in_stage == 0

def test_bo_dem_chuong_khong_chay_theo_canh():
    """C7: 6 cảnh trong một chương chỉ được tính là MỘT chương."""
    st = RelationshipState(a="A", b="B", chapters_in_stage=0,
                           last_counted_chapter=None)
    for _ in range(6):
        apply_scene_effects(st, {}, chapter=9)
    assert st.chapters_in_stage == 1           # bản cũ cho 6
```

---

## 16. Phụ lục

### 16.1 Human-in-the-loop — bốn checkpoint

Lỗ hổng L7. Hệ thống chạy 40 chương rồi tác giả mới phát hiện arc sai từ chương 6 là kịch bản tệ nhất. Bốn điểm dừng bắt buộc:

| Checkpoint | Khi nào | Tác giả quyết định gì |
|-----------|---------|----------------------|
| **CP-1 Arc gate** | Trước mỗi arc (8–12 chương) | Duyệt outline arc + danh sách manh mối sẽ cài |
| **CP-2 Canon commit** | Trước `reconcile_node` mỗi chương | Duyệt diff của StateDelta — đây là `interrupt_before` trong §9.4 |
| **CP-3 Escalation** | Khi bounded loop hết lượt | Giải quyết BLOCKER mà agent không sửa được |
| **CP-4 Plan patch** | Khi có IMPROVEMENT đụng kế hoạch hạ nguồn | Giữ chi tiết mới & sửa kế hoạch, hay gỡ chi tiết |

CP-2 nên hiển thị dạng diff ngắn, không phải JSON thô:

```
Chương 14 đề xuất ghi vào canon:
  + Thực thể mới: "Xưởng đúc Vệ Đà" (location)      [ENRICHMENT]
  + Quan hệ: Serena -[SUSPECTS 0.6]-> Kaelen        [ENRICHMENT]
  ~ Clue CLUE_042: planted → reinforced             [ENRICHMENT]
  ! Assertion: "Kaelen có em gái"                   [IMPROVEMENT]
      ⚠ Xung đột với beat chương 30: "người cuối cùng của dòng họ"
      [G] Giữ chi tiết mới, sửa chương 30
      [B] Bỏ chi tiết, viết lại đoạn 3 chương 14
```

### 16.2 Checklist trước khi viết chương đầu tiên

- [ ] Mỗi nhân vật chính có `want` và `need` **mâu thuẫn nhau** (không phải hai cách nói của cùng một thứ)
- [ ] Mỗi `fatal_flaw` có `distortion_rule` mô tả bằng **hành vi**, không bằng tính từ
- [ ] Mỗi `VoiceFingerprint` có ≥5 từ trong `forbidden_lexicon` (danh sách cấm quan trọng hơn danh sách nên dùng)
- [ ] `somatic_signature` không chứa bất kỳ mục nào trong `CLICHE_SOMATICS`
- [ ] Mỗi manh mối có `payoff_deadline`, không chỉ `payoff_threshold`
- [ ] Mỗi manh mối có ≥3 `surface_forms` khác nhau về hình thức
- [ ] Đồ thị manh mối là DAG thật — chạy `networkx.is_directed_acyclic_graph`
- [ ] Mỗi nhân vật đồng hành có `private_agenda` không trùng với mục tiêu của nhân vật chính
- [ ] Mỗi cặp quan hệ chính có `stake_conflict` **cấu trúc**, không chỉ `friction`
- [ ] Có ít nhất 3 `THEMATIC_MIRROR` NPC được lên kế hoạch trước cho toàn truyện
- [ ] Đồ thị tuyến đường có `latency_ticks` cho mọi cặp địa điểm mà nhân vật di chuyển giữa chúng (§3.6.1 cần nó, §5.6 cũng vậy)
- [ ] Đã chọn độ hạt `TICKS_PER_HOUR` và ghi vào một chỗ duy nhất
- [ ] `mean_sentence_len` trong mọi `VoiceFingerprint` đã hiệu chỉnh theo tiếng Việt, không lấy nguyên ngưỡng tiếng Anh (§10.3.3)
- [ ] Mỗi `NewsItem` quan trọng có ≥2 kênh truyền với độ trễ khác nhau
- [ ] Đã chạy thử `verify_spans` trên một chương viết tay để kiểm ngưỡng fuzzy 0,90 không quá chặt

### 16.3 Bảng thuật ngữ

| Thuật ngữ | Nghĩa trong hệ thống này |
|-----------|--------------------------|
| **Canon** | Tập sự thật đã được xác nhận về thế giới truyện |
| **StateDelta** | Gói thay đổi canon do một chương sinh ra |
| **Assertion** | Một mệnh đề trích từ văn xuôi, kèm trích dẫn làm bằng chứng |
| **Epistemic tag** | Nhãn phân biệt sự thật khách quan / điều nhân vật tin / điều nhân vật nói |
| **POV Firewall** | Lớp lọc đảm bảo Writer chỉ thấy điều nhân vật góc nhìn biết |
| **Salience** | Độ "còn trong trí nhớ độc giả" của một manh mối, suy giảm theo hàm mũ |
| **Surface form** | Một cách hiện hình cụ thể của manh mối trong văn xuôi |
| **Narrative debt** | Nợ tự sự: manh mối chưa trả bài, NPC chưa tái xuất hiện |
| **Attention budget** | Trần tổng "trọng lượng chú ý" mà một chương chịu được |
| **Stake conflict** | Xung đột lợi ích cấu trúc — không hoà giải được bằng đối thoại |
| **Scar** | Tổn thương vĩnh viễn, đặt trần cho intimacy tương lai |
| **Drift reconciliation** | Quy trình xử lý khi Writer đi chệch kế hoạch |
| **Bounded critique** | Vòng phê bình có giới hạn số lượt, tránh lặp vô hạn |
| **Blind attribution** | Test đo độ phân biệt giọng nhân vật bằng cách giấu tên người nói |

### 16.4 Ba sai lầm phổ biến nhất khi triển khai

**Sai lầm 1 — Để LLM quyết định logic tự sự.** "Hỏi model xem chương này nên cài manh mối nào" nghe có vẻ linh hoạt, nhưng model không có trí nhớ về lịch trình 40 chương và sẽ chọn theo cái gì nổi bật trong prompt. Logic lịch trình phải là code.

**Sai lầm 2 — Prompt phình theo số chương.** Nếu prompt chương 30 dài gấp ba prompt chương 3, kiến trúc đã hỏng. Ngân sách token phải **hằng số** theo số chương — đó chính là điều Memory Hierarchy đảm bảo.

**Sai lầm 3 — Đánh giá bằng cảm giác.** "Chương này đọc hay hơn" không phải dữ liệu. Sau 20 chương bạn sẽ không còn nhớ chương 4 hay thế nào để so sánh. Regression set và 8 metric tồn tại chính vì lý do đó.

---

### Hướng đi tiếp theo

Ba nhánh có thể đào sâu, theo thứ tự lợi ích giảm dần đối với chất lượng cuối cùng:

1. **Foreshadowing Engine** — mở rộng sang mô hình nhiều tầng manh mối (manh mối dẫn tới manh mối), bài toán lập lịch với ràng buộc POV đa góc nhìn, và cơ chế "red herring" có kiểm soát.
2. **Character Engine** — mô hình arc nhân vật dài hạn: khi nào `fatal_flaw` chuyển từ `deepens` sang `transmutes`, và làm sao cưỡng chế sự chuyển hoá đó diễn ra dần dần thay vì đột ngột ở chương áp chót.
3. **World Graph** — sinh lịch sử thế giới bằng mô phỏng: chạy 200 năm lịch sử bằng agent-based simulation của các phe phái, lấy kết quả làm nền lore. Cách này cho ra mạng lưới thù oán có tính nội tại cao hơn nhiều so với viết tay.
