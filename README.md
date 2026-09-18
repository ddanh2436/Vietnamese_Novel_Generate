# Novel Engine

Động cơ sinh tiểu thuyết dài bằng LLM, viết theo `NOVEL_ENGINE_v2_Architecture.md`.
Chạy trên Windows, SQLite + NetworkX, không Docker.

Nguyên tắc chi phối mọi thứ bên dưới: **LLM diễn đạt, không quyết định.** Model
viết văn xuôi và đọc văn xuôi; mọi thứ quyết định tính đúng sai — thời gian,
tri thức nhân vật, trạng thái manh mối, quan hệ — do code giữ và do code suy ra
từ bằng chứng có trích dẫn nguyên văn.

## Chạy thử trong ba mươi giây

```bash
python cli.py write  --chapter 1            # backend `fake`: miễn phí, tất định
python cli.py show   --chapter 1
python cli.py review --chapter 1            # xem trước những gì sẽ vào canon
python cli.py commit --chapter 1
python cli.py status
```

Dùng Gemini thì thêm `--llm gemini` và đặt `GEMINI_API_KEY` trong `.env`. Mặc
định là `fake` **một cách có chủ ý**: chạy `cli.py` mà vô tình đốt hạn ngạch vì
quên đặt biến môi trường là chuyện không nên xảy ra.

## Vòng đời một chương

```
outline.yaml
    │
    ├─► Director    dựng SceneContract cho từng cảnh (POV, mục tiêu, nhịp,
    │               chỉ thị cài manh mối, chỉ thị quan hệ)
    │
    ├─► Writer      viết văn xuôi  ◄──┐
    │                                 │ vòng phê bình có trần (§9.3)
    ├─► Auditor     dò lỗi ──────────┘  blocker ≤2 lần viết lại rồi leo thang
    │                                   major 1 lần rồi chuyển polish
    ├─► Polish      sửa câu chữ, KHÔNG sửa nội dung
    │
    ├─► Extractor   rút mệnh đề + bằng chứng manh mối, mỗi mục kèm span nguyên văn
    │
    ├─► verify      span nào không tìm thấy trong văn xuôi thì LOẠI (NT-5)
    │
    └─► reconcile   phân loại delta → tác giả duyệt → ghi canon
```

Văn xuôi ra `output/chapters/`, báo cáo ra `output/reports/`, canon vào
`novel_storage.db`.

## Canon là một chuỗi sự kiện, không phải một bảng

`WorldState` không được lưu. Nó là kết quả **fold** toàn bộ `StateDelta` đã ghi,
tính lại mỗi lần dựng động cơ. Hệ quả thực tế: quay lui một chương = xoá delta
của nó rồi dựng lại, không có bước "sửa ngược" nào cần viết đúng.

Hai trục thời gian tách hẳn nhau, và trộn chúng là nguồn lỗi kinh điển:

| trục | là gì | dùng để |
|---|---|---|
| `epoch_tick` | thời gian TRUYỆN | nhân vật biết gì tại thời điểm này |
| `narrative_order` | thứ tự ĐỌC | độc giả đã biết gì tới lúc này |

Một cảnh hồi ức ở Chương 2 nằm ở tick 2480 trong khi truyện đã ở tick 20600. Lọc
tri thức theo *chương* thay vì theo *tick* khiến nhân vật trong hồi ức "biết"
mọi thứ học được suốt 18.000 tick sau đó — và văn xuôi vẫn đọc trôi chảy, nên
không ai phát hiện.

## Ba lớp tường lửa POV

1. **Bộ nhớ** (`filter_memory`) — sự thật lọc theo `known_by(pov, epoch_tick)`.
2. **Hợp đồng cảnh** (`filter_scene_contract`) — nội tâm của nhân vật không phải
   POV bị lột sạch; hành động ngầm chỉ còn một dấu vết vật lý mà POV hiểu sai.
3. **Sau khi viết** (`pov_leak_scan`) — dò mẫu rò rỉ trong văn xuôi đã sinh.

Tầng nhớ-lại-theo-nội-dung (L5, xem dưới) chịu chung kỷ luật này: nó chỉ trả về
cảnh của **chính POV**.

## Bộ nhớ năm tầng

| tầng | nội dung | chọn theo |
|---|---|---|
| L1 | 2 cảnh liền trước, nguyên digest | vị trí |
| L2 | 5 chương gần nhất | vị trí |
| L3 | các arc đã đóng | vị trí |
| L4 | sự thật liên quan, đã lọc qua tường lửa | điểm liên quan |
| **L5** | cảnh CŨ cùng POV, liên quan theo nội dung | BM25 |

L5 giải bài toán mà L1–L3 không với tới: một chi tiết ở Chương 3 được gọi lại ở
Chương 37. Dùng BM25 thuần Python thay vì vector store, vì máy đích 7,7 GB RAM
và vì bộ hồi quy đòi cùng seed cho cùng kết quả. Đánh đổi: không bắt được diễn
đạt khác chữ — xem docstring `memory/recall.py`.

Ngân sách token là **cứng**. L4 trả nhiều hơn thì Assembler cắt theo điểm liên
quan, không nới ngân sách.

## Kiểm toán: cái gì code đo, cái gì LLM đọc

Chia theo một ranh giới: **code đo được thì code đo**, và chỉ những gì cần đọc
hiểu mới đưa cho LLM.

Tất định (`novel_engine/audit/`): sáo ngữ, nhịp câu, lặp cảnh (3-gram
containment), ngân sách tật ngôn ngữ và cử chỉ, vệ sinh văn bản, rò rỉ POV.
LLM (Auditor Agent): tri thức POV, cảnh không đổi trạng thái, subtext, nhân vật
nhượng bộ quá dễ.

Kiểm tra **văn phong** không bao giờ kích hoạt viết lại — chỉ đi polish. Đưa
ngưỡng thống kê cho Writer sẽ sinh ra văn xuôi thoả mãn con số và đọc như máy.

Cùng lý do đó, `signature_lexicon` **không** vào prompt Writer. Nó là thước đo
của M3 và của ngân sách tật ngôn ngữ; đưa thước cho người bị đo thì đo cái gì.
Writer nhận `voice_exemplars` — câu mẫu dạy CÁCH nói — và câu mẫu không được
chứa cụm nào đang bị đo (có test chốt).

## Bộ chỉ số M1–M15

`python cli.py eval --from 1 --to 5 [--baseline eval_cu.json]`

M1–M12 theo §13. M13–M15 thêm sau một lượt đọc bằng mắt cho thấy bản đạt gần hết
M1–M12 vẫn có ba cảnh liên tiếp dựng lại cùng một cảnh:

| | đo | mục tiêu |
|---|---|---|
| M13 | trùng lặp cao nhất giữa hai cảnh | ≤ 0,085 |
| M14 | một cử chỉ nhận dạng lặp nhiều nhất trong một chương | ≤ 3 |
| M15 | rác còn sót (chỉ số beat, lỗi gõ, ký tự lạ) | 0 |

`value = None` nghĩa là **chưa đo được**, không phải 0. M6/M7 cần một LLM judge
(`--judge gemini`); không có judge thì chúng nói thế, chứ không báo điểm 0.

Bộ hồi quy biết metric nào **thấp thì tốt** — nếu không, trùng lặp giảm sẽ bị
tính là tụt.

## Dựng giao diện: dùng `novel_engine/api.py`

`cli.py` trộn ba việc: truy vấn động cơ, định dạng bảng terminal, trả mã thoát.
Giao diện chỉ cần việc đầu, nên có một mặt tiền riêng.

```python
from novel_engine import api

api.trang_thai(db)              # tổng quan + liên tục toàn cục
api.danh_sach_chuong(db)        # CẢ outline, kèm trạng thái thực tế
api.doc_chuong(db, 3)           # văn xuôi từng cảnh + chỉ số đọc được
api.nhan_vat(db)                # hồ sơ giọng, đủ để dựng khung chỉnh giọng
api.manh_moi(db)                # bảng theo dõi phục bút
api.cham_diem(db, 1, 5)         # M1–M15, không tốn hạn ngạch

api.viet_chuong(db, 4, "gemini", on_event=print)   # tiến độ từng bước
```

Ba giao kèo, đều có test:

- không `print`, không `sys.exit`, không `argparse` — lỗi ném ra như lỗi;
- mọi giá trị trả về `json.dumps` được;
- tên khoá là hợp đồng, đổi tên khoá là đổi API.

`viet_chuong` ném `FileExistsError` nếu chương đã có: viết đè là thao tác **phá
huỷ** (xoá frame, digest, delta), nên giao diện phải hỏi trước, y như CLI bắt
`--force`. Lỗi ném ra từ `on_event` không giết chương đang viết.

## Bible

```
bible/
  characters/{kaelen,serena,vhal}.yaml   giọng, cử chỉ, tri thức riêng, lằn ranh đạo đức
  outline.yaml      dàn ý từng chương/cảnh — đầu vào của TÁC GIẢ, hệ thống không tự bịa
  clues.yaml        manh mối, hạn trả bài, DAG tiền đề
  relationships.yaml, news.yaml, world.yaml, predicates.yaml
```

Nhân vật mới hay chương mới đều bắt đầu từ đây, không từ prompt.

## Kiểm thử

```bash
python -m pytest tests -q          # ~677 bài, khoảng hai phút, không gọi mạng
```

Mọi bài chạy bằng `FakeLLM`. Nhiều bài mang theo một con số đo được trên văn bản
thật trong docstring — đó là lý do tồn tại của ngưỡng, và là thứ cần đọc trước
khi đổi ngưỡng.
