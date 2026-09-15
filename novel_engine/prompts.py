"""Prompt Library (§12).

Bốn nguyên tắc áp cho toàn bộ thư viện:

1. Không đưa tính từ tính cách. Đưa ràng buộc hành vi.
2. Không đưa `clue.description`. Đưa `surface_form` + cường độ.
3. Không đưa thông tin ngoài tầm POV trừ khi kèm nhãn cấm rõ ràng.
4. Ràng buộc phủ định phải đi kèm phương án thay thế. "Đừng dùng 'tim đập
   thình thịch'" một mình sẽ khiến model dùng một sáo ngữ khác.

NT-7: chỉ số văn phong là công cụ CHẨN ĐOÁN, không phải mục tiêu đưa cho
Writer. Đưa ngưỡng thống kê (σ độ dài câu, tỉ lệ mệnh đề phụ) vào prompt sẽ
sinh ra văn xuôi thoả mãn con số và đọc như máy (§10.3.2). Vì vậy WRITER_TMPL
nhận `register` và `signature_lexicon` nhưng KHÔNG nhận các khoảng số.

Dùng `.format_map()` với dict, không `.format(**kw)`: template có dấu ngoặc
nhọn trong ví dụ JSON, và `.format` sẽ vấp chúng.
"""
from __future__ import annotations

WRITER_TMPL = """Bạn viết một cảnh tiểu thuyết bằng tiếng Việt. Viết văn xuôi, không viết tóm tắt.

## GÓC NHÌN — RÀNG BUỘC CỨNG
Người kể: {pov_name} ({pov_id}), ngôi thứ ba giới hạn.
Bạn CHỈ được viết những gì nhân vật này tri giác và suy nghĩ.

Nhân vật này KHÔNG biết những điều sau. Tuyệt đối không viết ra chúng,
kể cả dưới dạng "anh không biết rằng...":
{knowledge_boundary}

Những điều nhân vật này chỉ NGHI NGỜ — phải viết như phỏng đoán, kèm khả năng sai:
{hypotheses}

## NHÂN VẬT TRONG CẢNH
{active_characters}

Với mỗi nhân vật KHÔNG phải POV: nếu họ có `director_only.hidden_action`, bạn
biết hành động ngầm đó nhưng POV thì không. Hãy để nó tạo ra một DẤU VẾT QUAN
SÁT ĐƯỢC mà POV nhìn thấy nhưng diễn giải sai hoặc bỏ qua. Cấm mọi câu tường
thuật nội tâm, ý định hay cảm xúc của nhân vật đó.

## GIỌNG NHÂN VẬT
Mỗi nhân vật có `voice_reminder`:
- không được dùng từ trong `forbidden_lexicon`
- dấu vết của `signature_lexicon` xuất hiện trong THOẠI của chính nhân vật đó.
  KHÔNG rải các cụm này vào lời kể, kể cả khi nhân vật đó là người kể: lời kể
  mang giọng nhân vật qua cách chọn chi tiết và nhịp câu, không qua khẩu ngữ.
- giữ đúng `register` và `syntactic_tic`
Dùng phản ứng cơ thể trong `somatic_allowed`. CẤM mọi biểu hiện trong
`somatic_forbidden` — đó là sáo ngữ; thay bằng một mục trong `somatic_allowed`.

## NHIỆM VỤ TỰ SỰ
Câu hỏi kịch tính: {dramatic_question}
Cảnh BẮT ĐẦU ở trạng thái: {entry_state}
Cảnh PHẢI KẾT THÚC ở trạng thái: {exit_state}
Điều bắt buộc thay đổi: {scene_must_change}
Một cảnh kết thúc nguyên trạng là một cảnh hỏng.

Nhịp: {tension_mode} — {tension_note}
Loại áp lực chủ đạo: {pressure_type}

## MANH MỐI CẦN CÀI
{plant_directives}

Với mỗi mục: dùng đúng `surface_form` đã cho, đặt qua `carrier` đã chỉ định,
theo đúng `instruction`. Không thêm câu nào bình luận về ý nghĩa của nó.
Nếu `intensity` < 0.35: chi tiết phải xuất hiện khi sự chú ý của độc giả
đang hướng về chuyện khác.

## QUAN HỆ
{relationship_directives}
Với mỗi mục có `scene_requirement`: đó là ràng buộc, không phải gợi ý.

## VĂN PHONG
Ngân sách: {word_min}–{word_max} từ.
Tối đa {max_explicit_goal_statements} lần một nhân vật nói thẳng mục tiêu của
mình. Mọi mục tiêu khác phải lộ qua hành động hoặc qua điều nhân vật TRÁNH nói.
Cần ≥{sensory_channels_required} kênh giác quan, phải có ít nhất một kênh
không phải thị giác.
Không quá 3 câu trong cảnh được mở đầu bằng "Khi", "Trong khi" hoặc
"Sau khi". Với các câu còn lại, đặt chủ ngữ hoặc hành động lên đầu.
Subtext: {subtext_requirement}
Cụm từ bị cấm trong cảnh này: {forbidden_cliches}

## TIN TỨC NGƯỜI KỂ VỪA NGHE
{news}
Tin chỉ đến qua kênh ghi kèm (người đưa tin, tin đồn, tín hiệu, đoàn buôn). Cho
người kể NHẬN tin ngay trong cảnh, không kể lại như điều đã biết từ trước.

## BỐI CẢNH ĐÃ XẢY RA
Cảnh liền trước: {recent_scenes}
Các chương gần đây: {recent_chapters}
Bối cảnh xa: {arc_history}
Sự thật POV nắm được: {known_facts}

{feedback}

Viết cảnh. Chỉ xuất văn xuôi, không tiêu đề, không ghi chú.
"""


DIRECTOR_TMPL = """Bạn là trợ lý đạo diễn. Khung hợp đồng cảnh dưới đây đã
được hệ thống dựng sẵn; mọi quyết định về thời gian, manh mối, nhân vật và
địa điểm ĐÃ CHỐT. Việc của bạn chỉ là điền bốn trường tự sự còn để trống.

## KHUNG HỢP ĐỒNG (chỉ đọc)
{contract}

## BEAT
Chức năng: {beat_function}
Dàn ý chương: {outline}
Nợ tự sự đang treo: {debt}

Xuất JSON với ĐÚNG bốn khoá sau, không thêm khoá nào khác:

{{
  "scene_must_change": "điều gì PHẢI khác đi khi cảnh kết thúc — một mệnh đề cụ thể, kiểm chứng được",
  "entry_state": "trạng thái quan hệ/thông tin lúc cảnh mở ra",
  "exit_state": "trạng thái ấy đã đổi thành gì",
  "subtext_requirement": "điều không ai nói ra nhưng người đọc phải cảm được"
}}

Chỉ xuất JSON.
"""


# F2 + lỗi định danh: §9.2 truyền `[x["name"] for x in active_characters]` vào
# prompt này, trong khi `ContinuityFrame.locations` khoá theo `char_id` và
# `character_track(cid, frames)` (§3.6.1) tra bằng id. Model nhận tên thì sẽ
# khoá `locations` bằng tên, và MỌI luật liên tục trở thành mã chết: chúng
# không ném lỗi, chúng chỉ không khớp mãi mãi (NT-8).
#
# Sửa bằng cách đưa ID làm khoá và tên chỉ làm chú thích trong ngoặc, kèm một
# câu lệnh tường minh. Schema không diễn đạt được ràng buộc này (khoá dict là
# chuỗi tự do), nên prompt phải gánh — và `scene_boundary_node` kiểm lại sau.
#
# VÍ DỤ PHẢI DÙNG MÃ GIẢ. Bản đầu tiên lấy `CHAR_KAELEN` làm ví dụ minh hoạ,
# và lượt chạy thử đầu tiên cho thấy Kaelen xuất hiện trong `locations` của
# một cảnh anh ta KHÔNG có mặt — model (và FakeLLM) nhặt mã ấy ra từ chính ví
# dụ. Cùng cơ chế với cái bẫy mà `strip_fences` chặn ở §9.5: đưa dữ liệu thật
# vào phần minh hoạ thì không phân biệt được minh hoạ với kết quả.
SCENE_DIGEST_TMPL = """Bạn chốt sổ một cảnh vừa viết xong. Bạn là thư ký
trường quay, không phải biên tập viên: ghi lại cái đã xảy ra, không đánh giá.

## HỢP ĐỒNG CẢNH (mốc thời gian đã được hệ thống ấn định, ĐỪNG tự đặt lại)
scene_id: {scene_id}
epoch_tick bắt đầu: {epoch_tick}
duration_ticks dự kiến: {planned_duration}
Nhân vật có mặt: {present}
Địa điểm của cảnh: {location_id}

## MÃ ĐỊA ĐIỂM HỢP LỆ — TẬP ĐÓNG
Chỉ được dùng một trong các mã sau. KHÔNG tự đặt mã mới, kể cả khi văn xuôi
nhắc tới một góc phòng hay một khu vực nhỏ hơn — hãy quy nó về địa điểm bao
trùm trong danh sách này:
{valid_locations}

## VĂN XUÔI
{prose}

Xuất JSON đúng schema SceneClose:

1. `digest` — 5–8 câu: ai làm gì, đổi gì, kết thúc ở trạng thái nào. Viết cho
   một người sẽ đọc nó ở chương sau mà không đọc lại cảnh này. Bỏ mọi chi
   tiết văn phong; giữ mọi chi tiết có hệ quả.

2. `continuity` — trạng thái VẬT LÝ tại thời điểm cảnh kết thúc:
   - `locations`: KHOÁ PHẢI LÀ MÃ ĐỊNH DANH dạng CHAR_* đúng như trong
     "Nhân vật có mặt" ở trên — KHÔNG dùng tên thường. Giá trị là mã địa điểm
     dạng LOC_*. Chỉ liệt kê nhân vật CÓ TRONG danh sách trên, không thêm ai.
     Đúng:  {{"CHAR_MOT": "LOC_MOT"}}
     Sai:   {{"Tên Thường": "Tên Địa Điểm Thường"}}
   - `injuries`: vết thương CÒN HIỆU LỰC, cũng khoá theo mã CHAR_* (gồm cả
     vết thương có từ trước mà văn bản không nói là đã lành)
   - `possessions`: vật mang theo có vai trò trong truyện, khoá theo mã CHAR_*;
     bỏ qua quần áo và vật dụng thông thường
   - `weather`: chỉ điền nếu văn bản có nói

3. `actual_duration_ticks` — số giờ truyện thực sự trôi qua trong cảnh. Lệch
   nhiều so với dự kiến là thông tin hữu ích, không phải lỗi.

4. `unresolved` — những gì cảnh mở ra mà chưa đóng lại (một câu hỏi bị bỏ
   lửng, một người bước vào mà chưa rời đi, một tiếng động chưa ai kiểm tra).

Chỉ xuất JSON.
"""


AUDITOR_TMPL = """Bạn thẩm định một cảnh tiểu thuyết theo hợp đồng cảnh.
Phần kiểm tra máy móc (độ dài, sáo ngữ, giọng nhân vật, nhịp câu, mẫu rò rỉ
góc nhìn hiển nhiên) đã chạy rồi. Bạn CHỈ đánh giá những điều máy không làm được.

## HỢP ĐỒNG
{contract}

## VĂN XUÔI
{prose}

## KIỂM TRA — `check` chỉ được là một trong các tên sau
{checklist}

## MỨC ĐỘ
- blocker: CHỈ dùng cho `pov_knowledge` — văn xuôi tường thuật trực tiếp điều
  người kể không thể biết.
- major: cảnh hỏng chức năng (không có thay đổi trạng thái, bí mật bị nói toạc).
- minor: có thể tốt hơn nhưng cảnh vẫn đứng được.

## BẰNG CHỨNG
Mỗi mục PHẢI có `evidence`: một câu hoặc cụm TRÍCH NGUYÊN VĂN từ văn xuôi trên
— chép y hệt, không diễn đạt lại, không rút gọn. Mục không trích được nguyên văn
sẽ bị loại. Với `scene_must_change`, trích câu cuối của cảnh.

Xuất JSON: {{"findings": [{{"severity": "major", "check": "subtext",
"message": "vấn đề cụ thể", "evidence": "trích nguyên văn"}}]}}

Không có vấn đề thì trả {{"findings": []}}. Đừng bịa ra vấn đề để tỏ ra hữu ích.
Chỉ xuất JSON.
"""


POLISH_TMPL = """Bạn trau chuốt VĂN PHONG của một cảnh đã viết xong.

TUYỆT ĐỐI KHÔNG đổi nội dung: không thêm/bớt sự kiện, không đổi ai làm gì,
không thêm chi tiết mới về thế giới. Chỉ sửa câu chữ.

## GHI CHÚ CẦN XỬ LÝ
{notes}

## GIỌNG NHÂN VẬT
{voice_sheets}

## CỤM TỪ CẤM
{banned}

## VĂN XUÔI
{prose}

Chỉ xuất văn xuôi đã trau chuốt, không giải thích.
"""


# ═══════════════════ EXTRACTOR — HAI LƯỢT TÁCH BIỆT (§12.3) ═══════════════════
#
# Hai template dưới đây KHÔNG BAO GIỜ được gộp làm một: lượt 2 mất giá trị
# ngay khi nó nhìn thấy kế hoạch. Model bị neo vào contract sẽ bỏ qua đúng
# những thứ nằm ngoài kế hoạch — tức toàn bộ mục đích của lượt này.

EXTRACT_DIFF_TMPL = """Bạn là kiểm toán viên. Bạn có KẾ HOẠCH của chương và
VĂN BẢN đã viết. Nhiệm vụ: đối chiếu, không phải đọc hiểu tự do.

## KẾ HOẠCH (SceneContract của từng cảnh)
{contracts}

## VĂN BẢN (chương {chapter}) — mỗi cảnh có nhãn [SCENE_ID] ở đầu
{prose}

## DANH SÁCH VỊ TỪ HỢP LỆ — TẬP ĐÓNG
Chỉ áp cho mệnh đề `objective`. Lời nhân vật nói (`claimed_by`) và điều
nhân vật tin (`believed_by`) không bị giới hạn bởi danh sách này.
{predicates}

Trả lời đúng hai câu hỏi, theo thứ tự:

### CÂU 1 — KẾ HOẠCH CÓ ĐƯỢC THỰC HIỆN KHÔNG?
Với TỪNG mục trong `plant_directives` và `relationship_directives`:
- Nếu CÓ trong văn bản: trích `span` NGUYÊN VĂN (≥12 ký tự, sao chép chính xác
  từng chữ kể cả dấu, không diễn giải, không rút gọn).
- Nếu KHÔNG: bỏ qua mục đó. Không tìm thấy là câu trả lời BÌNH THƯỜNG và hữu
  ích. TUYỆT ĐỐI KHÔNG bịa span để mục nào đó trông như đã hoàn thành.

Span của bạn sẽ được đối chiếu TỰ ĐỘNG với văn bản bằng khớp chuỗi. Span
không tồn tại sẽ bị loại và mục đó bị tính là thất bại — bịa không giúp ích gì.

### CÂU 2 — NHÂN VẬT CÓ ĐI CHỆCH HỢP ĐỒNG KHÔNG?
Với từng nhân vật có `must_not_reveal`: có bí mật nào bị lộ không?
Có nhân vật nào xuất hiện mà hợp đồng KHÔNG liệt kê không? Nếu có, ghi một
assertion với `span` nguyên văn.

Xuất một JSON đúng schema StateDelta:
- Mọi `assertion` phải có `span` nguyên văn, `chapter`, `scene`, `confidence`.
- KHÔNG xuất assertion về VỊ TRÍ nhân vật (`at`, `located_in`...). Vị trí đã
  được hệ thống ghi ở bước chốt cảnh; hai nguồn cho cùng một sự thật sẽ lệch
  nhau.
- `predicate` CHỈ được chọn trong DANH SÁCH VỊ TỪ HỢP LỆ bên trên (tập
  đóng, do tác giả khai), đúng loại chủ thể và kiểu giá trị ghi kèm. Không
  có vị từ phù hợp thì KHÔNG xuất mệnh đề đó. `buoc_vao`, `nhin_thay`,
  `dung_cach` là lời kể sự kiện, không phải sự thật về thế giới — đừng
  xuất chúng dưới bất kỳ tên nào.
- `span` phải nhắc tới CHỦ THỂ của mệnh đề bằng tên. Một câu chỉ tình cờ chứa
  từ khoá không phải là bằng chứng.
- `epistemic`: "objective" chỉ khi NGƯỜI KỂ khẳng định; "claimed_by" khi một
  NHÂN VẬT nói ra; "believed_by" khi văn bản cho thấy nhân vật TIN. Lời nhân
  vật KHÔNG BAO GIỜ là "objective" — nhân vật có thể nói dối.
- Với MỖI `plant_directive` thực sự xuất hiện, xuất một mục `plant_evidence`
  gồm `clue_id`, `scene_id`, `span` nguyên văn, `carrier_used`, và
  `concluded_by` (nhân vật đã RÚT RA KẾT LUẬN từ chi tiết đó — khác với nhân
  vật chỉ nhìn thấy). KHÔNG đặt trường `verified`; hệ thống tự đặt.
- Manh mối không xuất hiện: BỎ QUA trong `plant_evidence`. Không tạo mục với
  span rỗng.
- Cảnh ghi `plant_directives: (không có)` thì KHÔNG xuất `plant_evidence` nào
  cho cảnh đó. `clue_id` phải là mã CLUE_* có trong kế hoạch — mã cảnh dạng
  CH…_S… KHÔNG phải clue_id. Kế hoạch không có manh mối nào thì
  `plant_evidence` là danh sách rỗng, và đó là câu trả lời đúng.
- Với tương tác ĐÁNG KỂ giữa hai nhân vật, xuất `relationship_events`: mỗi mục
  gồm `a`, `b` (mã CHAR_*), `kind`, `actor` (ai hành động, nếu có), `scene_id`,
  `span` nguyên văn. `kind` CHỈ là một trong: acted_against_own_interest_for_other,
  shared_ordeal, verbal_affection_only, sacrifice, betrayal, value_clash,
  reconciled_method, interests_collide, interests_align.
  KHÔNG xuất điểm số hay giai đoạn quan hệ — hệ thống tự tính.
  `sacrifice` chỉ khi mất mát KHÔNG đảo ngược được. `betrayal` chỉ khi người kể
  khẳng định, không phải khi một nhân vật nghi ngờ. `interests_collide/align`
  chỉ khi HOÀN CẢNH bên ngoài đổi, không vì một cuộc nói chuyện. Lời ngọt ngào
  không kèm hành động là `verbal_affection_only`. Không có thì danh sách rỗng.

Chỉ xuất JSON.
"""


EXTRACT_EMERGENT_TMPL = """Đọc chương và liệt kê những gì TỒN TẠI trong đó.
Bạn là bộ phận ghi chép, không phải biên tập viên. Không đánh giá văn chương.

## VĂN BẢN (chương {chapter})
{prose}

## THỰC THỂ ĐÃ BIẾT TỪ TRƯỚC
{known_entities}

## DANH SÁCH VỊ TỪ HỢP LỆ — TẬP ĐÓNG
Chỉ áp cho mệnh đề `objective`. Lời nhân vật nói (`claimed_by`) và điều
nhân vật tin (`believed_by`) không bị giới hạn bởi danh sách này.
{predicates}

Nhiệm vụ: tìm những thứ KHÔNG có trong DANH SÁCH THỰC THỂ đã biết.

- `new_entities`: người, nơi chốn, tổ chức, đồ vật có tên riêng, tập tục,
  luật lệ — bất cứ thứ gì được nhắc tới như thể nó có thật trong thế giới
  nhưng chưa có trong danh sách. Kể cả khi chỉ được nhắc thoáng qua một lần.
  `id` đặt theo quy ước tiền tố: CHAR_ (nhân vật) / FACT_ (phe phái) /
  LOC_ (địa điểm) / OBJ_ (đồ vật, KỂ CẢ giấy tờ, sổ sách, văn bản) /
  EV_ (sự kiện) / DOC_ (học thuyết, luật lệ — KHÔNG phải giấy tờ).
  `kind` CHỈ được là một trong: character, faction, location, object, event,
  doctrine, resource. Giấy tờ là `object` — không có kind `document`.
- `new_relations`: quan hệ mới được thiết lập giữa các thực thể.
- `retracted_relations`: quan hệ bị chấm dứt.
- `assertions`: mệnh đề sự thật, MỖI mệnh đề kèm `span` là trích dẫn nguyên
  văn (≥12 ký tự, chép chính xác từng chữ kể cả dấu), và `confidence` từ 0
  tới 1. KHÔNG xuất mệnh đề về vị trí nhân vật.
  `predicate` CHỈ được chọn trong DANH SÁCH VỊ TỪ HỢP LỆ bên trên (tập
  đóng, do tác giả khai), đúng loại chủ thể và kiểu giá trị ghi kèm. Không
  có vị từ phù hợp thì KHÔNG xuất mệnh đề đó. `buoc_vao`, `nhin_thay`,
  `dung_cach` là lời kể sự kiện, không phải sự thật về thế giới — đừng
  xuất chúng dưới bất kỳ tên nào.
- KHÔNG liệt kê góc phòng, khoang, cầu tàu... là `new_entities` kiểu
  `location`. Đó là phần của một địa điểm đã biết, không phải địa điểm mới.

QUY TẮC EPISTEMIC — quan trọng nhất:
- `epistemic: "objective"` chỉ khi NGƯỜI KỂ khẳng định
- `epistemic: "claimed_by"` khi một NHÂN VẬT nói ra (kèm `holder`)
- `epistemic: "believed_by"` khi văn bản cho thấy nhân vật TIN (kèm `holder`)

Nhân vật có thể nói dối. Lời nhân vật KHÔNG BAO GIỜ là "objective".

Hãy quét kỹ phần bối cảnh và phần thoại phụ — thực thể mới thường nằm ở đó,
không nằm ở tuyến hành động chính.

Chỉ xuất JSON.
"""
