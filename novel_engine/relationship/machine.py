"""Transition guards — cấm nhảy cóc giai đoạn (§7.2).

Hai vai trò, hai lần gọi (§7.3):
- `advance_or_hold` — CHỈ THỊ cho Director, tính trên trạng thái TRƯỚC khi viết.
- `dynamics.settle_chapter` — PHÁN QUYẾT sau khi Extractor đã cập nhật chỉ số.

Ba chỗ khác mã §7.2:

1. `MIN_CHAPTERS_IN_STAGE` không có RUPTURE và SEVERED, nên `can_advance` ném
   `KeyError` ngay dòng đầu với mọi cặp đã đổ vỡ. Director gọi hàm này cho MỌI
   cặp, nên một lần phản bội giết mọi chương sau đó. Ở đây hai giai đoạn này có
   guard riêng: RUPTURE có một lối ra lên (TRIAL), SEVERED không có lối nào.
2. Gợi ý mở khoá khi HOLD (`UNBLOCK_HINT`) đi vào `scene_requirement` — tức
   WRITER_TMPL coi nó là RÀNG BUỘC ("đó là ràng buộc, không phải gợi ý"). Mỗi
   chương cặp nào đang chờ cũng bị ép dựng "một nghịch cảnh buộc hai người phụ
   thuộc nhau". Giai đoạn chờ thành giai đoạn bị kích liên tục. Ở đây HOLD mang
   `hint` (cho Director và tác giả), không mang `scene_requirement`.
3. Thêm `not_yet` — điều CHƯA được viết ở giai đoạn hiện tại. Guard chặn nhãn
   giai đoạn, nhưng Writer không biết nhãn, nên "người lạ → tâm sự tuổi thơ"
   vẫn lên trang dù canon ghi STRANGERS. Guard phải có mặt ở chỗ văn xuôi sinh ra.
"""
from __future__ import annotations

from novel_engine.canon.models import RelationStage
from novel_engine.relationship.models import RelationshipState

R = RelationStage
ORDER = [R.STRANGERS, R.FRICTION, R.VULNERABILITY, R.TRIAL, R.CATHARSIS]

MIN_CHAPTERS_IN_STAGE = {
    R.STRANGERS: 1,
    R.FRICTION: 3,        # phải va chạm đủ lâu
    R.VULNERABILITY: 2,
    R.TRIAL: 2,
    R.CATHARSIS: 0,
    R.RUPTURE: 3,         # hàn gắn không xảy ra trong một chương
    R.SEVERED: 0,
}
RUPTURE_HEAL_MIN_INTIMACY = 25

SCENE_REQUIREMENT = {
    R.FRICTION: ("Cảnh phải chứa một BẤT ĐỒNG VỀ PHƯƠNG PHÁP mà cả hai đều có lý. "
                 "Không được để một bên rõ ràng đúng."),
    R.VULNERABILITY: ("Điểm yếu phải LỘ RA DO HOÀN CẢNH, không do tự nguyện kể. "
                      "Nhân vật phải cố che giấu và thất bại."),
    R.TRIAL: ("Phải có một lựa chọn mà MỌI phương án đều mất mát. Cấm phương án "
              "'vừa cứu được người vừa giữ được mục tiêu'."),
    R.CATHARSIS: ("Chấp nhận nhau KÈM vết sẹo. Phải có ít nhất một điều vĩnh viễn "
                  "không thể lấy lại được, và cả hai đều biết điều đó."),
}
# RUPTURE → TRIAL không phải "thử thách lần đầu": hàn gắn phải trả giá.
HEAL_REQUIREMENT = ("Hàn gắn phải có GIÁ: một bên làm điều bất lợi cho chính mình "
                    "vì bên kia, và vết đổ vỡ vẫn còn nguyên trong cách họ nói với nhau.")

UNBLOCK_HINT = {
    R.STRANGERS: "cần một cảnh hai người phải hợp tác dù bất đồng",
    R.FRICTION: "cần một nghịch cảnh buộc hai người phụ thuộc nhau",
    R.VULNERABILITY: ("cần nâng stake_conflict: đưa vào một sự kiện khiến mục tiêu "
                      "của hai người loại trừ nhau"),
    R.TRIAL: "cần một hy sinh có thật, để lại hậu quả không đảo ngược",
    R.RUPTURE: "cần thời gian và một hành động có giá — lời xin lỗi không đủ",
}

NOT_YET = {
    R.STRANGERS: "chưa tin nhau, chưa tâm sự, chưa có cử chỉ thân mật",
    R.FRICTION: "chưa bộc lộ điểm yếu thật, chưa hoà giải về giá trị",
    R.VULNERABILITY: "chưa đặt hai người vào thế phải chọn chống lại nhau, chưa hy sinh cho nhau",
    R.TRIAL: "chưa hoà giải trọn vẹn, chưa tha thứ dễ dàng",
    R.CATHARSIS: "không xoá được vết sẹo: điều đã mất vẫn mất",
    R.RUPTURE: "không hàn gắn bằng lời nói; mọi tiếp xúc đều mang dấu đổ vỡ",
    R.SEVERED: "quan hệ đã chấm dứt: không hàn gắn, không quay lại",
}

STAGE_VI = {
    R.STRANGERS: "người lạ", R.FRICTION: "va chạm", R.VULNERABILITY: "bộc lộ tổn thương",
    R.TRIAL: "thử thách lòng tin", R.CATHARSIS: "gắn kết có trả giá",
    R.RUPTURE: "đổ vỡ", R.SEVERED: "chấm dứt",
}


def next_stage(st: RelationshipState) -> RelationStage | None:
    if st.stage == R.RUPTURE:
        return R.TRIAL
    if st.stage in ORDER[:-1]:
        return ORDER[ORDER.index(st.stage) + 1]
    return None


def can_advance(st: RelationshipState, chapter: int) -> tuple[bool, str]:
    """Guard cứng. Đây là thứ ngăn 'người lạ → yêu say đắm' trong 2 chương."""
    if st.stage == R.SEVERED:
        return False, "quan hệ đã chấm dứt vĩnh viễn"
    if st.stage == R.CATHARSIS:
        return False, "đã ở trạng thái cuối"

    need = MIN_CHAPTERS_IN_STAGE[st.stage]
    if st.chapters_in_stage < need:
        return False, (f"mới ở {st.stage.value} {st.chapters_in_stage} chương, "
                       f"cần ≥{need}")

    if st.stage == R.RUPTURE:
        if st.intimacy < RUPTURE_HEAL_MIN_INTIMACY:
            return False, (f"intimacy {st.intimacy:.0f} < {RUPTURE_HEAL_MIN_INTIMACY} — "
                           f"không còn đủ gắn kết để thử hàn gắn")
        return True, "đủ thời gian và gắn kết để quay lại thử thách"

    if st.stage == R.STRANGERS:
        if st.friction < 15:
            return False, "chưa có va chạm giá trị nào đáng kể"
        return True, "đã có bất đồng đủ để bắt đầu"

    if st.stage == R.FRICTION:
        # Nghịch cảnh CHUNG, không phải tâm sự tự nguyện — nhân vật ngồi xuống
        # kể về tuổi thơ là thứ không ai làm.
        if not st.shared_ordeals:
            return False, "chưa cùng trải qua nghịch cảnh nào"
        if st.intimacy < 20:
            return False, f"intimacy {st.intimacy:.0f} < 20"
        return True, "có nghịch cảnh chung + đủ gắn kết để lộ điểm yếu"

    if st.stage == R.VULNERABILITY:
        if st.stake_conflict < 40:
            return False, (f"stake_conflict {st.stake_conflict:.0f} < 40 — "
                           f"chưa có gì thật sự phải đánh đổi")
        if st.intimacy < 45:
            return False, "chưa đủ gắn kết để sự phản bội có sức nặng"
        return True, "lợi ích đối đầu trực diện"

    # TRIAL
    if not st.scars:
        return False, "chưa ai trả giá gì — catharsis sẽ rỗng"
    if st.stake_conflict > 55:
        return False, ("xung đột lợi ích còn quá cao; catharsis lúc này là kết "
                       "thúc cổ tích giả tạo")
    return True, "đã trả giá, xung đột đã hạ xuống mức sống chung được"


def advance_or_hold(st: RelationshipState, chapter: int) -> dict:
    """CHỈ THỊ cho cảnh — tính trên trạng thái TRƯỚC khi viết. Không đổi `st`.

    Không chứa chỉ số (intimacy, friction…): NT-7 — con số là công cụ chẩn đoán,
    đưa vào prompt thì Writer viết để thoả con số.
    """
    ok, reason = can_advance(st, chapter)
    base = {"stage": st.stage.value, "stage_vi": STAGE_VI[st.stage],
            "reason": reason, "not_yet": NOT_YET[st.stage]}
    nxt = next_stage(st)
    if ok and nxt is not None:
        req = HEAL_REQUIREMENT if st.stage == R.RUPTURE else SCENE_REQUIREMENT[nxt]
        return {**base, "action": "advance", "to": nxt.value,
                "to_vi": STAGE_VI[nxt], "scene_requirement": req,
                # Đang được phép tiến tới giai đoạn sau: `not_yet` của giai
                # đoạn HIỆN TẠI không còn là giới hạn đúng.
                "not_yet": NOT_YET[nxt]}
    return {**base, "action": "hold", "hint": UNBLOCK_HINT.get(st.stage, "")}
