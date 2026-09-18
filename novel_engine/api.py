"""Mặt tiền lập trình cho giao diện — mọi thứ trả về dữ liệu, không in ra.

`cli.py` trộn ba việc: truy vấn động cơ, định dạng bảng cho terminal, và trả mã
thoát. Một giao diện đồ hoạ chỉ cần việc thứ nhất, và nếu nó phải gọi qua CLI
rồi bóc chữ từ stdout thì mọi lần đổi một dòng in sẽ làm hỏng giao diện.

Ba giao kèo của module này:

- KHÔNG `print`, KHÔNG `sys.exit`, KHÔNG đọc `argparse`. Lỗi ném ra như lỗi.
- Mọi giá trị trả về `json.dumps` được. Có test chốt điều đó, vì một `datetime`
  hay một `Enum` lọt vào sẽ chỉ nổ ở tầng HTTP, xa chỗ gây ra nó.
- Tên khoá là HỢP ĐỒNG với tầng giao diện. Đổi tên khoá là đổi API.

Không giữ trạng thái toàn cục: mỗi lời gọi tự dựng động cơ từ `db`. Một tiến
trình phục vụ nhiều dự án cùng lúc mà chia sẻ một `Engines` giữa các request là
con đường ngắn nhất tới việc canon của dự án này rò sang dự án kia.
"""
from __future__ import annotations

import re
from pathlib import Path

from novel_engine.audit.deterministic import audit_stats
from novel_engine.audit.timeline_rules import check_continuity
from novel_engine.eval.harness import evaluate
from novel_engine.graph.build import run_chapter
from novel_engine.graph.engines import build_engines
from novel_engine.llm.gemini import build_llm

DB_MAC_DINH = "novel_storage.db"
THU_MUC_CHUONG = Path("output/chapters")
THU_MUC_BAO_CAO = Path("output/reports")

_CANH_RE = re.compile(r"^## Cảnh (\d+)\s*$", re.M)


def _eng(db: str = DB_MAC_DINH, llm: str = "fake"):
    return build_engines(build_llm(llm), db_path=db)


def _so_chuong(scene_id: str) -> int:
    return int(scene_id.split("_")[0].removeprefix("CH"))


# ═══════════════════════ ĐỌC ═══════════════════════

def trang_thai(db: str = DB_MAC_DINH) -> dict:
    """Tổng quan một dự án: bao nhiêu chương, canon sạch không."""
    eng = _eng(db)
    frames = eng.store.get_frames()
    chuong = sorted({_so_chuong(f.scene_id) for f in frames})
    lien_tuc = check_continuity(frames, eng.graph) if frames else []
    return {
        "db": db,
        "so_chuong": len(chuong),
        "so_canh": len(frames),
        "chuong_da_viet": chuong,
        "tong_chuong_ke_hoach": eng.total_chapters,
        "lien_tuc": {
            "sach": not lien_tuc,
            "so_loi": len(lien_tuc),
            "chi_tiet": [{"severity": f.get("severity"), "check": f.get("check"),
                          "message": f.get("message")} for f in lien_tuc[:20]],
        },
    }


def danh_sach_chuong(db: str = DB_MAC_DINH) -> list[dict]:
    """Mọi chương trong outline, kèm trạng thái thực tế trong kho."""
    eng = _eng(db)
    theo_ch: dict[int, list] = {}
    for f in eng.store.get_frames():
        theo_ch.setdefault(_so_chuong(f.scene_id), []).append(f)

    ra = []
    for ch in sorted(eng.planner.chapters):
        fs = theo_ch.get(ch, [])
        ds = eng.store.get_deltas(ch)
        hien_tai = [f for f in fs if f.time.mode == "present"]
        ra.append({
            "chapter": ch,
            "title": eng.planner.title(ch),
            "outline_beat": eng.planner.outline_beat(ch),
            "so_canh_ke_hoach": eng.planner.n_scenes(ch),
            "so_canh_da_viet": len(fs),
            "trang_thai": ("đã ghi canon" if ds and ds[0].committed
                           else "chờ duyệt" if ds
                           else "đã viết" if fs else "chưa viết"),
            "epoch_tu": min((f.time.epoch_tick for f in hien_tai), default=None),
            "epoch_den": max((f.time.end_tick for f in hien_tai), default=None),
        })
    return ra


def doc_chuong(db: str = DB_MAC_DINH, chapter: int = 1,
               chapters_dir: Path | str = THU_MUC_CHUONG) -> dict:
    """Văn xuôi từng cảnh + chỉ số đọc được, cho khung soạn thảo."""
    eng = _eng(db)
    # Hai chuyện khác nhau, và giao diện cần phân biệt được: chương CÓ trong
    # dàn ý mà chưa viết (bấm "viết" được), với chương không có trong dàn ý
    # (một đường dẫn cũ, hoặc một lỗi ở phía gọi). Trả rỗng cho cả hai nhưng
    # nói rõ là cái nào — im lặng gộp chúng lại là giấu một lỗi thật.
    trong_outline = chapter in eng.planner.chapters
    p = Path(chapters_dir) / f"ch{chapter:03d}.md"
    if not p.exists():
        return {"chapter": chapter, "ton_tai": False,
                "trong_outline": trong_outline, "scenes": [], "so_tu": 0,
                "title": eng.planner.title(chapter) if trong_outline else ""}

    phan = _CANH_RE.split(p.read_text(encoding="utf-8"))
    scenes, truoc = [], []
    for i in range(1, len(phan), 2):
        si, van = int(phan[i]), phan[i + 1].strip()
        ct = {"scene_id": f"CH{chapter:03d}_S{si:02d}", "active_characters": []}
        scenes.append({"scene_index": si, "scene_id": ct["scene_id"],
                       "prose": van, "stats": audit_stats(van, ct, truoc)})
        truoc.append({"prose": van})
    return {
        "chapter": chapter, "ton_tai": True,
        "trong_outline": trong_outline,
        "title": eng.planner.title(chapter) if trong_outline else "",
        "so_tu": sum(len(s["prose"].split()) for s in scenes),
        "scenes": scenes,
    }


def nhan_vat(db: str = DB_MAC_DINH) -> list[dict]:
    """Hồ sơ giọng nhân vật — đủ để dựng khung chỉnh giọng trong giao diện.

    `signature_lexicon` có mặt ở đây vì tác giả cần THẤY và SỬA nó. Nó vẫn
    không đi vào prompt Writer (xem `_characters_brief`): hai đường khác nhau
    cho hai người đọc khác nhau.
    """
    ra = []
    for cid, p in _eng(db).chars.items():
        v = p.voice
        ra.append({
            "id": cid, "name": p.name,
            "register": v.register,
            "mean_sentence_len": list(v.mean_sentence_len),
            "max_sentence_len": v.max_sentence_len,
            "question_ratio": list(v.question_ratio),
            "signature_lexicon": list(v.signature_lexicon),
            "verbal_tics": list(v.verbal_tics),
            "voice_exemplars": list(v.voice_exemplars),
            "forbidden_lexicon": list(v.forbidden_lexicon),
            "syntactic_tic": v.syntactic_tic,
            "somatic_signature": list(p.somatic_signature),
        })
    return ra


def manh_moi(db: str = DB_MAC_DINH) -> list[dict]:
    """Trạng thái manh mối — bảng theo dõi phục bút cho giao diện."""
    ra = []
    for cid, c in _eng(db).graph.clues.items():
        st = getattr(c, "status", "")
        ra.append({
            "id": cid,
            "status": st.value if hasattr(st, "value") else str(st),
            "salience": round(float(getattr(c, "salience", 0.0)), 4),
            "planted_in_chapter": getattr(c, "planted_in_chapter", None),
            "last_touched_chapter": getattr(c, "last_touched_chapter", None),
            "understood_by": list(getattr(c, "understood_by_characters", [])),
            "payoff_deadline": getattr(c, "payoff_deadline", None),
        })
    return ra


def cham_diem(db: str = DB_MAC_DINH, tu: int = 1, den: int | None = None,
              chapters_dir: Path | str = THU_MUC_CHUONG,
              reports_dir: Path | str = THU_MUC_BAO_CAO) -> dict:
    """Bộ chỉ số M1–M15. Không judge LLM, nên giao diện gọi được mà không tốn gì.

    M6 và M7 sẽ báo "chưa đo được" — đó là câu trả lời đúng, không phải thiếu
    sót: chúng cần một LLM judge, và một lần bấm nút trong giao diện không nên
    âm thầm đốt hạn ngạch.
    """
    eng = _eng(db)
    if den is None:
        da_viet = [c["chapter"] for c in danh_sach_chuong(db)
                   if c["so_canh_da_viet"]]
        den = max(da_viet, default=tu)
    return evaluate(eng, range(tu, den + 1),
                    chapters_dir=Path(chapters_dir),
                    reports_dir=Path(reports_dir))


# ═══════════════════════ GHI ═══════════════════════

def viet_chuong(db: str = DB_MAC_DINH, chapter: int = 1, llm: str = "fake", *,
                force: bool = False, on_event=None,
                chapters_dir: Path | str | None = None) -> dict:
    """Viết một chương. `on_event` nhận tiến độ từng bước (xem `run_chapter`).

    Ném `FileExistsError` nếu chương đã có mà không `force` — giao diện phải
    HỎI trước khi xoá, y như CLI làm. Viết đè là thao tác phá huỷ: nó xoá
    frame, digest và delta của bản cũ.

    `chapters_dir`: `None` (mặc định) là KHÔNG ghi file gì cả — canon store là
    nguồn sự thật duy nhất được đụng tới. Truyền một thư mục thì hàm ghi thêm
    file markdown ở đó, cùng định dạng mà `doc_chuong` đọc lại — để một giao
    diện có thể viết rồi hiển thị ngay mà không cần tự lặp lại logic render
    của `cli.py`. Không bắt buộc: nhiều giao diện có thể chỉ cần canon, không
    cần file trên đĩa.
    """
    eng = _eng(db, llm)
    if eng.store.has_chapter(chapter):
        if not force:
            raise FileExistsError(
                f"chương {chapter} đã có trong {db}; "
                f"truyền force=True để xoá bản cũ và viết lại")
        eng.store.clear_chapter(chapter)
        eng.rebuild_canon()

    st = run_chapter(eng, chapter, on_event=on_event)
    canh = st.get("scene_outputs") or []

    if chapters_dir is not None and not st.get("escalation_reason"):
        from novel_engine.render import render_chapter_markdown
        thu_muc = Path(chapters_dir)
        thu_muc.mkdir(parents=True, exist_ok=True)
        md = render_chapter_markdown(eng, chapter, canh)
        (thu_muc / f"ch{chapter:03d}.md").write_text(md, encoding="utf-8")

    return {
        "chapter": chapter,
        "escalated": bool(st.get("escalation_reason")),
        "escalation_reason": st.get("escalation_reason") or "",
        "so_canh": len(canh),
        "so_tu": sum(len((s.get("prose") or "").split()) for s in canh),
        "hygiene_notes": st.get("hygiene_notes") or [],
        "unresolved": st.get("unresolved") or [],
    }


def sua_giong(char_id: str, thay_doi: dict, *,
              bible_dir: Path | str | None = None) -> dict:
    """Sửa hồ sơ giọng của một nhân vật trong bible, có kiểm tra trước khi ghi.

    Giao diện chỉnh giọng phải ghi được xuống bible, nếu không nó chỉ là một
    khung hiển thị. Nhưng ghi thẳng YAML từ một ô nhập liệu là cách chắc chắn
    để có một bible không nạp được — và lúc đó MỌI lệnh đều hỏng, không riêng
    cái vừa bấm.

    Nên: dựng lại `CharacterProfile` với giá trị mới, để Pydantic và các
    validator của nó phán xử, và chỉ khi qua được mới chạm vào file. Cùng lối
    đó, ràng buộc "câu mẫu không được chứa cụm đang bị đo" sẽ nổ Ở ĐÂY thay vì
    nổ âm thầm ba chương sau.
    """
    import yaml
    from novel_engine.canon.bible import DEFAULT_BIBLE
    from novel_engine.character.models import CharacterProfile

    goc = Path(bible_dir) if bible_dir else DEFAULT_BIBLE
    for f in sorted((goc / "characters").glob("*.yaml")):
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        if raw.get("id") != char_id:
            continue
        la_giong = set(CharacterProfile.model_fields["voice"]
                       .annotation.model_fields)
        moi = dict(raw)
        moi["voice"] = dict(raw.get("voice") or {})
        for k, v in thay_doi.items():
            (moi["voice"] if k in la_giong else moi)[k] = v

        CharacterProfile.model_validate(moi)      # ném thì KHÔNG ghi gì
        f.write_text(yaml.safe_dump(moi, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        return {"id": char_id, "file": str(f), "da_sua": sorted(thay_doi)}
    raise KeyError(f"không có nhân vật {char_id!r} trong {goc / 'characters'}")
