"""Máy chủ HTTP cho giao diện Xưởng viết — bọc mỏng quanh `novel_engine/api.py`.

Không route nào ở đây tự quyết định gì: mọi route gọi thẳng một hàm của
`api.py` và trả nguyên kết quả (hoặc lỗi của nó) dưới dạng JSON. Nếu một
tính năng cần logic mới, logic đó thuộc về `api.py`, không phải ở đây — chỗ
này chỉ có việc parse query string, chọn mã trạng thái HTTP, và (cho
`viet_chuong`) bắc cầu callback `on_event` đồng bộ sang một stream SSE.

`db` luôn là query param tường minh, không bao giờ suy ra từ session hay
biến toàn cục — lý do y hệt lý do `api.py` không giữ trạng thái (xem
docstring module đó): một tiến trình phục vụ nhiều dự án cùng lúc mà đoán
nhầm `db` là con đường ngắn nhất tới việc canon của dự án này rò sang dự án
kia.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from novel_engine import api

STATIC_DIR = Path(__file__).resolve().parent.parent / "webapp"


def _an_toan(fn, *args, **kwargs) -> Any:
    """Gọi một hàm `api.py` và đổi lỗi Python thành lỗi HTTP.

    `api.py` ném lỗi như lỗi (giao kèo của nó) — tầng HTTP là nơi phải bắt
    chúng, vì phía trên nó không còn ai bắt hộ nữa.
    """
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


class SuaGiongBody(BaseModel):
    thay_doi: dict[str, Any]


def create_app(default_db: str = api.DB_MAC_DINH) -> FastAPI:
    app = FastAPI(title="Novel Engine — Xưởng viết")

    def db_param(db: str = Query(default_db)) -> str:
        return db

    @app.get("/api/trang-thai")
    def trang_thai(db: str = Query(default_db)):
        return _an_toan(api.trang_thai, db)

    @app.get("/api/chuong")
    def danh_sach_chuong(db: str = Query(default_db)):
        return _an_toan(api.danh_sach_chuong, db)

    @app.get("/api/chuong/{chapter}")
    def doc_chuong(chapter: int, db: str = Query(default_db)):
        return _an_toan(api.doc_chuong, db, chapter)

    @app.get("/api/nhan-vat")
    def nhan_vat(db: str = Query(default_db)):
        return _an_toan(api.nhan_vat, db)

    @app.get("/api/manh-moi")
    def manh_moi(db: str = Query(default_db)):
        return _an_toan(api.manh_moi, db)

    @app.get("/api/cham-diem")
    def cham_diem(tu: int = Query(1), den: int | None = Query(None),
                  db: str = Query(default_db)):
        return _an_toan(api.cham_diem, db, tu, den)

    @app.patch("/api/nhan-vat/{char_id}")
    def sua_giong(char_id: str, body: SuaGiongBody):
        return _an_toan(api.sua_giong, char_id, body.thay_doi)

    @app.post("/api/chuong/{chapter}/viet")
    def viet_chuong(chapter: int, db: str = Query(default_db),
                    llm: str = Query("fake"), force: bool = Query(False)):
        """Viết một chương, phát tiến độ qua Server-Sent Events.

        Cầu nối bắt buộc: `api.viet_chuong` chạy engine LangGraph đồng bộ
        và chặn (có thể mất vài phút), còn HTTP request cần trả byte ngay
        khi có. Chạy nó trên một thread riêng, đẩy sự kiện qua `queue.Queue`
        thread-safe, generator phía dưới chỉ việc rút hàng đợi ra và
        format theo SSE — không có logic viết chương nào lặp lại ở đây.
        """
        hang_doi: queue.Queue = queue.Queue()
        XONG = object()

        def on_event(ev: dict) -> None:
            hang_doi.put(("progress", ev))

        def chay() -> None:
            try:
                ket_qua = api.viet_chuong(db, chapter, llm, force=force,
                                          on_event=on_event,
                                          chapters_dir=api.THU_MUC_CHUONG)
                hang_doi.put(("done", ket_qua))
            except Exception as e:                     # noqa: BLE001
                hang_doi.put(("error", {"message": str(e)}))
            finally:
                hang_doi.put((XONG, None))

        threading.Thread(target=chay, daemon=True).start()

        def gen():
            while True:
                loai, payload = hang_doi.get()
                if loai is XONG:
                    break
                data = json.dumps(payload, ensure_ascii=False)
                yield f"event: {loai}\ndata: {data}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="webapp")

    return app


app = create_app()
