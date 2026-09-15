"""Adapter Gemini Flash — bậc miễn phí của Google AI Studio.

Bậc free hiện chỉ còn dòng **Flash** (Pro đã vào sau billing từ 5/2026):
~1.500 request/ngày, 10–15 RPM tuỳ model.

Đối chiếu nhu cầu: một chương ≈ 26 lượt gọi (6 Director + ~8 Writer + ~4
Auditor + 6 Polish + 2 Extractor). Hạn ngạch NGÀY thừa sức (~57 chương/ngày).
Nút thắt thật là **RPM** — nên lớp này có bộ điều tiết nhịp sẵn, không phải
tuỳ chọn. Không có nó thì Writer bắn 8 lượt liên tiếp trong vài giây và lượt
thứ 3 trở đi ăn 429.

⚠ Google đã cắt free quota 50–80% hồi 12/2025 mà không báo trước, và giới hạn
thay đổi theo vùng lẫn theo trạng thái xác minh tài khoản. Đừng xây kế hoạch
dài hạn dựa vào con số cụ thể; `RPM` dưới đây chỉnh được.
"""
from __future__ import annotations

import os
import threading
import time
import warnings

from novel_engine.llm.env import GEMINI_KEY_NAMES, gemini_api_key, load_dotenv

# Nhiệt độ theo vai. Writer cần biến thiên, các lượt có cấu trúc thì không —
# JSON sinh ở nhiệt độ cao là JSON hỏng.
_TEMPERATURE: dict[str, float] = {
    "writer": 0.85,
    "polish": 0.7,
    "director": 0.3,
    "auditor": 0.2,
    "scene_digest": 0.1,
    "extractor_diff": 0.1,
    "extractor_emergent": 0.1,
    "repair": 0.0,
}
DEFAULT_TEMPERATURE = 0.4

# Ghim BẢN CỤ THỂ, không dùng alias `gemini-flash-latest`: alias trượt sang
# model mới bất cứ lúc nào, và bộ hồi quy §13.2 đòi cùng seed cho cùng kết quả.
#
# Vì sao là flash-LITE chứ không phải flash. Đo thực trên cùng một prompt văn
# xuôi tiếng Việt 90 từ:
#
#     gemini-3.5-flash-lite    4,3s   ~30 RPM  → 1 chương (26 lượt) ≈ 2 phút
#     gemini-3.1-flash-lite    2,2s   ~30 RPM  → ≈ 1 phút
#     gemini-3.8-flash        56,1s     5 RPM  → ≈ 24 phút + chờ điều tiết
#
# `3.8-flash` chậm gấp 13 lần và trần RPM thấp gấp 6, trong khi chênh lệch
# chất lượng không đáng kể: cả ba đều đạt ≥3 kênh giác quan, tránh được danh
# sách sáo ngữ cấm, và bám đúng tật nói-bằng-con-số của Kaelen. Với một vòng
# lặp 26 lượt/chương thì độ trễ là thứ quyết định, không phải điểm chất lượng
# trên một lượt.
#
# (`gemini-2.5-flash` đã bị khai tử — gọi vào trả 404.)
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Trần RPM thay đổi theo model VÀ theo trạng thái tài khoản. Dòng flash-lite
# rộng hơn hẳn dòng flash. Đặt thấp hơn trần thật một quãng: 429 giữa chương
# tốn nhiều hơn vài giây chờ.
RPM_BY_MODEL: dict[str, int] = {
    "gemini-3.5-flash-lite": 15,
    "gemini-3.1-flash-lite": 15,
    "gemini-flash-lite-latest": 15,
    "gemini-3.8-flash": 4,
    "gemini-3.7-flash": 4,
    "gemini-flash-latest": 4,
}
DEFAULT_RPM = 8

# google-genai kêu về automatic function calling ở mọi lượt generate_content.
# Ta không dùng AFC; cảnh báo này chỉ làm nhiễu log của mỗi lượt gọi.
warnings.filterwarnings("ignore", message=".*automatic function calling.*")


class _RateLimiter:
    """Cửa sổ trượt, chặn đủ lâu để không vượt `rpm` lượt mỗi phút.

    Thread-safe vì §14.3 cho phép song song hoá vài việc (prefetch context,
    sinh nhiều phương án cho cùng một cảnh).
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._hits: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._hits = [t for t in self._hits if now - t < 60.0]
            waited = 0.0
            if len(self._hits) >= self.rpm:
                waited = 60.0 - (now - self._hits[0]) + 0.05
                if waited > 0:
                    time.sleep(waited)
                now = time.monotonic()
                self._hits = [t for t in self._hits if now - t < 60.0]
            self._hits.append(now)
            return waited


class GeminiLLM:
    """Triển khai `LLMPort` trên Gemini Flash.

    Khoá API lấy miễn phí ở Google AI Studio, đặt vào biến môi trường
    `GOOGLE_API_KEY` (hoặc truyền thẳng `api_key`).
    """

    def __init__(self, model: str = DEFAULT_MODEL, *,
                 api_key: str | None = None, rpm: int | None = None,
                 max_output_tokens: int = 4000,
                 max_retries: int = 3) -> None:
        load_dotenv()                      # không ghi đè biến đã có sẵn
        key = api_key or gemini_api_key()
        if not key:
            raise RuntimeError(
                f"chưa có khoá Gemini (thử {', '.join(GEMINI_KEY_NAMES)}). "
                "Lấy khoá miễn phí ở https://aistudio.google.com/apikey rồi "
                "đặt vào .env, hoặc dùng FakeLLM để chạy pipeline không tốn gì.")
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._key = key
        # Trần RPM suy ra từ model nếu gọi không chỉ định: dùng chung một con
        # số cho mọi model là cách chắc chắn ăn 429 trên dòng flash, hoặc phí
        # nửa thời gian chờ vô ích trên dòng flash-lite.
        self.rpm = rpm if rpm is not None else RPM_BY_MODEL.get(model, DEFAULT_RPM)
        self._limiter = _RateLimiter(self.rpm)
        self._clients: dict[float, object] = {}      # nhiệt độ → client
        self.calls: list[dict] = []

    def _client(self, temperature: float):
        """Một client cho mỗi nhiệt độ, dựng lười. `ChatGoogleGenerativeAI`
        chốt nhiệt độ lúc khởi tạo, nên không tái dùng chung được."""
        if temperature not in self._clients:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._clients[temperature] = ChatGoogleGenerativeAI(
                model=self.model_name, google_api_key=self._key,
                temperature=temperature,
                max_output_tokens=self.max_output_tokens)
        return self._clients[temperature]

    def invoke(self, prompt: str, *, role: str = "") -> str:
        temp = _TEMPERATURE.get(role, DEFAULT_TEMPERATURE)
        client = self._client(temp)

        last: Exception | None = None
        for attempt in range(self.max_retries):
            waited = self._limiter.acquire()
            try:
                resp = client.invoke(prompt)
                self.calls.append({"role": role, "temperature": temp,
                                   "waited": round(waited, 2)})
                content = resp.content
                # Gemini có thể trả content dạng list block thay vì chuỗi.
                if isinstance(content, list):
                    return "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content)
                return content or ""
            except Exception as e:                     # noqa: BLE001
                last = e
                if "429" not in str(e) and "quota" not in str(e).lower():
                    raise
                # 429 dù đã điều tiết → hạn ngạch ngày, hoặc Google vừa siết.
                # Lùi theo cấp số nhân rồi thử lại; hết lượt thì ném lên để
                # `safe_node` biến thành escalation chứ không thành stack trace.
                time.sleep(2 ** attempt * 5)
        raise RuntimeError(
            f"Gemini từ chối sau {self.max_retries} lần thử (có thể đã hết hạn "
            f"ngạch ngày của bậc miễn phí): {last}")


def build_llm(backend: str | None = None, **kw):
    """Chọn backend theo biến môi trường `NOVEL_LLM`.

    `fake` (mặc định) chạy miễn phí và tất định; `gemini` dùng bậc free của
    Google AI Studio. Mặc định là `fake` một cách CÓ CHỦ Ý: chạy `cli.py` mà
    vô tình đốt hạn ngạch là chuyện không nên xảy ra vì quên set biến.
    """
    from novel_engine.llm.fake import FakeLLM

    backend = (backend or os.environ.get("NOVEL_LLM", "fake")).lower()
    if backend == "gemini":
        return GeminiLLM(**kw)
    if backend == "fake":
        return FakeLLM()
    raise ValueError(f"backend không biết: {backend!r} (dùng 'fake' | 'gemini')")
