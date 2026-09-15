"""News Dispatcher — lan truyền tin & Fog of War (§5.6).

`POVFirewall` trả lời chiều CẤM (nhân vật không được biết gì). Module này trả lời
chiều CHO PHÉP: nhân vật biết một sự kiện ở nơi khác TỪ LÚC NÀO, qua kênh nào,
và biết PHIÊN BẢN nào. Toàn bộ tất định, không gọi LLM (NT-1): code chọn toán tử
biến dạng, Writer diễn đạt.

═══ SÁU CHỖ KHÁC MÃ §5.6 ═══════════════════════════════════════════════════

1. `allowed_channels` LÀ MÃ CHẾT. `propagate` gọi `graph.routes_from(loc)` không
   truyền kênh, nên tin đồn đi qua khu Thánh đường (chỉ cho người đưa tin và tín
   hiệu) và người đưa tin đi đường biển. Chính đồ thị đã có tham số `channel=`.

2. CHỈ KÊNH ĐẦU TIÊN ĐƯỢC LAN. `frontier` dùng chung cho mọi kênh: kênh thứ hai
   bắt đầu ở điểm gốc với (tick gốc, fidelity 1,0) — đúng bản kênh đầu đã nạp —
   nên bị coi là "bị trội" ngay chặng đầu và không đi đâu cả. Với mặc định
   `channels=["courier", "rumor"]`, tin đồn không bao giờ lan. Ở đây biên Pareto
   TÌM KIẾM là của từng kênh (mỗi kênh đi được những cạnh khác nhau), còn biên
   Pareto TRI THỨC — bản nào đáng ghi nhận ở một nơi — là chung.

3. `MAX_FRONTIER` CẮT MẤT BẢN CHÍNH XÁC NHẤT. `keep[:4]` sau khi sắp theo tick giữ
   4 bản đến SỚM nhất. Trên biên Pareto, sắp theo tick thì fidelity tăng dần — bản
   muộn nhất là bản chính xác nhất, đúng bản mà lỗi "tin đồn chặn sứ giả" (chính
   mục §5.6.3 vừa sửa) cần giữ. Ở đây giữ bản sớm nhất VÀ các bản chính xác nhất.

4. `is_correction` PHỤ THUỘC THỨ TỰ DUYỆT KÊNH. Nó so với các bản đã có trong
   `arrivals` lúc đó — tức chỉ các kênh đã chạy trước. Đổi thứ tự `channels` là
   đổi beat đính chính. Ở đây đánh dấu SAU khi lan xong, theo thời gian.

5. NGƯỜI ĐẾN SAU KHÔNG BAO GIỜ BIẾT. `commit_arrivals` chỉ hỏi ai CÓ MẶT lúc tin
   tới (`characters_at(loc, tick)`, vốn không tồn tại). Tin không bay đi khi người
   đưa tin rời quán: ai tới sau đó vẫn nghe. Không có điều này là đúng lỗi mà phần
   mở đầu §5.6 cảnh báo — "mãi không biết dù tin đã lan khắp nơi".

6. `FLAW_RESISTANCE` TRA THEO TÊN KHUYẾT ĐIỂM ("sợ bị phản bội", "kiêu ngạo trí
   thức"...). Không nhân vật nào trong bible mang những tên đó, nên kháng cự luôn
   bằng 0 và "đính chính không tự động được chấp nhận" không bao giờ xảy ra. Tên
   là văn xuôi của tác giả, không phải khoá. `Flaw` khai số kháng cự, thẻ tin mà
   nó kháng cự, và độ tin theo kênh.

Thêm: rủi ro (tin rơi, tin méo) bốc thăm bằng hash của TỪNG CHẶNG, không bằng
một `random.Random` chạy tuần tự. §5.6.5 nói seed phải theo `news_id` để thêm tin
mới không đổi đường lan tin cũ; cùng lập luận đó, thêm một tuyến đường mới cũng
không được đổi kết quả trên những tuyến cũ (NT-16).
"""
from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from novel_engine.canon.timeline import ContinuityFrame, StoryTime


class Channel(BaseModel):
    kind: str
    latency_multiplier: float
    reliability: float = Field(ge=0.0, le=1.0)
    distortion_rate: float = Field(ge=0.0, le=1.0)
    interceptable: bool = True
    controlled_by: str | None = None


DEFAULT_CHANNELS: dict[str, Channel] = {
    "courier": Channel(kind="courier", latency_multiplier=1.0, reliability=0.92,
                       distortion_rate=0.04),
    "signal": Channel(kind="signal", latency_multiplier=0.02, reliability=0.75,
                      distortion_rate=0.01),
    "rumor": Channel(kind="rumor", latency_multiplier=2.4, reliability=0.99,
                     distortion_rate=0.38, interceptable=False),
    "trade_caravan": Channel(kind="trade_caravan", latency_multiplier=3.0,
                             reliability=0.88, distortion_rate=0.12),
    "ritual": Channel(kind="ritual", latency_multiplier=1.5, reliability=0.95,
                      distortion_rate=0.02),
}
CHANNEL_VI = {"courier": "người đưa tin", "signal": "tín hiệu", "rumor": "tin đồn",
              "trade_caravan": "đoàn buôn", "ritual": "nghi lễ"}
CHANNEL_TRUST = {"courier": 0.85, "signal": 0.75, "trade_caravan": 0.55,
                 "rumor": 0.25, "ritual": 0.70}


class NewsItem(BaseModel):
    news_id: str
    origin_location: str
    origin_tick: int
    truth: str                                  # nội dung THẬT
    subject_entities: list[str]
    channels: list[str] = Field(default_factory=lambda: ["courier", "rumor"])
    suppressed_by: list[str] = Field(default_factory=list)
    # Thẻ để `Flaw.resists_tags` so khớp — "exculpates:CHAR_KAELEN" v.v.
    tags: list[str] = Field(default_factory=list)


DISTORTION_OPERATORS = {
    "exaggerate": "nhân số lượng/quy mô lên 2–5 lần",
    "substitute_agent": "đổi người gây ra sang một phe/nhân vật khác",
    "invert_outcome": "đảo kết cục: bị bắt ↔ đã chết ↔ trốn thoát",
    "drop_qualifier": "bỏ điều kiện: 'nếu không đầu hàng sẽ bị truy nã' → 'đã bị truy nã'",
    "merge_with_prior": "trộn với một tin cũ về cùng nhân vật",
    "attribute_motive": "gán động cơ không có trong tin gốc",
}
OPS_VI = {"exaggerate": "phóng đại quy mô", "substitute_agent": "đổi người gây ra",
          "invert_outcome": "đảo kết cục", "drop_qualifier": "bỏ mất điều kiện",
          "merge_with_prior": "trộn với tin cũ", "attribute_motive": "gán động cơ không có"}

FIDELITY_DECAY = 0.72
MAX_FRONTIER = 4          # số phiên bản tối đa một nơi giữ lại cho một tin
CORRECTION_GAIN = 0.25    # mức tăng fidelity tối thiểu để coi là "đính chính"
INTEREST_THRESHOLD = 0.5


def _unit(*parts) -> float:
    """Số trong [0, 1) tất định theo nội dung — blake2b, không `hash()` (NT-16)."""
    h = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def distort(payload: dict, key: tuple) -> dict:
    ops = list(DISTORTION_OPERATORS)
    op = ops[int(_unit(*key, "op") * len(ops))]
    return {"content_ref": payload["content_ref"],
            "applied_ops": payload["applied_ops"] + [op],
            "fidelity": round(payload["fidelity"] * FIDELITY_DECAY, 3)}


class ParetoFrontier:
    """Tập bản (tick, fidelity) không bị trội ở mỗi nơi: sớm hơn VÀ chính xác hơn."""

    def __init__(self, cap: int | None = MAX_FRONTIER):
        self.cap = cap
        self.items: dict[str, list[tuple[int, float]]] = {}

    def dominated(self, loc: str, tick: int, fid: float) -> bool:
        return any(t <= tick and f >= fid for t, f in self.items.get(loc, []))

    def admit(self, loc: str, tick: int, fid: float) -> None:
        keep = [(t, f) for t, f in self.items.get(loc, []) if not (tick <= t and fid >= f)]
        keep.append((tick, fid))
        keep.sort(key=lambda x: (x[0], -x[1]))
        if self.cap is not None and len(keep) > self.cap:
            # Sắp theo tick thì fidelity TĂNG dần: bản cuối là bản chính xác nhất.
            keep = [keep[0]] + keep[-(self.cap - 1):]
        self.items[loc] = keep


def propagate(news: NewsItem, graph, upto_tick: int,
              channels: dict[str, Channel] | None = None) -> list[dict]:
    """Lan truyền ĐA MỤC TIÊU (sớm và chính xác). Một nơi có thể nhận NHIỀU phiên
    bản của cùng một tin, ở các thời điểm khác nhau."""
    channels = channels or DEFAULT_CHANNELS
    knowledge = ParetoFrontier()
    found: list[dict] = []

    for cname in news.channels:
        ch = channels.get(cname)
        if ch is None or (ch.controlled_by and ch.controlled_by in news.suppressed_by):
            continue
        search = ParetoFrontier(cap=None)          # của RIÊNG kênh này
        counter = 0
        pq = [(news.origin_tick, counter, news.origin_location,
               {"content_ref": news.news_id, "applied_ops": [], "fidelity": 1.0},
               (news.origin_location,))]
        while pq:
            tick, _, loc, payload, path = heapq.heappop(pq)
            fid = payload["fidelity"]
            if tick > upto_tick or search.dominated(loc, tick, fid):
                continue
            search.admit(loc, tick, fid)
            found.append({"location": loc, "tick": tick, "channel": cname,
                          "payload": payload})
            for edge in graph.routes_from(loc, channel=cname):
                if set(edge["blocked_by"]) & set(news.suppressed_by):
                    continue
                if edge["to"] in path:
                    continue                          # không quay vòng
                hop = (news.news_id, cname, loc, edge["to"], tick)
                if _unit(*hop, "reach") > ch.reliability:
                    continue
                nxt = distort(payload, hop) if _unit(*hop, "distort") < ch.distortion_rate \
                    else payload
                counter += 1
                heapq.heappush(pq, (tick + int(edge["latency_ticks"] * ch.latency_multiplier),
                                    counter, edge["to"], nxt, path + (edge["to"],)))

    # Biên TRI THỨC tính trên mọi kênh cùng lúc, theo thời gian — không theo thứ
    # tự duyệt kênh.
    found.sort(key=lambda a: (a["tick"], -a["payload"]["fidelity"], a["channel"]))
    for a in found:
        if not knowledge.dominated(a["location"], a["tick"], a["payload"]["fidelity"]):
            knowledge.admit(a["location"], a["tick"], a["payload"]["fidelity"])
    arrivals, seen = [], set()
    for a in found:
        key = (a["location"], a["tick"], a["payload"]["fidelity"])
        if key in seen or (key[1], key[2]) not in knowledge.items.get(key[0], []):
            continue
        seen.add(key)
        arrivals.append(a)
    return mark_corrections(arrivals)


def mark_corrections(arrivals: list[dict]) -> list[dict]:
    """Đính chính là một BEAT tự sự — "hoá ra anh ta vẫn còn sống" tự nó là một cảnh."""
    best: dict[str, float] = {}
    for a in sorted(arrivals, key=lambda x: (x["tick"], -x["payload"]["fidelity"])):
        fid = a["payload"]["fidelity"]
        prior = best.get(a["location"])
        a["is_correction"] = prior is not None and fid - prior >= CORRECTION_GAIN
        best[a["location"]] = max(prior or 0.0, fid)
    return arrivals


# ═══════════════════════ BỘ LỌC QUAN TÂM (§5.6.4) ═══════════════════════

def _factions(graph, eid: str) -> set[str]:
    if not graph.exists(eid):
        return set()
    return {dst for _, dst, key, d in graph.g.out_edges(eid, keys=True, data=True)
            if key == "MEMBER_OF" and d.get("until_chapter") is None}


def interest_score(char, news: NewsItem, graph) -> float:
    """§5.6.4 so `e in [b.proposition ...]` — MÃ thực thể với CÂU niềm tin, không
    bao giờ bằng nhau (NT-8). Ở đây tìm TÊN thực thể trong câu niềm tin."""
    s = 0.0
    mine = _factions(graph, char.id)
    belief_text = " ".join(b.proposition for b in
                           list(char.beliefs) + list(char.private_knowledge)).lower()
    for e in news.subject_entities:
        if e == char.id:
            s += 1.0
        if e in char.leverage_over or e in char.debt_to:
            s += 0.5
        if e in mine or (e != char.id and mine & _factions(graph, e)):
            s += 0.4
        if any(e in d.threatened_by or e in d.satisfied_by for d in char.desires):
            s += 0.6
        names = graph.names_of(e) if graph.exists(e) else []
        if any(n and n != e and n.lower() in belief_text for n in names):
            s += 0.3
    return s


# ═══════════════════════ AI NGHE, LÚC NÀO ═══════════════════════

def _as_frames(frames) -> list[ContinuityFrame]:
    return [f if isinstance(f, ContinuityFrame) else ContinuityFrame.model_validate(f)
            for f in (frames or [])]


def chapter_end_tick(frames) -> int:
    return max((f.time.end_tick for f in _as_frames(frames) if f.time.mode == "present"),
               default=0)


def learn_tick(cid: str, frames: list[ContinuityFrame], loc: str, arrival_tick: int,
               upto_tick: int) -> int | None:
    """Lúc nhân vật NGHE được bản tin đã tới `loc` ở `arrival_tick`.

    Vị trí tại T = frame hiện tại muộn nhất có `epoch_tick ≤ T` (ở lại tới cảnh
    sau). Đang ở đó lúc tin tới → nghe ngay. Không thì nghe khi lần đầu tới đó.
    """
    track = sorted((f for f in frames if cid in f.locations and f.time.mode == "present"),
                   key=lambda f: (f.time.epoch_tick, f.time.narrative_order))
    before = [f for f in track if f.time.epoch_tick <= arrival_tick]
    if before and before[-1].locations[cid] == loc:
        return arrival_tick
    for f in track:
        if arrival_tick < f.time.epoch_tick <= upto_tick and f.locations[cid] == loc:
            return f.time.epoch_tick
    return None


# ═══════════════════════ ĐÍNH CHÍNH (§5.6.6) ═══════════════════════

def accept_correction(char, old_fid: float, new_fid: float, channel: str,
                      tags=()) -> tuple[bool, str]:
    flaw = char.fatal_flaw
    trust = flaw.channel_trust.get(channel, CHANNEL_TRUST.get(channel, 0.5))
    applies = (not flaw.resists_tags
               or any(t.replace("{self}", char.id) in tags for t in flaw.resists_tags))
    resist = flaw.correction_resistance if applies else 0.0
    gain = (new_fid - old_fid) * trust
    threshold = 0.15 + resist
    if gain > threshold:
        return True, f"chấp nhận: gain {gain:.2f} > ngưỡng {threshold:.2f}"
    return False, (f"GIỮ NGUYÊN niềm tin cũ: '{flaw.name}' đặt ngưỡng {threshold:.2f}, "
                   f"tin qua '{channel}' chỉ đạt {gain:.2f}")


def render_version(news: NewsItem, held: dict, graph) -> str:
    """Bản nhân vật GIỮ, dạng đưa được vào ngữ cảnh. Bản méo KHÔNG chứa `truth`:
    người nghe tin đồn mà thấy sự thật trong ngữ cảnh là rò rỉ POV."""
    kenh = CHANNEL_VI.get(held["channel"], held["channel"])
    if not held["ops"]:
        return f"Tin ({kenh}): {news.truth}"
    names = []
    for e in news.subject_entities:
        ns = graph.names_of(e) if graph.exists(e) else []
        names.append(next((n for n in ns if n != e), e))
    ops = ", ".join(OPS_VI[o] for o in held["ops"])
    return (f"Tin {kenh} về {', '.join(names)} — một bản đã sai lệch ({ops}); "
            f"nhân vật tin bản này")


class NewsDispatcher:
    def __init__(self, items: list[NewsItem] | None = None,
                 channels: dict[str, Channel] | None = None):
        self.items = {n.news_id: n for n in (items or [])}
        self.channels = dict(channels or DEFAULT_CHANNELS)

    @classmethod
    def from_bible(cls, bible_dir: Path | str) -> "NewsDispatcher":
        p = Path(bible_dir) / "news.yaml"
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        chans = dict(DEFAULT_CHANNELS)
        for name, over in (raw.get("channels") or {}).items():
            base = chans[name].model_dump() if name in chans else {"kind": name}
            chans[name] = Channel.model_validate({**base, **(over or {})})
        return cls([NewsItem.model_validate(n) for n in raw.get("news", [])], chans)

    def hearings(self, char, graph, frames: list[ContinuityFrame], upto_tick: int) -> list[dict]:
        out = []
        for nid in sorted(self.items):
            news = self.items[nid]
            if news.origin_tick > upto_tick or interest_score(char, news, graph) < INTEREST_THRESHOLD:
                continue
            for a in propagate(news, graph, upto_tick, self.channels):
                t = learn_tick(char.id, frames, a["location"], a["tick"], upto_tick)
                if t is not None:
                    out.append({"news_id": nid, "tick": t, "arrival_tick": a["tick"],
                                "location": a["location"], "channel": a["channel"],
                                "fidelity": a["payload"]["fidelity"],
                                "ops": list(a["payload"]["applied_ops"])})
        out.sort(key=lambda h: (h["tick"], h["news_id"], -h["fidelity"], h["channel"]))
        return out

    def judge(self, char, hearings: list[dict]) -> list[dict]:
        """Chuỗi quyết định theo thời gian: nghe lần đầu → tin; bản chính xác hơn
        HẲN → `accept_correction`; bản kém hơn → bỏ qua."""
        held: dict[str, dict] = {}
        events = []
        for h in hearings:
            nid = h["news_id"]
            news = self.items[nid]
            cur = held.get(nid)
            if cur is None:
                kind, reason = "first", "nghe lần đầu"
                held[nid] = h
            elif h["fidelity"] - cur["fidelity"] >= CORRECTION_GAIN:
                ok, reason = accept_correction(char, cur["fidelity"], h["fidelity"],
                                               h["channel"], news.tags)
                kind = "correction_accepted" if ok else "correction_rejected"
                if ok:
                    held[nid] = h
            else:
                continue
            events.append({**h, "kind": kind, "reason": reason, "held": held[nid],
                           "truth": news.truth,
                           "ops_vi": [OPS_VI[o] for o in h["ops"]],
                           "channel_vi": CHANNEL_VI.get(h["channel"], h["channel"])})
        return events

    def scene_directives(self, *, graph, chars: dict, past_frames,
                         planned: list[dict]) -> dict[int, list[dict]]:
        """Tin chạm tới POV của từng cảnh — vị trí THẬT của các chương đã viết cộng
        vị trí DỰ KIẾN của chương đang viết."""
        past = [f for f in _as_frames(past_frames) if f.time.mode == "present"]
        synth = []
        times = []
        for p in planned:
            t = p["time"] if isinstance(p["time"], StoryTime) else StoryTime.model_validate(p["time"])
            times.append(t)
            if t.mode == "present":
                synth.append(ContinuityFrame(scene_id=p["scene_id"], time=t,
                                             locations={c: p["location_id"] for c in p["present"]}))
        frames = past + synth
        out: dict[int, list[dict]] = defaultdict(list)
        for si, (p, t) in enumerate(zip(planned, times)):
            char = chars.get(p["pov"])
            if t.mode != "present" or char is None:
                continue
            prev = [f.time.end_tick for f in frames
                    if p["pov"] in f.locations and f.scene_id != p["scene_id"]
                    and f.time.epoch_tick < t.epoch_tick]
            lo = max(prev) if prev else None
            for ev in self.judge(char, self.hearings(char, graph, frames, t.end_tick)):
                if (lo is None or ev["tick"] > lo) and ev["tick"] <= t.end_tick:
                    out[si].append({k: ev[k] for k in (
                        "news_id", "kind", "channel", "channel_vi", "ops", "ops_vi",
                        "truth", "tick")})
        return out


def settle_news(graph, chars: dict, frames, upto_tick: int) -> dict:
    """Ghi tri thức tin tức vào canon — gọi lúc commit VÀ lúc replay.

    Dựng lại TỪ ĐẦU tới `upto_tick` mỗi lần: kết quả chỉ phụ thuộc bible + vị trí
    nhân vật, nên chạy lại cho ra đúng như cũ (idempotent), và rollback một chương
    tự động rút lại những gì nhân vật đã nghe trong chương đó.
    """
    disp = getattr(graph, "news", None)
    if disp is None or not disp.items:
        return {"records": 0, "rejected": 0}
    from novel_engine.character.beliefs import update_belief

    fs = [f for f in _as_frames(frames)
          if f.time.mode == "present" and f.time.epoch_tick <= upto_tick]
    knowledge: dict[tuple[str, str], dict] = {}
    rejected = 0
    for cid in sorted(chars):
        char = chars[cid]
        char.beliefs = [b for b in char.beliefs if not b.source.startswith("news:")]
        by_news: dict[str, list[dict]] = defaultdict(list)
        for ev in disp.judge(char, disp.hearings(char, graph, fs, upto_tick)):
            by_news[ev["news_id"]].append(ev)
        for nid, evs in by_news.items():
            news = disp.items[nid]
            held = evs[-1]["held"]
            n_rej = sum(1 for e in evs if e["kind"] == "correction_rejected")
            rejected += n_rej
            trust = char.fatal_flaw.channel_trust.get(
                held["channel"], CHANNEL_TRUST.get(held["channel"], 0.5))
            # Từ chối đính chính KHÔNG đưa confidence về 0 — chỉ giảm nhẹ (§5.6.6).
            conf = min(0.95, 0.35 + 0.6 * held["fidelity"] * trust) * (0.9 ** n_rej)
            render = render_version(news, held, graph)
            update_belief(char, render, round(conf, 3),
                          source=f"news:{nid}:{held['channel']}",
                          is_actually_true=not held["ops"])
            knowledge[(cid, nid)] = {
                "since_tick": evs[0]["tick"], "held": held, "rejected": n_rej,
                "timeline": [{"tick": e["tick"],
                              "render": render_version(news, e["held"], graph)}
                             for e in evs]}
    graph.news_knowledge = knowledge
    return {"records": len(knowledge), "rejected": rejected}
