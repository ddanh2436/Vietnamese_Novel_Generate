"use strict";

/* Trạng thái phía giao diện — không có logic nghiệp vụ ở đây, mọi quyết định
 * (chương nào chờ duyệt, canon sạch hay không, v.v.) đến nguyên từ
 * novel_engine/api.py qua các endpoint /api/*. File này chỉ gọi, render, và
 * bắt lỗi để hiển thị. */
const state = {
  db: localStorage.getItem("ne_db") || "novel_storage.db",
  llm: localStorage.getItem("ne_llm") || "fake",
  chapters: [],
  selected: null,
  characters: null,
  clues: null,
  continuity: null,
  writing: false,
};

const STATUS = {
  "chưa viết":     { dot: "dot-none",    pill: "pill-none" },
  "đã viết":       { dot: "dot-written", pill: "pill-written" },
  "chờ duyệt":     { dot: "dot-review",  pill: "pill-review" },
  "đã ghi canon":  { dot: "dot-canon",   pill: "pill-canon" },
};

function statusOf(trangThai) {
  return STATUS[trangThai] || STATUS["chưa viết"];
}

async function apiGet(path) {
  const sep = path.includes("?") ? "&" : "?";
  const res = await fetch(`${path}${sep}db=${encodeURIComponent(state.db)}`);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

async function apiPatch(path, body) {
  const res = await fetch(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

/* ── Tải dữ liệu ─────────────────────────────────────────────────────── */

async function refreshAll() {
  await Promise.all([loadProjectSummary(), loadChapterList(), loadCharacters(), loadClues()]);
}

async function loadProjectSummary() {
  const box = document.getElementById("project-summary");
  try {
    const t = await apiGet("/api/trang-thai");
    state.continuity = t.lien_tuc;
    box.innerHTML = `
      <div class="title">${t.so_chuong} / ${t.tong_chuong_ke_hoach} chương</div>
      <div>${t.so_canh} cảnh đã dựng · ${state.db}</div>
      <div style="color:${t.lien_tuc.sach ? "var(--status-canon)" : "var(--danger)"}">
        ${t.lien_tuc.sach ? "Liên tục sạch ✓" : `${t.lien_tuc.so_loi} lỗi liên tục`}
      </div>`;
    renderContinuityTab();
  } catch (e) {
    box.innerHTML = `<div class="continuity-issue">${escapeHtml(e.message)}</div>`;
  }
}

async function loadChapterList() {
  const list = document.getElementById("chapter-list");
  try {
    state.chapters = await apiGet("/api/chuong");
  } catch (e) {
    list.innerHTML = `<div class="continuity-issue">${escapeHtml(e.message)}</div>`;
    return;
  }
  list.innerHTML = "";
  for (const ch of state.chapters) {
    const st = statusOf(ch.trang_thai);
    const row = el(`
      <button class="chapter-row ${ch.so_canh_da_viet ? "" : "unwritten"} ${state.selected === ch.chapter ? "selected" : ""}">
        <span class="dot ${st.dot}"></span>
        <span class="num">${String(ch.chapter).padStart(2, "0")}</span>
        <span class="title">${escapeHtml(ch.title)}</span>
      </button>`);
    row.addEventListener("click", () => selectChapter(ch.chapter));
    list.appendChild(row);
  }
  if (state.selected == null && state.chapters.length) {
    selectChapter(state.chapters[0].chapter);
  }
}

async function loadCharacters() {
  try {
    state.characters = await apiGet("/api/nhan-vat");
  } catch (e) {
    state.characters = [];
  }
  renderCharactersTab();
}

async function loadClues() {
  try {
    state.clues = await apiGet("/api/manh-moi");
  } catch (e) {
    state.clues = [];
  }
  renderCluesTab();
}

/* ── Cột giữa: chương đang chọn ──────────────────────────────────────── */

async function selectChapter(n) {
  if (state.writing) return;
  state.selected = n;
  document.querySelectorAll(".chapter-row").forEach((r, i) => {
    r.classList.toggle("selected", state.chapters[i]?.chapter === n);
  });
  const header = document.getElementById("chapter-header");
  const body = document.getElementById("chapter-body");
  header.innerHTML = "";
  body.innerHTML = `<div class="muted">Đang tải…</div>`;

  const meta = state.chapters.find((c) => c.chapter === n);
  let chapter;
  try {
    chapter = await apiGet(`/api/chuong/${n}`);
  } catch (e) {
    body.innerHTML = `<div class="continuity-issue">${escapeHtml(e.message)}</div>`;
    return;
  }
  renderChapterHeader(meta, chapter);
  renderChapterBody(meta, chapter);
}

function renderChapterHeader(meta, chapter) {
  const header = document.getElementById("chapter-header");
  const st = statusOf(meta.trang_thai);
  const written = chapter.ton_tai;
  header.innerHTML = "";

  const left = el(`
    <div>
      <span class="pill ${st.pill}"><span class="dot ${st.dot}"></span>${escapeHtml(meta.trang_thai)}</span>
      <h1 class="h-serif">${escapeHtml(chapter.title || meta.title)}</h1>
      <div class="sub">Chương ${meta.chapter} · ${meta.so_canh_da_viet} / ${meta.so_canh_ke_hoach} cảnh${written ? ` · ${chapter.so_tu} từ` : ""}</div>
    </div>`);

  const actions = el(`<div class="actions"></div>`);
  const writeBtn = el(`<button class="btn btn-primary">${written ? "Viết lại (force)" : "Viết chương →"}</button>`);
  writeBtn.addEventListener("click", () => writeChapter(meta.chapter, written));
  if (written) writeBtn.classList.replace("btn-primary", "btn-ghost");
  actions.appendChild(writeBtn);

  header.appendChild(left);
  header.appendChild(actions);
}

function renderChapterBody(meta, chapter) {
  const body = document.getElementById("chapter-body");
  if (!chapter.ton_tai) {
    body.innerHTML = "";
    body.appendChild(el(`
      <div class="empty-state">
        <div>Chương này chưa được viết.</div>
        ${meta.outline_beat ? `<div class="beat">${escapeHtml(meta.outline_beat)}</div>` : ""}
      </div>`));
    return;
  }
  body.innerHTML = "";
  for (const s of chapter.scenes) {
    const stats = s.stats || {};
    const statChips = Object.entries(stats)
      .slice(0, 4)
      .map(([k, v]) => `<span class="chip">${escapeHtml(k)}: ${escapeHtml(JSON.stringify(v))}</span>`)
      .join("");
    body.appendChild(el(`
      <div class="scene-block">
        <div class="chips">
          <span class="chip">Cảnh ${s.scene_index}</span>
          <span class="chip">${escapeHtml(s.scene_id)}</span>
        </div>
        <p>${escapeHtml(s.prose)}</p>
        ${statChips ? `<div class="stat-chips">${statChips}</div>` : ""}
      </div>`));
  }
}

/* ── Viết chương (SSE) ───────────────────────────────────────────────── */

async function writeChapter(chapter, force) {
  if (state.writing) return;
  state.writing = true;
  const banner = document.getElementById("progress-banner");
  banner.hidden = false;
  banner.className = "progress-banner";
  banner.textContent = "Đang khởi động pipeline…";

  const url = `/api/chuong/${chapter}/viet?db=${encodeURIComponent(state.db)}` +
              `&llm=${encodeURIComponent(state.llm)}&force=${force ? "true" : "false"}`;

  try {
    const res = await fetch(url, { method: "POST" });
    if (!res.ok || !res.body) {
      throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        handleSseChunk(chunk, banner);
      }
    }
  } catch (e) {
    banner.className = "progress-banner error";
    banner.textContent = `Lỗi: ${e.message}`;
  } finally {
    state.writing = false;
  }
}

function handleSseChunk(chunk, banner) {
  const lines = chunk.split("\n");
  let event = "message", data = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  let payload = {};
  try { payload = JSON.parse(data); } catch { /* dòng trống, bỏ qua */ }

  if (event === "progress") {
    banner.className = "progress-banner";
    banner.textContent = `Đang viết cảnh ${payload.scenes_done}/${payload.scenes_total}` +
      (payload.revision_count ? ` · vòng phê bình ${payload.revision_count}` : "") +
      (payload.escalated ? " · đã leo thang" : "");
  } else if (event === "done") {
    banner.className = "progress-banner done";
    banner.textContent = `Xong: ${payload.so_canh} cảnh, ${payload.so_tu} từ` +
      (payload.escalated ? ` · leo thang: ${payload.escalation_reason}` : "");
    refreshAll().then(() => { if (state.selected != null) selectChapter(state.selected); });
  } else if (event === "error") {
    banner.className = "progress-banner error";
    banner.textContent = `Lỗi: ${payload.message}`;
  }
}

/* ── Cột phải: tab ───────────────────────────────────────────────────── */

function renderContinuityTab() {
  const box = document.getElementById("tab-lien-tuc");
  const c = state.continuity;
  if (!c) { box.innerHTML = `<div class="muted small">Đang tải…</div>`; return; }
  if (c.sach) {
    box.innerHTML = `<div class="continuity-ok">Không có lỗi liên tục nào</div>`;
    return;
  }
  box.innerHTML = c.chi_tiet.map((f) => `
    <div class="continuity-issue">
      <div style="font-weight:700;text-transform:uppercase;font-size:10.5px;">${escapeHtml(f.severity)} · ${escapeHtml(f.check)}</div>
      <div style="margin-top:4px;">${escapeHtml(f.message)}</div>
    </div>`).join("");
}

function renderCharactersTab() {
  const box = document.getElementById("tab-nhan-vat");
  if (!state.characters) { box.innerHTML = `<div class="muted small">Đang tải…</div>`; return; }
  if (!state.characters.length) {
    box.innerHTML = `<div class="muted small">Không có nhân vật nào trong bible.</div>`;
    return;
  }
  box.innerHTML = "";
  for (const c of state.characters) {
    const card = el(`
      <div class="card char-card">
        <div class="row">
          <span class="name">${escapeHtml(c.name)}</span>
          <span class="chip">${escapeHtml(c.register || "—")}</span>
        </div>
        ${c.voice_exemplars?.[0] ? `<div class="exemplar">"${escapeHtml(c.voice_exemplars[0])}"</div>` : ""}
        <button class="btn btn-ghost btn-small" style="align-self:flex-start;">Chỉnh giọng</button>
      </div>`);
    card.querySelector("button").addEventListener("click", () => openVoiceModal(c));
    box.appendChild(card);
  }
}

function renderCluesTab() {
  const box = document.getElementById("tab-manh-moi");
  if (!state.clues) { box.innerHTML = `<div class="muted small">Đang tải…</div>`; return; }
  if (!state.clues.length) {
    box.innerHTML = `<div class="muted small">Không có manh mối nào trong bible.</div>`;
    return;
  }
  const header = `<div class="eyebrow" style="margin-bottom:2px;">Manh mối · toàn dự án</div>`;
  box.innerHTML = header + state.clues.map((c) => `
    <div class="clue-row">
      <div>
        <div class="id">${escapeHtml(c.id)}</div>
        <div class="meta">trạng thái ${escapeHtml(c.status)} · salience ${c.salience}</div>
      </div>
      <span class="chip">${c.payoff_deadline != null ? `hạn ch.${c.payoff_deadline}` : "không hạn"}</span>
    </div>`).join("");
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) =>
      p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
  });
});

/* ── Modal: chỉnh giọng ──────────────────────────────────────────────── */

let voiceTarget = null;

function openVoiceModal(character) {
  voiceTarget = character;
  document.getElementById("voice-char-name").textContent = character.name;
  document.getElementById("voice-register").value = character.register || "";
  document.getElementById("voice-exemplars").value = (character.voice_exemplars || []).join("\n");
  document.getElementById("voice-forbidden").value = (character.forbidden_lexicon || []).join(", ");
  document.getElementById("voice-error").hidden = true;
  document.getElementById("voice-modal").hidden = false;
}

document.getElementById("btn-voice-cancel").addEventListener("click", () => {
  document.getElementById("voice-modal").hidden = true;
});

document.getElementById("btn-voice-save").addEventListener("click", async () => {
  const errBox = document.getElementById("voice-error");
  errBox.hidden = true;
  const thayDoi = {
    register: document.getElementById("voice-register").value.trim(),
    voice_exemplars: document.getElementById("voice-exemplars").value
      .split("\n").map((s) => s.trim()).filter(Boolean),
    forbidden_lexicon: document.getElementById("voice-forbidden").value
      .split(",").map((s) => s.trim()).filter(Boolean),
  };
  try {
    await apiPatch(`/api/nhan-vat/${encodeURIComponent(voiceTarget.id)}`, { thay_doi: thayDoi });
    document.getElementById("voice-modal").hidden = true;
    await loadCharacters();
  } catch (e) {
    errBox.textContent = e.message;
    errBox.hidden = false;
  }
});

/* ── Modal: cài đặt dự án ────────────────────────────────────────────── */

document.getElementById("btn-settings").addEventListener("click", () => {
  document.getElementById("input-db").value = state.db;
  document.getElementById("input-llm").value = state.llm;
  document.getElementById("settings-modal").hidden = false;
});
document.getElementById("btn-settings-cancel").addEventListener("click", () => {
  document.getElementById("settings-modal").hidden = true;
});
document.getElementById("btn-settings-save").addEventListener("click", () => {
  state.db = document.getElementById("input-db").value.trim() || "novel_storage.db";
  state.llm = document.getElementById("input-llm").value;
  localStorage.setItem("ne_db", state.db);
  localStorage.setItem("ne_llm", state.llm);
  document.getElementById("settings-modal").hidden = true;
  state.selected = null;
  refreshAll();
});

refreshAll();
