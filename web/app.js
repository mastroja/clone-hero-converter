// RockWizard web app: the same screens in the desktop window and on the phone.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const desk = () => window.pywebview && window.pywebview.api;  // only inside the desktop window

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((data && (data.detail || data.error)) || `${r.status} ${r.statusText}`);
  return data;
}
const post = (path, body = {}) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const qs = (o) => new URLSearchParams(o).toString();

let toastTimer;
function toast(msg) {
  let t = $(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.append(t); }
  t.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 4200);
}

// ---------- icons (24px line icons) ----------
const P = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
const ICON = {
  create: P('<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/><circle cx="12" cy="12" r="3"/>'),
  queue: P('<path d="M4 6h11M4 12h11M4 18h7"/><path d="m17 15 3 3-3 3"/>'),
  staging: P('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m8 12 3 3 5-6"/>'),
  library: P('<path d="M4 4h4v16H4zM10 4h4v16h-4z"/><path d="m16.5 4.5 3.8 1 -3.9 14.9-3.8-1z"/>'),
  settings: P('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>'),
  upload: P('<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>'),
  play: P('<path d="m7 4 13 8-13 8z"/>'),
  folder: P('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
  rocket: P('<path d="M5 15c-1.5 1.5-2 5-2 5s3.5-.5 5-2"/><path d="M14 4c3-1 6-1 6-1s0 3-1 6l-6 6-5-5z"/><path d="M9 10 5 9l-2 2 4 1M14 15l1 4 2-2-1-4"/>'),
};
const VIEWS = [["create", "Create"], ["queue", "Queue"], ["staging", "Staging"], ["library", "Library"], ["settings", "Settings"]];

// ---------- state + shell ----------
const S = { settings: {}, jobs: [], status: {}, staging: null };
const view = () => (location.hash.slice(1).split("?")[0] || "create");

function navHTML(cls) {
  const active = S.jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  return VIEWS.map(([id, label]) =>
    `<button class="${cls}${view() === id ? " on" : ""}" data-go="${id}">${ICON[id]}<span>${label}</span>` +
    (id === "queue" && active ? `<span class="count">${active}</span>` : "") + `</button>`).join("");
}
function renderNav() {
  $("#nav").innerHTML = navHTML("nav");
  $("#tabbar").innerHTML = navHTML("");
}
document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]");
  if (go) { location.hash = go.dataset.go; }
});
window.addEventListener("hashchange", () => { closeDrawer(); renderNav(); render(); });

function statusPills() {
  const st = S.status || {};
  const pills = [];
  if (st.busy) pills.push(["busy", `Converting${st.queued ? ` (+${st.queued} waiting)` : ""}`]);
  else if (st.training && st.training.running) pills.push(["busy", "GPU training a model"]);
  else pills.push(["good", "Ready"]);
  const sy = st.sync;
  if (sy) {
    if (sy.state === "not shared") pills.push(["", "Live folder not synced"]);
    else if (sy.devices && sy.devices.length) {
      const d = sy.devices[0];
      if (!d.connected) pills.push(["warn", `${d.name}: offline`]);
      else if (d.completion >= 100) pills.push(["good", `${d.name}: up to date`]);
      else pills.push(["busy", `${d.name}: syncing ${Math.floor(d.completion)}%`]);
    } else pills.push(["", `Sync: ${sy.state}`]);
  }
  $("#pills").innerHTML = pills.map(([c, t]) => `<div class="pill"><span class="dot ${c}"></span><span>${esc(t)}</span></div>`).join("");
  const [c, t] = pills[0];
  $("#mpill").innerHTML = `<span class="dot ${c}"></span><span>${esc(t)}</span>`;
}

async function refresh() {
  try {
    const [jobs, status] = await Promise.all([api("/api/jobs"), api("/api/status")]);
    const before = JSON.stringify(S.jobs.map((j) => [j.id, j.status, j.step, Math.round(j.progress * 50)]));
    S.jobs = jobs; S.status = status;
    renderNav(); statusPills();
    if (view() === "queue" && before !== JSON.stringify(jobs.map((j) => [j.id, j.status, j.step, Math.round(j.progress * 50)]))) renderQueue();
  } catch (e) { /* server restarting */ }
}

function page(title, sub, body, actions = "") {
  $("#main").innerHTML = `<div class="top"><div><h1>${title}</h1>${sub ? `<p class="sub">${sub}</p>` : ""}</div><div class="row">${actions}</div></div>${body}`;
}
const render = () => ({ create: renderCreate, queue: renderQueue, staging: renderStaging, library: renderLibrary, settings: renderSettings }[view()] || renderCreate)();

// ---------- create ----------
function namesFromFile(stem) {  // quick guess; the PC looks names up properly while converting
  let s = stem.replace(/_/g, " ").replace(/[\(\[].*?[\)\]]/g, "").trim();
  const parts = s.split(" - ").map((x) => x.trim()).filter(Boolean);
  if (parts.length > 2 && (!parts[parts.length - 1].includes(" ") || parts[parts.length - 1] === parts[parts.length - 1].toUpperCase())) parts.pop();
  return parts.length > 1 ? [parts[0], parts.slice(1).join(" - ")] : ["", parts[0] || stem];
}

function renderCreate() {
  page("Create", "Drop in a song, search for one, or paste a YouTube link. Your PC does the work.", `
    <div class="card">
      <div class="drop" id="drop">
        <div class="icon">${ICON.upload}</div>
        <b>Drop an MP3 or WAV here</b>
        <div class="muted">or click to choose a file</div>
        <input type="file" id="file" accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.opus" hidden>
      </div>
      <div id="pending" class="hidden stack" style="margin-top:14px"></div>
    </div>
    <div class="card">
      <h2>Find a song</h2>
      <div class="searchbox">
        <input id="q" type="search" placeholder="Artist - Song title, or paste a YouTube link" autocomplete="off">
        <button class="btn primary" id="go">Search</button>
      </div>
      <div id="results"></div>
    </div>
    <div class="card desktop-only" id="batchcard">
      <div class="row spread wrap">
        <div><h2 style="margin:0">Convert a whole folder</h2><div class="faint">Every song in a folder, one after another.</div></div>
        <button class="btn" id="pickbatch">${ICON.folder} Choose folder…</button>
      </div>
      <div id="batch"></div>
    </div>`);
  const drop = $("#drop"), file = $("#file");
  drop.onclick = () => file.click();
  file.onchange = () => file.files[0] && pending(file.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) pending(f); };
  $("#go").onclick = doSearch;
  $("#q").onkeydown = (e) => e.key === "Enter" && doSearch();
  if (!desk()) $("#batchcard").classList.add("hidden");  // needs the desktop window's folder picker
  $("#pickbatch").onclick = pickBatch;
}

function pending(f) {
  const [a, t] = namesFromFile(f.name.replace(/\.[^.]+$/, ""));
  const box = $("#pending");
  box.classList.remove("hidden");
  box.innerHTML = `
    <div class="row"><span class="chip accent">${esc(f.name)}</span><span class="faint">${(f.size / 1e6).toFixed(1)} MB</span></div>
    <div class="grid2">
      <div><label class="field">Artist</label><input id="pa" value="${esc(a)}"></div>
      <div><label class="field">Song title</label><input id="pt" value="${esc(t)}"></div>
    </div>
    <button class="btn primary big" id="make">${ICON.rocket} Make my chart</button>`;
  $("#make").onclick = async () => {
    const fd = new FormData();
    fd.append("file", f); fd.append("artist", $("#pa").value); fd.append("title", $("#pt").value);
    $("#make").disabled = true; $("#make").innerHTML = `<span class="spinner"></span> Uploading…`;
    try {
      await api("/api/jobs/upload", { method: "POST", body: fd });
      toast("Added to the queue"); location.hash = "queue";
    } catch (e) { toast(e.message); $("#make").disabled = false; $("#make").textContent = "Make my chart"; }
  };
}

const ytId = (url) => { try { return new URL(url).searchParams.get("v") || url.split("/").pop(); } catch (e) { return ""; } };
const mmss = (s) => s ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "";

async function doSearch() {
  const q = $("#q").value.trim();
  if (!q) return;
  const out = $("#results");
  if (/^https?:\/\//.test(q)) {
    await post("/api/jobs/convert", { url: q });
    toast("Added to the queue"); location.hash = "queue"; return;
  }
  out.innerHTML = `<div class="empty"><span class="spinner"></span><div>Searching charts, song details and YouTube…</div></div>`;
  try {
    const r = await api("/api/search?" + qs({ q }));
    const d = r.details;
    out.innerHTML = `
      ${d ? `<p class="muted" style="margin:14px 0 4px">${esc(d.artist)} – <b style="color:var(--text)">${esc(d.title)}</b> · ${esc([d.album, d.year, d.genre].filter(Boolean).join(" · "))}</p>` : ""}
      ${r.errors.length ? `<p class="chip yellow">${esc(r.errors.join("; "))}</p>` : ""}
      <h2 style="margin-top:16px">Already charted by the community</h2>
      ${r.charts.length ? r.charts.map((c, i) => `
        <div class="result"><div class="grow"><div class="ellipsis"><b>${esc(c.artist)} – ${esc(c.name)}</b></div>
          <div class="faint">charted by ${esc(c.charter)}${c.parts ? " · " + esc(c.parts) : ""}</div></div>
          <button class="btn small" data-chart="${i}">Download</button></div>`).join("")
        : `<p class="faint">No human-made chart found, so making one is the way to go.</p>`}
      <h2 style="margin-top:16px">Make a chart from YouTube</h2>
      ${r.videos.map((v, i) => `
        <div class="result">
          <img class="thumb" src="https://i.ytimg.com/vi/${esc(ytId(v.url))}/mqdefault.jpg" alt="" loading="lazy">
          <div class="grow"><div class="ellipsis"><b>${esc(v.title)}</b></div>
            <div class="faint">${esc(v.channel)} · ${mmss(v.seconds)} ${v.best ? `<span class="chip green">Best match</span>` : ""}</div></div>
          <button class="btn primary small" data-video="${i}">Chart this</button></div>`).join("") || `<p class="faint">No videos found.</p>`}`;
    $$("[data-chart]", out).forEach((b) => b.onclick = async () => {
      const c = r.charts[b.dataset.chart];
      b.disabled = true; b.innerHTML = `<span class="spinner"></span>`;
      try { const s = await post("/api/charts/download", c); b.textContent = "Saved ✓"; toast("Saved to " + s.saved); }
      catch (e) { toast(e.message); b.disabled = false; b.textContent = "Download"; }
    });
    $$("[data-video]", out).forEach((b) => b.onclick = async () => {
      const v = r.videos[b.dataset.video];
      await post("/api/jobs/convert", { url: v.url, artist: v.artist, title: v.song });
      toast(`Charting ${v.artist} – ${v.song}`); location.hash = "queue";
    });
  } catch (e) { out.innerHTML = `<p class="chip red">${esc(e.message)}</p>`; }
}

async function pickBatch() {
  const d = await desk().pick_folder();
  if (!d) return;
  const box = $("#batch");
  box.innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  const r = await api("/api/library/batch?" + qs({ folder: d }));
  box.innerHTML = `
    <div class="faint" style="margin:12px 0 6px">${r.files.length} song${r.files.length === 1 ? "" : "s"} in ${esc(r.folder)} · names are looked up again while converting</div>
    <div class="list">${r.files.map((f, i) => `
      <div class="item"><input type="checkbox" class="bf" data-i="${i}" checked style="width:18px;height:18px;accent-color:var(--accent)">
        <div class="grow"><div class="grid2"><input class="ba" value="${esc(f.artist)}" placeholder="Artist"><input class="bt" value="${esc(f.title)}" placeholder="Song title"></div>
        <div class="faint" style="margin-top:4px">${esc(f.file)}${f.existing ? ` · <span style="color:var(--warn)">already converted: the old version is backed up and replaced</span>` : ""}</div></div></div>`).join("")}</div>
    <button class="btn primary" id="runbatch" style="margin-top:12px" ${r.files.length ? "" : "disabled"}>Convert ${r.files.length} songs</button>`;
  $("#runbatch").onclick = async () => {
    const rows = $$(".item", box);
    let n = 0;
    for (const row of rows) {
      if (!$(".bf", row).checked) continue;
      const f = r.files[$(".bf", row).dataset.i];
      await post("/api/jobs/convert", { path: f.path, artist: $(".ba", row).value, title: $(".bt", row).value });
      n++;
    }
    toast(`${n} songs added to the queue`); location.hash = "queue";
  };
}

// ---------- queue ----------
const STATUS_CHIP = { queued: ["", "Waiting"], running: ["accent", "Working"], done: ["green", "Done"], failed: ["red", "Failed"], cancelled: ["", "Cancelled"] };
const VERDICT = { green: ["green", "Looks good"], yellow: ["yellow", "Check flagged spots"], red: ["red", "Has problems"] };

function renderQueue() {
  const jobs = S.jobs;
  page("Queue", "Songs being made, one at a time on your PC's graphics card.",
    jobs.length ? `<div class="stack">${jobs.map((j) => {
      const [cc, ct] = STATUS_CHIP[j.status] || ["", j.status];
      const step = j.status === "running" ? (j.steps[j.step] || "Starting") : j.status === "done" ? "Saved to staging" : j.message || "";
      const v = j.result && j.result.verdict && VERDICT[j.result.verdict];
      return `<div class="card">
        <div class="row spread"><div class="grow"><b class="ellipsis" style="display:block">${esc(j.title)}</b>
          <div class="faint">${j.kind === "update" ? "Updating chart · " : ""}${esc(step)}</div></div>
          <span class="chip ${cc}">${ct}</span></div>
        ${j.status === "running" || j.status === "queued" ? `<div class="bar" style="margin-top:12px"><i style="width:${Math.round(j.progress * 100)}%"></i></div>` : ""}
        <div class="row" style="margin-top:12px">
          ${v ? `<span class="chip ${v[0]}">${v[1]}</span>` : ""}<span class="grow"></span>
          ${j.status === "running" || j.status === "queued" ? `<button class="btn small" data-cancel="${j.id}">Cancel</button>` : ""}
          ${j.status === "done" ? `<button class="btn primary small" data-review="${esc(j.result.folder)}">Review</button>` : ""}
        </div></div>`;
    }).join("")}</div>` : `<div class="card empty">Nothing in the queue. Make a chart from the Create tab.</div>`,
    jobs.some((j) => ["done", "failed", "cancelled"].includes(j.status)) ? `<button class="btn" id="clear">Clear finished</button>` : "");
  $$("[data-cancel]").forEach((b) => b.onclick = async () => { await post(`/api/jobs/${b.dataset.cancel}/cancel`); refresh(); });
  $$("[data-review]").forEach((b) => b.onclick = () => openReview(b.dataset.review));
  const c = $("#clear"); if (c) c.onclick = async () => { await post("/api/jobs/clear"); refresh().then(renderQueue); };
}

// ---------- staging ----------
let stagingFilter = "all";
const initials = (s) => (s || "?").split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
const coverHTML = (s, cls = "cover") => s.has_cover
  ? `<img class="${cls}" src="/api/cover?${qs({ folder: s.folder })}" alt="" loading="lazy">`
  : `<div class="${cls} ph">${esc(initials(s.title || s.name))}</div>`;

async function renderStaging(reviewAll = false) {
  page("Staging", "New charts land here first. Check them, then publish to your live library.",
    `<div class="empty"><span class="spinner"></span>${reviewAll ? "<div>Checking every song…</div>" : ""}</div>`,
    `<button class="btn" id="checkall">Check all</button>`);
  $("#checkall").onclick = () => renderStaging(true);
  let r;
  try { r = await api("/api/staging?" + qs({ review_all: reviewAll })); } catch (e) { $("#main").innerHTML += `<p class="chip red">${esc(e.message)}</p>`; return; }
  S.staging = r;
  const songs = r.songs.filter((s) => stagingFilter === "all" || (stagingFilter === "draft" ? s.status === "draft" : s.status === stagingFilter));
  const counts = { all: r.songs.length, draft: r.songs.filter((s) => s.status === "draft").length, reviewed: r.songs.filter((s) => s.status === "reviewed").length };
  $("#main").querySelector(".empty").outerHTML = `
    <div class="row spread wrap" style="margin-bottom:14px">
      <div class="tabs">${[["all", "All"], ["draft", "To review"], ["reviewed", "Reviewed"]].map(([k, l]) =>
        `<button class="${stagingFilter === k ? "on" : ""}" data-filter="${k}">${l} <span class="faint">${counts[k]}</span></button>`).join("")}</div>
      <span class="faint ellipsis">${esc(r.folder)}</span>
    </div>
    ${songs.length ? `<div class="songs">${songs.map((s) => {
      const v = s.verdict && VERDICT[s.verdict];
      return `<div class="song" data-open="${esc(s.folder)}">${coverHTML(s)}
        <div class="meta"><div class="t ellipsis">${esc(s.title || s.name)}</div>
          <div class="faint ellipsis">${esc(s.artist || "Unknown artist")}</div>
          <div class="row wrap" style="margin-top:8px;gap:6px">
            ${v ? `<span class="chip ${v[0]}">${v[0] === "green" ? "✓" : v[0] === "yellow" ? "!" : "✕"} ${v[0] === "green" ? "OK" : v[0] === "yellow" ? "Check" : "Problems"}</span>` : `<span class="chip">Not checked</span>`}
            ${s.status === "reviewed" ? `<span class="chip accent">Reviewed</span>` : ""}
          </div></div></div>`;
    }).join("")}</div>` : `<div class="card empty">No songs here yet.</div>`}`;
  $$("[data-filter]").forEach((b) => b.onclick = () => { stagingFilter = b.dataset.filter; renderStaging(); });
  $$("[data-open]").forEach((c) => c.onclick = () => openReview(c.dataset.open));
}

function closeDrawer() { $$(".scrim, .drawer").forEach((e) => e.remove()); }

async function openReview(folder) {
  closeDrawer();
  const scrim = document.createElement("div"); scrim.className = "scrim"; scrim.onclick = closeDrawer;
  const dr = document.createElement("div"); dr.className = "drawer";
  dr.innerHTML = `<div class="empty"><span class="spinner"></span><div>Checking the chart…</div></div>`;
  document.body.append(scrim, dr);
  let s;
  try { s = await api("/api/staging/review?" + qs({ folder })); } catch (e) { dr.innerHTML = `<p class="chip red">${esc(e.message)}</p>`; return; }
  const v = VERDICT[s.verdict];
  const order = { red: 0, yellow: 1, info: 2 };
  const findings = [...s.findings].sort((a, b) => order[a.level] - order[b.level]);
  const local = S.settings.local;
  dr.innerHTML = `
    <div class="row spread"><span class="faint">Review</span><button class="btn small" id="x">Close</button></div>
    <div class="head" style="margin:14px 0 16px">${coverHTML(s, s.has_cover ? "" : "ph cover")}
      <div class="grow"><h2 style="margin:0">${esc(s.title || s.name)}</h2>
        <div class="muted">${esc(s.artist)}${s.album ? ` · ${esc(s.album)}` : ""}${s.year ? ` · ${esc(s.year)}` : ""}</div>
        <div class="row wrap" style="margin-top:8px;gap:6px"><span class="chip ${v[0]}">${v[1]}</span>
          <span class="chip">${esc(s.status)}</span></div>
        <div class="tiers" style="margin-top:8px">${["guitar", "bass", "drums"].map((k) => `<span>${k} <b>${esc(s.tiers[k] || "–")}</b></span>`).join("")}</div>
      </div></div>
    <div class="card"><h2>Checks</h2>
      ${findings.map((f) => `<div class="finding"><span class="lv ${f.level}"></span><div>${esc(f.text)}</div></div>`).join("")}</div>
    <div class="card"><h2>Listen</h2><div class="stems">
      ${["guitar", "bass", "drums", "vocals"].filter((k) => s.stems.includes(k)).map((k) =>
        `<span>${k}</span><audio controls preload="none" src="/api/audio?${qs({ folder: s.folder, stem: k })}"></audio>`).join("")}</div></div>
    <div class="card"><h2>Next</h2>
      <div class="row wrap">
        <button class="btn primary" id="pub">${ICON.rocket} Publish to live</button>
        ${s.status !== "reviewed" ? `<button class="btn" id="rev">Mark reviewed</button>` : ""}
        <button class="btn" id="rechart">Re-chart</button>
      </div>
      ${local ? `<div class="row wrap" style="margin-top:10px">
        <button class="btn small" data-open-pc="clonehero">${ICON.play} Play in Clone Hero</button>
        <button class="btn small" data-open-pc="moonscraper">Open in Moonscraper</button>
        <button class="btn small" data-open-pc="folder">${ICON.folder} Folder</button></div>` : ""}
      <p class="faint" style="margin-bottom:0">Publishing moves the song into your live folder; Syncthing takes it to the Deck.</p>
    </div>`;
  $("#x", dr).onclick = closeDrawer;
  $("#pub", dr).onclick = async () => {
    let r = await post("/api/staging/publish", { folder: s.folder });
    if (!r.ok && confirm(`${r.message}.\n\nPublish anyway?`)) r = await post("/api/staging/publish", { folder: s.folder, force: true });
    toast(r.message);
    if (r.ok) { closeDrawer(); if (view() === "staging") renderStaging(); }
  };
  const rev = $("#rev", dr);
  if (rev) rev.onclick = async () => { await post("/api/staging/status", { folder: s.folder, status: "reviewed" }); toast("Marked as reviewed"); openReview(folder); if (view() === "staging") renderStaging(); };
  $("#rechart", dr).onclick = async () => { await post("/api/jobs/update", { folders: [s.folder] }); toast("Re-charting with the newest models"); closeDrawer(); location.hash = "queue"; };
  $$("[data-open-pc]", dr).forEach((b) => b.onclick = async () => {
    try { await post("/api/open", { what: b.dataset.openPc, folder: s.folder }); } catch (e) { toast(e.message); }
  });
}

// ---------- library ----------
let libTab = "update";
function folderField(id, value) {
  return `<div class="row"><input id="${id}" value="${esc(value)}" class="grow">
    ${desk() ? `<button class="btn" data-browse="${id}">${ICON.folder}</button>` : ""}
    <button class="btn primary" data-load="${id}">Load</button></div>`;
}
function wireFolder(load) {
  $$("[data-browse]").forEach((b) => b.onclick = async () => {
    const d = await desk().pick_folder(); if (d) { $("#" + b.dataset.browse).value = d; load(); }
  });
  $$("[data-load]").forEach((b) => b.onclick = load);
}

function renderLibrary() {
  page("Library", "Keep your songs tidy and up to date.", `
    <div class="tabs" style="margin-bottom:14px">${[["update", "Update charts"], ["tidy", "Tidy names"], ["dupes", "Duplicates"]].map(([k, l]) =>
      `<button class="${libTab === k ? "on" : ""}" data-lib="${k}">${l}</button>`).join("")}</div>
    <div id="lib"></div>`);
  $$("[data-lib]").forEach((b) => b.onclick = () => { libTab = b.dataset.lib; renderLibrary(); });
  ({ update: libUpdate, tidy: libTidy, dupes: libDupes })[libTab]();
}

function libUpdate() {
  $("#lib").innerHTML = `<div class="card"><p class="muted" style="margin-top:0">Remake charts with the newest models, from the audio already in each song folder. The old chart is backed up first. Only songs RockWizard made are listed; other charts are never touched.</p>
    ${folderField("uf", S.settings.songs_dir || "")}<div id="ulist" style="margin-top:12px"></div></div>`;
  const load = async () => {
    const box = $("#ulist"); box.innerHTML = `<span class="spinner"></span>`;
    try {
      const r = await api("/api/library/ours?" + qs({ folder: $("#uf").value }));
      box.innerHTML = r.songs.length ? `<div class="list">${r.songs.map((s) => `
        <label class="item check"><input type="checkbox" class="us" value="${esc(s.folder)}" checked><span class="grow ellipsis">${esc(s.name)}</span>
        <span class="faint">${new Date(s.modified * 1000).toLocaleDateString()}</span></label>`).join("")}</div>
        <button class="btn primary" id="urun" style="margin-top:12px">Update selected</button>` : `<div class="empty">No RockWizard songs in that folder.</div>`;
      const run = $("#urun"); if (run) run.onclick = async () => {
        const folders = $$(".us:checked").map((c) => c.value);
        await post("/api/jobs/update", { folders }); toast(`${folders.length} charts queued`); location.hash = "queue";
      };
    } catch (e) { box.innerHTML = `<p class="chip red">${esc(e.message)}</p>`; }
  };
  wireFolder(load); load();
}

function libTidy() {
  const o = S.tidyOpts || (S.tidyOpts = { ini: true, parens: true, brackets: true });
  $("#lib").innerHTML = `<div class="card"><p class="muted" style="margin-top:0">Give every song the same “Artist - Song Title” name. Nothing changes until you press Rename, and it can be undone.</p>
    ${folderField("tf", S.settings.tidy_dir || S.settings.live_dir || S.settings.songs_dir || "")}
    <div class="row wrap" style="margin-top:12px;gap:18px">
      <label class="check"><input type="checkbox" id="oini" ${o.ini ? "checked" : ""}> Artist - Song Title (from song.ini)</label>
      <label class="check"><input type="checkbox" id="opar" ${o.parens ? "checked" : ""}> Remove (parentheses)</label>
      <label class="check"><input type="checkbox" id="obra" ${o.brackets ? "checked" : ""}> Remove [brackets]</label></div>
    <div id="tlist" style="margin-top:12px"></div></div>`;
  const opts = () => ({ ini: $("#oini").checked, parens: $("#opar").checked, brackets: $("#obra").checked });
  const load = async () => {
    Object.assign(o, opts());
    const box = $("#tlist"); box.innerHTML = `<span class="spinner"></span>`;
    try {
      const r = await api("/api/tidy?" + qs({ folder: $("#tf").value, ...o }));
      box.innerHTML = (r.proposals.length ? `<div class="list">${r.proposals.map((p) => `
        <label class="item check"><input type="checkbox" class="tp" value="${esc(p.path)}" ${p.note ? "disabled" : "checked"}>
          <div class="grow"><div class="faint ellipsis">${esc(p.name)}</div><div class="ellipsis">→ ${esc(p.new_name)}${p.note ? ` <span style="color:var(--warn)">(skipped: ${esc(p.note)})</span>` : ""}</div></div></label>`).join("")}</div>`
        : `<div class="empty">All names are already tidy.</div>`) + `
        <div class="row" style="margin-top:12px"><button class="btn primary" id="tapply" ${r.proposals.length ? "" : "disabled"}>Rename selected</button>
        <button class="btn" id="tundo" ${r.can_undo ? "" : "disabled"}>Undo last rename</button></div>`;
      $("#tapply").onclick = async () => {
        const res = await post("/api/tidy/apply", { folder: r.folder, paths: $$(".tp:checked").map((c) => c.value), ...o });
        toast(`Renamed ${res.renamed}` + (res.errors.length ? ` · ${res.errors[0]}` : "")); load();
      };
      $("#tundo").onclick = async () => { const res = await post("/api/tidy/undo", { folder: r.folder }); toast(`Restored ${res.restored}`); load(); };
    } catch (e) { box.innerHTML = `<p class="chip red">${esc(e.message)}</p>`; }
  };
  $$("#oini, #opar, #obra").forEach((c) => c.onchange = load);
  wireFolder(load); load();
}

function libDupes() {
  $("#lib").innerHTML = `<div class="card"><p class="muted" style="margin-top:0">The same song more than once. Copies with exactly the same chart are ticked for you; different versions (live, another charter…) are listed but left alone. Removed copies go to the Recycle Bin.</p>
    ${folderField("df", S.settings.tidy_dir || S.settings.live_dir || S.settings.songs_dir || "")}<div id="dlist" style="margin-top:12px"></div></div>`;
  const load = async () => {
    const box = $("#dlist"); box.innerHTML = `<span class="spinner"></span>`;
    try {
      const r = await api("/api/dupes?" + qs({ folder: $("#df").value }));
      box.innerHTML = r.groups.length ? r.groups.map((g) => `<div style="margin-top:12px"><b>${esc(g[0].artist ? g[0].artist + " - " : "")}${esc(g[0].title)}</b>
        ${g.map((c) => `<label class="item check"><input type="checkbox" class="dp" value="${esc(c.path)}" ${c.identical ? "checked" : ""}>
          <div class="grow"><div class="ellipsis">${esc(c.name)}</div>
          <div class="faint">${c.keep ? `<span style="color:var(--good)">keep (suggested)</span>` : c.identical ? "same chart as the kept copy" : "different version"} · ${esc(c.charter)} · ${esc(c.parts.join(", ") || "parts not listed")} · ${c.size_mb} MB</div></div></label>`).join("")}</div>`).join("")
        + `<button class="btn primary" id="drm" style="margin-top:14px">Move ticked copies to the Recycle Bin</button>`
        : `<div class="empty">No duplicates found.</div>`;
      const b = $("#drm"); if (b) b.onclick = async () => {
        const paths = $$(".dp:checked").map((c) => c.value);
        if (!paths.length || !confirm(`Move ${paths.length} folder${paths.length === 1 ? "" : "s"} to the Recycle Bin?`)) return;
        const res = await post("/api/dupes/remove", { paths }); toast(`Moved ${res.removed} to the Recycle Bin` + (res.errors.length ? ` · ${res.errors[0]}` : "")); load();
      };
    } catch (e) { box.innerHTML = `<p class="chip red">${esc(e.message)}</p>`; }
  };
  wireFolder(load); load();
}

// ---------- settings ----------
async function renderSettings() {
  const st = S.settings, local = st.local;
  const field = (id, label, hint, kind = "folder") => `<div style="margin-top:14px"><label class="field">${label}</label>
    <div class="row"><input id="${id}" value="${esc(st[id] || "")}" class="grow">${desk() && kind ? `<button class="btn" data-pick="${id}" data-kind="${kind}">Browse</button>` : ""}</div>
    ${hint ? `<div class="faint" style="margin-top:4px">${hint}</div>` : ""}</div>`;
  page("Settings", "", `
    <div class="card"><h2>Folders and programs</h2>
      ${field("songs_dir", "Staging folder", "New charts are saved here for review first.")}
      ${field("live_dir", "Live folder", "Where published songs go (your Quickplay folder, which Syncthing sends to the Deck).")}
      ${field("charter", "Charter name", "Written into each song.", "")}
      ${field("clonehero_exe", "Clone Hero", "Clone Hero.exe, for Play in Clone Hero.", "file")}
      ${field("moonscraper_exe", "Moonscraper", "Moonscraper Chart Editor.exe, for Open in Moonscraper.", "file")}
      <button class="btn primary" id="save" style="margin-top:16px">Save</button></div>
    <div class="card" id="phone"><h2>Use RockWizard on your phone</h2><div class="empty"><span class="spinner"></span></div></div>
    <div class="card"><h2>Steam Deck sync</h2><div id="syncbox" class="muted"></div></div>
    <div class="card"><h2>YouTube downloader</h2><p class="muted" style="margin-top:0">YouTube changes often and older downloader versions stop working. If searching or downloading fails, update it.</p>
      <button class="btn" id="dl">Update downloader</button> <span id="dlmsg" class="faint"></span></div>`);
  $$("[data-pick]").forEach((b) => b.onclick = async () => {
    const v = b.dataset.kind === "file" ? await desk().pick_file() : await desk().pick_folder();
    if (v) $("#" + b.dataset.pick).value = v;
  });
  $("#save").onclick = async () => {
    const body = {}; ["songs_dir", "live_dir", "charter", "clonehero_exe", "moonscraper_exe"].forEach((k) => body[k] = $("#" + k).value);
    Object.assign(S.settings, await post("/api/settings", body)); toast("Saved");
  };
  $("#dl").onclick = async () => { $("#dlmsg").textContent = "Updating…"; const r = await post("/api/downloader/update"); $("#dlmsg").textContent = r.message; };
  if (!("sync" in S.status)) { try { S.status = await api("/api/status"); } catch (e) { /* shown as not set up */ } }
  const sy = S.status.sync;
  $("#syncbox").innerHTML = !sy ? "Syncthing isn't set up on this PC, or no live folder is set."
    : sy.state === "not shared" ? "The live folder isn't one of Syncthing's shared folders."
    : `Folder “${esc(sy.label)}”: ${esc(sy.state)}. ${(sy.devices || []).map((d) => `${esc(d.name)} ${d.connected ? `${Math.floor(d.completion)}% synced` : "offline"}`).join(" · ")}
       <div style="margin-top:10px"><button class="btn small" id="rescan">Sync now</button></div>`;
  const rs = $("#rescan"); if (rs) rs.onclick = async () => { try { await post("/api/sync/rescan"); toast("Syncthing is checking for changes"); } catch (e) { toast(e.message); } };
  if (!local) { $("#phone").innerHTML = `<h2>Use RockWizard on your phone</h2><p class="muted">This phone is paired. Tip: in Safari, Share → Add to Home Screen.</p>`; return; }
  const p = await api("/api/pairing");
  $("#phone").innerHTML = `<h2>Use RockWizard on your phone</h2>
    <div class="row wrap" style="align-items:flex-start;gap:20px">
      ${p.qr ? `<div class="qr">${p.qr}</div>` : ""}
      <div class="grow" style="min-width:240px">
        <p style="margin-top:0">Scan with your iPhone's camera, then in Safari tap <b>Share → Add to Home Screen</b>. Your PC still does the work.</p>
        <p class="faint">At home this works over Wi-Fi. Away from home, install <b>Tailscale</b> (free) on this PC and your iPhone and sign in to both; the link then uses the Tailscale address (100.x.x.x).</p>
        ${p.urls.map((u) => `<div style="margin-top:6px"><code>${esc(u)}</code></div>`).join("")}
        <button class="btn small" id="newkey" style="margin-top:12px">New pairing key (unpairs other phones)</button></div></div>`;
  $("#newkey").onclick = async () => { if (confirm("Make a new key? Phones paired before will need to scan again.")) { await post("/api/pairing/reset"); renderSettings(); } };
}

// ---------- start ----------
(async () => {
  try { S.settings = await api("/api/settings"); } catch (e) { $("#main").innerHTML = `<div class="card empty">${esc(e.message)}</div>`; return; }
  renderNav(); render(); refresh();
  setInterval(refresh, 1500);
  const h = location.hash.slice(1);
  if (h.startsWith("review=")) openReview(decodeURIComponent(h.slice(7)));
})();
