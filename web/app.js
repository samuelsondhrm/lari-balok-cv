"use strict";
// Lari Balok CV dashboard: polls /api/state, renders the scoreboard, plays start/alert sounds,
// and drives calibration and the referee's frame-by-frame review.

const $ = (s) => document.querySelector(s);
const LANES = [
  { name: "Merah", color: "#ef4444", ink: "#fff" },
  { name: "Biru", color: "#3b82f6", ink: "#fff" },
  { name: "Kuning", color: "#facc15", ink: "#231a00" },
];
const STATUS = { siap: "SIAP", kosong: "KOSONG", berlaga: "BERLAGA", finis: "FINIS", gugur: "GUGUR", dnf: "TIDAK FINIS" };
const HINTS = ["Klik ujung DEKAT garis START", "Klik ujung JAUH garis START", "Klik ujung DEKAT garis FINIS", "Klik ujung JAUH garis FINIS"];
const PT_LABELS = ["START dekat", "START jauh", "FINIS dekat", "FINIS jauh"];
const CTA = `<li class="cta">Arena belum dikalibrasi.<br>Tekan <b>Kalibrasi</b> untuk menandai garis START, FINIS, dan warna balok.</li>`;

let S = null;               // latest server state
let gotAt = 0;              // performance.now() when S arrived
let announced = null;       // event ids already announced with a toast and sound
let countdownFor = null;    // race_id whose countdown beeps are scheduled
let podiumClosedFor = null; // race_id whose podium the operator closed
let audio = null;
let cal = null;             // calibration session
let rv = null;              // review session
let resumedCalib = false;

const fmt = (t) => {
  const cs = Math.round(Math.max(0, t || 0) * 100);
  const m = Math.floor(cs / 6000);
  return `${String(m).padStart(2, "0")}:${((cs % 6000) / 100).toFixed(2).padStart(5, "0")}`;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const laneStyle = (i) => `--c:${LANES[i].color};--ink:${LANES[i].ink}`;
const setHTML = (el, html) => { if (el._html !== html) { el.innerHTML = html; el._html = html; } };

async function api(action, body = {}) {
  try {
    const r = await fetch("/api/cmd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...body }),
    });
    const j = await r.json();
    if (!j.ok) toast(j.error || "Perintah gagal.", "bad");
    return j;
  } catch {
    toast("Server tidak terjangkau.", "bad");
    return { ok: false };
  }
}

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    S = await r.json();
    gotAt = performance.now();
    render();
  } catch {
    $("#live").textContent = "● TERPUTUS";
    $("#live").className = "pill";
  }
  setTimeout(poll, 150);
}

function render() {
  renderHeader();
  renderLanes();
  renderLog();
  announce();
  scheduleCountdown();
  renderPodium();
  if (rv && $("#review").open) reviewStatus();
  if (S.calibrating && !cal && !resumedCalib) { resumedCalib = true; calibBegin(); } // page reloaded mid-calibration
}

function renderHeader() {
  const live = $("#live");
  const name = S.source.split(/[\\/]/).pop();
  live.textContent = S.is_file ? `▶ VIDEO · ${name}${S.eof ? " · habis" : ""}` : `● LIVE · Kamera ${S.source}`;
  live.className = `pill ${S.is_file ? "file" : "on"}`;
  $("#fps").textContent = `${Math.round(S.fps)} fps · ${S.model.replace(/\.pt$/, "")}`;
  const st = S.race?.state;
  $("#btnStart").disabled = !S.race || st !== "siap" || S.calibrating || S.eof;
  $("#btnEnd").disabled = !["hitung", "berlaga"].includes(st);
  $("#btnReset").disabled = S.calibrating || (!S.race && !S.eof);
  $("#btnCalib").disabled = S.calibrating || ["hitung", "berlaga"].includes(st);
}

function renderLanes() {
  const el = $("#lanes");
  if (!S.race) { el._n = 0; return setHTML(el, CTA); }
  const lanes = S.race.lanes;
  if (el._n !== lanes.length) {
    el._n = lanes.length;
    setHTML(el, lanes.map((l, i) => `<li class="lane" style="${laneStyle(i)}">
      <div class="lane-head"><span class="badge">${l.lane}</span>
        <span class="lane-name">Lintasan ${l.lane}<small>${LANES[i].name}</small></span><span class="chips"></span></div>
      <div class="bar"><i></i></div><div class="lane-foot"></div><div class="viols"></div></li>`).join(""));
  }
  const clips = new Set(S.clips);
  lanes.forEach((l, i) => {
    const card = el.children[i];
    card.className = `lane s-${l.status}`;
    const hidden = l.status === "berlaga" && !l.visible;
    setHTML(card.querySelector(".chips"),
      (hidden ? `<span class="chip warn" title="Kaki pemain tidak terlihat kamera; momen ini tidak dinilai">TERTUTUP</span>` : "") +
      `<span class="chip st">${STATUS[l.status]}</span>`);
    card.querySelector(".bar i").style.width = `${(l.progress * 100).toFixed(1)}%`;
    const time = l.finish != null ? fmt(l.finish) : l.status === "berlaga" ? `${Math.round(l.progress * 100)}%` : "—";
    const rank = l.rank ? `<span class="rank r${Math.min(l.rank, 4)}">${l.tie ? "SERI · " : ""}JUARA ${l.rank}</span>` : "";
    setHTML(card.querySelector(".lane-foot"), `<span class="time">${time}</span>${rank}`);
    // the card shows the touch that puts the lane out; overturned and later touches stay in the log
    const active = S.race.events.filter((e) => e.type === "pelanggaran" && e.lane === i && e.review !== "batal");
    const e = active[0], ready = e && clips.has(e.id);
    setHTML(card.querySelector(".viols"), e ? `<div class="viol"><span>${esc(cap(e.joint))} menyentuh tanah · ${fmt(e.t)}` +
      `${e.review === "sah" ? " · disahkan" : ""}${active.length > 1 ? ` · +${active.length - 1} lagi` : ""}</span>
      <button class="btn mini" data-review="${esc(e.id)}" ${ready ? "" : "disabled"}>${ready ? "Tinjau" : "Menyimpan…"}</button></div>` : "");
  });
}

function renderLog() {
  const ev = S.race ? [...S.race.events].reverse() : [];
  const clips = new Set(S.clips);
  setHTML($("#log"), ev.length ? ev.map((e) => {
    const finish = e.type === "finis";
    const cls = finish ? "ok" : e.review === "batal" ? "muted" : "bad";
    const what = finish ? "finis" : `${esc(e.joint)} menyentuh tanah`;
    const note = e.review === "sah" ? " <em>(disahkan)</em>" : e.review === "batal" ? " <em>(dibatalkan)</em>" : "";
    const btn = clips.has(e.id) ? `<button class="btn mini" data-review="${esc(e.id)}">Tinjau</button>` : "<span></span>";
    return `<li class="${cls}"><time>${fmt(e.t)}</time><span class="dot" style="${laneStyle(e.lane)}"></span>
      <span>L${e.lane + 1} · ${what}${note}</span>${btn}</li>`;
  }).join("") : `<li class="empty">Belum ada kejadian.</li>`);
}

function announce() {
  const events = S.race ? S.race.events : [];
  if (announced === null || events.length === 0) { announced = new Set(events.map((e) => e.id)); return; }
  for (const e of events) {
    if (announced.has(e.id)) continue;
    announced.add(e.id);
    if (e.type === "finis") { toast(`Lintasan ${e.lane + 1} finis · ${fmt(e.t)}`, "ok"); chime(); }
    else { toast(`Lintasan ${e.lane + 1}: ${e.joint} menyentuh tanah · ${fmt(e.t)}`, "bad"); alarm(); }
  }
}

function scheduleCountdown() {
  const r = S.race;
  if (!r || r.state !== "hitung" || r.countdown == null || countdownFor === S.race_id) return;
  countdownFor = S.race_id;
  for (const k of [3, 2, 1]) if (r.countdown - k > -0.2) tone(660, 0.14, r.countdown - k);
  tone(988, 0.6, r.countdown, "triangle", 0.3);
}

function renderPodium() {
  const r = S.race, pod = $("#podium");
  const active = r ? r.lanes.filter((l) => !["kosong", "siap"].includes(l.status)) : [];
  const settled = active.length > 0 && active.every((l) => ["finis", "gugur", "dnf"].includes(l.status));
  const show = r && (r.state === "selesai" || (r.state === "berlaga" && settled)) && active.length > 0 && podiumClosedFor !== S.race_id;
  pod.hidden = !show;
  if (!show) return;
  const ranked = r.lanes.filter((l) => l.rank).sort((a, b) => a.rank - b.rank || a.finish - b.finish);
  const place = (l, cls) => (l
    ? `<div class="place ${cls}" style="${laneStyle(l.lane - 1)}"><span class="medal">${l.rank}</span><b>Lintasan ${l.lane}</b>
        <small>${LANES[l.lane - 1].name}</small><time>${fmt(l.finish)}</time>${l.tie ? "<em>SERI · tinjau klip</em>" : ""}<div class="block"></div></div>`
    : `<div class="place ${cls} empty"></div>`);
  setHTML($("#podiumStage"), place(ranked[1], "second") + place(ranked[0], "first") + place(ranked[2], "third"));
  setHTML($("#podiumRest"), r.lanes.filter((l) => !l.rank && l.status !== "kosong").map((l) =>
    `<li><span class="badge" style="${laneStyle(l.lane - 1)}">${l.lane}</span>Lintasan ${l.lane} · ${STATUS[l.status]}</li>`).join(""));
}

function tick() {
  if (S?.race) {
    const r = S.race, dt = Math.min((performance.now() - gotAt) / 1000, 0.5);
    const t = r.state === "berlaga" ? r.time + dt : r.time;
    $("#clock").textContent = fmt(t);
    $("#clock").classList.toggle("running", r.state === "berlaga");
    let text = null;
    if (r.state === "hitung" && r.countdown != null) { const c = r.countdown - dt; text = c > 0 ? String(Math.ceil(c)) : "MULAI!"; }
    else if (r.state === "berlaga" && t < 0.8) text = "MULAI!";
    const cd = $("#countdown");
    cd.hidden = text === null;
    if (text !== null && cd.textContent !== text) {
      cd.textContent = text;
      cd.classList.toggle("go", text === "MULAI!");
      cd.classList.remove("pop");
      void cd.offsetWidth; // restart the pop animation
      cd.classList.add("pop");
    }
  }
  requestAnimationFrame(tick);
}

// ---------- sound and toasts ----------
function ensureAudio() {
  audio ??= new AudioContext();
  if (audio.state === "suspended") audio.resume();
}
function tone(freq, dur, when = 0, type = "sine", vol = 0.25) {
  if (!audio) return;
  const t = audio.currentTime + Math.max(0, when);
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = type;
  o.frequency.value = freq;
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(audio.destination);
  o.start(t);
  o.stop(t + dur + 0.05);
}
const alarm = () => { tone(880, 0.16, 0, "square", 0.12); tone(622, 0.3, 0.18, "square", 0.12); };
const chime = () => { tone(784, 0.18, 0, "sine", 0.22); tone(1175, 0.4, 0.14, "sine", 0.22); };

function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").prepend(el);
  setTimeout(() => el.classList.add("out"), 4200);
  setTimeout(() => el.remove(), 4700);
}

// ---------- referee review: frame-by-frame evidence clip ----------
const nearest = (frames, t) => frames.reduce((b, f, i) => (Math.abs(f.t - t) < Math.abs(frames[b].t - t) ? i : b), 0);

async function openReview(id) {
  const ev = S?.race?.events.find((e) => e.id === id);
  if (!ev || !S.run) return;
  const base = `/runs/${encodeURIComponent(S.run)}/clips/${encodeURIComponent(id)}/`;
  let clip;
  try { clip = await (await fetch(base + "clip.json", { cache: "no-store" })).json(); } catch { clip = null; }
  if (!clip?.frames?.length) return toast("Klip belum tersedia.", "bad");
  if (rv) stopPlay();
  rv = { id, ev, base, frames: clip.frames, idx: 0, timer: null, at: nearest(clip.frames, ev.t) };
  clip.frames.forEach((f) => { new Image().src = base + f.file; }); // preload for smooth stepping
  const badge = $("#rvLane");
  badge.textContent = ev.lane + 1;
  badge.style.cssText = laneStyle(ev.lane);
  $("#rvTitle").textContent = ev.type === "finis"
    ? `Lintasan ${ev.lane + 1} · finis ${fmt(ev.t)}`
    : `Lintasan ${ev.lane + 1} · ${cap(ev.joint)} menyentuh tanah · ${fmt(ev.t)}`;
  $("#rvDecide").hidden = ev.type !== "pelanggaran";
  $("#rvSlider").max = clip.frames.length - 1;
  reviewStatus();
  showFrame(rv.at);
  $("#review").showModal();
}
function showFrame(i) {
  rv.idx = Math.max(0, Math.min(rv.frames.length - 1, i));
  const f = rv.frames[rv.idx];
  $("#rvImg").src = rv.base + f.file;
  $("#rvSlider").value = rv.idx;
  $("#rvTime").innerHTML = `${fmt(f.t)} · frame ${rv.idx + 1}/${rv.frames.length}${rv.idx === rv.at ? " · <b>saat kejadian</b>" : ""}`;
}
function reviewStatus() {
  const ev = S.race?.events.find((e) => e.id === rv.id) || rv.ev;
  $("#rvStatus").textContent = { menunggu: "Menunggu keputusan wasit", sah: "Pelanggaran disahkan", batal: "Pelanggaran dibatalkan" }[ev.review] || "";
}
function togglePlay() {
  if (rv.timer) return stopPlay();
  if (rv.idx >= rv.frames.length - 1) showFrame(0);
  $("#rvPlay").textContent = "Jeda";
  rv.timer = setInterval(() => (rv.idx >= rv.frames.length - 1 ? stopPlay() : showFrame(rv.idx + 1)), 1000 / 15); // half speed
}
function stopPlay() {
  if (rv?.timer) { clearInterval(rv.timer); rv.timer = null; }
  $("#rvPlay").textContent = "Putar ½×";
}
async function decide(decision) {
  if (!(await api("review", { id: rv.id, decision })).ok) return;
  toast(decision === "sah" ? "Pelanggaran disahkan." : "Pelanggaran dibatalkan; peringkat dihitung ulang.", decision === "sah" ? "bad" : "ok");
  $("#review").close();
}

// ---------- calibration ----------
const sizes = () => ({ lanes: +$("#inLanes").value, length_m: +$("#inLen").value, block_h_cm: +$("#inBlock").value });

async function calibBegin() {
  const j = await api("calib_begin");
  if (!j.ok) return;
  const c = j.calib || {};
  cal = { pts: (c.pts || []).map((p) => [...p]), color: !!c.hsv, picking: false };
  $("#inLanes").value = c.lanes ?? 3;
  $("#inLen").value = c.length_m ?? 5;
  $("#inBlock").value = c.block_h_cm ?? 10;
  $("#colorInfo").textContent = c.hsv ? "Tersimpan dari kalibrasi sebelumnya." : "Belum diambil.";
  $("#board").hidden = true;
  $("#calib").hidden = false;
  document.body.classList.add("calibrating");
  await preview();
  calibUI();
}
async function preview() {
  if (cal.pts.length === 4) await api("calib_draft", { pts: cal.pts, ...sizes() }); // server drops rejected points
  $("#video").src = `/api/snapshot?draft=1&t=${Date.now()}`;
}
function calibUI() {
  const n = cal.pts.length;
  $("#hint").hidden = false;
  $("#hint").textContent = cal.picking ? "Klik sisi balok yang berwarna untuk mengambil warnanya"
    : n < 4 ? `${n + 1}/4 · ${HINTS[n]}`
    : !cal.color ? "Titik lengkap · tekan Ambil warna balok"
    : "Periksa pita lintasan dan kotak balok, lalu tekan Simpan";
  $("#ptsInfo").textContent = n < 4 ? `${n}/4 titik · ${HINTS[n]}.` : "4/4 titik ditandai.";
  $("#st1").classList.toggle("done", n === 4);
  $("#st3").classList.toggle("done", cal.color);
  $("#btnColor").classList.toggle("active", cal.picking);
  $("#btnCalSave").disabled = !(n === 4 && cal.color);
  drawMarks();
}
function drawMarks() {
  const svg = $("#marks"), img = $("#video");
  svg.toggleAttribute("hidden", !cal); // SVG elements have no .hidden property
  if (!cal) return;
  const W = img.naturalWidth || 1920, H = img.naturalHeight || 1080, R = W / 110, p = cal.pts;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  let h = "";
  if (p.length >= 2) h += `<line x1="${p[0][0]}" y1="${p[0][1]}" x2="${p[1][0]}" y2="${p[1][1]}" stroke="#fff"/>`;
  if (p.length >= 4) h += `<line x1="${p[2][0]}" y1="${p[2][1]}" x2="${p[3][0]}" y2="${p[3][1]}" stroke="#f59e0b"/>`;
  p.forEach(([x, y], i) => {
    h += `<circle cx="${x}" cy="${y}" r="${R}" class="${i < 2 ? "s" : "f"}"/>` +
      `<text x="${x + R * 1.5}" y="${y - R * 1.3}" font-size="${R * 2.1}">${PT_LABELS[i]}</text>`;
  });
  svg.innerHTML = h;
}
function exitCalib() {
  cal = null;
  $("#calib").hidden = true;
  $("#board").hidden = false;
  $("#hint").hidden = true;
  document.body.classList.remove("calibrating");
  drawMarks();
  $("#video").src = `/stream?t=${Date.now()}`;
}

$("#video").addEventListener("click", async (ev) => {
  if (!cal) return;
  const img = ev.currentTarget, r = img.getBoundingClientRect();
  const x = ((ev.clientX - r.left) * img.naturalWidth) / r.width;
  const y = ((ev.clientY - r.top) * img.naturalHeight) / r.height;
  if (cal.picking) {
    cal.picking = false;
    const j = await api("calib_sample", { x: Math.round(x), y: Math.round(y) });
    if (j.ok) {
      cal.color = true;
      $("#colorInfo").innerHTML = `Diambil <span class="swatch" style="background:rgb(${j.rgb.join(",")})"></span>`;
      await preview();
    }
  } else if (cal.pts.length < 4) {
    cal.pts.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
    if (cal.pts.length === 4) await preview();
  }
  calibUI();
});
$("#video").addEventListener("load", drawMarks);
$("#video").addEventListener("error", () => setTimeout(() => { if (!cal) $("#video").src = `/stream?t=${Date.now()}`; }, 1000));
for (const id of ["#inLanes", "#inLen", "#inBlock"]) $(id).addEventListener("change", () => cal && preview());
$("#btnRedo").onclick = async () => {
  cal.pts = [];
  cal.picking = false;
  await api("calib_draft", { pts: [] });
  await preview();
  calibUI();
};
$("#btnColor").onclick = () => { cal.picking = !cal.picking; calibUI(); };
$("#btnCalSave").onclick = async () => {
  if (!(await api("calib_draft", { pts: cal.pts, ...sizes() })).ok) return;
  if ((await api("calib_save")).ok) { exitCalib(); toast("Kalibrasi tersimpan. Siapkan pemain, lalu tekan Mulai.", "ok"); }
};
$("#btnCalCancel").onclick = async () => { await api("calib_cancel"); exitCalib(); };

// ---------- race controls ----------
$("#btnCalib").onclick = () => { ensureAudio(); calibBegin(); };
$("#btnStart").onclick = () => { ensureAudio(); api("start"); };
$("#btnEnd").onclick = () => api("end");
$("#btnReset").onclick = () => api("reset");
$("#podiumClose").onclick = () => { podiumClosedFor = S.race_id; $("#podium").hidden = true; };
document.addEventListener("pointerdown", ensureAudio);
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-review]");
  if (b && !b.disabled) openReview(b.dataset.review);
});
$("#rvPrev").onclick = () => { stopPlay(); showFrame(rv.idx - 1); };
$("#rvNext").onclick = () => { stopPlay(); showFrame(rv.idx + 1); };
$("#rvPlay").onclick = () => togglePlay();
$("#rvSlider").oninput = (e) => { stopPlay(); showFrame(+e.target.value); };
$("#rvSah").onclick = () => decide("sah");
$("#rvBatal").onclick = () => decide("batal");
$("#rvClose").onclick = () => $("#review").close();
$("#review").addEventListener("close", stopPlay);
document.addEventListener("keydown", (e) => {
  if (!rv || !$("#review").open) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") { stopPlay(); showFrame(rv.idx + (e.key === "ArrowLeft" ? -1 : 1)); }
  else if (e.key === " ") togglePlay();
  else return;
  e.preventDefault();
});

stopPlay();
poll();
requestAnimationFrame(tick);
