import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

// ---------------------------------------------------------------- constants
const CLASSES = ['N', 'S', 'V', 'F', 'Q'];
const NAMES = {
  N: 'Normal', S: 'Supraventricular ectopic', V: 'Ventricular ectopic',
  F: 'Fusion', Q: 'Unclassifiable',
};
const css = getComputedStyle(document.documentElement);
const INK = Object.fromEntries(CLASSES.map(c => [c, css.getPropertyValue(`--c-${c}`).trim()]));
const PAPER = '#FBEDE8';
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// 3D layout: one beat spans W world units; ECG paper's 0.2 s major square scales with it
const W = 10, X0 = -W / 2, AMP = 0.42, BASE = 3.2, DZ = 1.15, MAX_BEATS = 42;
const BEAT_SECONDS = 250 / 360;
const MAJOR = W / (BEAT_SECONDS / 0.2);
const R_INDEX = 45;                     // R-peak position in the 125-point beat

const $ = id => document.getElementById(id);
const fmt = new Intl.NumberFormat('en');
const pct = v => `${(v * 100).toFixed(1)}%`;

const state = {
  status: null, data: null, idx: 0, playing: false, speed: 1, acc: 0,
  selected: null, follow: true, seen: [0, 0, 0, 0, 0], agree: 0, judged: 0,
};

// ---------------------------------------------------------------- scene
const host = $('stage-canvas');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(PAPER);
scene.fog = new THREE.Fog(PAPER, 18, 50);

const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);
camera.position.set(-4.5, 7.2, 12.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.6, 1.0, -7);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 7;
controls.maxDistance = 32;
controls.maxPolarAngle = Math.PI * 0.46;
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0xf0cfc6, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.9);
sun.position.set(-7, 15, 9);
sun.target.position.set(0, 0, -10);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -18, right: 18, top: 26, bottom: -26, near: 1, far: 70 });
sun.shadow.radius = 5;
sun.shadow.bias = -0.0006;
scene.add(sun, sun.target);

function paperTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 500;
  const g = c.getContext('2d');
  g.fillStyle = PAPER; g.fillRect(0, 0, 500, 500);
  g.strokeStyle = '#F2CBC2'; g.lineWidth = 2;
  for (let i = 1; i < 5; i++) {
    g.beginPath();
    g.moveTo(i * 100, 0); g.lineTo(i * 100, 500);
    g.moveTo(0, i * 100); g.lineTo(500, i * 100);
    g.stroke();
  }
  g.strokeStyle = '#E39A90'; g.lineWidth = 6; g.strokeRect(0, 0, 500, 500);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
const PLANE = 140;
const tex = paperTexture();
tex.repeat.set(PLANE / MAJOR, PLANE / MAJOR);
const paper = new THREE.Mesh(
  new THREE.PlaneGeometry(PLANE, PLANE),
  new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }),
);
paper.rotation.x = -Math.PI / 2;
paper.position.set(X0 + PLANE / 2, 0, -PLANE / 2 + 6);   // grid lines start at the beat's left edge
paper.receiveShadow = true;
scene.add(paper);

// materials per class, plus a highlighted variant for the selected beat
const mats = Object.fromEntries(CLASSES.map(c => [c, {
  tube: new THREE.MeshStandardMaterial({ color: INK[c], roughness: 0.4, metalness: 0.05 }),
  tubeSel: new THREE.MeshStandardMaterial({ color: INK[c], roughness: 0.3, emissive: INK[c], emissiveIntensity: 0.55 }),
  curtain: new THREE.MeshBasicMaterial({ color: INK[c], transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }),
  curtainSel: new THREE.MeshBasicMaterial({ color: INK[c], transparent: true, opacity: 0.24, side: THREE.DoubleSide, depthWrite: false }),
  flag: new THREE.MeshStandardMaterial({ color: INK[c], roughness: 0.6, side: THREE.DoubleSide }),
}]));
const poleMat = new THREE.MeshStandardMaterial({ color: 0x1F2233, roughness: 0.5 });
const poleGeo = new THREE.CylinderGeometry(0.018, 0.018, 1.1, 8);
const flagGeo = new THREE.PlaneGeometry(0.5, 0.3);

const pen = new THREE.Mesh(
  new THREE.SphereGeometry(0.11, 20, 16),
  new THREE.MeshStandardMaterial({ color: 0x1F2233, roughness: 0.2, metalness: 0.3 }),
);
pen.castShadow = true;
pen.visible = false;
scene.add(pen);

const TUBE_SEGS = 248, RADIAL = 6;
const live = [];                 // beats currently in the scene

function beatPoints(vals) {
  const n = vals.length;
  return vals.map((v, i) => new THREE.Vector3(X0 + (i / (n - 1)) * W, Math.max(0.03, (v + BASE) * AMP), 0));
}

function curtainGeometry(pts) {
  const pos = new Float32Array(pts.length * 6);
  pts.forEach((p, i) => pos.set([p.x, 0.005, 0, p.x, p.y, 0], i * 6));
  const idx = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const a = i * 2;
    idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

function addBeat(i, animate = true) {
  const d = state.data;
  const c = CLASSES[d.labels[i]];
  const pts = beatPoints(d.beats[i]);
  const group = new THREE.Group();

  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), TUBE_SEGS, 0.05, RADIAL, false),
    mats[c].tube,
  );
  tube.castShadow = true;
  tube.userData.beat = i;
  const curtain = new THREE.Mesh(curtainGeometry(pts), mats[c].curtain);
  curtain.userData.beat = i;
  group.add(curtain, tube);

  const missed = d.pred && d.pred[i] !== d.labels[i];
  if (missed) {
    const peak = pts[R_INDEX];
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.set(peak.x, peak.y + 0.75, 0);
    const flag = new THREE.Mesh(flagGeo, mats[CLASSES[d.pred[i]]].flag);
    flag.position.set(peak.x + 0.26, peak.y + 1.15, 0);
    pole.castShadow = flag.castShadow = true;
    group.add(pole, flag);
  }

  for (const b of live) b.targetZ -= DZ;
  group.position.z = 0;
  scene.add(group);
  const b = { i, c, group, tube, curtain, pts, targetZ: 0, drawT: animate && !reduceMotion ? 0 : 1 };
  b.drawDur = Math.min(0.45, Math.max(0.12, (d.rr[i] || 0.8) / state.speed * 0.6));
  applyDraw(b);
  live.push(b);

  while (live.length > MAX_BEATS) {
    const old = live.shift();
    scene.remove(old.group);
    old.tube.geometry.dispose();
    old.curtain.geometry.dispose();
  }

  // tallies
  state.seen[d.labels[i]] += 1;
  if (d.pred) { state.judged += 1; if (!missed) state.agree += 1; }
  if (state.follow) select(i, false);
  renderLegend();
  renderProgress();
}

function applyDraw(b) {
  const t = b.drawT;
  b.tube.geometry.setDrawRange(0, Math.floor(t * TUBE_SEGS) * RADIAL * 6);
  b.curtain.geometry.setDrawRange(0, Math.floor(t * (b.pts.length - 1)) * 6);
  if (t < 1) {
    const p = b.pts[Math.min(b.pts.length - 1, Math.floor(t * (b.pts.length - 1)))];
    pen.position.set(p.x, p.y + 0.06, b.group.position.z);
    pen.visible = true;
  }
}

function clearScene() {
  for (const b of live) {
    scene.remove(b.group);
    b.tube.geometry.dispose();
    b.curtain.geometry.dispose();
  }
  live.length = 0;
  pen.visible = false;
}

function highlight() {
  for (const b of live) {
    const sel = b.i === state.selected;
    b.tube.material = sel ? mats[b.c].tubeSel : mats[b.c].tube;
    b.curtain.material = sel ? mats[b.c].curtainSel : mats[b.c].curtain;
  }
}

// ---------------------------------------------------------------- loop
function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w / h < 1 ? 58 : 36;   // narrow screens need a wider lens to fit a beat
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  const d = state.data;

  if (state.playing && d) {
    state.acc += dt * state.speed;
    const wait = Math.min(2, Math.max(0.25, d.rr[state.idx] || 0.8));
    if (state.acc >= wait) {
      state.acc = 0;
      addBeat(state.idx);
      state.idx += 1;
      if (state.idx >= d.labels.length) setPlaying(false);
    }
  }

  let drawing = false;
  for (const b of live) {
    b.group.position.z += (b.targetZ - b.group.position.z) * (reduceMotion ? 1 : Math.min(1, dt * 9));
    if (b.drawT < 1) {
      b.drawT = Math.min(1, b.drawT + dt / b.drawDur);
      applyDraw(b);
      drawing = true;
    }
  }
  if (!drawing) pen.visible = false;

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- picking
const ray = new THREE.Raycaster();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
    -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObjects(live.flatMap(b => [b.tube, b.curtain]))[0];
  if (hit) {
    state.follow = false;
    select(hit.object.userData.beat, true);
  }
});

// ---------------------------------------------------------------- inspector
function select(i, manual) {
  state.selected = i;
  $('follow').hidden = state.follow;
  highlight();
  const d = state.data;
  const t = CLASSES[d.labels[i]];
  $('insp-num').textContent = `${fmt.format(i + 1)} of ${fmt.format(d.labels.length)}`;
  $('insp-true').textContent = `${t}, ${NAMES[t]}`;
  $('insp-true').style.color = INK[t];
  const predEl = $('insp-pred');
  if (d.pred) {
    const p = CLASSES[d.pred[i]];
    const conf = d.proba[i][d.pred[i]];
    predEl.textContent = p === t ? `${p}, agrees (${pct(conf)})` : `${p}, disagrees (${pct(conf)})`;
    predEl.classList.toggle('miss', p !== t);
  } else {
    predEl.textContent = 'No trained model';
    predEl.classList.remove('miss');
  }
  $('insp-rr').textContent = d.rr[i] ? `${d.rr[i].toFixed(2)} s` : '–';
  drawWave(d.beats[i], INK[t]);
  renderProba(i);
  if (manual) renderProgress();
}

function drawWave(vals, color) {
  const cv = $('insp-canvas');
  const dpr = Math.min(devicePixelRatio, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext('2d');
  g.scale(dpr, dpr);
  g.fillStyle = PAPER; g.fillRect(0, 0, w, h);
  const minor = w / (BEAT_SECONDS / 0.04);
  for (let k = 0, x = 0; x <= w + 1; k++, x = k * minor) {
    g.strokeStyle = k % 5 === 0 ? '#E39A90' : '#F2CBC2';
    g.lineWidth = k % 5 === 0 ? 1.2 : 0.6;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
  for (let k = 0, y = h; y >= -1; k++, y = h - k * minor) {
    g.strokeStyle = k % 5 === 0 ? '#E39A90' : '#F2CBC2';
    g.lineWidth = k % 5 === 0 ? 1.2 : 0.6;
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const yOf = v => h - 12 - ((v - lo) / (hi - lo || 1)) * (h - 24);
  g.strokeStyle = color; g.lineWidth = 2.2; g.lineJoin = 'round';
  g.beginPath();
  vals.forEach((v, k) => {
    const x = (k / (vals.length - 1)) * w;
    k ? g.lineTo(x, yOf(v)) : g.moveTo(x, yOf(v));
  });
  g.stroke();
}

function renderProba(i) {
  const ul = $('insp-proba');
  const d = state.data;
  if (!d.proba) { ul.innerHTML = ''; return; }
  if (!ul.children.length) {
    ul.innerHTML = CLASSES.map(c => `<li><span>${c}</span><span class="track"><span class="fill" style="background:${INK[c]}"></span></span><span class="pct"></span></li>`).join('');
  }
  [...ul.children].forEach((li, k) => {
    const v = d.proba[i][k];
    li.querySelector('.fill').style.width = `${v * 100}%`;
    li.querySelector('.pct').textContent = `${Math.round(v * 100)}%`;
  });
}

$('follow').addEventListener('click', () => {
  state.follow = true;
  if (live.length) select(live[live.length - 1].i, false);
});

// ---------------------------------------------------------------- legend + progress
function renderLegend() {
  const rows = CLASSES.map((c, k) => `<li><span class="swatch" style="background:${INK[c]}"></span><span>${NAMES[c]}</span><span class="count">${fmt.format(state.seen[k])}</span></li>`);
  if (state.data?.pred && state.judged) {
    rows.push(`<li class="agree">Agrees with the cardiologist on ${pct(state.agree / state.judged)} of beats so far</li>`);
  }
  $('legend').innerHTML = rows.join('');
}

function renderProgress() {
  const d = state.data;
  if (!d) return;
  $('progress').textContent = `Beat ${fmt.format(state.idx)} of ${fmt.format(d.labels.length)}`;
}

// ---------------------------------------------------------------- playback
function setPlaying(on) {
  const d = state.data;
  if (on && d && state.idx >= d.labels.length) restart();
  state.playing = on;
  $('play').textContent = on ? 'Pause' : 'Play';
}

function restart() {
  clearScene();
  state.idx = 0; state.acc = 0;
  state.seen = [0, 0, 0, 0, 0]; state.agree = 0; state.judged = 0;
  state.follow = true;
  renderLegend(); renderProgress();
}

$('play').addEventListener('click', () => setPlaying(!state.playing));

document.querySelectorAll('.speed button').forEach(btn => btn.addEventListener('click', () => {
  state.speed = Number(btn.dataset.speed);
  document.querySelectorAll('.speed button').forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
}));

$('next-miss').addEventListener('click', () => {
  const d = state.data;
  if (!d?.pred) return;
  let j = state.idx;
  while (j < d.labels.length && d.pred[j] === d.labels[j]) j++;
  if (j >= d.labels.length) { $('progress').textContent = 'No more disagreements for this patient'; return; }
  setPlaying(false);
  // lay down the beats leading up to the miss instantly, so it arrives in context
  const from = Math.max(state.idx, j - MAX_BEATS + 1);
  for (let k = state.idx; k < from; k++) {   // count skipped beats without drawing them
    state.seen[d.labels[k]] += 1;
    state.judged += 1;
    if (d.pred[k] === d.labels[k]) state.agree += 1;
  }
  for (let k = from; k <= j; k++) addBeat(k, k === j);
  state.idx = j + 1;
  state.follow = false;
  select(j, true);
});

document.addEventListener('keydown', e => {
  if (e.target.closest('select, input')) return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!state.playing); }
  if (e.code === 'ArrowRight' && state.data && state.idx < state.data.labels.length) {
    addBeat(state.idx); state.idx += 1; renderProgress();
  }
});

// ---------------------------------------------------------------- data loading
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} returned ${r.status}`);
  return r.json();
}

async function loadPatient(id, meta) {
  setPlaying(false);
  $('patient-title').textContent = `Patient ${id}`;
  $('patient-sub').textContent = 'Loading recording…';
  state.data = await getJSON(`/api/patients/${id}`);
  $('insp-proba').innerHTML = '';
  restart();
  $('patient-sub').textContent = `${fmt.format(meta.n)} heartbeats, never seen during training.`;
  $('next-miss').disabled = !state.data.pred;
  // seed the scene so the first view is never empty
  const seed = Math.min(14, state.data.labels.length);
  for (let k = 0; k < seed; k++) addBeat(k, false);
  state.idx = seed;
  renderProgress();
  if (!reduceMotion) setPlaying(true);
}

function showNotice(status) {
  const msgs = [];
  if (status.mode === 'demo') msgs.push('Demo mode: these beats are synthetic. Run <code>python -m src.data</code> to load the MIT-BIH recordings.');
  if (!status.model) msgs.push('No trained model found, so only the cardiologist labels are shown. Run <code>python -m src.train</code>.');
  else msgs.push(`Predictions from the ${status.model}.`);
  const el = $('notice');
  el.innerHTML = msgs.join(' ');
  el.hidden = false;
}

// ---------------------------------------------------------------- results section
const MODEL_LABELS = { cnn: '1D CNN', baseline_random_forest: 'Random forest', baseline_logreg: 'Logistic regression' };

function renderResults(metrics) {
  const models = Object.keys(MODEL_LABELS).filter(k => metrics[k]);
  $('results-empty').hidden = models.length > 0 || !!metrics.leakage;
  $('results-body').hidden = models.length === 0;
  if (models.length) {
    const tabs = $('model-tabs');
    tabs.innerHTML = models.map((k, n) => `<button role="tab" type="button" data-model="${k}" aria-selected="${n === 0}">${MODEL_LABELS[k]}</button>`).join('');
    tabs.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
      tabs.querySelectorAll('button').forEach(o => o.setAttribute('aria-selected', String(o === b)));
      renderModel(metrics[b.dataset.model]);
    }));
    renderModel(metrics[models[0]]);
  }
  if (metrics.leakage) renderLeakage(metrics.leakage);
}

function renderModel(m) {
  $('metrics-body').innerHTML = CLASSES.map(c => {
    const v = m.per_class[c];
    const dash = v.support === 0;
    return `<tr><td><span class="type"><i style="background:${INK[c]}"></i>${c}, ${NAMES[c]}</span></td>
      <td>${dash ? '–' : pct(v.sensitivity)}</td><td>${dash ? '–' : pct(v.ppv)}</td>
      <td>${dash ? '–' : v.f1.toFixed(2)}</td><td>${fmt.format(v.support)}</td></tr>`;
  }).join('');
  $('headline-metric').textContent = `Macro-F1 ${m.macro_f1.toFixed(3)}, balanced accuracy ${pct(m.balanced_accuracy)}`;

  const cm = m.confusion_matrix;
  const grid = $('confusion');
  let html = '<span></span>' + CLASSES.map(c => `<span class="lab" style="color:${INK[c]}">${c}</span>`).join('');
  cm.forEach((row, r) => {
    const total = row.reduce((a, b) => a + b, 0);
    html += `<span class="lab" style="color:${INK[CLASSES[r]]}">${CLASSES[r]}</span>`;
    row.forEach(v => {
      const share = total ? v / total : 0;
      const bg = `color-mix(in srgb, var(--ink) ${Math.round(share * 92)}%, var(--paper))`;
      const fg = share > 0.45 ? 'var(--paper)' : 'var(--ink)';
      html += `<span class="cell" style="background:${bg};color:${fg}"><b>${total ? Math.round(share * 100) : '–'}${total ? '%' : ''}</b>${fmt.format(v)}</span>`;
    });
  });
  grid.innerHTML = html;
  grid.setAttribute('aria-label', 'Confusion matrix: ' + cm.map((row, r) => `${CLASSES[r]} row ${row.join(', ')}`).join('; '));
}

function renderLeakage(L) {
  $('leakage').hidden = false;
  const rows = [
    ['Accuracy', m => m.accuracy],
    ['Macro-F1', m => m.macro_f1],
    ['Supraventricular beats caught', m => m.per_class.S.sensitivity],
    ['Ventricular beats caught', m => m.per_class.V.sensitivity],
  ];
  $('leakage-bars').innerHTML = rows.map(([label, get]) => {
    const a = get(L.random_beat_split), b = get(L.patient_split);
    return `<div class="leak-row"><h3>${label}</h3>
      <div class="leak-bar"><span>Random beat split</span><span class="track"><span class="fill" style="width:${a * 100}%;background:var(--c-V)"></span></span><span class="val">${pct(a)}</span></div>
      <div class="leak-bar"><span>Patient-level split</span><span class="track"><span class="fill" style="width:${b * 100}%;background:var(--c-S)"></span></span><span class="val">${pct(b)}</span></div></div>`;
  }).join('');
}

// ---------------------------------------------------------------- boot
async function boot() {
  resize();
  requestAnimationFrame(frame);
  try {
    const [status, patients, metrics] = await Promise.all([
      getJSON('/api/status'), getJSON('/api/patients'), getJSON('/api/metrics'),
    ]);
    state.status = status;
    showNotice(status);
    renderResults(metrics);
    const sel = $('patient-select');
    sel.innerHTML = patients.map(p => `<option value="${p.id}">Patient ${p.id} (${fmt.format(p.n)} beats)</option>`).join('');
    const byId = Object.fromEntries(patients.map(p => [p.id, p]));
    sel.addEventListener('change', () => loadPatient(sel.value, byId[sel.value]));
    // open on a patient with plenty of ectopic beats, so the first view shows variety
    const start = patients.slice().sort((a, b) => (b.counts[1] + b.counts[2]) - (a.counts[1] + a.counts[2]))[0];
    sel.value = start.id;
    await loadPatient(start.id, start);
  } catch (err) {
    $('patient-title').textContent = 'Server not reachable';
    $('patient-sub').textContent = 'Start it with python -m app.server, then reload this page.';
    console.error(err);
  }
}
boot();
