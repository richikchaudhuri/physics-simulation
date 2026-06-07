import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { initWasm, World, makePositions, makeExtra } from './wasm.js';

await initWasm();

// --- sim registry ---------------------------------------------------------
// Each entry describes how a Rust sim (by `kind`) is presented: body-count
// range, environment (box/grid), camera framing, and rendering (per-instance
// scale, speed coloring). The SceneManager (loadSim/buildWorld below) reads
// these to (re)configure the scene without per-sim branching elsewhere.
const SIMS = {
  collisions: {
    kind: 0,
    accent: '#e0a155', // warm amber — energetic bounces
    defaultCount: 200,
    minCount: 2,
    maxCount: 2000, // uniform-grid broad-phase keeps high counts smooth
    boxed: true, // walls + floor box, camera framed on the box
    showGrid: true,
    colorBySpeed: true, // tint balls by speed (hot = fast)
    speedScale: 12,
    centralScale: 1,
    hasGravityParam: true,
    gravLabel: 'Gravity',
    gravMin: 0,
    gravMax: 2, // multiplier on Earth gravity; 0 = weightless gas
    gravStep: 0.1,
    gravDefault: 1,
    grabbable: true,
    rods: false,
    camPos: new THREE.Vector3(11, 9, 15),
    hint: 'Drag a ball to grab & throw · left-drag to orbit · scroll to zoom',
  },
  gravity: {
    kind: 1,
    accent: '#9d8cff', // cosmic violet
    defaultCount: 500,
    minCount: 50,
    maxCount: 1500,
    boxed: false, // open space centered on the origin
    showGrid: false,
    colorBySpeed: true,
    speedScale: 16,
    centralScale: 6, // draw the heavy central body (index 0) larger
    hasGravityParam: true,
    gravLabel: 'Gravity G',
    gravMin: 0,
    gravMax: 3,
    gravStep: 0.1,
    gravDefault: 1,
    grabbable: true,
    rods: false,
    camPos: new THREE.Vector3(0, 15, 22),
    hint: 'Drag a star to grab & throw · left-drag to orbit · scroll to zoom',
  },
  pendulum: {
    kind: 2,
    accent: '#57c8c2', // teal
    defaultCount: 24, // number of pendulums (bobs = 2x this)
    minCount: 1,
    maxCount: 200,
    boxed: false,
    showGrid: false,
    colorBySpeed: false,
    colorByIndex: true, // rainbow by pendulum, so chaotic divergence is visible
    speedScale: 1,
    centralScale: 1,
    rods: true, // draw the two arms of each pendulum
    grabbable: false, // bobs are angle-constrained; left-drag always orbits
    hasGravityParam: true,
    gravLabel: 'Gravity g',
    gravMin: 0,
    gravMax: 30,
    gravStep: 0.5,
    gravDefault: 9.81,
    camPos: new THREE.Vector3(0, -0.5, 8),
    camTarget: new THREE.Vector3(0, -1, 0),
    hint: 'Identical pendulums, tiny angle offsets — watch chaos diverge · left-drag to orbit',
  },
  cloth: {
    kind: 3,
    accent: '#6ca0ff', // blue — matches the sheet
    // Count here is the grid resolution R (Rust builds an R×R sheet); the slider
    // shows it as "Resolution" and the renderer derives R = sqrt(count).
    defaultCount: 30,
    minCount: 8,
    maxCount: 60,
    countName: 'Resolution',
    boxed: false,
    showGrid: false,
    bobs: false, // the sheet is a deformable mesh, not instanced spheres
    rods: false,
    cloth: true,
    grabbable: false,
    colorBySpeed: false,
    centralScale: 1,
    speedScale: 1,
    hasGravityParam: true,
    gravLabel: 'Gravity g',
    gravMin: 0,
    gravMax: 20,
    gravStep: 0.5,
    gravDefault: 9.81,
    hasWind: true,
    windLabel: 'Wind',
    windMin: 0,
    windMax: 30,
    windStep: 0.5,
    windDefault: 6,
    hasPin: true,
    camPos: new THREE.Vector3(5, 1, 11),
    camTarget: new THREE.Vector3(0, -0.3, 0),
    hint: 'A pinned mass-spring cloth billowing in the wind · toggle Pin · left-drag to orbit',
  },
};

let simKey = 'collisions';
let cfg = SIMS[simKey];

// --- simulation state (rebuilt on sim / count change) ---------------------
let world = null;
let positions = null;
let extra = null;
let simRadius = 0.4;
let simBounds = 4;
let count = 0;
let bodyCount = cfg.defaultCount;

// --- renderer / scene -----------------------------------------------------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  1000,
);
camera.position.copy(cfg.camPos);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// --- lights ---------------------------------------------------------------
scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.3);
key.position.set(8, 16, 6);
scene.add(key);

// --- environment: ground grid + box (sized to the boxed sim) --------------
const ENV_BOUNDS = 4; // collisions bounds; box/grid only show for that sim
const grid = new THREE.GridHelper(ENV_BOUNDS * 2, 20, 0x2a2a30, 0x161619);
scene.add(grid);

const boxGeo = new THREE.BoxGeometry(ENV_BOUNDS * 2, ENV_BOUNDS * 2, ENV_BOUNDS * 2);
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(boxGeo),
  new THREE.LineBasicMaterial({ color: 0x2a2a30 }),
);
box.position.y = ENV_BOUNDS; // rests on the grid (floor at y = 0)
scene.add(box);

// --- instanced spheres (unit sphere, scaled per instance) -----------------
const sphereGeo = new THREE.SphereGeometry(1, 16, 12);
const sphereMat = new THREE.MeshStandardMaterial({
  color: 0xffffff, // actual tint comes from per-instance colors
  roughness: 0.4,
  metalness: 0.1,
});
const BASE_COLOR = new THREE.Color(0xe8e8e8);
const HELD_COLOR = new THREE.Color(0x6ca0ff);
const tmpColor = new THREE.Color();
const dummy = new THREE.Object3D();
let mesh = null;

// Rods for the pendulum sim: one THREE.LineSegments holding both arms of every
// pendulum (pivot->bob1, bob1->bob2). Per-vertex colored to match its bobs;
// positions are rewritten each frame from the WASM bob coords.
const rodMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
let rodGeo = null;
let rodPos = null; // Float32Array backing rodGeo's position attribute
let rodLines = null;

// Cloth (sim #3): a single deformable triangle mesh whose vertices ARE the WASM
// particle positions. We copy the position buffer in each frame and recompute
// normals so lighting follows the billowing surface. DoubleSide so the back of
// the sheet is lit too when the wind flips it toward the camera.
const clothMat = new THREE.MeshStandardMaterial({
  color: 0x6ca0ff,
  roughness: 0.65,
  metalness: 0.0,
  side: THREE.DoubleSide,
});
let clothGeo = null;
let clothPos = null; // Float32Array backing clothGeo's position attribute
let clothMesh = null;

// Hue ramp by pendulum index (0..n) -> a spread-out rainbow.
function indexToColor(p, total, out) {
  return out.setHSL((p / Math.max(total, 1)) * 0.92, 0.7, 0.56);
}

// (Re)build the pendulum rod geometry for the current pendulum count. Colors are
// fixed here (per pendulum); only positions change per frame.
function buildRods() {
  if (rodLines) {
    scene.remove(rodLines);
    rodGeo.dispose();
    rodLines = null;
  }
  const pend = count / 2; // two bobs per pendulum
  const nVerts = pend * 4; // 2 segments * 2 endpoints
  rodPos = new Float32Array(nVerts * 3);
  const colArr = new Float32Array(nVerts * 3);
  for (let pi = 0; pi < pend; pi++) {
    indexToColor(pi, pend, tmpColor);
    for (let v = 0; v < 4; v++) {
      const c = (pi * 4 + v) * 3;
      colArr[c] = tmpColor.r;
      colArr[c + 1] = tmpColor.g;
      colArr[c + 2] = tmpColor.b;
    }
  }
  rodGeo = new THREE.BufferGeometry();
  rodGeo.setAttribute('position', new THREE.BufferAttribute(rodPos, 3));
  rodGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  rodLines = new THREE.LineSegments(rodGeo, rodMat);
  rodLines.frustumCulled = false; // bobs can swing past the initial bounds
  scene.add(rodLines);
}

// Rewrite rod endpoints from the bob positions `p` (shared pivot at the origin).
function updateRods(p) {
  const pend = count / 2;
  for (let pi = 0; pi < pend; pi++) {
    const b1 = 2 * pi * 3; // bob1
    const b2 = (2 * pi + 1) * 3; // bob2
    const o = pi * 12; // 4 verts * 3
    // segment: pivot -> bob1
    rodPos[o] = 0;
    rodPos[o + 1] = 0;
    rodPos[o + 2] = 0;
    rodPos[o + 3] = p[b1];
    rodPos[o + 4] = p[b1 + 1];
    rodPos[o + 5] = p[b1 + 2];
    // segment: bob1 -> bob2
    rodPos[o + 6] = p[b1];
    rodPos[o + 7] = p[b1 + 1];
    rodPos[o + 8] = p[b1 + 2];
    rodPos[o + 9] = p[b2];
    rodPos[o + 10] = p[b2 + 1];
    rodPos[o + 11] = p[b2 + 2];
  }
  rodGeo.attributes.position.needsUpdate = true;
}

// (Re)build the cloth mesh for an R×R grid: a flat position buffer (filled each
// frame from WASM) plus a static triangle index (two tris per quad).
function buildCloth(R) {
  if (clothMesh) {
    scene.remove(clothMesh);
    clothGeo.dispose();
    clothMesh = null;
  }
  clothPos = new Float32Array(R * R * 3);
  const index = [];
  for (let row = 0; row < R - 1; row++) {
    for (let col = 0; col < R - 1; col++) {
      const a = row * R + col;
      const b = row * R + col + 1;
      const c = (row + 1) * R + col;
      const d = (row + 1) * R + col + 1;
      index.push(a, c, b, b, c, d); // two triangles per cell
    }
  }
  clothGeo = new THREE.BufferGeometry();
  clothGeo.setAttribute('position', new THREE.BufferAttribute(clothPos, 3));
  clothGeo.setIndex(index);
  clothMesh = new THREE.Mesh(clothGeo, clothMat);
  clothMesh.frustumCulled = false; // the sheet swings outside its initial box
  scene.add(clothMesh);
}

// Copy the WASM particle positions into the cloth vertices and recompute normals.
function updateCloth(p) {
  clothPos.set(p);
  clothGeo.attributes.position.needsUpdate = true;
  clothGeo.computeVertexNormals();
}

// (Re)create the InstancedMesh for the current body count. instanceColor must
// be fully initialized or untouched instances render black. For sims that render
// their own geometry (cloth) the instanced spheres are built but hidden.
function buildMesh() {
  if (mesh) {
    scene.remove(mesh);
    mesh.dispose();
  }
  mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < count; i++) mesh.setColorAt(i, BASE_COLOR);
  mesh.instanceColor.needsUpdate = true;
  mesh.visible = cfg.bobs !== false; // cloth hides the spheres
  scene.add(mesh);

  if (cfg.rods) buildRods();
  else if (rodLines) rodLines.visible = false;

  if (cfg.cloth) buildCloth(Math.round(Math.sqrt(count)));
  else if (clothMesh) clothMesh.visible = false;
}

// Rebuild the active sim with a new body count (slider / sim switch).
function buildWorld(n) {
  releaseGrab(); // indices change; drop anything held
  if (world && typeof world.free === 'function') world.free();
  bodyCount = n;
  world = new World(cfg.kind, n);
  positions = makePositions(world);
  extra = makeExtra(world);
  simRadius = world.radius();
  simBounds = world.bounds();
  count = world.count();
  buildMesh();
  statCount.textContent = count;
}

// Frame the camera on the active sim (used on sim load and by Reset View).
function frameCamera() {
  camera.position.copy(cfg.camPos);
  if (cfg.camTarget) controls.target.copy(cfg.camTarget);
  else controls.target.set(0, cfg.boxed ? simBounds * 0.5 : 0, 0);
  controls.update();
}

// Switch to a different sim: reconfigure environment, controls, UI, then build.
function loadSim(nextKey) {
  if (!SIMS[nextKey]) return;
  simKey = nextKey;
  cfg = SIMS[nextKey];

  // Restrained per-sim signature accent: drives slider fill, active tab, focus.
  document.documentElement.style.setProperty('--accent', cfg.accent || '#6ca0ff');

  grid.visible = cfg.showGrid;
  box.visible = cfg.boxed;

  countName.textContent = cfg.countName || 'Count';
  countSlider.min = String(cfg.minCount);
  countSlider.max = String(cfg.maxCount);
  countSlider.value = String(cfg.defaultCount);
  countLabel.textContent = String(cfg.defaultCount);

  // The gravity control is shared; each sim relabels/rescales it (and a fresh
  // World starts at the matching default, so no set_param push is needed here).
  gravRow.classList.toggle('is-collapsed', !cfg.hasGravityParam);
  if (cfg.hasGravityParam) {
    gravName.textContent = cfg.gravLabel;
    gravSlider.min = String(cfg.gravMin);
    gravSlider.max = String(cfg.gravMax);
    gravSlider.step = String(cfg.gravStep);
    gravSlider.value = String(cfg.gravDefault);
    gravLabel.textContent = cfg.gravDefault.toFixed(1);
  }

  // Wind control (cloth only): same shared-default trick as gravity.
  windRow.classList.toggle('is-collapsed', !cfg.hasWind);
  if (cfg.hasWind) {
    windName.textContent = cfg.windLabel;
    windSlider.min = String(cfg.windMin);
    windSlider.max = String(cfg.windMax);
    windSlider.step = String(cfg.windStep);
    windSlider.value = String(cfg.windDefault);
    windLabel.textContent = cfg.windDefault.toFixed(1);
  }

  // Pin toggle (cloth only): always starts at corners-pinned.
  pinMode = 0;
  pinWrap.classList.toggle('is-collapsed', !cfg.hasPin);
  btnPin.classList.remove('is-on');
  if (cfg.hasPin) btnPin.querySelector('.lbl').textContent = 'Pin: Corners';

  buildWorld(cfg.defaultCount);

  frameCamera();

  refreshFills(); // sliders were just re-min/maxed; repaint their accent fills

  if (cfg.hint) hintEl.textContent = cfg.hint;
  for (const b of segBtns) b.classList.toggle('is-active', b.dataset.sim === nextKey);
  positionSeg();
}

// --- ui -------------------------------------------------------------------
const statCount = document.getElementById('stat-count');
const statFps = document.getElementById('stat-fps');
const countSlider = document.getElementById('count-slider');
const countLabel = document.getElementById('count-label');
const countName = document.getElementById('count-name');
const speedSlider = document.getElementById('speed-slider');
const speedLabel = document.getElementById('speed-label');
const gravSlider = document.getElementById('grav-slider');
const gravLabel = document.getElementById('grav-label');
const gravName = document.getElementById('grav-name');
const gravRow = document.getElementById('grav-row');
const windSlider = document.getElementById('wind-slider');
const windLabel = document.getElementById('wind-label');
const windName = document.getElementById('wind-name');
const windRow = document.getElementById('wind-row');
const btnPin = document.getElementById('btn-pin');
const btnPause = document.getElementById('btn-pause');
const btnStep = document.getElementById('btn-step');
const btnView = document.getElementById('btn-view');
const btnFull = document.getElementById('btn-fullscreen');
const segBtns = Array.from(document.querySelectorAll('.seg__btn'));
const segNav = document.getElementById('sim-switch');
const segHL = document.querySelector('.seg__hl');
const pinWrap = document.querySelector('.pin-wrap');
const hintEl = document.querySelector('.hint');

// Position a sliding/resizing highlighter to wrap the active button in a pill nav.
// Geometry is computed deterministically (collapsed icon width + the label's
// natural scrollWidth) so it's correct immediately, even while the label is still
// animating open — no measuring mid-transition. Pass activeIdx = -1 to hide it.
function positionHL(nav, btns, hl, activeIdx) {
  if (!nav || !hl || !btns.length) return;
  if (activeIdx < 0) {
    hl.style.width = '0px';
    nav.classList.remove('has-open');
    return;
  }
  nav.classList.add('has-open');
  const ncs = getComputedStyle(nav);
  const padL = parseFloat(ncs.paddingLeft) || 0;
  const padT = parseFloat(ncs.paddingTop) || 0;
  const gap = parseFloat(ncs.columnGap || ncs.gap) || 0;
  const active = btns[activeIdx];
  const bcs = getComputedStyle(active);
  const ico = active.querySelector('.ico');
  // Collapsed (icon-only) width is identical for every button and independent of
  // any in-flight label transition, so derive it from the always-visible icon +
  // the button's horizontal padding/border (NOT a sibling's offsetWidth, which is
  // still mid-collapse right after a switch).
  const iconW =
    (ico ? ico.offsetWidth : 0) +
    (parseFloat(bcs.paddingLeft) || 0) +
    (parseFloat(bcs.paddingRight) || 0) +
    (parseFloat(bcs.borderLeftWidth) || 0) +
    (parseFloat(bcs.borderRightWidth) || 0);
  const lbl = active.querySelector('.lbl');
  const labelW = lbl ? lbl.scrollWidth : 0; // natural width regardless of clip
  const left = padL + activeIdx * (iconW + gap);
  hl.style.transform = `translate(${left}px, ${padT}px)`;
  hl.style.width = `${iconW + labelW}px`;
  hl.style.height = `${active.offsetHeight}px`;
}

// Sim-mode highlighter: always tracks the currently-active mode.
function positionSeg() {
  const i = segBtns.findIndex((b) => b.classList.contains('is-active'));
  positionHL(segNav, segBtns, segHL, i);
}

let speed = 1; // sim-time multiplier
let paused = false;
let pinMode = 0; // cloth pin mode: 0 = corners, 1 = top edge

// Paint a slider's accent "fill" up to its current value (CSS reads --fill for the
// WebKit track gradient; Firefox uses native ::-moz-range-progress and ignores it).
function setFill(el) {
  const min = parseFloat(el.min);
  const max = parseFloat(el.max);
  const v = parseFloat(el.value);
  const pct = max > min ? ((v - min) / (max - min)) * 100 : 0;
  el.style.setProperty('--fill', `${pct}%`);
}
function refreshFills() {
  for (const el of document.querySelectorAll('input[type="range"]')) setFill(el);
}
// One delegated listener keeps every slider's fill in sync as it's dragged.
document.querySelector('.panel').addEventListener('input', (e) => {
  if (e.target.matches('input[type="range"]')) setFill(e.target);
});

for (const b of segBtns) {
  b.addEventListener('click', () => loadSim(b.dataset.sim));
}

// Live label while dragging; rebuild only on release so we don't thrash.
countSlider.addEventListener('input', () => {
  countLabel.textContent = countSlider.value;
});
countSlider.addEventListener('change', () => {
  buildWorld(parseInt(countSlider.value, 10));
});

speedSlider.addEventListener('input', () => {
  speed = parseFloat(speedSlider.value);
  speedLabel.textContent = `${speed.toFixed(1)}×`;
});

gravSlider.addEventListener('input', () => {
  const g = parseFloat(gravSlider.value);
  gravLabel.textContent = g.toFixed(1);
  // id 0 is each sim's primary tunable: gravity multiplier (collisions),
  // gravitational constant G (n-body), or gravity g (pendulum / cloth).
  world.set_param(0, g);
});

// Wind strength (cloth): id 1.
windSlider.addEventListener('input', () => {
  const w = parseFloat(windSlider.value);
  windLabel.textContent = w.toFixed(1);
  world.set_param(1, w);
});

// Pin mode toggle (cloth): id 2, 0 = corners, 1 = whole top edge.
btnPin.addEventListener('click', () => {
  pinMode = pinMode === 0 ? 1 : 0;
  world.set_param(2, pinMode);
  btnPin.querySelector('.lbl').textContent = pinMode === 1 ? 'Pin: Top Edge' : 'Pin: Corners';
  btnPin.classList.toggle('is-on', pinMode === 1);
});

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.querySelector('.lbl').textContent = paused ? 'Play' : 'Pause';
  btnPause.classList.toggle('is-on', paused);
});

// Step: advance exactly one fixed tick and repaint (handy while paused to scrub
// chaos/cloth frame by frame).
btnStep.addEventListener('click', () => {
  world.step(FIXED);
  render();
});

// Reset View: re-frame the camera on the current sim without rebuilding it.
btnView.addEventListener('click', frameCamera);

// Fullscreen toggle (the resize listener handles the canvas/aspect update).
// requestFullscreen rejects in sandboxed iframes; swallow that so it's a no-op.
btnFull.addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
});
document.addEventListener('fullscreenchange', () => {
  // The button is icon-only now, so reflect state via a class + label rather than text.
  const fs = !!document.fullscreenElement;
  btnFull.classList.toggle('is-on', fs);
  const fsLabel = fs ? 'Exit fullscreen' : 'Fullscreen';
  btnFull.title = fsLabel;
  btnFull.setAttribute('aria-label', fsLabel);
});

// --- About / Settings disclosure (expanding pill + flow-down sheet) -------
const disclosure = document.getElementById('disclosure');
const navPill = disclosure.querySelector('.navpill');
const navHL = disclosure.querySelector('.navpill__hl');
const btnAbout = document.getElementById('btn-about');
const btnSettings = document.getElementById('btn-settings');
const navBtns = [btnAbout, btnSettings];
const PANELS = {
  about: { btn: btnAbout, sheet: document.getElementById('sheet-about') },
  settings: { btn: btnSettings, sheet: document.getElementById('sheet-settings') },
};
let openPanel = null;
const sheetTimers = new WeakMap();

// Highlighter that slides between About and Settings (hidden when both closed).
function positionNav() {
  const i = navBtns.findIndex((b) => b.classList.contains('is-open'));
  positionHL(navPill, navBtns, navHL, i);
}

// Expand/collapse one panel: the pill label opens and the sheet flows down. The
// sheet is unhidden, reflowed, then transitioned (rAF stalls in headless tabs).
function setPanel(key, open) {
  const p = PANELS[key];
  if (!p) return;
  p.btn.classList.toggle('is-open', open);
  p.btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) {
    clearTimeout(sheetTimers.get(p.sheet));
    p.sheet.hidden = false;
    void p.sheet.offsetWidth;
    p.sheet.classList.add('is-open');
  } else {
    p.sheet.classList.remove('is-open');
    sheetTimers.set(
      p.sheet,
      setTimeout(() => {
        p.sheet.hidden = true;
      }, 440),
    );
  }
}

function openDisclosure(key) {
  if (openPanel && openPanel !== key) setPanel(openPanel, false);
  openPanel = key;
  setPanel(key, true);
  positionNav();
}

function closeDisclosure() {
  if (!openPanel) return;
  setPanel(openPanel, false);
  openPanel = null;
  positionNav();
}

btnAbout.addEventListener('click', () =>
  openPanel === 'about' ? closeDisclosure() : openDisclosure('about'),
);
btnSettings.addEventListener('click', () =>
  openPanel === 'settings' ? closeDisclosure() : openDisclosure('settings'),
);

// A pointer-down anywhere outside the disclosure dismisses the open sheet.
document.addEventListener('pointerdown', (e) => {
  if (openPanel && !e.target.closest('#disclosure')) closeDisclosure();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeDisclosure();
});

// --- settings toggles -----------------------------------------------------
// High quality: full device pixel ratio (capped at 2) vs 1 for performance.
const setQuality = document.getElementById('set-quality');
setQuality.addEventListener('change', () => {
  renderer.setPixelRatio(setQuality.checked ? Math.min(window.devicePixelRatio, 2) : 1);
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Vignette: toggle the decorative edge darkening.
const vignetteEl = document.querySelector('.vignette');
const setVignette = document.getElementById('set-vignette');
setVignette.addEventListener('change', () => {
  vignetteEl.style.display = setVignette.checked ? '' : 'none';
});

// Reduce motion: kill UI transitions/animations regardless of the OS preference.
const setMotion = document.getElementById('set-motion');
setMotion.addEventListener('change', () => {
  document.body.classList.toggle('reduce-motion', setMotion.checked);
});

// Re-push the current slider/toggle state onto a freshly-built World (reset spins
// up a new sim at its built-in defaults, which can drift from the live controls).
function reapplyParams() {
  if (cfg.hasGravityParam) world.set_param(0, parseFloat(gravSlider.value));
  if (cfg.hasWind) world.set_param(1, parseFloat(windSlider.value));
  if (cfg.hasPin) world.set_param(2, pinMode);
}

document.getElementById('btn-reset').addEventListener('click', () => {
  world.reset();
  reapplyParams();
});

// --- grab & throw ---------------------------------------------------------
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const hitPoint = new THREE.Vector3();
const camDir = new THREE.Vector3();
const instVel = new THREE.Vector3();
const lastDragPos = new THREE.Vector3();
const throwVel = new THREE.Vector3();
let grabbed = -1; // instance id being dragged, -1 = none
let lastDragTime = 0;
const MAX_THROW = 40;

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

function pointerNDC(e) {
  const rect = canvas.getBoundingClientRect();
  ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

function clampToBox(v) {
  if (cfg.boxed) {
    const lim = simBounds - simRadius;
    v.x = clamp(v.x, -lim, lim);
    v.z = clamp(v.z, -lim, lim);
    v.y = clamp(v.y, simRadius, 2 * simBounds - simRadius);
  } else {
    const lim = simBounds;
    v.x = clamp(v.x, -lim, lim);
    v.y = clamp(v.y, -lim, lim);
    v.z = clamp(v.z, -lim, lim);
  }
}

// Nearest body to the click ray within a generous radius. More forgiving than
// exact mesh raycasting and works for the tiny N-body stars too.
function pickBody() {
  raycaster.setFromCamera(ndc, camera);
  const o = raycaster.ray.origin;
  const d = raycaster.ray.direction; // normalized
  const p = positions();
  const pickR = Math.max(simRadius * 1.6, 0.35);
  const pickR2 = pickR * pickR;
  let best = -1;
  let bestT = Infinity;
  for (let i = 0; i < count; i++) {
    const k = i * 3;
    const wx = p[k] - o.x;
    const wy = p[k + 1] - o.y;
    const wz = p[k + 2] - o.z;
    const t = wx * d.x + wy * d.y + wz * d.z; // distance along ray to closest point
    if (t < 0 || t >= bestT) continue; // behind camera, or farther than current best
    const perp2 = wx * wx + wy * wy + wz * wz - t * t;
    if (perp2 <= pickR2) {
      bestT = t;
      best = i;
    }
  }
  return best;
}

function releaseGrab() {
  if (grabbed < 0) return;
  mesh.setColorAt(grabbed, BASE_COLOR);
  mesh.instanceColor.needsUpdate = true;
  world.set_held(-1);
  grabbed = -1;
  controls.enabled = true;
}

// Capture phase on window so this runs BEFORE OrbitControls' canvas listener;
// when we grab a body we stopPropagation so the orbit gesture never starts.
function onPointerDown(e) {
  if (e.button !== 0 || e.target !== canvas) return;
  if (!cfg.grabbable) return; // e.g. pendulum: let OrbitControls orbit instead
  pointerNDC(e);
  const id = pickBody();
  if (id < 0) return; // missed -> let OrbitControls orbit

  grabbed = id;
  controls.enabled = false;
  e.stopPropagation();

  world.set_held(grabbed);
  mesh.setColorAt(grabbed, HELD_COLOR);
  mesh.instanceColor.needsUpdate = true;

  const p = positions();
  const k = grabbed * 3;
  lastDragPos.set(p[k], p[k + 1], p[k + 2]);
  camera.getWorldDirection(camDir);
  dragPlane.setFromNormalAndCoplanarPoint(camDir, lastDragPos);
  throwVel.set(0, 0, 0);
  lastDragTime = performance.now();
}

function onPointerMove(e) {
  if (grabbed < 0) return;
  pointerNDC(e);
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(dragPlane, hitPoint)) return;
  clampToBox(hitPoint);

  const now = performance.now();
  const dt = Math.max((now - lastDragTime) / 1000, 1e-3);
  instVel.copy(hitPoint).sub(lastDragPos).divideScalar(dt);
  throwVel.lerp(instVel, 0.6); // smooth cursor jitter
  lastDragTime = now;
  lastDragPos.copy(hitPoint);

  world.set_pos(grabbed, hitPoint.x, hitPoint.y, hitPoint.z);
}

function onPointerUp() {
  if (grabbed < 0) return;
  if (performance.now() - lastDragTime > 120) throwVel.set(0, 0, 0); // paused -> drop
  if (throwVel.length() > MAX_THROW) throwVel.setLength(MAX_THROW);
  world.set_vel(grabbed, throwVel.x, throwVel.y, throwVel.z);
  releaseGrab();
}

window.addEventListener('pointerdown', onPointerDown, true);
window.addEventListener('pointermove', onPointerMove);
window.addEventListener('pointerup', onPointerUp);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  positionSeg(); // pill geometry is layout-dependent
  positionNav();
});

// Map a normalized speed [0,1] to a blue(slow) -> orange(fast) color.
function speedToColor(t, out) {
  t = clamp(t, 0, 1);
  return out.setHSL(0.62 - 0.54 * t, 0.72, 0.38 + 0.34 * t);
}

// --- render + fixed-timestep loop -----------------------------------------
const FIXED = 1 / 120;
let last = performance.now();
let acc = 0;
let fpsTime = 0;
let fpsFrames = 0;

function render() {
  const p = positions(); // re-fetched AFTER stepping (step may reallocate)
  const ex = cfg.colorBySpeed ? extra() : null;
  const byIndex = cfg.colorByIndex;
  const recolor = ex || byIndex;
  const cs = cfg.centralScale;
  const pend = count / 2; // only used for pendulum index coloring
  // Cloth renders its own mesh; the instanced spheres stay hidden, so skip them.
  if (cfg.bobs !== false) {
    for (let i = 0; i < count; i++) {
      const k = i * 3;
      dummy.position.set(p[k], p[k + 1], p[k + 2]);
      dummy.scale.setScalar(simRadius * (cs > 1 && i === 0 ? cs : 1));
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      if (recolor) {
        if (i === grabbed) tmpColor.copy(HELD_COLOR);
        else if (ex) speedToColor(ex[i] / cfg.speedScale, tmpColor);
        else indexToColor((i / 2) | 0, pend, tmpColor); // two bobs share a pendulum hue
        mesh.setColorAt(i, tmpColor);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (recolor) mesh.instanceColor.needsUpdate = true;
  }
  if (cfg.rods && rodLines) updateRods(p);
  if (cfg.cloth && clothMesh) updateCloth(p);
  controls.update();
  renderer.render(scene, camera);
}

function frame(now) {
  let dt = (now - last) / 1000;
  last = now;
  if (dt > 0.1) dt = 0.1; // clamp after tab-switch / stalls
  if (paused) {
    acc = 0;
  } else {
    acc += dt * speed;
    while (acc >= FIXED) {
      world.step(FIXED);
      acc -= FIXED;
    }
  }
  render();

  fpsFrames++;
  fpsTime += dt;
  if (fpsTime >= 0.5) {
    statFps.textContent = Math.round(fpsFrames / fpsTime);
    fpsFrames = 0;
    fpsTime = 0;
  }

  schedule();
}

// Prefer requestAnimationFrame (vsync-smooth) when the tab is visible. Some
// embedded/preview webviews report the tab as hidden, which pauses rAF entirely;
// fall back to setTimeout so the scene still animates there.
function schedule() {
  if (document.hidden) {
    setTimeout(() => frame(performance.now()), 1000 / 60);
  } else {
    requestAnimationFrame(frame);
  }
}

loadSim('collisions'); // build the initial sim + scene
render(); // paint the first frame before the loop ticks
schedule();

// Label widths shift when the brand/UI webfonts finish loading — re-measure the
// sim-mode highlighter so it keeps hugging the active mode after the font swap.
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(positionSeg);
}

// Engine is up and the first frame is on screen — fade out the boot loader.
const loaderEl = document.getElementById('loader');
if (loaderEl) {
  loaderEl.classList.add('is-hidden');
  setTimeout(() => loaderEl.remove(), 700);
}

document.addEventListener('visibilitychange', () => {
  last = performance.now(); // avoid a huge dt spike when visibility flips
});

// Dev-only: headless previews keep the tab hidden, which pauses rAF. This hook
// lets the harness manually advance the sim and inspect state for verification.
if (import.meta.env.DEV) {
  window.__sandbox = {
    tick(steps = 60) {
      for (let i = 0; i < steps; i++) world.step(FIXED);
      render();
    },
    render,
    loadSim,
    buildWorld,
    setSpeed(v) {
      speed = v;
    },
    setG(v) {
      world.set_param(0, v);
    },
    setPaused(v) {
      paused = v;
    },
    rods() {
      return rodLines
        ? { visible: rodLines.visible, verts: rodGeo.attributes.position.count }
        : null;
    },
    cam() {
      return {
        pos: [camera.position.x, camera.position.y, camera.position.z],
        target: [controls.target.x, controls.target.y, controls.target.z],
      };
    },
    moveCam(x, y, z) {
      camera.position.set(x, y, z);
      controls.update();
    },
    cloth() {
      if (!clothMesh) return null;
      const p = positions();
      let topY = -Infinity;
      let botY = Infinity;
      let maxZ = -Infinity;
      let minZ = Infinity;
      for (let i = 0; i < count; i++) {
        const y = p[i * 3 + 1];
        const z = p[i * 3 + 2];
        if (y > topY) topY = y;
        if (y < botY) botY = y;
        if (z > maxZ) maxZ = z;
        if (z < minZ) minZ = z;
      }
      return {
        visible: clothMesh.visible,
        meshVisible: mesh.visible,
        verts: clothGeo.attributes.position.count,
        tris: clothGeo.index.count / 3,
        topY,
        botY,
        maxZ,
        minZ,
      };
    },
    grab(i, x, y, z) {
      world.set_held(i);
      world.set_pos(i, x, y, z);
    },
    throw(i, vx, vy, vz) {
      world.set_vel(i, vx, vy, vz);
      world.set_held(-1);
    },
    sample() {
      const p = positions();
      let minY = Infinity;
      let maxY = -Infinity;
      let maxAbsXZ = 0;
      let maxAbs = 0;
      let minPairDist = Infinity;
      let nan = false;
      for (let i = 0; i < count; i++) {
        const x = p[i * 3];
        const y = p[i * 3 + 1];
        const z = p[i * 3 + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) nan = true;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        maxAbsXZ = Math.max(maxAbsXZ, Math.abs(x), Math.abs(z));
        maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
      }
      if (count <= 400) {
        for (let i = 0; i < count; i++) {
          for (let j = i + 1; j < count; j++) {
            const dx = p[i * 3] - p[j * 3];
            const dy = p[i * 3 + 1] - p[j * 3 + 1];
            const dz = p[i * 3 + 2] - p[j * 3 + 2];
            const d = Math.hypot(dx, dy, dz);
            if (d < minPairDist) minPairDist = d;
          }
        }
      }
      const ex = extra();
      let maxSpeed = 0;
      for (let i = 0; i < ex.length; i++) if (ex[i] > maxSpeed) maxSpeed = ex[i];
      // Spatial spread (stddev of x,y) — for the pendulum this grows from ~0 to
      // large as nearly-identical initial states diverge (a chaos signature).
      let mx = 0;
      let my = 0;
      for (let i = 0; i < count; i++) {
        mx += p[i * 3];
        my += p[i * 3 + 1];
      }
      mx /= count || 1;
      my /= count || 1;
      let vx = 0;
      let vy = 0;
      for (let i = 0; i < count; i++) {
        const ddx = p[i * 3] - mx;
        const ddy = p[i * 3 + 1] - my;
        vx += ddx * ddx;
        vy += ddy * ddy;
      }
      const spreadXY = Math.sqrt((vx + vy) / (count || 1));
      return {
        sim: simKey,
        count,
        radius: simRadius,
        bounds: simBounds,
        minY,
        maxY,
        maxAbsXZ,
        maxAbs,
        minPairDist,
        nan,
        maxSpeed,
        spreadXY,
        extraLen: ex.length,
        first: [p[0], p[1], p[2]],
      };
    },
  };
}
