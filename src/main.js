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
const grid = new THREE.GridHelper(ENV_BOUNDS * 2, 20, 0x444444, 0x222222);
scene.add(grid);

const boxGeo = new THREE.BoxGeometry(ENV_BOUNDS * 2, ENV_BOUNDS * 2, ENV_BOUNDS * 2);
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(boxGeo),
  new THREE.LineBasicMaterial({ color: 0x333333 }),
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

// (Re)create the InstancedMesh for the current body count. instanceColor must
// be fully initialized or untouched instances render black.
function buildMesh() {
  if (mesh) {
    scene.remove(mesh);
    mesh.dispose();
  }
  mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < count; i++) mesh.setColorAt(i, BASE_COLOR);
  mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);

  if (cfg.rods) buildRods();
  else if (rodLines) rodLines.visible = false;
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

// Switch to a different sim: reconfigure environment, controls, UI, then build.
function loadSim(nextKey) {
  if (!SIMS[nextKey]) return;
  simKey = nextKey;
  cfg = SIMS[nextKey];

  grid.visible = cfg.showGrid;
  box.visible = cfg.boxed;

  countSlider.min = String(cfg.minCount);
  countSlider.max = String(cfg.maxCount);
  countSlider.value = String(cfg.defaultCount);
  countLabel.textContent = String(cfg.defaultCount);

  // The gravity control is shared; each sim relabels/rescales it (and a fresh
  // World starts at the matching default, so no set_param push is needed here).
  gravRow.style.display = cfg.hasGravityParam ? '' : 'none';
  if (cfg.hasGravityParam) {
    gravName.textContent = cfg.gravLabel;
    gravSlider.min = String(cfg.gravMin);
    gravSlider.max = String(cfg.gravMax);
    gravSlider.step = String(cfg.gravStep);
    gravSlider.value = String(cfg.gravDefault);
    gravLabel.textContent = cfg.gravDefault.toFixed(1);
  }

  buildWorld(cfg.defaultCount);

  camera.position.copy(cfg.camPos);
  if (cfg.camTarget) controls.target.copy(cfg.camTarget);
  else controls.target.set(0, cfg.boxed ? simBounds * 0.5 : 0, 0);
  controls.update();

  if (cfg.hint) hintEl.textContent = cfg.hint;
  for (const b of segBtns) b.classList.toggle('is-active', b.dataset.sim === nextKey);
}

// --- ui -------------------------------------------------------------------
const statCount = document.getElementById('stat-count');
const statFps = document.getElementById('stat-fps');
const countSlider = document.getElementById('count-slider');
const countLabel = document.getElementById('count-label');
const speedSlider = document.getElementById('speed-slider');
const speedLabel = document.getElementById('speed-label');
const gravSlider = document.getElementById('grav-slider');
const gravLabel = document.getElementById('grav-label');
const gravName = document.getElementById('grav-name');
const gravRow = document.getElementById('grav-row');
const btnPause = document.getElementById('btn-pause');
const segBtns = Array.from(document.querySelectorAll('.seg__btn'));
const hintEl = document.querySelector('.hint');

let speed = 1; // sim-time multiplier
let paused = false;

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
  // id 0 is each sim's primary tunable: gravity multiplier (collisions) or
  // gravitational constant G (n-body).
  world.set_param(0, g);
});

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? 'Play' : 'Pause';
});

document.getElementById('btn-reset').addEventListener('click', () => world.reset());

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
  if (cfg.rods && rodLines) updateRods(p);
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
