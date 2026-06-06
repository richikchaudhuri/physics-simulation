import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { initWasm, World, makePositions } from './wasm.js';

await initWasm();

// --- simulation -----------------------------------------------------------
let bodyCount = 200;
let world = new World(0, bodyCount);
let positions = makePositions(world);
const radius = world.radius(); // constant across rebuilds
const bounds = world.bounds(); // constant across rebuilds
let count = world.count();

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
camera.position.set(11, 9, 15);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, bounds * 0.5, 0);

// --- lights ---------------------------------------------------------------
scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.3);
key.position.set(8, 16, 6);
scene.add(key);

// --- environment: ground grid + box ---------------------------------------
const grid = new THREE.GridHelper(bounds * 2, 20, 0x444444, 0x222222);
scene.add(grid);

const boxGeo = new THREE.BoxGeometry(bounds * 2, bounds * 2, bounds * 2);
const box = new THREE.LineSegments(
  new THREE.EdgesGeometry(boxGeo),
  new THREE.LineBasicMaterial({ color: 0x333333 }),
);
box.position.y = bounds; // rests on the grid (floor at y = 0)
scene.add(box);

// --- instanced spheres ----------------------------------------------------
const sphereGeo = new THREE.SphereGeometry(radius, 18, 14);
const sphereMat = new THREE.MeshStandardMaterial({
  color: 0xffffff, // actual tint comes from per-instance colors
  roughness: 0.4,
  metalness: 0.1,
});
const BASE_COLOR = new THREE.Color(0xe8e8e8);
const HELD_COLOR = new THREE.Color(0x6ca0ff);
const dummy = new THREE.Object3D();
let mesh = null;

// (Re)create the InstancedMesh for the current body count and tint every
// instance with the base color (instanceColor must be fully initialized or
// untouched instances render black).
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
}
buildMesh();

// Tear down the sim and rebuild it with a new body count (slider).
function rebuild(n) {
  releaseGrab(); // drop anything held before the indices change
  bodyCount = n;
  if (typeof world.free === 'function') world.free(); // free old WASM instance
  world = new World(0, bodyCount);
  positions = makePositions(world);
  count = world.count();
  buildMesh();
  statCount.textContent = count;
}

// --- ui -------------------------------------------------------------------
const statCount = document.getElementById('stat-count');
const statFps = document.getElementById('stat-fps');
const countSlider = document.getElementById('count-slider');
const countLabel = document.getElementById('count-label');
statCount.textContent = count;
countSlider.value = String(bodyCount);
countLabel.textContent = String(bodyCount);

document.getElementById('btn-reset').addEventListener('click', () => world.reset());

// Live label while dragging; rebuild only on release so we don't thrash the
// sim/mesh on every intermediate value.
countSlider.addEventListener('input', () => {
  countLabel.textContent = countSlider.value;
});
countSlider.addEventListener('change', () => {
  rebuild(parseInt(countSlider.value, 10));
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
const MAX_THROW = 40; // cap fling speed (world units / s)

function pointerNDC(e) {
  const rect = canvas.getBoundingClientRect();
  ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

function clampToBox(v) {
  const lim = bounds - radius;
  v.x = Math.max(-lim, Math.min(lim, v.x));
  v.z = Math.max(-lim, Math.min(lim, v.z));
  v.y = Math.max(radius, Math.min(2 * bounds - radius, v.y));
}

function releaseGrab() {
  if (grabbed < 0) return;
  mesh.setColorAt(grabbed, BASE_COLOR);
  mesh.instanceColor.needsUpdate = true;
  world.set_held(-1);
  grabbed = -1;
  controls.enabled = true;
}

// Capture-phase on window so we run BEFORE OrbitControls' canvas listener; when
// we grab a ball we stopPropagation so the orbit gesture never starts.
function onPointerDown(e) {
  if (e.button !== 0 || e.target !== canvas) return; // left button, on canvas
  pointerNDC(e);
  raycaster.setFromCamera(ndc, camera);
  mesh.computeBoundingSphere(); // matrices are current from the last render()
  const hits = raycaster.intersectObject(mesh);
  if (hits.length === 0 || hits[0].instanceId == null) return; // missed -> orbit

  grabbed = hits[0].instanceId;
  controls.enabled = false;
  e.stopPropagation();

  world.set_held(grabbed);
  mesh.setColorAt(grabbed, HELD_COLOR);
  mesh.instanceColor.needsUpdate = true;

  // Drag plane faces the camera, through the ball's current position, so the
  // ball tracks the cursor at a constant depth while dragging.
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
  throwVel.lerp(instVel, 0.6); // smooth out cursor jitter
  lastDragTime = now;
  lastDragPos.copy(hitPoint);

  world.set_pos(grabbed, hitPoint.x, hitPoint.y, hitPoint.z);
}

function onPointerUp() {
  if (grabbed < 0) return;
  // Releasing after a pause should drop, not fling: ignore stale velocity.
  if (performance.now() - lastDragTime > 120) throwVel.set(0, 0, 0);
  if (throwVel.length() > MAX_THROW) throwVel.setLength(MAX_THROW);
  world.set_vel(grabbed, throwVel.x, throwVel.y, throwVel.z);
  releaseGrab();
}

// Capture phase so this runs before OrbitControls' canvas listener.
window.addEventListener('pointerdown', onPointerDown, true);
window.addEventListener('pointermove', onPointerMove);
window.addEventListener('pointerup', onPointerUp);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- render + fixed-timestep loop -----------------------------------------
const FIXED = 1 / 120;
let last = performance.now();
let acc = 0;
let fpsTime = 0;
let fpsFrames = 0;

function render() {
  const p = positions(); // re-fetched AFTER stepping (step may reallocate)
  for (let i = 0; i < count; i++) {
    dummy.position.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  controls.update();
  renderer.render(scene, camera);
}

function frame(now) {
  let dt = (now - last) / 1000;
  last = now;
  if (dt > 0.1) dt = 0.1; // clamp after tab-switch / stalls
  acc += dt;
  while (acc >= FIXED) {
    world.step(FIXED);
    acc -= FIXED;
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

render(); // paint the initial frame right away, before the loop ticks
schedule();

document.addEventListener('visibilitychange', () => {
  last = performance.now(); // avoid a huge dt spike when visibility flips
});

// Dev-only: headless previews keep the tab hidden, which pauses rAF. This hook
// lets the harness manually advance the sim and draw a frame for screenshots.
if (import.meta.env.DEV) {
  window.__sandbox = {
    tick(steps = 60) {
      for (let i = 0; i < steps; i++) world.step(FIXED);
      render();
    },
    render,
    grab(i, x, y, z) {
      world.set_held(i);
      world.set_pos(i, x, y, z);
    },
    throw(i, vx, vy, vz) {
      world.set_vel(i, vx, vy, vz);
      world.set_held(-1);
    },
    rebuild,
    sample() {
      const p = positions();
      let minY = Infinity;
      let maxY = -Infinity;
      let maxAbsXZ = 0;
      let minPairDist = Infinity;
      for (let i = 0; i < count; i++) {
        const y = p[i * 3 + 1];
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        maxAbsXZ = Math.max(maxAbsXZ, Math.abs(p[i * 3]), Math.abs(p[i * 3 + 2]));
      }
      for (let i = 0; i < count; i++) {
        for (let j = i + 1; j < count; j++) {
          const dx = p[i * 3] - p[j * 3];
          const dy = p[i * 3 + 1] - p[j * 3 + 1];
          const dz = p[i * 3 + 2] - p[j * 3 + 2];
          const d = Math.hypot(dx, dy, dz);
          if (d < minPairDist) minPairDist = d;
        }
      }
      return { count, radius, bounds, minY, maxY, maxAbsXZ, minPairDist, first: [p[0], p[1], p[2]] };
    },
  };
}
