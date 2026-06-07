# Physics Sandbox

A real-time, interactive **3D physics sandbox** that runs a physics engine written
in **Rust, compiled to WebAssembly**, and renders it with **Three.js** at 60 fps.
Four simulations are switchable live, with a boxy dark-glass UI and an orbit camera.

**Live demo:** https://richikchaudhuri.github.io/physics-simulation/

> Built a real-time 3D physics engine in Rust compiled to WebAssembly — four
> interactive simulations (elastic collisions, N-body gravity, chaotic double
> pendulums, and a mass-spring cloth) rendered with Three.js at 60 fps via
> zero-copy WASM↔JS memory sharing and GPU-instanced rendering.

## Simulations

| Sim | Physics | Integrator | Highlights |
| --- | --- | --- | --- |
| **Collisions** | Sphere–sphere elastic collisions + wall bounces in a box | Semi-implicit Euler | Uniform-grid broad-phase (counting sort, 27-cell neighborhood) scales to ~2000 bodies; grab & throw any ball |
| **Gravity** | Newtonian N-body with Plummer softening | Velocity-Verlet (symplectic) | Rotating disk around a heavy central mass develops spiral structure; grab & throw stars |
| **Pendulum** | Lagrangian double-pendulum equations of motion | Classical RK4 | Many near-identical pendulums fan out — sensitive dependence on initial conditions made visible |
| **Cloth** | Mass-spring grid (structural + shear + bend constraints) | Verlet + Jakobsen position-based relaxation | Pinned sheet billowing in oscillating wind; toggle corner/top-edge pinning |

## Interactions

- **Left-drag empty space** to orbit, **scroll** to zoom.
- **Drag a body** (collisions / gravity) to grab and throw it — it becomes an
  immovable obstacle while held, then flings on release.
- Per-sim controls: body count / resolution, simulation speed, gravity, wind, pin mode.
- Global controls: **Pause**, **Step** (advance one tick), **Reset** (rebuild the sim),
  **Reset View** (re-frame the camera), and **Fullscreen**.

## Architecture

- **Rust engine (`crates/physics`)** exposes one concrete `World` over the
  `wasm-bindgen` boundary (traits/generics can't cross it), holding a
  `Box<dyn Simulation>`. Each sim implements a small `Simulation` trait
  (`step`, `positions`, `reset`, `set_param`, grab hooks, …) and a numeric
  `kind` selects it at construction.
- **Zero-copy state streaming.** Each frame JS reads body positions as a
  `Float32Array` view directly over `wasm.memory.buffer` at `world.positions_ptr()`
  — no per-frame copy. The view is rebuilt only when the buffer identity, pointer,
  or length changes (WASM memory growth detaches the old `ArrayBuffer`, and a
  `Vec` realloc can move the pointer). See `src/wasm.js`.
- **GPU-instanced rendering.** Spheres/particles are a single `InstancedMesh`
  (per-instance transforms + colors); the pendulum arms are one `LineSegments`;
  the cloth is a deformable `BufferGeometry` whose vertices are the WASM
  particle positions with normals recomputed each frame.
- **Stepping.** A fixed-timestep accumulator (1/120 s) with `dt` clamping keeps
  the integrators stable regardless of frame rate, with a sim-speed multiplier.

## Tech stack

Rust → WebAssembly (`wasm-pack`) · Three.js · Vite 8 (Rolldown) · vanilla JavaScript.

## Local development

Prerequisites: a recent **Rust** toolchain, the **wasm32 target**, **wasm-pack**, and **Node** ≥ 18.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack          # or: cargo binstall wasm-pack

npm install
npm run dev                      # builds the WASM crate, then starts Vite
```

`npm run dev` and `npm run build` both run `wasm-pack` first (there is no Rust HMR),
so editing Rust requires a re-run / refresh. Build output goes to `dist/`.

```
physics-sandbox/
  index.html              # dark shell: <canvas> + glass UI overlay
  vite.config.js          # vite-plugin-wasm, esnext target, Pages base path
  src/
    main.js               # entry: WASM init, Three.js scene/camera, sim registry, render loop
    wasm.js               # init() + memory-growth-aware Float32Array views
    styles.css            # boxy dark-glass UI
    physics-pkg/          # wasm-pack output (generated, gitignored)
  crates/physics/
    src/lib.rs            # Simulation trait + the four sims + World (wasm-bindgen)
  .github/workflows/      # GitHub Pages deploy
```

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the WASM
engine and the Vite front-end and publishes `dist/` to GitHub Pages.

**One-time setup (repo owner):** Settings → Pages → Build and deployment →
**Source = "GitHub Actions"**. The site is then served at
`https://<user>.github.io/physics-simulation/`. The production base path lives in
`vite.config.js` (override via the `VITE_BASE` env var if the repo is renamed or
hosted elsewhere, e.g. a custom domain or Netlify/Vercel at root: `VITE_BASE=/`).
