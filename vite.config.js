import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

// Vite 8 (Rolldown) supports top-level await natively with target 'esnext',
// so vite-plugin-top-level-await is not needed (and is incompatible with Rolldown).
//
// `base` is the repo name in production so assets resolve under GitHub Project
// Pages (https://<user>.github.io/physics-simulation/), but stays '/' for local
// dev/preview. Override with VITE_BASE if the repo is renamed or hosted elsewhere.
export default defineConfig(({ command }) => ({
  base: process.env.VITE_BASE ?? (command === 'build' ? '/physics-simulation/' : '/'),
  plugins: [wasm()],
  build: { target: 'esnext' },
}));
