import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

// Vite 8 (Rolldown) supports top-level await natively with target 'esnext',
// so vite-plugin-top-level-await is not needed (and is incompatible with Rolldown).
export default defineConfig({
  plugins: [wasm()],
  build: { target: 'esnext' },
});
