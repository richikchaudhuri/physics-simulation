import init, { World } from './physics-pkg/physics.js';

let wasm = null;

export async function initWasm() {
  // `init()` returns the wasm exports, including `.memory` (the WebAssembly.Memory).
  wasm = await init();
  return wasm;
}

export { World };

// Returns a function giving a live Float32Array over the sim's position buffer.
//
// The buffer must be re-derived whenever WASM linear memory grows (the old
// ArrayBuffer detaches and any view over it becomes empty) OR whenever the Rust
// Vec is reallocated (e.g. after reset, the pointer can move). We cheaply poll
// ptr/len each frame and only rebuild the typed-array view when something changes.
export function makePositions(world) {
  let view = null;
  let lastBuffer = null;
  let lastPtr = -1;
  let lastLen = -1;
  return function positions() {
    const buffer = wasm.memory.buffer;
    const ptr = world.positions_ptr();
    const len = world.len();
    if (buffer !== lastBuffer || ptr !== lastPtr || len !== lastLen) {
      lastBuffer = buffer;
      lastPtr = ptr;
      lastLen = len;
      view = new Float32Array(buffer, ptr, len);
    }
    return view;
  };
}
