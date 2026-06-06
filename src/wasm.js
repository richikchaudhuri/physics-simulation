import init, { World } from './physics-pkg/physics.js';

let wasm = null;

export async function initWasm() {
  // `init()` returns the wasm exports, including `.memory` (the WebAssembly.Memory).
  wasm = await init();
  return wasm;
}

export { World };

const EMPTY = new Float32Array(0);

// Returns a function giving a live Float32Array over a WASM-owned f32 buffer.
//
// The buffer must be re-derived whenever WASM linear memory grows (the old
// ArrayBuffer detaches and any view over it becomes empty) OR whenever the Rust
// Vec is reallocated (e.g. after reset, the pointer can move). We cheaply poll
// ptr/len each frame and only rebuild the typed-array view when something changes.
function makeFloatView(ptrFn, lenFn) {
  let view = null;
  let lastBuffer = null;
  let lastPtr = -1;
  let lastLen = -1;
  return function () {
    const len = lenFn();
    if (len === 0) return EMPTY;
    const buffer = wasm.memory.buffer;
    const ptr = ptrFn();
    if (buffer !== lastBuffer || ptr !== lastPtr || len !== lastLen) {
      lastBuffer = buffer;
      lastPtr = ptr;
      lastLen = len;
      view = new Float32Array(buffer, ptr, len);
    }
    return view;
  };
}

// Live view over the sim's flat [x,y,z,...] position buffer.
export function makePositions(world) {
  return makeFloatView(
    () => world.positions_ptr(),
    () => world.len(),
  );
}

// Live view over the sim's optional per-body scalar buffer (e.g. speed). Empty
// (length 0) when the active sim doesn't expose one.
export function makeExtra(world) {
  return makeFloatView(
    () => world.extra_ptr(),
    () => world.extra_len(),
  );
}
