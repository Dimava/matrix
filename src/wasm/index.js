import { kernels } from './kernels';

// The WebAssembly kernels (wasm/kernels.ts) compute exactly what the
// JavaScript loops compute, faster, using SIMD. Where WebAssembly or its SIMD
// extension is unavailable (old engines, a Content-Security-Policy without
// 'wasm-unsafe-eval'), `getKernels` returns undefined and the JavaScript loops
// run instead.

const PAGE_BYTES = 65536;
// Calls needing more memory than this get an instance of their own, dropped
// afterwards, so that one large product does not leave a large memory behind.
const SHARED_MAX_BYTES = 16 * 1024 * 1024;

let module;
let enabled = true;
let shared;

function decode(base64) {
  if (typeof atob === 'function') {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  // eslint-disable-next-line no-undef
  return new Uint8Array(Buffer.from(base64, 'base64'));
}

function compile() {
  if (module === undefined) {
    module = null;
    try {
      if (typeof WebAssembly === 'object') {
        const bytes = decode(kernels);
        if (WebAssembly.validate(bytes)) {
          module = new WebAssembly.Module(bytes);
        }
      }
    } catch {
      module = null;
    }
  }
  return module;
}

// Each instance gets a new memory of the size it needs, never a grown one:
// growing detaches the previous buffer, and from the first detached buffer on,
// V8 checks for detachment on every typed array access in the process, which
// made unrelated code such as EVD 15% slower.
class Kernels {
  constructor(compiled, bytes) {
    this.memory = new WebAssembly.Memory({
      initial: Math.max(1, Math.ceil(bytes / PAGE_BYTES)),
    });
    this.exports = new WebAssembly.Instance(compiled, {
      env: { memory: this.memory },
    }).exports;
    this.f64 = new Float64Array(this.memory.buffer);
    this.i32 = new Int32Array(this.memory.buffer);
  }

  get bytes() {
    return this.memory.buffer.byteLength;
  }
}

/**
 * Kernels with at least `bytes` bytes of memory, or undefined if WebAssembly
 * SIMD is unavailable or disabled. The memory is scratch space: its content
 * is only valid until the next call.
 * @param {number} bytes
 * @returns {Kernels | undefined}
 */
export function getKernels(bytes) {
  if (!enabled) return undefined;
  const compiled = compile();
  if (!compiled) return undefined;
  if (bytes > SHARED_MAX_BYTES) {
    return new Kernels(compiled, bytes);
  }
  if (!shared || shared.bytes < bytes) {
    // At least double, so that growing sizes replace it only a few times.
    const size = Math.max(
      bytes,
      Math.min(2 * (shared?.bytes ?? 0), SHARED_MAX_BYTES),
    );
    shared = new Kernels(compiled, size);
  }
  return shared;
}

/**
 * Turns the WebAssembly kernels on or off, so that tests can compare them
 * with the JavaScript loops. Not part of the public API.
 * @param {boolean} value
 */
export function setWasmEnabled(value) {
  enabled = value;
}
