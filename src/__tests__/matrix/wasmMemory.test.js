import { describe, expect, it } from 'vitest';

import { getKernels } from '../../wasm/index';

// Growing a WebAssembly memory detaches its previous buffer, and one detached
// buffer makes V8 add a detachment check to every typed array access in the
// process. The kernels must never cause that.
describe('WebAssembly kernel memory', () => {
  it('replaces the memory instead of growing it', () => {
    const small = getKernels(1024);
    if (!small) return; // No WebAssembly SIMD here: nothing to check.
    const buffer = small.memory.buffer;
    const large = getKernels(small.bytes + 1);
    expect(large.bytes).toBeGreaterThan(small.bytes);
    expect(buffer.byteLength).toBeGreaterThan(0);
    expect(getKernels(64 * 1024 * 1024).bytes).toBeGreaterThanOrEqual(
      64 * 1024 * 1024,
    );
    expect(large.memory.buffer.byteLength).toBeGreaterThan(0);
  });
});
