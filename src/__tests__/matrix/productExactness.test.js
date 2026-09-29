import { afterAll, describe, expect, it } from 'vitest';

import { Matrix, MatrixTransposeView } from '../..';
import { setWasmEnabled } from '../../wasm/index';

// The products are computed with register tiles and, when available,
// WebAssembly SIMD, but each cell must still be the sum the textbook loop
// computes, in the same order, down to the last bit. These references are the
// loops the library used before.

function textbookMmul(a, b) {
  const result = new Matrix(a.rows, b.columns);
  const column = new Float64Array(a.columns);
  for (let j = 0; j < b.columns; j++) {
    for (let k = 0; k < a.columns; k++) column[k] = b.get(k, j);
    for (let i = 0; i < a.rows; i++) {
      let s = 0;
      for (let k = 0; k < a.columns; k++) s += a.get(i, k) * column[k];
      result.set(i, j, s);
    }
  }
  return result;
}

function textbookTransposeMultiply(a, b) {
  const result = new Matrix(a.columns, b.columns);
  for (let r = 0; r < a.rows; r++) {
    for (let i = 0; i < a.columns; i++) {
      const value = a.get(r, i);
      if (value === 0) continue;
      for (let j = 0; j < b.columns; j++) {
        result.set(i, j, result.get(i, j) + value * b.get(r, j));
      }
    }
  }
  return result;
}

function textbookGram(a) {
  const n = a.columns;
  const result = new Matrix(n, n);
  for (let r = 0; r < a.rows; r++) {
    for (let i = 0; i < n; i++) {
      const value = a.get(r, i);
      if (value === 0) continue;
      for (let j = i; j < n; j++) {
        result.set(i, j, result.get(i, j) + value * a.get(r, j));
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) result.set(j, i, result.get(i, j));
  }
  return result;
}

function textbookMmulByTranspose(a, scale) {
  const m = a.rows;
  const n = a.columns;
  const result = new Matrix(m, m);
  const rowj = new Float64Array(n);
  for (let j = 0; j < m; j++) {
    for (let k = 0; k < n; k++) {
      rowj[k] = scale === undefined ? a.get(j, k) : scale[k] * a.get(j, k);
    }
    for (let i = j; i < m; i++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += a.get(i, k) * rowj[k];
      result.set(i, j, s);
      result.set(j, i, s);
    }
  }
  return result;
}

function random(seed) {
  let state = seed;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296 - 0.5;
  };
}

const specials = [0, -0, NaN, Infinity, -Infinity, 1e308, -1e308, 5e-324];

/** A rows × columns matrix: random, with zeros, or with special values. */
function sample(rows, columns, kind, seed) {
  const next = random(seed);
  const matrix = new Matrix(rows, columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      let value = next();
      if (kind === 'sparse' && Math.abs(value) < 0.3) value = 0;
      if (kind === 'special' && Math.abs(value) < 0.05) {
        value = specials[Math.floor((value + 0.05) * 80)];
      }
      if (kind === 'integers') value = Math.round(value * 8);
      matrix.set(i, j, value);
    }
  }
  return matrix;
}

function expectIdentical(actual, expected) {
  expect(actual.rows).toBe(expected.rows);
  expect(actual.columns).toBe(expected.columns);
  for (let i = 0; i < expected.rows; i++) {
    for (let j = 0; j < expected.columns; j++) {
      const a = actual.get(i, j);
      const e = expected.get(i, j);
      if (!Object.is(a, e)) {
        expect.fail(`(${i}, ${j}): ${a} instead of ${e}`);
      }
    }
  }
}

// Sizes around the tile edges (2, 4 and 8) and above the WebAssembly threshold.
const shapes = [
  [1, 1, 1],
  [3, 5, 2],
  [7, 9, 13],
  [16, 16, 16],
  [17, 23, 30],
  [33, 1, 31],
  [64, 65, 66],
];
const kinds = ['random', 'sparse', 'special', 'integers'];

describe.each([
  ['WebAssembly', true],
  ['JavaScript', false],
])('products are bit-identical to the textbook loops (%s)', (_, wasm) => {
  setWasmEnabled(wasm);
  afterAll(() => setWasmEnabled(true));

  it.each(shapes)('mmul %i×%i · %i×…', (m, n, p) => {
    setWasmEnabled(wasm);
    for (const kind of kinds) {
      const a = sample(m, n, kind, m * 1000 + n);
      const b = sample(n, p, kind, n * 1000 + p);
      expectIdentical(a.mmul(b), textbookMmul(a, b));
      const view = new MatrixTransposeView(b.transpose());
      expectIdentical(a.mmul(view), textbookMmul(a, b));
    }
  });

  it.each(shapes)('transposeMultiply %i×%i, %i', (m, n, p) => {
    setWasmEnabled(wasm);
    for (const kind of kinds) {
      const a = sample(m, n, kind, m * 1000 + n);
      const b = sample(m, p, kind, m * 1000 + p);
      expectIdentical(a.transposeMultiply(b), textbookTransposeMultiply(a, b));
    }
  });

  it.each(shapes)('gram and mmulByTranspose %i×%i', (m, n) => {
    setWasmEnabled(wasm);
    for (const kind of kinds) {
      const a = sample(m, n, kind, m * 1000 + n);
      expectIdentical(a.gram(), textbookGram(a));
      expectIdentical(a.mmulByTranspose(), textbookMmulByTranspose(a));
      const scale = a.getRow(0);
      expectIdentical(
        a.mmulByTranspose(scale),
        textbookMmulByTranspose(a, scale),
      );
    }
  });
});
