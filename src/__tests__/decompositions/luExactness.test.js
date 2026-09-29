import { describe, expect, it } from 'vitest';

import { LU, Matrix, determinant, inverse, solve } from '../..';

// The factorization and the solves are blocked, but every value must still
// come out of the same operations in the same order as JAMA's loops, which the
// library used before. These are those loops.

function jamaLu(matrix) {
  const lu = matrix.clone();
  const rows = lu.rows;
  const columns = lu.columns;
  const pivotVector = new Float64Array(rows);
  let pivotSign = 1;
  for (let i = 0; i < rows; i++) pivotVector[i] = i;
  const LUcolj = new Float64Array(rows);
  for (let j = 0; j < columns; j++) {
    for (let i = 0; i < rows; i++) LUcolj[i] = lu.get(i, j);
    for (let i = 0; i < rows; i++) {
      const kmax = Math.min(i, j);
      let s = 0;
      for (let k = 0; k < kmax; k++) s += lu.get(i, k) * LUcolj[k];
      LUcolj[i] -= s;
      lu.set(i, j, LUcolj[i]);
    }
    let p = j;
    for (let i = j + 1; i < rows; i++) {
      if (Math.abs(LUcolj[i]) > Math.abs(LUcolj[p])) p = i;
    }
    if (p !== j) {
      for (let k = 0; k < columns; k++) {
        const t = lu.get(p, k);
        lu.set(p, k, lu.get(j, k));
        lu.set(j, k, t);
      }
      const v = pivotVector[p];
      pivotVector[p] = pivotVector[j];
      pivotVector[j] = v;
      pivotSign = -pivotSign;
    }
    if (j < rows && lu.get(j, j) !== 0) {
      for (let i = j + 1; i < rows; i++) {
        lu.set(i, j, lu.get(i, j) / lu.get(j, j));
      }
    }
  }
  return { LU: lu, pivotVector, pivotSign };
}

function jamaSolve({ LU: lu, pivotVector }, value) {
  const count = value.columns;
  const X = value.subMatrixRow(pivotVector, 0, count - 1);
  const columns = lu.columns;
  for (let k = 0; k < columns; k++) {
    for (let i = k + 1; i < columns; i++) {
      for (let j = 0; j < count; j++) {
        X.set(i, j, X.get(i, j) - X.get(k, j) * lu.get(i, k));
      }
    }
  }
  for (let k = columns - 1; k >= 0; k--) {
    for (let j = 0; j < count; j++) {
      X.set(k, j, X.get(k, j) / lu.get(k, k));
    }
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < count; j++) {
        X.set(i, j, X.get(i, j) - X.get(k, j) * lu.get(i, k));
      }
    }
  }
  return X;
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

function sample(rows, columns, kind, seed) {
  const next = random(seed);
  const matrix = new Matrix(rows, columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      let value = next();
      if (kind === 'integers') value = Math.round(value * 6);
      if (kind === 'sparse' && Math.abs(value) < 0.3) value = 0;
      if (kind === 'scaled') value *= 10 ** ((i % 5) * 60 - 120);
      matrix.set(i, j, value);
    }
  }
  if (kind === 'singular' && rows > 1) {
    matrix.setRow(rows - 1, matrix.getRow(0));
  }
  return matrix;
}

function expectIdentical(actual, expected) {
  expect(actual.length ?? actual.rows).toBe(expected.length ?? expected.rows);
  const a = actual.to1DArray ? actual.to1DArray() : Array.from(actual);
  const e = expected.to1DArray ? expected.to1DArray() : Array.from(expected);
  for (let i = 0; i < e.length; i++) {
    if (!Object.is(a[i], e[i])) {
      expect.fail(`element ${i}: ${a[i]} instead of ${e[i]}`);
    }
  }
}

// Sizes around the block width (4) and the blocked threshold (80 rows and
// columns).
const shapes = [
  [1, 1],
  [3, 3],
  [5, 3],
  [3, 5],
  [8, 8],
  [9, 9],
  [17, 17],
  [20, 13],
  [13, 20],
  [40, 40],
  [67, 67],
  [80, 80],
  [85, 81],
  [81, 85],
  [97, 97],
];
const kinds = ['random', 'integers', 'sparse', 'scaled', 'singular'];

describe('LU is bit-identical to the JAMA loops', () => {
  it.each(shapes)('factors of %i×%i', (rows, columns) => {
    for (const kind of kinds) {
      const matrix = sample(rows, columns, kind, rows * 100 + columns);
      const expected = jamaLu(matrix);
      const lu = new LU(matrix);
      expectIdentical(lu.LU, expected.LU);
      expectIdentical(lu.pivotVector, expected.pivotVector);
      expect(lu.pivotSign).toBe(expected.pivotSign);
    }
  });

  it.each(shapes.filter(([r, c]) => r === c))(
    'solve and inverse %i×%i',
    (n) => {
      for (const kind of kinds) {
        const matrix = sample(n, n, kind, n * 7);
        const expected = jamaLu(matrix);
        const singular = expected.LU.diag().includes(0);
        for (const rhs of [
          sample(n, 1, 'random', 3),
          sample(n, 11, 'integers', 5),
          Matrix.eye(n),
        ]) {
          if (singular) {
            expect(() => solve(matrix, rhs)).toThrow('LU matrix is singular');
          } else {
            expectIdentical(solve(matrix, rhs), jamaSolve(expected, rhs));
          }
        }
        if (!singular) {
          expectIdentical(inverse(matrix), jamaSolve(expected, Matrix.eye(n)));
        }
        if (n > 3) {
          let det = expected.pivotSign;
          for (let j = 0; j < n; j++) det *= expected.LU.get(j, j);
          expect(determinant(matrix)).toBe(det);
        }
      }
    },
  );
});
