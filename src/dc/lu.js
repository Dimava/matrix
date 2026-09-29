import Matrix from '../matrix';
import { getKernels } from '../wasm/index';
import WrapperMatrix2D from '../wrap/WrapperMatrix2D';

// Columns factored together. Wider blocks give the dot products more
// independent sums to overlap but leave more work to the column-by-column tail.
const BLOCK = 4;
// Smaller problems stay in JavaScript, where they finish before copying them to
// WebAssembly and back would pay off: factorizations of fewer values than
// WASM_MIN_FACTOR_SIZE (20 × 20), and solves of fewer multiply-adds than
// WASM_MIN_SOLVE_WORK (rows² × right-hand-side columns).
const WASM_MIN_FACTOR_SIZE = 400;
const WASM_MIN_SOLVE_WORK = 1024;
// Below this many rows or columns, the dot products are too short for the
// blocked algorithm to make up for its extra copying.
const MIN_BLOCKED_SIZE = 80;

export default class LuDecomposition {
  constructor(matrix) {
    matrix = WrapperMatrix2D.checkMatrix(matrix);

    // A plain Matrix copy: the clone of a SymmetricMatrix would mirror every
    // write.
    let lu = new Matrix(matrix);
    let rows = lu.rows;
    let columns = lu.columns;
    let pivotVector = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      pivotVector[i] = i;
    }

    let pivotSign = 0;
    if (rows * columns >= WASM_MIN_FACTOR_SIZE) {
      pivotSign = factorWasm(lu.data, rows, columns, pivotVector);
    }
    if (pivotSign === 0) {
      pivotSign =
        Math.min(rows, columns) < MIN_BLOCKED_SIZE
          ? factorByColumn(lu.data, rows, columns, pivotVector)
          : factorBlocked(lu.data, rows, columns, pivotVector);
    }

    this.LU = lu;
    this.pivotVector = pivotVector;
    this.pivotSign = pivotSign;
  }

  isSingular() {
    let data = this.LU;
    let col = data.columns;
    for (let j = 0; j < col; j++) {
      if (data.get(j, j) === 0) {
        return true;
      }
    }
    return false;
  }

  solve(value) {
    value = Matrix.checkMatrix(value);

    let lu = this.LU;
    let rows = lu.rows;

    if (rows !== value.rows) {
      throw new Error('Invalid matrix dimensions');
    }
    if (this.isSingular()) {
      throw new Error('LU matrix is singular');
    }

    let count = value.columns;
    if (lu.isSquare() && rows * rows * count >= WASM_MIN_SOLVE_WORK) {
      const X = solvesIdentity(lu, value)
        ? solveIdentityWasm(lu.data, this.pivotVector, rows)
        : solveWasm(
            lu.data,
            rows,
            value.subMatrixRow(this.pivotVector, 0, count - 1),
          );
      if (X) return X;
    }
    if (rows >= MIN_IDENTITY_SOLVE_SIZE && solvesIdentity(lu, value)) {
      return solveIdentity(lu.data, this.pivotVector, rows);
    }
    let X = value.subMatrixRow(this.pivotVector, 0, count - 1);
    for (let j0 = 0; j0 < count; j0 += SOLVE_BLOCK) {
      const j1 = Math.min(j0 + SOLVE_BLOCK, count);
      forward(lu.data, lu.columns, X.data, j0, j1);
      backward(lu.data, lu.columns, X.data, j0, j1);
    }
    return X;
  }

  get determinant() {
    let data = this.LU;
    if (!data.isSquare()) {
      throw new Error('Matrix must be square');
    }
    let determinant = this.pivotSign;
    let col = data.columns;
    for (let j = 0; j < col; j++) {
      determinant *= data.get(j, j);
    }
    return determinant;
  }

  get lowerTriangularMatrix() {
    let data = this.LU;
    let rows = data.rows;
    let columns = data.columns;
    let X = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        if (i > j) {
          X.set(i, j, data.get(i, j));
        } else if (i === j) {
          X.set(i, j, 1);
        } else {
          X.set(i, j, 0);
        }
      }
    }
    return X;
  }

  get upperTriangularMatrix() {
    let data = this.LU;
    let rows = data.rows;
    let columns = data.columns;
    let X = new Matrix(rows, columns);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        if (i <= j) {
          X.set(i, j, data.get(i, j));
        } else {
          X.set(i, j, 0);
        }
      }
    }
    return X;
  }

  get pivotPermutationVector() {
    return Array.from(this.pivotVector);
  }
}

/**
 * The Crout-style algorithm of JAMA, in place on the rows `data`: column j is
 * finished with
 *   LU[i][j] -= sum over k < min(i, j) of LU[i][k] * LU[k][j]
 * where each sum starts at 0 and adds its terms in ascending k, followed by a
 * partial pivoting row swap and the division of the column below the diagonal.
 * @param {Float64Array[]} data
 * @param {number} rows
 * @param {number} columns
 * @param {Float64Array} pivotVector
 * @returns {number} the pivot sign
 */
function factorByColumn(data, rows, columns, pivotVector) {
  let pivotSign = 1;
  const column = new Float64Array(rows);
  for (let j = 0; j < columns; j++) {
    for (let i = 0; i < rows; i++) {
      column[i] = data[i][j];
    }
    for (let i = 0; i < rows; i++) {
      const row = data[i];
      const kmax = Math.min(i, j);
      let s = 0;
      for (let k = 0; k < kmax; k++) {
        s += row[k] * column[k];
      }
      column[i] -= s;
      row[j] = column[i];
    }

    let p = j;
    for (let i = j + 1; i < rows; i++) {
      if (Math.abs(column[i]) > Math.abs(column[p])) {
        p = i;
      }
    }
    if (p !== j) {
      const t = data[p];
      data[p] = data[j];
      data[j] = t;
      const v = pivotVector[p];
      pivotVector[p] = pivotVector[j];
      pivotVector[j] = v;
      pivotSign = -pivotSign;
    }

    if (j < rows && data[j][j] !== 0) {
      const diagonal = data[j][j];
      for (let i = j + 1; i < rows; i++) {
        data[i][j] = data[i][j] / diagonal;
      }
    }
  }
  return pivotSign;
}

/**
 * `factorByColumn` with the columns processed in blocks of BLOCK. Computing
 * one sum at a time leaves every addition waiting for the previous one, so the
 * sums of a block are interleaved. Each sum still adds the same terms in the
 * same order, so the factors are identical.
 * @param {Float64Array[]} data
 * @param {number} rows
 * @param {number} columns
 * @param {Float64Array} pivotVector
 * @returns {number} the pivot sign
 */
function factorBlocked(data, rows, columns, pivotVector) {
  let pivotSign = 1;
  const panel = new Float64Array(rows * BLOCK);
  const partial = new Float64Array(rows * BLOCK);
  for (let j0 = 0; j0 < columns; j0 += BLOCK) {
    const width = Math.min(BLOCK, columns - j0);
    const top = Math.min(j0, rows);

    // `panel` holds the block columns, one row of BLOCK values per matrix row.
    for (let i = 0; i < rows; i++) {
      const row = data[i];
      const offset = i * BLOCK;
      for (let c = 0; c < width; c++) {
        panel[offset + c] = row[j0 + c];
      }
    }

    // Rows above the block: forward substitution with the finished rows of L,
    // each row depending on the rows above it, for all block columns at once.
    if (width === BLOCK) {
      for (let i = 0; i < top; i++) {
        const row = data[i];
        let s0 = 0;
        let s1 = 0;
        let s2 = 0;
        let s3 = 0;
        for (let k = 0, q = 0; k < i; k++, q += BLOCK) {
          const l = row[k];
          s0 += l * panel[q];
          s1 += l * panel[q + 1];
          s2 += l * panel[q + 2];
          s3 += l * panel[q + 3];
        }
        const q = i * BLOCK;
        panel[q] -= s0;
        panel[q + 1] -= s1;
        panel[q + 2] -= s2;
        panel[q + 3] -= s3;
      }
    } else {
      for (let i = 0; i < top; i++) {
        const row = data[i];
        for (let c = 0; c < width; c++) {
          let s = 0;
          for (let k = 0; k < i; k++) {
            s += row[k] * panel[k * BLOCK + c];
          }
          panel[i * BLOCK + c] -= s;
        }
      }
    }

    // Rows from the block down: the part of their sums over the finished
    // columns (k < j0), two rows at a time. The rest is added below.
    let i = j0;
    if (width === BLOCK) {
      for (; i + 1 < rows; i += 2) {
        const row0 = data[i];
        const row1 = data[i + 1];
        let s00 = 0;
        let s01 = 0;
        let s02 = 0;
        let s03 = 0;
        let s10 = 0;
        let s11 = 0;
        let s12 = 0;
        let s13 = 0;
        for (let k = 0, q = 0; k < j0; k++, q += BLOCK) {
          const u0 = panel[q];
          const u1 = panel[q + 1];
          const u2 = panel[q + 2];
          const u3 = panel[q + 3];
          const l0 = row0[k];
          const l1 = row1[k];
          s00 += l0 * u0;
          s01 += l0 * u1;
          s02 += l0 * u2;
          s03 += l0 * u3;
          s10 += l1 * u0;
          s11 += l1 * u1;
          s12 += l1 * u2;
          s13 += l1 * u3;
        }
        const q = i * BLOCK;
        partial[q] = s00;
        partial[q + 1] = s01;
        partial[q + 2] = s02;
        partial[q + 3] = s03;
        partial[q + BLOCK] = s10;
        partial[q + BLOCK + 1] = s11;
        partial[q + BLOCK + 2] = s12;
        partial[q + BLOCK + 3] = s13;
      }
    }
    for (; i < rows; i++) {
      const row = data[i];
      for (let c = 0; c < width; c++) {
        let s = 0;
        for (let k = 0; k < j0; k++) {
          s += row[k] * panel[k * BLOCK + c];
        }
        partial[i * BLOCK + c] = s;
      }
    }

    // Finish the block column by column, as the unblocked algorithm does.
    for (let c = 0; c < width; c++) {
      const j = j0 + c;
      for (let i = j0; i < rows; i++) {
        const row = data[i];
        const kmax = Math.min(i, j);
        let s = partial[i * BLOCK + c];
        for (let k = j0; k < kmax; k++) {
          s += row[k] * panel[k * BLOCK + c];
        }
        panel[i * BLOCK + c] -= s;
      }
      for (let i = 0; i < rows; i++) {
        data[i][j] = panel[i * BLOCK + c];
      }

      let p = j;
      for (let i = j + 1; i < rows; i++) {
        if (Math.abs(panel[i * BLOCK + c]) > Math.abs(panel[p * BLOCK + c])) {
          p = i;
        }
      }

      if (p !== j) {
        // `lu` is our own copy, so its rows can be swapped by reference. The
        // block's working values move with their rows.
        const t = data[p];
        data[p] = data[j];
        data[j] = t;
        for (let d = 0; d < BLOCK; d++) {
          const a = p * BLOCK + d;
          const b = j * BLOCK + d;
          let v = panel[a];
          panel[a] = panel[b];
          panel[b] = v;
          v = partial[a];
          partial[a] = partial[b];
          partial[b] = v;
        }

        const v = pivotVector[p];
        pivotVector[p] = pivotVector[j];
        pivotVector[j] = v;

        pivotSign = -pivotSign;
      }

      if (j < rows && data[j][j] !== 0) {
        const diagonal = data[j][j];
        for (let i = j + 1; i < rows; i++) {
          data[i][j] = data[i][j] / diagonal;
        }
      }
    }
  }
  return pivotSign;
}

// Right-hand-side columns handled together: every column is an independent
// system, so splitting them keeps the rows being updated in cache.
const SOLVE_BLOCK = 256;
// Below this many rows, allocating the scratch matrix of solveIdentity costs
// more than the third of the forward substitution it skips.
const MIN_IDENTITY_SOLVE_SIZE = 20;

/**
 * Forward substitution with the unit lower factor, in place on the columns
 * [from, to) of `x` (rows already permuted): `x[i] -= x[k] * L[i][k]` for
 * ascending k, as the textbook loop does for each element.
 * @param {Float64Array[]} lu
 * @param {number} columns - columns of the factors
 * @param {Float64Array[]} x
 * @param {number} from
 * @param {number} to
 */
function forward(lu, columns, x, from, to) {
  for (let k = 0; k < columns; k++) {
    const xk = x[k];
    for (let i = k + 1; i < columns; i++) {
      const xi = x[i];
      const l = lu[i][k];
      for (let j = from; j < to; j++) {
        xi[j] -= xk[j] * l;
      }
    }
  }
}

/**
 * Back substitution with the upper factor, in place on the columns [from, to)
 * of `x`: for descending k, `x[k] /= U[k][k]` then `x[i] -= x[k] * U[i][k]`.
 * @param {Float64Array[]} lu
 * @param {number} columns - columns of the factors
 * @param {Float64Array[]} x
 * @param {number} from
 * @param {number} to
 */
function backward(lu, columns, x, from, to) {
  for (let k = columns - 1; k >= 0; k--) {
    const xk = x[k];
    const diagonal = lu[k][k];
    for (let j = from; j < to; j++) {
      xk[j] /= diagonal;
    }
    for (let i = 0; i < k; i++) {
      const xi = x[i];
      const u = lu[i][k];
      for (let j = from; j < to; j++) {
        xi[j] -= xk[j] * u;
      }
    }
  }
}

/**
 * Whether solving the factors `lu` with `value` can take solveIdentity: `value`
 * is the identity, and the multipliers are finite.
 * @param {Matrix} lu
 * @param {Matrix} value
 * @returns {boolean}
 */
function solvesIdentity(lu, value) {
  return (
    lu.isSquare() &&
    isIdentity(value) &&
    hasFiniteMultipliers(lu.data, lu.columns)
  );
}

function isIdentity(matrix) {
  if (!(matrix instanceof Matrix) || !matrix.isSquare()) return false;
  for (let i = 0; i < matrix.rows; i++) {
    const row = matrix.data[i];
    for (let j = 0; j < matrix.columns; j++) {
      if (row[j] !== (i === j ? 1 : 0)) return false;
    }
  }
  return true;
}

function hasFiniteMultipliers(lu, columns) {
  for (let i = 1; i < columns; i++) {
    const row = lu[i];
    for (let k = 0; k < i; k++) {
      if (!Number.isFinite(row[k])) return false;
    }
  }
  return true;
}

/**
 * `solve` with the identity as right-hand side, i.e. the inverse, skipping the
 * operations that cannot change any bit of the result.
 *
 * The permuted identity has, in row r, a single 1 in column pivotVector[r].
 * Ordering the columns that way turns it into the identity `y`, and forward
 * substitution keeps row k of `y` exactly +0 right of column k. The skipped
 * updates are all `y[i][q] -= (+0) * L[i][k]`: with finite multipliers that
 * subtracts a zero, and `y` never holds a -0 for it to flip, so every element
 * ends as in the full loop. Back substitution then runs in full.
 * @param {Float64Array[]} lu
 * @param {Float64Array} pivotVector
 * @param {number} n
 * @returns {Matrix}
 */
function solveIdentity(lu, pivotVector, n) {
  const X = new Matrix(n, n);
  const x = X.data;
  const y = Matrix.eye(n).data;
  for (let q0 = 0; q0 < n; q0 += SOLVE_BLOCK) {
    const q1 = Math.min(q0 + SOLVE_BLOCK, n);
    for (let k = q0; k < n; k++) {
      const yk = y[k];
      const end = Math.min(k + 1, q1);
      for (let i = k + 1; i < n; i++) {
        const yi = y[i];
        const l = lu[i][k];
        for (let q = q0; q < end; q++) {
          yi[q] -= yk[q] * l;
        }
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = y[i];
    for (let q = 0; q < n; q++) {
      xi[pivotVector[q]] = yi[q];
    }
  }
  for (let j0 = 0; j0 < n; j0 += SOLVE_BLOCK) {
    backward(lu, n, x, j0, Math.min(j0 + SOLVE_BLOCK, n));
  }
  return X;
}

// Block width of the WebAssembly factorization (W in wasm/kernels.ts).
const WASM_BLOCK = 8;

/**
 * Factors `data` in place with the WebAssembly kernel, which runs the same
 * blocked algorithm as the constructor. Returns the pivot sign, or 0 if
 * WebAssembly is unavailable.
 */
function factorWasm(data, rows, columns, pivotVector) {
  const lda = rowStride(columns);
  const aLength = rows * lda;
  const workOffset = aLength + Math.ceil(rows / 2);
  const kernels = getKernels((workOffset + 2 * rows * WASM_BLOCK) * 8);
  if (!kernels) return 0;
  const f = kernels.f64;
  for (let i = 0; i < rows; i++) {
    f.set(data[i], i * lda);
  }
  const sign = kernels.exports.luFactor(
    0,
    rows,
    columns,
    lda,
    aLength * 8,
    workOffset * 8,
  );
  for (let i = 0; i < rows; i++) {
    data[i].set(f.subarray(i * lda, i * lda + columns));
  }
  const pivots = kernels.i32;
  for (let i = 0; i < rows; i++) {
    pivotVector[i] = pivots[aLength * 2 + i];
  }
  return sign;
}

/**
 * Row stride, in values, for rows of `count` values in WebAssembly memory: a
 * multiple of 8, as the solve kernels need, and one cache line more when that
 * is a multiple of 64, whose rows would compete for the same cache sets (see
 * allocateRows in src/matrix.js).
 * @param {number} count
 * @returns {number}
 */
function rowStride(count) {
  const stride = Math.ceil(count / 8) * 8;
  return stride % 64 === 0 ? stride + 8 : stride;
}

/**
 * Copies the n × n factors into WebAssembly memory, followed by `bytes` of
 * room. Returns the kernels, their row stride `ldl` and the offset, in values,
 * of the room.
 */
function loadFactors(lu, n, bytes) {
  const ldl = rowStride(n);
  const offset = n * ldl;
  const kernels = getKernels(offset * 8 + bytes);
  if (!kernels) return undefined;
  const f = kernels.f64;
  for (let i = 0; i < n; i++) {
    f.set(lu[i], i * ldl);
  }
  return { kernels, f, ldl, offset };
}

/** `solve` in WebAssembly for a right-hand side of any width. */
function solveWasm(lu, n, X) {
  const count = X.columns;
  // The kernels work on tiles of 8 columns; the columns past `count` are zero.
  const width = Math.ceil(count / 8) * 8;
  const ldx = rowStride(count);
  const loaded = loadFactors(lu, n, n * ldx * 8);
  if (!loaded) return undefined;
  const { kernels, f, ldl, offset: xOffset } = loaded;
  for (let i = 0; i < n; i++) {
    const start = xOffset + i * ldx;
    f.set(X.data[i], start);
    f.fill(0, start + count, start + width);
  }
  kernels.exports.forward(0, n, ldl, xOffset * 8, ldx, 0, width, 0);
  kernels.exports.backward(0, n, ldl, xOffset * 8, ldx, 0, width);
  for (let i = 0; i < n; i++) {
    const start = xOffset + i * ldx;
    X.data[i].set(f.subarray(start, start + count));
  }
  return X;
}

/** `solveIdentity` in WebAssembly. */
function solveIdentityWasm(lu, pivotVector, n) {
  const width = Math.ceil(n / 8) * 8;
  const ldx = rowStride(n);
  const loaded = loadFactors(lu, n, 2 * n * ldx * 8 + 4 * n);
  if (!loaded) return undefined;
  const { kernels, f, ldl, offset: xOffset } = loaded;
  const yOffset = xOffset + n * ldx;
  const pivotsOffset = (yOffset + n * ldx) * 8;
  f.fill(0, yOffset, yOffset + n * ldx);
  for (let i = 0; i < n; i++) {
    f[yOffset + i * ldx + i] = 1;
  }
  const pivots = kernels.i32;
  for (let i = 0; i < n; i++) {
    pivots[pivotsOffset / 4 + i] = pivotVector[i];
  }
  kernels.exports.forwardIdentity(0, n, ldl, yOffset * 8, ldx);
  f.fill(0, xOffset, xOffset + n * ldx);
  kernels.exports.scatterColumns(
    yOffset * 8,
    xOffset * 8,
    n,
    ldx,
    pivotsOffset,
  );
  kernels.exports.backward(0, n, ldl, xOffset * 8, ldx, 0, width);
  const X = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    const start = xOffset + i * ldx;
    X.data[i].set(f.subarray(start, start + n));
  }
  return X;
}
