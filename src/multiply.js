import { getKernels } from './wasm/index';

// Below this many multiply-adds, copying the operands into WebAssembly memory
// costs more than it saves.
export const WASM_MIN_WORK = 4096;
// It also costs more for an mmul result with fewer columns, whose copied
// values are each used too few times.
const WASM_MIN_COLUMNS = 8;
// gram and mmulByTranspose copy their one operand once for both sides, which
// pays off from a result of this many rows and columns.
export const WASM_MIN_SYMMETRIC_SIZE = 4;

// A Float64Array of more than 8 values is allocated outside the JS heap, which
// takes longer than a small product, so small products share this panel.
const sharedPanel = new Float64Array(1024);

/**
 * Writes the product `a · b` into `c`. All three are arrays of rows: `a` is
 * m × n, `b` has at least n rows of p values, and `c` is m × p, zero-filled.
 *
 * Every cell is summed as the textbook loop sums it, `0 + a[i][0] * b[0][j] +
 * a[i][1] * b[1][j] + ...` in ascending k, so the result is identical to the
 * last bit. Only the schedule differs: a tile of 2 rows by 4 columns is
 * computed at once, which gives 8 independent sums whose additions the CPU can
 * overlap, where a single sum makes each addition wait for the previous one.
 * The 4 columns of `b` are first copied side by side so they are read in
 * memory order.
 * @param {ArrayLike<number>[]} a
 * @param {ArrayLike<number>[]} b
 * @param {Float64Array[]} c
 * @param {number} m
 * @param {number} n
 * @param {number} p
 */
export function multiply(a, b, c, m, n, p) {
  // A single row stays here too: the kernel computes 4 rows at a time.
  if (
    m >= 2 &&
    p >= WASM_MIN_COLUMNS &&
    m * n * p >= WASM_MIN_WORK &&
    multiplyWasm(a, b, c, m, n, p)
  ) {
    return;
  }
  const panel =
    4 * n <= sharedPanel.length ? sharedPanel : new Float64Array(4 * n);
  const fullColumns = p - (p % 4);
  for (let j = 0; j < fullColumns; j += 4) {
    for (let k = 0, q = 0; k < n; k++, q += 4) {
      const row = b[k];
      panel[q] = row[j];
      panel[q + 1] = row[j + 1];
      panel[q + 2] = row[j + 2];
      panel[q + 3] = row[j + 3];
    }
    let i = 0;
    for (; i + 1 < m; i += 2) {
      const a0 = a[i];
      const a1 = a[i + 1];
      let c00 = 0;
      let c01 = 0;
      let c02 = 0;
      let c03 = 0;
      let c10 = 0;
      let c11 = 0;
      let c12 = 0;
      let c13 = 0;
      for (let k = 0, q = 0; k < n; k++, q += 4) {
        const b0 = panel[q];
        const b1 = panel[q + 1];
        const b2 = panel[q + 2];
        const b3 = panel[q + 3];
        const x0 = a0[k];
        const x1 = a1[k];
        c00 += x0 * b0;
        c01 += x0 * b1;
        c02 += x0 * b2;
        c03 += x0 * b3;
        c10 += x1 * b0;
        c11 += x1 * b1;
        c12 += x1 * b2;
        c13 += x1 * b3;
      }
      const r0 = c[i];
      const r1 = c[i + 1];
      r0[j] = c00;
      r0[j + 1] = c01;
      r0[j + 2] = c02;
      r0[j + 3] = c03;
      r1[j] = c10;
      r1[j + 1] = c11;
      r1[j + 2] = c12;
      r1[j + 3] = c13;
    }
    if (i < m) {
      const a0 = a[i];
      let c00 = 0;
      let c01 = 0;
      let c02 = 0;
      let c03 = 0;
      for (let k = 0, q = 0; k < n; k++, q += 4) {
        const x0 = a0[k];
        c00 += x0 * panel[q];
        c01 += x0 * panel[q + 1];
        c02 += x0 * panel[q + 2];
        c03 += x0 * panel[q + 3];
      }
      const r0 = c[i];
      r0[j] = c00;
      r0[j + 1] = c01;
      r0[j + 2] = c02;
      r0[j + 3] = c03;
    }
  }
  // The last p % 4 columns, one at a time, 4 rows at once.
  for (let j = fullColumns; j < p; j++) {
    for (let k = 0; k < n; k++) {
      panel[k] = b[k][j];
    }
    let i = 0;
    for (; i + 3 < m; i += 4) {
      const a0 = a[i];
      const a1 = a[i + 1];
      const a2 = a[i + 2];
      const a3 = a[i + 3];
      let c0 = 0;
      let c1 = 0;
      let c2 = 0;
      let c3 = 0;
      for (let k = 0; k < n; k++) {
        const x = panel[k];
        c0 += a0[k] * x;
        c1 += a1[k] * x;
        c2 += a2[k] * x;
        c3 += a3[k] * x;
      }
      c[i][j] = c0;
      c[i + 1][j] = c1;
      c[i + 2][j] = c2;
      c[i + 3][j] = c3;
    }
    for (; i < m; i++) {
      const row = a[i];
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += row[k] * panel[k];
      }
      c[i][j] = s;
    }
  }
}

/**
 * `multiply` with the WebAssembly kernel. Returns false if unavailable.
 */
function multiplyWasm(a, b, c, m, n, p) {
  return productWasm(
    { rows: a, transposed: false },
    { rows: b, transposed: true },
    c,
    m,
    n,
    p,
  );
}

/**
 * Computes X · Yᵀ in WebAssembly and writes it into the rows `c`, where X is
 * m × n and Y is p × n. Each operand is `{ rows, transposed, scale }`:
 * X[i][k] is `rows[i][k]`, or `rows[k][i]` if `transposed`, times `scale[k]`
 * (as `scale[k] * value`) if a scale is given. The same operand object as
 * both X and Y is copied once. The cells are summed as in `multiply`,
 * `0 + X[i][0] * Y[j][0] + X[i][1] * Y[j][1] + ...`.
 *
 * Options:
 * - `lowerOnly`: only the cells with i >= j are needed.
 * - `finiteY`: give up (return false) if Y has an infinite or NaN value.
 * Returns false if WebAssembly is unavailable or `finiteY` fails.
 * @returns {boolean}
 */
export function productWasm(x, y, c, m, n, p, options = {}) {
  const { lowerOnly = false, finiteY = false } = options;
  const mPanels = Math.ceil(m / 4);
  const pPanels = Math.ceil(p / 4);
  const width = pPanels * 4;
  const xLength = mPanels * 4 * n;
  const yLength = pPanels * 4 * n;
  const packY = y !== x;
  const offsetY = packY ? xLength : 0;
  const offsetC = offsetY + yLength;
  const kernels = getKernels((offsetC + mPanels * 4 * width) * 8);
  if (!kernels) return false;
  const f = kernels.f64;

  pack(f, 0, x, m, n);
  if (packY) pack(f, offsetY, y, p, n);
  if (finiteY && !allFinite(f, offsetY, offsetY + yLength)) return false;

  kernels.exports.gemm(
    0,
    offsetY * 8,
    offsetC * 8,
    mPanels,
    n,
    pPanels,
    lowerOnly,
  );

  for (let i = 0; i < m; i++) {
    const start = offsetC + i * width;
    c[i].set(f.subarray(start, start + p));
  }
  return true;
}

/**
 * Packs the count × n operand into f from `offset`, in panels of 4 rows:
 * f[offset + (panel * n + k) * 4 + r] holds row 4 * panel + r, column k, and
 * rows past `count` are zero.
 */
function pack(f, offset, operand, count, n) {
  const { rows, transposed, scale } = operand;
  const panels = Math.ceil(count / 4);
  if (!transposed) {
    for (let panel = 0; panel < panels; panel++) {
      const base = offset + panel * 4 * n;
      for (let r = 0; r < 4; r++) {
        const i = panel * 4 + r;
        if (i >= count) {
          for (let k = 0; k < n; k++) f[base + 4 * k + r] = 0;
        } else if (scale) {
          const row = rows[i];
          for (let k = 0; k < n; k++) f[base + 4 * k + r] = scale[k] * row[k];
        } else {
          const row = rows[i];
          for (let k = 0; k < n; k++) f[base + 4 * k + r] = row[k];
        }
      }
    }
    return;
  }
  // Transposed: the operand's row i is column i of `rows`, so each row of
  // `rows` fills one position k in every panel.
  const last = panels - 1;
  const lastCount = count - last * 4;
  for (let k = 0; k < n; k++) {
    const row = rows[k];
    let o = offset + 4 * k;
    for (let panel = 0; panel < last; panel++, o += 4 * n) {
      const i = 4 * panel;
      f[o] = row[i];
      f[o + 1] = row[i + 1];
      f[o + 2] = row[i + 2];
      f[o + 3] = row[i + 3];
    }
    for (let r = 0; r < 4; r++) {
      f[o + r] = r < lastCount ? row[last * 4 + r] : 0;
    }
  }
}

function allFinite(f, from, to) {
  for (let i = from; i < to; i++) {
    if (!Number.isFinite(f[i])) return false;
  }
  return true;
}
