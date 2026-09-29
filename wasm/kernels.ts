// WebAssembly kernels for ml-matrix, compiled by tools/buildWasm.mjs into
// src/wasm/kernels.js. Build: npm run build-wasm
//
// Every kernel computes each value with exactly the floating-point operations,
// in exactly the order, of the JavaScript loop it replaces. SIMD is only used
// across independent values (lanes never meet), and a multiply followed by an
// add or subtract stays two instructions, never a fused multiply-add. So the
// results are identical to the last bit; only the speed differs.
//
// Pointers are byte offsets into the exported memory, which the JavaScript side
// grows and lays out. Matrices are row-major f64.

// ---------------------------------------------------------------------------
// Matrix product

/**
 * C = A · B on packed operands, with a 4 × 4 tile of C in registers.
 *   A in panels of 4 rows:    a[(ip * n + k) * 4 + ii] = A[4 * ip + ii][k]
 *   B in panels of 4 columns: b[(jp * n + k) * 4 + jj] = B[k][4 * jp + jj]
 *   C row-major, 4 * pPanels columns.
 * With lowerOnly, only the cells on or below the diagonal are guaranteed.
 * Each cell is 0 + A[i][0] * B[0][j] + A[i][1] * B[1][j] + ... in ascending k.
 */
export function gemm(
  a: usize,
  b: usize,
  c: usize,
  mPanels: i32,
  n: i32,
  pPanels: i32,
  lowerOnly: bool,
): void {
  const rowBytes = (<usize>pPanels) << 5;
  const panelBytes = (<usize>n) << 5;
  for (let ip = 0; ip < mPanels; ip++) {
    const pa0 = a + <usize>ip * panelBytes;
    // With lowerOnly, the tiles entirely above the diagonal are skipped.
    const jpEnd = lowerOnly ? min(ip + 1, pPanels) : pPanels;
    for (let jp = 0; jp < jpEnd; jp++) {
      let pa = pa0;
      let pb = b + <usize>jp * panelBytes;
      let c00 = f64x2.splat(0);
      let c01 = f64x2.splat(0);
      let c10 = f64x2.splat(0);
      let c11 = f64x2.splat(0);
      let c20 = f64x2.splat(0);
      let c21 = f64x2.splat(0);
      let c30 = f64x2.splat(0);
      let c31 = f64x2.splat(0);
      for (let k = 0; k < n; k++) {
        const b0 = v128.load(pb);
        const b1 = v128.load(pb, 16);
        let x = v128.load64_splat(pa);
        c00 = f64x2.add(c00, f64x2.mul(x, b0));
        c01 = f64x2.add(c01, f64x2.mul(x, b1));
        x = v128.load64_splat(pa, 8);
        c10 = f64x2.add(c10, f64x2.mul(x, b0));
        c11 = f64x2.add(c11, f64x2.mul(x, b1));
        x = v128.load64_splat(pa, 16);
        c20 = f64x2.add(c20, f64x2.mul(x, b0));
        c21 = f64x2.add(c21, f64x2.mul(x, b1));
        x = v128.load64_splat(pa, 24);
        c30 = f64x2.add(c30, f64x2.mul(x, b0));
        c31 = f64x2.add(c31, f64x2.mul(x, b1));
        pa += 32;
        pb += 32;
      }
      const pc = c + ((<usize>ip * rowBytes) << 2) + ((<usize>jp) << 5);
      v128.store(pc, c00);
      v128.store(pc, c01, 16);
      v128.store(pc + rowBytes, c10);
      v128.store(pc + rowBytes, c11, 16);
      v128.store(pc + 2 * rowBytes, c20);
      v128.store(pc + 2 * rowBytes, c21, 16);
      v128.store(pc + 3 * rowBytes, c30);
      v128.store(pc + 3 * rowBytes, c31, 16);
    }
  }
}

// ---------------------------------------------------------------------------
// LU decomposition

// Columns factored together, 4 f64x2 lanes wide.
const W = 8;
const W_BYTES: usize = W * 8;

/**
 * JAMA's Crout LU with partial pivoting, in place on a rows × columns matrix
 * with row stride `lda` values, blocked as in src/dc/lu.js (see there for why
 * this is the same arithmetic). `pivots` receives rows i32 values, `work` must
 * hold 2 * rows * W f64. Returns the pivot sign.
 */
export function luFactor(
  a: usize,
  rows: i32,
  columns: i32,
  lda: i32,
  pivots: usize,
  work: usize,
): i32 {
  const stride = (<usize>lda) << 3;
  const rowBytes = (<usize>columns) << 3;
  const panel = work;
  const partial = work + <usize>rows * W_BYTES;
  let sign = 1;
  for (let i = 0; i < rows; i++) store<i32>(pivots + ((<usize>i) << 2), i);

  for (let j0 = 0; j0 < columns; j0 += W) {
    const width = min(W, columns - j0);
    const top = min(j0, rows);

    // The block columns, W per row, zero-padded past the last column.
    for (let i = 0; i < rows; i++) {
      const row = a + <usize>i * stride + ((<usize>j0) << 3);
      const p = panel + <usize>i * W_BYTES;
      for (let c = 0; c < W; c++) {
        store<f64>(
          p + ((<usize>c) << 3),
          c < width ? load<f64>(row + ((<usize>c) << 3)) : 0,
        );
      }
    }

    // Rows above the block: forward substitution, two rows at a time. Row
    // i + 1 needs row i finished for its last term, which is added after.
    let i = 0;
    for (; i + 1 < top; i += 2) {
      const row0 = a + <usize>i * stride;
      const row1 = row0 + stride;
      let s0 = f64x2.splat(0);
      let s1 = f64x2.splat(0);
      let s2 = f64x2.splat(0);
      let s3 = f64x2.splat(0);
      let t0 = f64x2.splat(0);
      let t1 = f64x2.splat(0);
      let t2 = f64x2.splat(0);
      let t3 = f64x2.splat(0);
      let pk = panel;
      for (let k = 0; k < i; k++) {
        const u0 = v128.load(pk);
        const u1 = v128.load(pk, 16);
        const u2 = v128.load(pk, 32);
        const u3 = v128.load(pk, 48);
        const l0 = v128.load64_splat(row0 + ((<usize>k) << 3));
        const l1 = v128.load64_splat(row1 + ((<usize>k) << 3));
        s0 = f64x2.add(s0, f64x2.mul(l0, u0));
        s1 = f64x2.add(s1, f64x2.mul(l0, u1));
        s2 = f64x2.add(s2, f64x2.mul(l0, u2));
        s3 = f64x2.add(s3, f64x2.mul(l0, u3));
        t0 = f64x2.add(t0, f64x2.mul(l1, u0));
        t1 = f64x2.add(t1, f64x2.mul(l1, u1));
        t2 = f64x2.add(t2, f64x2.mul(l1, u2));
        t3 = f64x2.add(t3, f64x2.mul(l1, u3));
        pk += W_BYTES;
      }
      // Finish row i.
      const u0 = f64x2.sub(v128.load(pk), s0);
      const u1 = f64x2.sub(v128.load(pk, 16), s1);
      const u2 = f64x2.sub(v128.load(pk, 32), s2);
      const u3 = f64x2.sub(v128.load(pk, 48), s3);
      v128.store(pk, u0);
      v128.store(pk, u1, 16);
      v128.store(pk, u2, 32);
      v128.store(pk, u3, 48);
      // Row i + 1: its term k = i, then finish.
      const l1 = v128.load64_splat(row1 + ((<usize>i) << 3));
      t0 = f64x2.add(t0, f64x2.mul(l1, u0));
      t1 = f64x2.add(t1, f64x2.mul(l1, u1));
      t2 = f64x2.add(t2, f64x2.mul(l1, u2));
      t3 = f64x2.add(t3, f64x2.mul(l1, u3));
      pk += W_BYTES;
      v128.store(pk, f64x2.sub(v128.load(pk), t0));
      v128.store(pk, f64x2.sub(v128.load(pk, 16), t1), 16);
      v128.store(pk, f64x2.sub(v128.load(pk, 32), t2), 32);
      v128.store(pk, f64x2.sub(v128.load(pk, 48), t3), 48);
    }
    for (; i < top; i++) {
      const row = a + <usize>i * stride;
      let s0 = f64x2.splat(0);
      let s1 = f64x2.splat(0);
      let s2 = f64x2.splat(0);
      let s3 = f64x2.splat(0);
      let pk = panel;
      for (let k = 0; k < i; k++) {
        const l = v128.load64_splat(row + ((<usize>k) << 3));
        s0 = f64x2.add(s0, f64x2.mul(l, v128.load(pk)));
        s1 = f64x2.add(s1, f64x2.mul(l, v128.load(pk, 16)));
        s2 = f64x2.add(s2, f64x2.mul(l, v128.load(pk, 32)));
        s3 = f64x2.add(s3, f64x2.mul(l, v128.load(pk, 48)));
        pk += W_BYTES;
      }
      v128.store(pk, f64x2.sub(v128.load(pk), s0));
      v128.store(pk, f64x2.sub(v128.load(pk, 16), s1), 16);
      v128.store(pk, f64x2.sub(v128.load(pk, 32), s2), 32);
      v128.store(pk, f64x2.sub(v128.load(pk, 48), s3), 48);
    }

    // Rows from the block down: their sums over the finished columns (k < j0),
    // two rows at a time.
    i = j0;
    for (; i + 1 < rows; i += 2) {
      const row0 = a + <usize>i * stride;
      const row1 = row0 + stride;
      let s0 = f64x2.splat(0);
      let s1 = f64x2.splat(0);
      let s2 = f64x2.splat(0);
      let s3 = f64x2.splat(0);
      let t0 = f64x2.splat(0);
      let t1 = f64x2.splat(0);
      let t2 = f64x2.splat(0);
      let t3 = f64x2.splat(0);
      let pk = panel;
      for (let k = 0; k < j0; k++) {
        const u0 = v128.load(pk);
        const u1 = v128.load(pk, 16);
        const u2 = v128.load(pk, 32);
        const u3 = v128.load(pk, 48);
        const l0 = v128.load64_splat(row0 + ((<usize>k) << 3));
        const l1 = v128.load64_splat(row1 + ((<usize>k) << 3));
        s0 = f64x2.add(s0, f64x2.mul(l0, u0));
        s1 = f64x2.add(s1, f64x2.mul(l0, u1));
        s2 = f64x2.add(s2, f64x2.mul(l0, u2));
        s3 = f64x2.add(s3, f64x2.mul(l0, u3));
        t0 = f64x2.add(t0, f64x2.mul(l1, u0));
        t1 = f64x2.add(t1, f64x2.mul(l1, u1));
        t2 = f64x2.add(t2, f64x2.mul(l1, u2));
        t3 = f64x2.add(t3, f64x2.mul(l1, u3));
        pk += W_BYTES;
      }
      const q = partial + <usize>i * W_BYTES;
      v128.store(q, s0);
      v128.store(q, s1, 16);
      v128.store(q, s2, 32);
      v128.store(q, s3, 48);
      v128.store(q, t0, 64);
      v128.store(q, t1, 80);
      v128.store(q, t2, 96);
      v128.store(q, t3, 112);
    }
    for (; i < rows; i++) {
      const row = a + <usize>i * stride;
      let s0 = f64x2.splat(0);
      let s1 = f64x2.splat(0);
      let s2 = f64x2.splat(0);
      let s3 = f64x2.splat(0);
      let pk = panel;
      for (let k = 0; k < j0; k++) {
        const l = v128.load64_splat(row + ((<usize>k) << 3));
        s0 = f64x2.add(s0, f64x2.mul(l, v128.load(pk)));
        s1 = f64x2.add(s1, f64x2.mul(l, v128.load(pk, 16)));
        s2 = f64x2.add(s2, f64x2.mul(l, v128.load(pk, 32)));
        s3 = f64x2.add(s3, f64x2.mul(l, v128.load(pk, 48)));
        pk += W_BYTES;
      }
      const q = partial + <usize>i * W_BYTES;
      v128.store(q, s0);
      v128.store(q, s1, 16);
      v128.store(q, s2, 32);
      v128.store(q, s3, 48);
    }

    // Finish the block column by column, as the unblocked algorithm does.
    for (let c = 0; c < width; c++) {
      const j = j0 + c;
      const cOffset = (<usize>c) << 3;
      for (let i = j0; i < rows; i++) {
        const row = a + <usize>i * stride;
        const kmax = min(i, j);
        let s = load<f64>(partial + <usize>i * W_BYTES + cOffset);
        for (let k = j0; k < kmax; k++) {
          s +=
            load<f64>(row + ((<usize>k) << 3)) *
            load<f64>(panel + <usize>k * W_BYTES + cOffset);
        }
        const p = panel + <usize>i * W_BYTES + cOffset;
        store<f64>(p, load<f64>(p) - s);
      }
      for (let i = 0; i < rows; i++) {
        store<f64>(
          a + <usize>i * stride + ((<usize>j) << 3),
          load<f64>(panel + <usize>i * W_BYTES + cOffset),
        );
      }

      let p = j;
      for (let i = j + 1; i < rows; i++) {
        if (
          abs<f64>(load<f64>(panel + <usize>i * W_BYTES + cOffset)) >
          abs<f64>(load<f64>(panel + <usize>p * W_BYTES + cOffset))
        ) {
          p = i;
        }
      }

      if (p !== j) {
        swapBytes(a + <usize>p * stride, a + <usize>j * stride, rowBytes);
        swapBytes(
          panel + <usize>p * W_BYTES,
          panel + <usize>j * W_BYTES,
          W_BYTES,
        );
        swapBytes(
          partial + <usize>p * W_BYTES,
          partial + <usize>j * W_BYTES,
          W_BYTES,
        );
        const pp = pivots + ((<usize>p) << 2);
        const pj = pivots + ((<usize>j) << 2);
        const v = load<i32>(pp);
        store<i32>(pp, load<i32>(pj));
        store<i32>(pj, v);
        sign = -sign;
      }

      if (j < rows) {
        const diagonal = load<f64>(a + <usize>j * stride + ((<usize>j) << 3));
        if (diagonal !== 0) {
          for (let i = j + 1; i < rows; i++) {
            const e = a + <usize>i * stride + ((<usize>j) << 3);
            store<f64>(e, load<f64>(e) / diagonal);
          }
        }
      }
    }
  }
  return sign;
}

function swapBytes(x: usize, y: usize, bytes: usize): void {
  let offset: usize = 0;
  for (; offset + 16 <= bytes; offset += 16) {
    const t = v128.load(x + offset);
    v128.store(x + offset, v128.load(y + offset));
    v128.store(y + offset, t);
  }
  for (; offset < bytes; offset += 8) {
    const t = load<f64>(x + offset);
    store<f64>(x + offset, load<f64>(y + offset));
    store<f64>(y + offset, t);
  }
}

// ---------------------------------------------------------------------------
// Triangular solves with the factors of luFactor, on an n × n factor with row
// stride `ldl` and a right-hand side of n rows whose row stride `ldx` is a
// multiple of 8 values (padding columns are computed and ignored). Row-oriented: each tile of 8
// columns of one or two rows is kept in registers while the rows it depends on
// stream past, applying for every value the same operations in the same order
// as the column-oriented loops in src/dc/lu.js.

/**
 * Forward substitution with the unit lower factor:
 * x[i] -= x[k] * L[i][k] for k from `from` to i - 1, ascending, for the rows
 * i >= `from`. `from` is 0 for a general right-hand side.
 */
export function forward(
  lu: usize,
  n: i32,
  ldl: i32,
  x: usize,
  ldx: i32,
  q0: i32,
  q1: i32,
  from: i32,
): void {
  const luStride = (<usize>ldl) << 3;
  const xStride = (<usize>ldx) << 3;
  for (let q = q0; q < q1; q += 8) {
    const col = (<usize>q) << 3;
    let i = from;
    for (; i + 1 < n; i += 2) {
      const l0row = lu + <usize>i * luStride;
      const l1row = l0row + luStride;
      const x0 = x + <usize>i * xStride + col;
      const x1 = x0 + xStride;
      let s0 = v128.load(x0);
      let s1 = v128.load(x0, 16);
      let s2 = v128.load(x0, 32);
      let s3 = v128.load(x0, 48);
      let t0 = v128.load(x1);
      let t1 = v128.load(x1, 16);
      let t2 = v128.load(x1, 32);
      let t3 = v128.load(x1, 48);
      let xk = x + <usize>from * xStride + col;
      for (let k = from; k < i; k++) {
        const v0 = v128.load(xk);
        const v1 = v128.load(xk, 16);
        const v2 = v128.load(xk, 32);
        const v3 = v128.load(xk, 48);
        const l0 = v128.load64_splat(l0row + ((<usize>k) << 3));
        const l1 = v128.load64_splat(l1row + ((<usize>k) << 3));
        s0 = f64x2.sub(s0, f64x2.mul(v0, l0));
        s1 = f64x2.sub(s1, f64x2.mul(v1, l0));
        s2 = f64x2.sub(s2, f64x2.mul(v2, l0));
        s3 = f64x2.sub(s3, f64x2.mul(v3, l0));
        t0 = f64x2.sub(t0, f64x2.mul(v0, l1));
        t1 = f64x2.sub(t1, f64x2.mul(v1, l1));
        t2 = f64x2.sub(t2, f64x2.mul(v2, l1));
        t3 = f64x2.sub(t3, f64x2.mul(v3, l1));
        xk += xStride;
      }
      v128.store(x0, s0);
      v128.store(x0, s1, 16);
      v128.store(x0, s2, 32);
      v128.store(x0, s3, 48);
      const l1 = v128.load64_splat(l1row + ((<usize>i) << 3));
      v128.store(x1, f64x2.sub(t0, f64x2.mul(s0, l1)));
      v128.store(x1, f64x2.sub(t1, f64x2.mul(s1, l1)), 16);
      v128.store(x1, f64x2.sub(t2, f64x2.mul(s2, l1)), 32);
      v128.store(x1, f64x2.sub(t3, f64x2.mul(s3, l1)), 48);
    }
    if (i < n) {
      const lrow = lu + <usize>i * luStride;
      const x0 = x + <usize>i * xStride + col;
      let s0 = v128.load(x0);
      let s1 = v128.load(x0, 16);
      let s2 = v128.load(x0, 32);
      let s3 = v128.load(x0, 48);
      let xk = x + <usize>from * xStride + col;
      for (let k = from; k < i; k++) {
        const l = v128.load64_splat(lrow + ((<usize>k) << 3));
        s0 = f64x2.sub(s0, f64x2.mul(v128.load(xk), l));
        s1 = f64x2.sub(s1, f64x2.mul(v128.load(xk, 16), l));
        s2 = f64x2.sub(s2, f64x2.mul(v128.load(xk, 32), l));
        s3 = f64x2.sub(s3, f64x2.mul(v128.load(xk, 48), l));
        xk += xStride;
      }
      v128.store(x0, s0);
      v128.store(x0, s1, 16);
      v128.store(x0, s2, 32);
      v128.store(x0, s3, 48);
    }
  }
}

/**
 * Back substitution with the upper factor: for every row i, descending,
 * x[i] -= x[k] * U[i][k] for k from n - 1 down to i + 1, then x[i] /= U[i][i].
 */
export function backward(
  lu: usize,
  n: i32,
  ldl: i32,
  x: usize,
  ldx: i32,
  q0: i32,
  q1: i32,
): void {
  const luStride = (<usize>ldl) << 3;
  const xStride = (<usize>ldx) << 3;
  for (let q = q0; q < q1; q += 8) {
    const col = (<usize>q) << 3;
    let i = n - 1;
    for (; i >= 1; i -= 2) {
      // Rows i and h = i - 1.
      const uiRow = lu + <usize>i * luStride;
      const uhRow = uiRow - luStride;
      const xi = x + <usize>i * xStride + col;
      const xh = xi - xStride;
      let s0 = v128.load(xi);
      let s1 = v128.load(xi, 16);
      let s2 = v128.load(xi, 32);
      let s3 = v128.load(xi, 48);
      let t0 = v128.load(xh);
      let t1 = v128.load(xh, 16);
      let t2 = v128.load(xh, 32);
      let t3 = v128.load(xh, 48);
      let xk = x + <usize>(n - 1) * xStride + col;
      for (let k = n - 1; k > i; k--) {
        const v0 = v128.load(xk);
        const v1 = v128.load(xk, 16);
        const v2 = v128.load(xk, 32);
        const v3 = v128.load(xk, 48);
        const ui = v128.load64_splat(uiRow + ((<usize>k) << 3));
        const uh = v128.load64_splat(uhRow + ((<usize>k) << 3));
        s0 = f64x2.sub(s0, f64x2.mul(v0, ui));
        s1 = f64x2.sub(s1, f64x2.mul(v1, ui));
        s2 = f64x2.sub(s2, f64x2.mul(v2, ui));
        s3 = f64x2.sub(s3, f64x2.mul(v3, ui));
        t0 = f64x2.sub(t0, f64x2.mul(v0, uh));
        t1 = f64x2.sub(t1, f64x2.mul(v1, uh));
        t2 = f64x2.sub(t2, f64x2.mul(v2, uh));
        t3 = f64x2.sub(t3, f64x2.mul(v3, uh));
        xk -= xStride;
      }
      const di = v128.load64_splat(uiRow + ((<usize>i) << 3));
      s0 = f64x2.div(s0, di);
      s1 = f64x2.div(s1, di);
      s2 = f64x2.div(s2, di);
      s3 = f64x2.div(s3, di);
      v128.store(xi, s0);
      v128.store(xi, s1, 16);
      v128.store(xi, s2, 32);
      v128.store(xi, s3, 48);
      const uh = v128.load64_splat(uhRow + ((<usize>i) << 3));
      const dh = v128.load64_splat(uhRow + ((<usize>(i - 1)) << 3));
      v128.store(xh, f64x2.div(f64x2.sub(t0, f64x2.mul(s0, uh)), dh));
      v128.store(xh, f64x2.div(f64x2.sub(t1, f64x2.mul(s1, uh)), dh), 16);
      v128.store(xh, f64x2.div(f64x2.sub(t2, f64x2.mul(s2, uh)), dh), 32);
      v128.store(xh, f64x2.div(f64x2.sub(t3, f64x2.mul(s3, uh)), dh), 48);
    }
    if (i === 0) {
      const x0 = x + col;
      let s0 = v128.load(x0);
      let s1 = v128.load(x0, 16);
      let s2 = v128.load(x0, 32);
      let s3 = v128.load(x0, 48);
      let xk = x + <usize>(n - 1) * xStride + col;
      for (let k = n - 1; k > 0; k--) {
        const u = v128.load64_splat(lu + ((<usize>k) << 3));
        s0 = f64x2.sub(s0, f64x2.mul(v128.load(xk), u));
        s1 = f64x2.sub(s1, f64x2.mul(v128.load(xk, 16), u));
        s2 = f64x2.sub(s2, f64x2.mul(v128.load(xk, 32), u));
        s3 = f64x2.sub(s3, f64x2.mul(v128.load(xk, 48), u));
        xk -= xStride;
      }
      const d = v128.load64_splat(lu);
      v128.store(x0, f64x2.div(s0, d));
      v128.store(x0, f64x2.div(s1, d), 16);
      v128.store(x0, f64x2.div(s2, d), 32);
      v128.store(x0, f64x2.div(s3, d), 48);
    }
  }
}

/**
 * The identity-right-hand-side solve of src/dc/lu.js (solveIdentity): `y`
 * holds the identity with row stride ldx; forward substitution touches, in the
 * column tile starting at q, only the rows from q down (the rows above are
 * zero there), then the columns are moved to x[i][pivots[q]].
 */
export function forwardIdentity(
  lu: usize,
  n: i32,
  ldl: i32,
  y: usize,
  ldx: i32,
): void {
  for (let q = 0; q < n; q += 8) {
    forward(lu, n, ldl, y, ldx, q, q + 8, q);
  }
}

/** x[i][pivots[q]] = y[i][q] for an n × n block, both with row stride ldx. */
export function scatterColumns(
  y: usize,
  x: usize,
  n: i32,
  ldx: i32,
  pivots: usize,
): void {
  const stride = (<usize>ldx) << 3;
  for (let i = 0; i < n; i++) {
    const yi = y + <usize>i * stride;
    const xi = x + <usize>i * stride;
    for (let q = 0; q < n; q++) {
      const target = <usize>load<i32>(pivots + ((<usize>q) << 2));
      store<f64>(xi + (target << 3), load<f64>(yi + ((<usize>q) << 3)));
    }
  }
}
