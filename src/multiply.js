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
