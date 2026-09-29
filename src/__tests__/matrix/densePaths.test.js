import { describe, expect, it } from 'vitest';

import { Matrix, MatrixTransposeView, wrap } from '../..';

// Matrix methods read and write the Float64Array rows of a Matrix directly and
// go through get/set for other AbstractMatrix subclasses. Both paths must give
// the same results: here the same values are given as a Matrix and as a view
// (the transpose of the transpose), which only has get/set.

function sample(rows, columns) {
  const values = [0, -0, 1, -2.5, 3e-300, NaN, 7, Infinity, 0.1, -9];
  const matrix = new Matrix(rows, columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      matrix.set(i, j, values[(i * 7 + j * 3) % values.length] + i - j / 3);
    }
  }
  return matrix;
}

function asView(matrix) {
  return new MatrixTransposeView(new MatrixTransposeView(matrix));
}

function plain(value) {
  if (value && typeof value.to2DArray === 'function') {
    return {
      rows: value.rows,
      columns: value.columns,
      data: value.to2DArray(),
    };
  }
  return value;
}

// Each case returns a result, possibly after mutating its matrix.
const cases = {
  to1DArray: (m) => m.to1DArray(),
  to2DArray: (m) => m.to2DArray(),
  clone: (m) => m.clone(),
  transpose: (m) => m.transpose(),
  copyConstructor: (m) => new Matrix(m),
  isSymmetric: (m) => m.isSymmetric(),
  getRow: (m) => m.getRow(1),
  getColumn: (m) => m.getColumn(2),
  setRow: (m) => m.setRow(1, m.getRow(0)),
  setColumn: (m) => m.setColumn(0, m.getColumn(2)),
  swapRows: (m) => m.swapRows(0, 2),
  swapColumns: (m) => m.swapColumns(0, 2),
  addRowVector: (m) => m.addRowVector(m.getRow(0)),
  subColumnVector: (m) => m.subColumnVector(m.getColumn(1)),
  mulRowVector: (m) => m.mulRowVector(m.getRow(1)),
  divColumnVector: (m) => m.divColumnVector(m.getColumn(0)),
  maxMin: (m) => [m.max(), m.min(), m.max('row'), m.min('column')],
  maxMinIndex: (m) => [m.maxIndex(), m.minIndex()],
  cumulativeSum: (m) => m.cumulativeSum(),
  dot: (m) => m.dot(m),
  kroneckerProduct: (m) => m.kroneckerProduct(m),
  subMatrix: (m) => m.subMatrix(1, 3, 0, 2),
  selection: (m) => m.selection([2, 0], [3, 1, 1]),
  setSubMatrix: (m) => m.setSubMatrix([[1, 2]], 1, 1),
  copy: (m) => Matrix.copy(m, new Matrix(m.rows, m.columns)),
  sum: (m) => [m.sum(), m.sum('row'), m.sum('column')],
  product: (m) => [m.product(), m.product('row'), m.product('column')],
  variance: (m) => [m.variance(), m.variance('row'), m.variance('column')],
  center: (m) => m.center('column'),
  scale: (m) => m.scale('row'),
  scaleAll: (m) => m.scale(),
  addScalar: (m) => m.add(0.5),
  subMatrixArgument: (m) => m.sub(m.clone().mul(3)),
  pow: (m) => m.pow(2),
  abs: (m) => m.abs(),
  mmul: (m) => m.mmul(m.transpose()),
};

describe('Matrix and views give identical results', () => {
  it.each(Object.keys(cases))('%s', (name) => {
    const dense = sample(4, 5);
    const view = asView(sample(4, 5));
    const fromDense = plain(cases[name](dense));
    const fromView = plain(cases[name](view));
    expect(fromDense).toStrictEqual(fromView);
    expect(dense.to2DArray()).toStrictEqual(view.to2DArray());
  });

  it('elementwise operations accept any operand type', () => {
    const matrix = sample(3, 3);
    const other = sample(3, 3).mul(2);
    const expected = matrix.clone().add(other).to2DArray();
    expect(matrix.clone().add(asView(other)).to2DArray()).toStrictEqual(
      expected,
    );
    expect(matrix.clone().add(other.to2DArray()).to2DArray()).toStrictEqual(
      expected,
    );
    expect(
      matrix.clone().add(wrap(other.to2DArray())).to2DArray(),
    ).toStrictEqual(expected);
  });
});
