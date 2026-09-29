import { describe, expect, it } from 'vitest';

import {
  CHO,
  DistanceMatrix,
  LU,
  Matrix,
  QR,
  SymmetricMatrix,
  determinant,
  inverse,
} from '../..';

// SymmetricMatrix.set writes both (i, j) and (j, i), so its clone cannot hold
// the non-symmetric intermediate values of a decomposition. Every result must
// be the one for a plain Matrix with the same values.

const values = [
  [4, 1, 2, 0.5],
  [1, 5, 3, 1],
  [2, 3, 6, 2],
  [0.5, 1, 2, 7],
];
const plain = new Matrix(values);
const symmetric = new SymmetricMatrix(values);

describe('decompositions of a SymmetricMatrix', () => {
  it('LU, determinant and inverse', () => {
    expect(new LU(symmetric).LU.to2DArray()).toStrictEqual(
      new LU(plain).LU.to2DArray(),
    );
    expect(determinant(symmetric)).toBe(determinant(plain));
    expect(inverse(symmetric).to2DArray()).toStrictEqual(
      inverse(plain).to2DArray(),
    );
  });

  it('QR', () => {
    const qr = new QR(symmetric);
    const expected = new QR(plain);
    expect(qr.orthogonalMatrix.to2DArray()).toStrictEqual(
      expected.orthogonalMatrix.to2DArray(),
    );
    expect(qr.upperTriangularMatrix.to2DArray()).toStrictEqual(
      expected.upperTriangularMatrix.to2DArray(),
    );
  });

  it('as the right-hand side of QR and Cholesky solves', () => {
    const qr = new QR(plain);
    expect(qr.solve(symmetric).to2DArray()).toStrictEqual(
      qr.solve(plain).to2DArray(),
    );
    const cholesky = new CHO(plain);
    expect(cholesky.solve(symmetric).to2DArray()).toStrictEqual(
      cholesky.solve(plain).to2DArray(),
    );
  });

  it('echelonForm', () => {
    expect(symmetric.echelonForm().to2DArray()).toStrictEqual(
      plain.echelonForm().to2DArray(),
    );
  });

  it('inverse of a DistanceMatrix', () => {
    const distances = [
      [0, 1, 2],
      [1, 0, 3],
      [2, 3, 0],
    ];
    expect(inverse(new DistanceMatrix(distances)).to2DArray()).toStrictEqual(
      inverse(new Matrix(distances)).to2DArray(),
    );
  });
});
