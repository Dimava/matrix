import { describe, expectTypeOf, it } from 'vitest';

import {
  AbstractMatrix,
  DistanceMatrix,
  Matrix,
  MatrixTransposeView,
  SymmetricMatrix,
  WrapperMatrix2D,
} from '../../../matrix';

describe('clone', () => {
  it('returns a Matrix, except for symmetric matrices', () => {
    const matrix = new Matrix(2, 2);
    expectTypeOf(matrix.clone()).toEqualTypeOf<Matrix>();
    expectTypeOf(
      new MatrixTransposeView(matrix).clone(),
    ).toEqualTypeOf<Matrix>();
    expectTypeOf(new WrapperMatrix2D([[1]]).clone()).toEqualTypeOf<Matrix>();
    expectTypeOf(
      new SymmetricMatrix(2).clone(),
    ).toEqualTypeOf<SymmetricMatrix>();
    expectTypeOf(new DistanceMatrix(2).clone()).toEqualTypeOf<DistanceMatrix>();
    expectTypeOf(
      (matrix as AbstractMatrix).clone(),
    ).toEqualTypeOf<AbstractMatrix>();
  });
});
