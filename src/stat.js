import { newArray } from './util';

// Every function receives `data`, the Float64Array rows of a Matrix, to read
// (and write) directly, or `undefined` for other subclasses, which go through
// get (and set) as before. Both loops accumulate every value in the same order.

export function sumByRow(matrix, data) {
  let sum = newArray(matrix.rows);
  if (data) {
    for (let i = 0; i < matrix.rows; ++i) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; ++j) {
        sum[i] += row[j];
      }
    }
    return sum;
  }
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[i] += matrix.get(i, j);
    }
  }
  return sum;
}

export function sumByColumn(matrix, data) {
  let sum = newArray(matrix.columns);
  if (data) {
    for (let i = 0; i < matrix.rows; ++i) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; ++j) {
        sum[j] += row[j];
      }
    }
    return sum;
  }
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[j] += matrix.get(i, j);
    }
  }
  return sum;
}

export function sumAll(matrix, data) {
  let v = 0;
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        v += row[j];
      }
    }
    return v;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      v += matrix.get(i, j);
    }
  }
  return v;
}

export function productByRow(matrix, data) {
  let sum = newArray(matrix.rows, 1);
  if (data) {
    for (let i = 0; i < matrix.rows; ++i) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; ++j) {
        sum[i] *= row[j];
      }
    }
    return sum;
  }
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[i] *= matrix.get(i, j);
    }
  }
  return sum;
}

export function productByColumn(matrix, data) {
  let sum = newArray(matrix.columns, 1);
  if (data) {
    for (let i = 0; i < matrix.rows; ++i) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; ++j) {
        sum[j] *= row[j];
      }
    }
    return sum;
  }
  for (let i = 0; i < matrix.rows; ++i) {
    for (let j = 0; j < matrix.columns; ++j) {
      sum[j] *= matrix.get(i, j);
    }
  }
  return sum;
}

export function productAll(matrix, data) {
  let v = 1;
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        v *= row[j];
      }
    }
    return v;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      v *= matrix.get(i, j);
    }
  }
  return v;
}

export function varianceByRow(matrix, data, unbiased, mean) {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const variance = [];

  for (let i = 0; i < rows; i++) {
    let sum1 = 0;
    let sum2 = 0;
    let x = 0;
    if (data) {
      const row = data[i];
      for (let j = 0; j < cols; j++) {
        x = row[j] - mean[i];
        sum1 += x;
        sum2 += x * x;
      }
    } else {
      for (let j = 0; j < cols; j++) {
        x = matrix.get(i, j) - mean[i];
        sum1 += x;
        sum2 += x * x;
      }
    }
    if (unbiased) {
      variance.push((sum2 - (sum1 * sum1) / cols) / (cols - 1));
    } else {
      variance.push((sum2 - (sum1 * sum1) / cols) / cols);
    }
  }
  return variance;
}

export function varianceByColumn(matrix, data, unbiased, mean) {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const variance = [];

  if (data) {
    // One pair of sums per column, filled row by row so the rows are scanned
    // in memory order; each column still sums its rows from first to last.
    const sum1 = zeros(cols);
    const sum2 = zeros(cols);
    for (let i = 0; i < rows; i++) {
      const row = data[i];
      for (let j = 0; j < cols; j++) {
        const x = row[j] - mean[j];
        sum1[j] += x;
        sum2[j] += x * x;
      }
    }
    for (let j = 0; j < cols; j++) {
      if (unbiased) {
        variance.push((sum2[j] - (sum1[j] * sum1[j]) / rows) / (rows - 1));
      } else {
        variance.push((sum2[j] - (sum1[j] * sum1[j]) / rows) / rows);
      }
    }
    return variance;
  }

  for (let j = 0; j < cols; j++) {
    let sum1 = 0;
    let sum2 = 0;
    let x = 0;
    for (let i = 0; i < rows; i++) {
      x = matrix.get(i, j) - mean[j];
      sum1 += x;
      sum2 += x * x;
    }
    if (unbiased) {
      variance.push((sum2 - (sum1 * sum1) / rows) / (rows - 1));
    } else {
      variance.push((sum2 - (sum1 * sum1) / rows) / rows);
    }
  }
  return variance;
}

export function varianceAll(matrix, data, unbiased, mean) {
  const rows = matrix.rows;
  const cols = matrix.columns;
  const size = rows * cols;

  let sum1 = 0;
  let sum2 = 0;
  let x = 0;
  if (data) {
    for (let i = 0; i < rows; i++) {
      const row = data[i];
      for (let j = 0; j < cols; j++) {
        x = row[j] - mean;
        sum1 += x;
        sum2 += x * x;
      }
    }
  } else {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        x = matrix.get(i, j) - mean;
        sum1 += x;
        sum2 += x * x;
      }
    }
  }
  if (unbiased) {
    return (sum2 - (sum1 * sum1) / size) / (size - 1);
  } else {
    return (sum2 - (sum1 * sum1) / size) / size;
  }
}

export function centerByRow(matrix, mean, data) {
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        row[j] = row[j] - mean[i];
      }
    }
    return;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) - mean[i]);
    }
  }
}

export function centerByColumn(matrix, mean, data) {
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        row[j] = row[j] - mean[j];
      }
    }
    return;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) - mean[j]);
    }
  }
}

export function centerAll(matrix, mean, data) {
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        row[j] = row[j] - mean;
      }
    }
    return;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) - mean);
    }
  }
}

export function getScaleByRow(matrix, data) {
  const scale = [];
  for (let i = 0; i < matrix.rows; i++) {
    let sum = 0;
    if (data) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        sum += row[j] ** 2 / (matrix.columns - 1);
      }
    } else {
      for (let j = 0; j < matrix.columns; j++) {
        sum += matrix.get(i, j) ** 2 / (matrix.columns - 1);
      }
    }
    scale.push(Math.sqrt(sum));
  }
  return scale;
}

export function scaleByRow(matrix, scale, data) {
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        row[j] = row[j] / scale[i];
      }
    }
    return;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) / scale[i]);
    }
  }
}

export function getScaleByColumn(matrix, data) {
  const scale = [];
  if (data) {
    const rows = matrix.rows;
    const columns = matrix.columns;
    // One sum per column, filled row by row (see varianceByColumn).
    const sum = zeros(columns);
    for (let i = 0; i < rows; i++) {
      const row = data[i];
      for (let j = 0; j < columns; j++) {
        sum[j] += row[j] ** 2 / (rows - 1);
      }
    }
    for (let j = 0; j < columns; j++) {
      scale.push(Math.sqrt(sum[j]));
    }
    return scale;
  }
  for (let j = 0; j < matrix.columns; j++) {
    let sum = 0;
    for (let i = 0; i < matrix.rows; i++) {
      sum += matrix.get(i, j) ** 2 / (matrix.rows - 1);
    }
    scale.push(Math.sqrt(sum));
  }
  return scale;
}

export function scaleByColumn(matrix, scale, data) {
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        row[j] = row[j] / scale[j];
      }
    }
    return;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) / scale[j]);
    }
  }
}

export function getScaleAll(matrix, data) {
  const divider = matrix.size - 1;
  let sum = 0;
  // Column by column, the order the sum has always been accumulated in.
  for (let j = 0; j < matrix.columns; j++) {
    for (let i = 0; i < matrix.rows; i++) {
      sum += (data ? data[i][j] : matrix.get(i, j)) ** 2 / divider;
    }
  }
  return Math.sqrt(sum);
}

export function scaleAll(matrix, scale, data) {
  if (data) {
    for (let i = 0; i < matrix.rows; i++) {
      const row = data[i];
      for (let j = 0; j < matrix.columns; j++) {
        row[j] = row[j] / scale;
      }
    }
    return;
  }
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      matrix.set(i, j, matrix.get(i, j) / scale);
    }
  }
}

/**
 * A plain array of zeros, for per-column sums: a Float64Array of more than 8
 * values is allocated outside the JS heap, which costs more than summing a
 * small matrix.
 * @param {number} length
 * @returns {number[]}
 */
function zeros(length) {
  const array = [];
  for (let i = 0; i < length; i++) {
    array.push(0);
  }
  return array;
}
