import fs from 'fs';

const operators = [
  // Arithmetic operators
  ['+', 'add'],
  ['-', 'sub', 'subtract'],
  ['*', 'mul', 'multiply'],
  ['/', 'div', 'divide'],
  ['%', 'mod', 'modulus'],
  // Bitwise operators
  ['&', 'and'],
  ['|', 'or'],
  ['^', 'xor'],
  ['<<', 'leftShift'],
  ['>>', 'signPropagatingRightShift'],
  ['>>>', 'rightShift', 'zeroFillRightShift'],
];

const methods = [['~', 'not']];
[
  'abs',
  'acos',
  'acosh',
  'asin',
  'asinh',
  'atan',
  'atanh',
  'cbrt',
  'ceil',
  'clz32',
  'cos',
  'cosh',
  'exp',
  'expm1',
  'floor',
  'fround',
  'log',
  'log1p',
  'log10',
  'log2',
  'round',
  'sign',
  'sin',
  'sinh',
  'sqrt',
  'tan',
  'tanh',
  'trunc',
].forEach((mathMethod) => {
  methods.push([`Math.${mathMethod}`, mathMethod]);
});

const methodsWithArgs = [['Math.pow', 1, 'pow']];

function fillTemplateFunction(template, values) {
  for (const value in values) {
    template = template.replace(new RegExp(`%${value}%`, 'g'), values[value]);
  }
  return template;
}

const inplaceOperator = `
  AbstractMatrix.prototype.%name% = function %name%(value) {
    if (typeof value === 'number') return this.%name%S(value);
    return this.%name%M(value);
  };
`;

// Matrix keeps one Float64Array per row, so when every operand is a Matrix the
// loop reads and writes the rows directly instead of dispatching through
// get/set per element. Other AbstractMatrix subclasses (views, wrappers) take
// the generic path. Both visit the elements in the same order.
// In `expression`, `X` is the element of `this` and `Y` the element of `matrix`.
function elementLoop(expression, withMatrix) {
  const dense = expression
    .replaceAll('X', 'row[j]')
    .replaceAll('Y', 'other[j]');
  const generic = expression
    .replaceAll('X', 'this.get(i, j)')
    .replaceAll('Y', 'matrix.get(i, j)');
  return `    if (this instanceof Matrix${withMatrix ? ' && matrix instanceof Matrix' : ''}) {
      const rows = this.rows;
      const columns = this.columns;
      for (let i = 0; i < rows; i++) {
        const row = this.data[i];${withMatrix ? '\n        const other = matrix.data[i];' : ''}
        for (let j = 0; j < columns; j++) {
          row[j] = ${dense};
        }
      }
      return this;
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.set(i, j, ${generic});
      }
    }
    return this;`;
}

const checkSameSize = `    matrix = Matrix.checkMatrix(matrix);
    if (this.rows !== matrix.rows ||
      this.columns !== matrix.columns) {
      throw new RangeError('Matrices dimensions must be equal');
    }`;

const inplaceOperatorScalar = `
  AbstractMatrix.prototype.%name% = function %name%(value) {
${elementLoop('X %op% value', false)}
  };
`;

const inplaceOperatorMatrix = `
  AbstractMatrix.prototype.%name% = function %name%(matrix) {
${checkSameSize}
${elementLoop('X %op% Y', true)}
  };
`;

const staticOperator = `
  AbstractMatrix.%name% = function %name%(matrix, value) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.%name%(value);
  };
`;

const inplaceMethod = `
  AbstractMatrix.prototype.%name% = function %name%() {
${elementLoop('%method%(X)', false)}
  };
`;

const staticMethod = `
  AbstractMatrix.%name% = function %name%(matrix) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.%name%();
  };
`;

const inplaceMethodWithArgs = `
  AbstractMatrix.prototype.%name% = function %name%(%args%) {
${elementLoop('%method%(X, %args%)', false)}
  };
`;

const staticMethodWithArgs = `
  AbstractMatrix.%name% = function %name%(matrix, %args%) {
    const newMatrix = new Matrix(matrix);
    return newMatrix.%name%(%args%);
  };
`;

const inplaceMethodWithOneArgScalar = `
  AbstractMatrix.prototype.%name% = function %name%(value) {
${elementLoop('%method%(X, value)', false)}
  };
`;
const inplaceMethodWithOneArgMatrix = `
  AbstractMatrix.prototype.%name% = function %name%(matrix) {
${checkSameSize}
${elementLoop('%method%(X, Y)', true)}
  };
`;

const inplaceMethodWithOneArg = `
  AbstractMatrix.prototype.%name% = function %name%(value) {
    if (typeof value === 'number') return this.%name%S(value);
    return this.%name%M(value);
  };
`;

const staticMethodWithOneArg = staticMethodWithArgs;

const mathOperations = [
  'export function installMathOperations(AbstractMatrix, Matrix) {',
];

for (const operator of operators) {
  mathOperations.push(
    fillTemplateFunction(inplaceOperator, {
      name: operator[1],
      op: operator[0],
    }),
  );
  mathOperations.push(
    fillTemplateFunction(inplaceOperatorScalar, {
      name: `${operator[1]}S`,
      op: operator[0],
    }),
  );
  mathOperations.push(
    fillTemplateFunction(inplaceOperatorMatrix, {
      name: `${operator[1]}M`,
      op: operator[0],
    }),
  );
  mathOperations.push(
    fillTemplateFunction(staticOperator, { name: operator[1] }),
  );
  for (let i = 2; i < operator.length; i++) {
    mathOperations.push(
      `  AbstractMatrix.prototype.${operator[i]} = AbstractMatrix.prototype.${operator[1]};\n`,
    );
    mathOperations.push(
      `  AbstractMatrix.prototype.${operator[i]}S = AbstractMatrix.prototype.${operator[1]}S;\n`,
    );
    mathOperations.push(
      `  AbstractMatrix.prototype.${operator[i]}M = AbstractMatrix.prototype.${operator[1]}M;\n`,
    );
    mathOperations.push(
      `  AbstractMatrix.${operator[i]} = AbstractMatrix.${operator[1]};\n`,
    );
  }
}

for (const method of methods) {
  mathOperations.push(
    fillTemplateFunction(inplaceMethod, {
      name: method[1],
      method: method[0],
    }),
  );
  mathOperations.push(
    fillTemplateFunction(staticMethod, {
      name: method[1],
    }),
  );
  for (let i = 2; i < method.length; i++) {
    mathOperations.push(
      `  AbstractMatrix.prototype.${method[i]} = AbstractMatrix.prototype.${method[1]}`,
    );
    mathOperations.push(
      `  AbstractMatrix.${method[i]} = AbstractMatrix.${method[1]}`,
    );
  }
}

for (const methodWithArg of methodsWithArgs) {
  let args = 'arg0';
  for (let i = 1; i < methodWithArg[1]; i++) {
    args += `, arg${i}`;
  }
  if (methodWithArg[1] !== 1) {
    mathOperations.push(
      fillTemplateFunction(inplaceMethodWithArgs, {
        name: methodWithArg[2],
        method: methodWithArg[0],
        args,
      }),
    );
    mathOperations.push(
      fillTemplateFunction(staticMethodWithArgs, {
        name: methodWithArg[2],
        args,
      }),
    );
    for (let i = 3; i < methodWithArg.length; i++) {
      mathOperations.push(
        `  AbstractMatrix.prototype.${methodWithArg[i]} = AbstractMatrix.prototype.${methodWithArg[2]}`,
      );
      mathOperations.push(
        `  AbstractMatrix.${methodWithArg[i]} = AbstractMatrix.${methodWithArg[2]}`,
      );
    }
  } else {
    const tmplVar = {
      name: methodWithArg[2],
      args,
      method: methodWithArg[0],
    };
    mathOperations.push(fillTemplateFunction(staticMethodWithOneArg, tmplVar));
    mathOperations.push(fillTemplateFunction(inplaceMethodWithOneArg, tmplVar));
    tmplVar.name = `${methodWithArg[2]}S`;
    mathOperations.push(
      fillTemplateFunction(inplaceMethodWithOneArgScalar, tmplVar),
    );
    tmplVar.name = `${methodWithArg[2]}M`;
    mathOperations.push(
      fillTemplateFunction(inplaceMethodWithOneArgMatrix, tmplVar),
    );
    for (let i = 3; i < methodWithArg.length; i++) {
      mathOperations.push(
        `  AbstractMatrix.prototype.${methodWithArg[i]} = AbstractMatrix.prototype.${methodWithArg[2]};\n`,
      );
      mathOperations.push(
        `  AbstractMatrix.${methodWithArg[i]} = AbstractMatrix.${methodWithArg[2]};\n`,
      );
      mathOperations.push(
        `  AbstractMatrix.prototype.${methodWithArg[i]}S = AbstractMatrix.prototype.${methodWithArg[2]}S;\n`,
      );
      mathOperations.push(
        `  AbstractMatrix.prototype.${methodWithArg[i]}M = AbstractMatrix.prototype.${methodWithArg[2]}M;\n`,
      );
    }
  }
}

// Math.pow(x, y) is written x ** y, as the prefer-exponentiation-operator lint
// rule wants.
const result = `${mathOperations.join('')}}\n`.replaceAll(
  /Math\.pow\((?<base>.+?), (?<exponent>value|other\[j\]|matrix\.get\(i, j\))\)/g,
  '$<base> ** $<exponent>',
);
fs.writeFileSync('src/mathOperations.js', result);
