const { defineConfig } = require('vitest/config');

module.exports = defineConfig({
  test: {
    typecheck: { enabled: true },
  },
});
