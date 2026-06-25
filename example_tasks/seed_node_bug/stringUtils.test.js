const test = require('node:test');
const assert = require('node:assert');
const { capitalize } = require('./stringUtils');

test('capitalizes first letter', () => {
  assert.strictEqual(capitalize('hello'), 'Hello');
});

test('handles empty string', () => {
  assert.strictEqual(capitalize(''), '');
});

test('handles already-capitalized input', () => {
  assert.strictEqual(capitalize('World'), 'World');
});
