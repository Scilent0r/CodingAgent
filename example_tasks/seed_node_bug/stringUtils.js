function capitalize(str) {
  // BUG: throws on empty string instead of returning '' — str[0] is
  // undefined for '', and undefined.toUpperCase() throws.
  return str[0].toUpperCase() + str.slice(1);
}

module.exports = { capitalize };
