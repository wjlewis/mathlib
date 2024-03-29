export function zip(xs, ys, fn = (x, y) => [x, y]) {
  const out = new Array(Math.min(xs.length, ys.length));
  for (let i = 0; i < out.length; i++) {
    out[i] = fn(xs[i], ys[i]);
  }
  return out;
}

export function range(lo, hi) {
  const len = hi - lo;
  const out = new Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = lo + i;
  }
  return out;
}

export function uniform(min, max) {
  return min + Math.random() * (max - min);
}
