/** Seeded random numbers, so every noisy or randomised run can be repeated exactly. */
export function createRandom(seed = 1) {
  let a = seed >>> 0;
  const uniform = () => {                       // mulberry32
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {                         // Box–Muller, standard normal
    let u = 0;
    while (u === 0) u = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
  };
  return { uniform, gauss, range: (lo, hi) => lo + (hi - lo) * uniform() };
}
