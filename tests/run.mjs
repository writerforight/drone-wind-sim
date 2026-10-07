// Run every tests/*.test.mjs:   node tests/run.mjs        (no dependencies)
// Each test file exports `tests`: { 'name': () => { ... throws on failure ... } }.
import { readdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
let failed = 0, passed = 0;
for (const f of readdirSync(here).filter((n) => n.endsWith('.test.mjs')).sort()) {
  const { tests } = await import(pathToFileURL(join(here, f)).href);
  for (const [name, fn] of Object.entries(tests)) {
    const t0 = performance.now();
    try { await fn(); passed++; console.log(`ok    ${f.replace('.test.mjs', '')} · ${name}  (${(performance.now() - t0).toFixed(0)} ms)`); }
    catch (e) { failed++; console.log(`FAIL  ${f.replace('.test.mjs', '')} · ${name}\n      ${e.message}`); }
  }
}
console.log(failed ? `${failed} failed, ${passed} passed` : `all ${passed} tests passed`);
process.exit(failed ? 1 : 0);
