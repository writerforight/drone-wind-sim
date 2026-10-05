// Compare the JS port with the Python simulator on deterministic scenarios (no turbulence, ideal sensors).
// Run:  python web/parity_ref.py > /tmp/ref.json && node web/parity_test.js /tmp/ref.json
const fs = require('fs');
const S = require('./sim.js');
const ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let worst = 0;
for (const c of ref.cases) {
  const af = S.PRESETS[c.airframe]();
  if (c.health) for (const [i, h] of c.health) af.rotors[i].motor.health = h;
  const wind = new S.WindField({ meanSpeed: c.wind, directionDeg: 250, gusts: [new S.Gust(6, 2.5, 7, 180)] });
  const sim = new S.Simulation(af, new S.Atmosphere(c.temp), wind, new S.Sensors('ideal'));
  sim.reset(c.start);
  const ctrl = new S.CascadedPID(S.PRESETS[c.airframe](), c.gains);
  const guid = new S.PathFollower(S.figureEight([0, 0, 10], 15, 8), 4);
  let maxd = 0;
  for (let k = 0; k < c.p.length; k++) {
    const d = Math.hypot(...sim.state.p.map((x, i) => x - c.p[k][i]));
    maxd = Number.isFinite(d) ? Math.max(maxd, d) : Infinity;
    const est = sim.estimate();
    sim.step(ctrl.update(est, guid.reference(sim.t, est.p), sim.ctrlDt));
  }
  worst = Math.max(worst, maxd);
  console.log(`${c.airframe.padEnd(16)} ${c.gains.padEnd(8)} wind ${c.wind} ${c.temp}°C${c.health ? ' motor fault' : ''}: max position difference ${maxd.toExponential(2)} m over ${(c.p.length / 100).toFixed(0)} s`);
}
if (worst > 1e-6) { console.error('PARITY FAILED'); process.exit(1); }
console.log('parity ok');
