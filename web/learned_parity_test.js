// Fly an exported learned loop in the JS simulator and compare with the Python simulator (same flights).
// Run:  python web/learned_parity_ref.py web/models/X.json > /tmp/lref.json && node web/learned_parity_test.js web/models/X.json /tmp/lref.json
const fs = require('fs');
const S = require('./sim.js');
const L = require('./learned.js');
const m = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const ref = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const net = new L.LearnedNet(m);
console.log(`self-check vs PyTorch: ${net.selfCheck().toExponential(1)}`);
let worst = 0;
for (const c of ref.cases) {
  const wind = new S.WindField({ meanSpeed: c.wind, directionDeg: 250, gusts: [new S.Gust(3, 2, 4, 180)] });
  const sim = new S.Simulation(S.PRESETS[m.airframe](), new S.Atmosphere(c.temp), wind, new S.Sensors('ideal'), 0.002, m.ctrl_dt);
  sim.reset([0, 0, 10]);
  const ctrl = new L.LearnedController(net);
  let maxd = 0;
  for (let k = 0; k < c.p.length; k++) {
    const d = Math.hypot(...sim.state.p.map((x, i) => x - c.p[k][i]));
    maxd = Number.isFinite(d) ? Math.max(maxd, d) : Infinity;
    sim.step(ctrl.update(sim.estimate(), { p: c.target, v: [0, 0, 0] }, sim.ctrlDt));
  }
  worst = Math.max(worst, maxd);
  console.log(`wind ${c.wind} m/s, ${c.temp} °C, target [${c.target}]: max position difference ${maxd.toExponential(2)} m over ${(c.p.length * m.ctrl_dt).toFixed(0)} s`);
}
// Math.tanh / Math.exp and torch differ in the last bit; a feedback loop amplifies that, so 0.1 mm, not 1e-13
if (worst > 1e-4) { console.error('PARITY FAILED'); process.exit(1); }
console.log('parity ok');
