/**
 * input/ — keyboard, gamepad (Gamepad API) and touch joysticks, merged into one stick state:
 *
 *   { roll, pitch, yaw, throttle }  each in [−1, 1]   (mode 2: left stick throttle/yaw, right stick pitch/roll)
 *
 * Positive roll = right, positive pitch = forward, positive yaw = turn left (counter-clockwise from above),
 * throttle 0 = hover thrust. A mode decides what the sticks mean (angle mode maps them to tilt angles).
 *
 *   keyboard   W/S pitch · A/D roll · ↑/↓ throttle · ←/→ yaw   (Shift = finer)
 *   gamepad    standard mapping, left stick (axes 0, 1), right stick (axes 2, 3)
 *   touch      two on-screen sticks (pointer events), created by attachTouchSticks()
 */
const clamp = (x) => Math.max(-1, Math.min(1, x));
const dead = (x, d = 0.08) => (Math.abs(x) < d ? 0 : (x - Math.sign(x) * d) / (1 - d));

export function createInput() {
  const keys = new Set();
  const touch = { left: [0, 0], right: [0, 0] };
  const down = (e) => {
    if (/input|select|textarea/i.test(e.target.tagName)) return;
    keys.add(e.code);
    if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
  };
  window.addEventListener('keydown', down);
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => keys.clear());

  return {
    touch,
    /** The merged stick state right now. */
    read() {
      const k = (a, b) => (keys.has(a) ? 1 : 0) - (keys.has(b) ? 1 : 0);
      const f = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 0.4 : 1;
      let roll = f * k('KeyD', 'KeyA'), pitch = f * k('KeyW', 'KeyS'), yaw = f * k('ArrowLeft', 'ArrowRight'), thr = f * k('ArrowUp', 'ArrowDown');
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      for (const gp of pads) {
        if (!gp || gp.axes.length < 4) continue;
        thr += dead(-gp.axes[1]); yaw += dead(-gp.axes[0]); pitch += dead(-gp.axes[3]); roll += dead(gp.axes[2]);
        break;
      }
      thr += touch.left[1]; yaw += -touch.left[0]; pitch += touch.right[1]; roll += touch.right[0];
      return { roll: clamp(roll), pitch: clamp(pitch), yaw: clamp(yaw), throttle: clamp(thr) };
    },
    gamepadName() {
      const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
      return pads.length ? pads[0].id : null;
    },
  };
}

/** Two virtual sticks for touch screens (any pointer: finger, pen or mouse). They spring back to centre. */
export function attachTouchSticks(container, input) {
  const make = (side) => {
    const base = document.createElement('div');
    base.className = `stick stick-${side}`;
    base.innerHTML = '<div class="knob"></div>';
    container.appendChild(base);
    const knob = base.firstChild, out = input.touch[side];
    let id = null;
    const move = (e) => {
      const r = base.getBoundingClientRect(), R = r.width / 2;
      let dx = e.clientX - (r.left + R), dy = e.clientY - (r.top + R);
      const d = Math.hypot(dx, dy);
      if (d > R) { dx *= R / d; dy *= R / d; }
      knob.style.transform = `translate(${dx}px, ${dy}px)`;
      out[0] = dx / R; out[1] = -dy / R;
    };
    base.addEventListener('pointerdown', (e) => { id = e.pointerId; base.setPointerCapture(id); move(e); });
    base.addEventListener('pointermove', (e) => { if (e.pointerId === id) move(e); });
    const end = (e) => { if (e.pointerId !== id) return; id = null; knob.style.transform = ''; out[0] = 0; out[1] = 0; };
    base.addEventListener('pointerup', end);
    base.addEventListener('pointercancel', end);
    return base;
  };
  return [make('left'), make('right')];
}
