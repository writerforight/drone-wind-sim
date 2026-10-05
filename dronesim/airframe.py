"""Airframe: rigid body + a list of rotors at arbitrary positions.

Body frame is FLU (x forward, y left, z up); every rotor pushes along body +z.
A rotor i at position r_i = (x_i, y_i, z_i) with thrust T_i and spin s_i (+1 = counter-clockwise seen
from above) produces

    force   (0, 0, T_i)
    torque  r_i x (0, 0, T_i) + (0, 0, -s_i k_i T_i)        k_i = Q_i / T_i  (reaction torque)

Stacking these for all rotors gives the allocation (mixer) matrix B:

    [F_z, tau_x, tau_y, tau_z]^T = B [T_1 ... T_n]^T,   B[:, i] = (1, y_i, -x_i, -s_i k_i)

so any geometry - quad, hexa, octo, asymmetric, a frame with a moved battery - works with the same
controller.  The inertia is built from a central body plus point masses at the motors unless it is
given explicitly.
"""
from dataclasses import dataclass, field

import numpy as np

from .motor import MotorModel


@dataclass
class Rotor:
    position: np.ndarray            # [m] in the body frame, relative to the centre of mass
    spin: int                       # +1 counter-clockwise (seen from above), -1 clockwise
    motor: MotorModel = field(default_factory=MotorModel)


@dataclass
class Airframe:
    name: str
    mass: float                                 # [kg]
    rotors: list
    inertia: np.ndarray = None                  # 3x3 [kg m^2]; computed if None
    drag_area: np.ndarray = field(default_factory=lambda: np.array([0.012, 0.012, 0.03]))   # C_d*A per body axis [m^2]
    rotor_drag: float = 1.0e-3                  # rotor (blade-flapping) drag per rotor speed [N s/m per rev/s]
    ang_damping: np.ndarray = None              # aerodynamic rotational damping [N m s]; None = 0.07 * diag(J)
    body_mass_fraction: float = 0.6             # for the inertia estimate: share of mass in the central body
    body_size: tuple = (0.15, 0.10, 0.06)       # central body box [m]

    def __post_init__(self):
        if self.inertia is None:
            self.inertia = self._estimate_inertia()
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        if self.ang_damping is None:
            # scales with size: the same decay rate (~0.07 1/s) for a 27 g and a 3 kg drone
            self.ang_damping = 0.07 * np.diag(self.inertia)

    # -------------------------------------------------------------------------------------------
    def _estimate_inertia(self):
        m_body = self.body_mass_fraction * self.mass
        a, b, c = self.body_size
        J = m_body / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        m_rot = (1.0 - self.body_mass_fraction) * self.mass / len(self.rotors)
        for r in self.rotors:
            p = np.asarray(r.position, dtype=float)
            J += m_rot * (p @ p * np.eye(3) - np.outer(p, p))
        return J

    @property
    def n(self):
        return len(self.rotors)

    def allocation(self):
        """4 x n matrix mapping rotor thrusts to (total thrust, roll, pitch, yaw torque)."""
        B = np.zeros((4, self.n))
        for i, r in enumerate(self.rotors):
            x, y, _ = r.position
            B[:, i] = (1.0, y, -x, -r.spin * r.motor.torque_per_thrust())
        return B

    def hover_thrust(self, g=9.80665):
        return self.mass * g

    def thrust_to_weight(self, rho=1.225, g=9.80665):
        return sum(r.motor.max_thrust(rho) for r in self.rotors) / (self.mass * g)

    def summary(self, rho=1.225):
        return (f"{self.name}: {self.n} rotors, mass {self.mass:.2f} kg, thrust/weight {self.thrust_to_weight(rho):.2f}, "
                f"inertia diag [{', '.join(f'{v:.3g}' for v in np.diag(self.inertia))}] kg m²")


# -------------------------------------------------------------------------------------------------
# Presets
# -------------------------------------------------------------------------------------------------

def ring(n, arm, start_deg, spins, motor=None, z=0.0):
    """n rotors evenly on a circle of radius ``arm``; spins is a list of +1/-1 repeated around the ring."""
    motor = motor or MotorModel()
    rotors = []
    for i in range(n):
        a = np.radians(start_deg + 360.0 * i / n)
        rotors.append(Rotor(np.array([arm * np.cos(a), arm * np.sin(a), z]), spins[i % len(spins)], motor.copy()))
    return rotors


def quad_x(mass=1.5, arm=0.225, motor=None, **kw):
    """Quadrotor in X configuration (450 mm class)."""
    return Airframe("quad-x", mass, ring(4, arm, 45.0, [-1, 1], motor), **kw)


def quad_plus(mass=1.5, arm=0.225, motor=None, **kw):
    return Airframe("quad-+", mass, ring(4, arm, 0.0, [-1, 1], motor), **kw)


def hexa_x(mass=2.2, arm=0.275, motor=None, **kw):
    return Airframe("hexa-x", mass, ring(6, arm, 30.0, [-1, 1], motor), **kw)


def octo_x(mass=3.2, arm=0.35, motor=None, **kw):
    return Airframe("octo-x", mass, ring(8, arm, 22.5, [-1, 1], motor), **kw)


def quad_asymmetric(mass=1.6, motor=None, **kw):
    """Quad with long front arms and short rear arms, and the centre of mass shifted forward
    (e.g. a camera in the nose) - the rotors no longer share the load equally."""
    motor = motor or MotorModel()
    pos = [(0.20, -0.26), (0.20, 0.26), (-0.22, 0.17), (-0.22, -0.17)]     # FR, FL, RL, RR (before CoM shift)
    com = np.array([0.04, 0.0])                                             # CoM 4 cm forward of the frame centre
    spins = [-1, 1, -1, 1]
    rotors = [Rotor(np.array([x - com[0], y - com[1], 0.0]), s, motor.copy()) for (x, y), s in zip(pos, spins)]
    return Airframe("quad-asym", mass, rotors, **kw)


PRESETS = {"quad_x": quad_x, "quad_plus": quad_plus, "hexa_x": hexa_x, "octo_x": octo_x,
           "quad_asymmetric": quad_asymmetric}


def make(name, **kw):
    """Any generic preset above, or a real drone with published parameters: 'iris', 'x500', 'crazyflie'."""
    from .presets_real import REAL
    return {**PRESETS, **REAL}[name](**kw)


def all_names():
    from .presets_real import REAL
    return list(PRESETS) + list(REAL)
