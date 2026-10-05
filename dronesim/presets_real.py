"""Airframes with published parameters, converted into this simulator's model.

Sources (read directly from the model files):
    iris        PX4-SITL_gazebo-classic, models/iris/iris.sdf.jinja           (3DR Iris, PX4's classic SITL drone)
    x500        PX4-gazebo-models, models/x500_base/model.sdf + x500/model.sdf (Holybro X500, PX4's current default)
    crazyflie   gym-pybullet-drones, assets/cf2x.urdf                          (Bitcraze Crazyflie 2.X; values
                from the identification by J. Förster, ETH Zurich 2015)

Conversions
    Gazebo:     T = k_motor w^2,  Q = k_moment T          (w in rad/s, no air density in the formula)
    pybullet:   T = k_f rpm^2,    Q = k_m rpm^2
    here:       T = C_T rho n^2 D^4,  Q = C_Q rho n^2 D^5  (n in rev/s)
    => C_T = k_motor (2 pi)^2 / (rho0 D^4) at rho0 = 1.225, so at standard conditions the thrust is exactly
       the source's, and it still drops in hot / thin air.
    Rotor drag:  Gazebo / pybullet apply F = -c w v_perp per rotor (w in rad/s)  =>  rotor_drag = 2 pi c.

Not in the sources (our estimates, marked below): body drag C_d*A (the Gazebo models have none), the
idle speed, and the Crazyflie's motor time constant (pybullet's motors respond instantly).
"""
import numpy as np

from .airframe import Airframe, Rotor
from .motor import MotorModel

RHO0 = 1.225
TWO_PI = 2.0 * np.pi


def from_gazebo(k_motor, k_moment, diameter, w_max, tau_up, tau_down, idle=0.08):
    c_t = k_motor * TWO_PI ** 2 / (RHO0 * diameter ** 4)
    c_q = k_moment * c_t / diameter
    n_max = w_max / TWO_PI
    return MotorModel(diameter=diameter, c_t=c_t, c_q=c_q, tau=tau_up, tau_down=tau_down,
                      n_min=idle * n_max, n_max=n_max)


def iris(**kw):
    """3DR Iris as in PX4 SITL (Gazebo classic).  1.535 kg, T/W 1.92, 10 in propellers."""
    m = from_gazebo(k_motor=5.84e-06, k_moment=0.06, diameter=2 * 0.128, w_max=1100.0, tau_up=0.0125, tau_down=0.025)
    layout = [((0.13, -0.22), +1), ((-0.13, 0.20), +1), ((0.13, 0.22), -1), ((-0.13, -0.20), -1)]   # ccw = +1
    rotors = [Rotor(np.array([x, y, 0.023]), s, m.copy()) for (x, y), s in layout]
    return Airframe("iris (PX4)", mass=1.5 + 4 * 0.005 + 0.015, rotors=rotors,
                    inertia=np.diag([0.029125, 0.029125, 0.055225]),
                    rotor_drag=TWO_PI * 0.000175,
                    drag_area=np.array([0.012, 0.012, 0.03]),            # estimate (not in the sdf)
                    **kw)


def x500(**kw):
    """Holybro X500 as in PX4's current Gazebo models.  2.06 kg, T/W 1.69, 11 in propellers."""
    m = from_gazebo(k_motor=8.54858e-06, k_moment=0.016, diameter=0.2792, w_max=1000.0, tau_up=0.0125, tau_down=0.025)
    layout = [((0.174, -0.174), +1), ((-0.174, 0.174), +1), ((0.174, 0.174), -1), ((-0.174, -0.174), -1)]
    rotors = [Rotor(np.array([x, y, 0.06]), s, m.copy()) for (x, y), s in layout]
    return Airframe("x500 (PX4)", mass=2.0 + 4 * 0.016076923, rotors=rotors,
                    inertia=np.diag([0.0216667, 0.0216667, 0.04]),
                    rotor_drag=TWO_PI * 8.06428e-05,
                    drag_area=np.array([0.015, 0.015, 0.04]),            # estimate (not in the sdf)
                    **kw)


def crazyflie(**kw):
    """Bitcraze Crazyflie 2.X (cf2x from gym-pybullet-drones).  27 g, T/W 2.25, 46 mm propellers."""
    mass, kf, km, D = 0.027, 3.16e-10, 7.94e-12, 2 * 2.31348e-2
    rpm_max = np.sqrt(2.25 * 9.8 * mass / (4 * kf))                    # as in gym-pybullet-drones
    c_t = kf * 3600.0 / (RHO0 * D ** 4)                                  # rpm = 60 n
    c_q = km * 3600.0 / (RHO0 * D ** 5)
    n_max = rpm_max / 60.0
    m = MotorModel(diameter=D, c_t=c_t, c_q=c_q, tau=0.02,              # tau: estimate (instant in pybullet)
                   n_min=0.08 * n_max, n_max=n_max)
    # prop0..3 positions from the urdf; yaw torque there is -Q0 + Q1 - Q2 + Q3  ->  spins +1, -1, +1, -1
    layout = [((0.028, -0.028), +1), ((-0.028, -0.028), -1), ((-0.028, 0.028), +1), ((0.028, 0.028), -1)]
    rotors = [Rotor(np.array([x, y, 0.0]), s, m.copy()) for (x, y), s in layout]
    return Airframe("crazyflie 2.x", mass=mass, rotors=rotors,
                    inertia=np.diag([1.4e-5, 1.4e-5, 2.17e-5]),
                    rotor_drag=TWO_PI * 9.1785e-7,
                    # estimate: frontal area ~ 9 cm x 2.5 cm with motors and battery, C_d ~ 1
                    drag_area=np.array([2e-3, 2e-3, 4e-3]),
                    **kw)


REAL = {"iris": iris, "x500": x500, "crazyflie": crazyflie}
