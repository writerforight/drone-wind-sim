"""International Standard Atmosphere (ISA, troposphere) with adjustable ground conditions.

    T(h)   = T0 - L h                                  temperature  [K]
    p(h)   = p0 (T(h) / T0)^(g / (R L))                pressure     [Pa]
    rho(h) = p(h) / (R T(h))                           air density  [kg/m^3]

T0 and p0 are the conditions at the ground (h = 0).  The defaults are the ISA sea-level values; a hot
day is simply ``Atmosphere(temperature_c=35)`` and a high-altitude site ``Atmosphere(ground_altitude=2000)``.
Air density matters for a multirotor because propeller thrust and aerodynamic drag both scale with rho:
on a hot day at altitude the same motor speed gives noticeably less thrust.
"""
from dataclasses import dataclass

import numpy as np

R_AIR = 287.05287      # specific gas constant of dry air [J/(kg K)]
G0 = 9.80665           # standard gravity [m/s^2]
LAPSE = 0.0065         # temperature lapse rate in the troposphere [K/m]
T_SL = 288.15          # ISA sea-level temperature [K]
P_SL = 101325.0        # ISA sea-level pressure [Pa]


@dataclass
class Atmosphere:
    temperature_c: float = 15.0     # temperature at the ground [°C]
    ground_altitude: float = 0.0    # altitude of the ground above sea level [m]
    sea_level_pressure: float = P_SL

    def __post_init__(self):
        # ground pressure follows from the ISA pressure law at the site altitude
        t_isa_site = T_SL - LAPSE * self.ground_altitude
        self.T0 = self.temperature_c + 273.15
        self.p0 = self.sea_level_pressure * (t_isa_site / T_SL) ** (G0 / (R_AIR * LAPSE))

    def temperature(self, h):
        """Temperature [K] at height h [m] above the ground."""
        return self.T0 - LAPSE * np.asarray(h, dtype=float)

    def pressure(self, h):
        return self.p0 * (self.temperature(h) / self.T0) ** (G0 / (R_AIR * LAPSE))

    def density(self, h):
        return self.pressure(h) / (R_AIR * self.temperature(h))

    def speed_of_sound(self, h):
        return np.sqrt(1.4 * R_AIR * self.temperature(h))

    def describe(self, h=0.0):
        return (f"T = {self.temperature(h) - 273.15:.1f} °C, p = {self.pressure(h) / 100:.1f} hPa, "
                f"rho = {self.density(h):.4f} kg/m³")
