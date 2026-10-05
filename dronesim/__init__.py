"""dronesim - multirotor flight simulator with a realistic outdoor environment.

    from dronesim import *
    sim = Simulation(airframe.quad_x(), Atmosphere(temperature_c=30),
                     WindField(mean_speed=6, direction_deg=270, turbulence="moderate"))
"""
from . import airframe
from .airframe import Airframe, Rotor
from .atmosphere import Atmosphere
from .control import CascadedPID, Mixer
from .guidance import Hover, PathFollower, Waypoints, box_mission, figure_eight
from .motor import MotorModel
from .sensors import SensorModel
from .simulation import Simulation, metrics
from .wind import Gust, WindField
from .env import DroneEnv, Randomization

__all__ = ["airframe", "Airframe", "Rotor", "Atmosphere", "CascadedPID", "Mixer", "Hover", "PathFollower",
           "Waypoints", "box_mission", "figure_eight", "MotorModel", "SensorModel", "Simulation", "metrics",
           "Gust", "WindField", "DroneEnv", "Randomization"]
