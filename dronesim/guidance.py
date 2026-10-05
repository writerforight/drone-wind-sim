"""Guidance: turns a mission into a reference for the position controller at every instant.

    Hover(point)                     hold one point
    Waypoints(points, radius)        fly to each point in turn; switch when inside the acceptance radius
    PathFollower(points, speed)      follow a continuous path: the reference is the closest point of the
                                     path (with progress that only moves forward), plus feed-forward
                                     velocity along the path and centripetal acceleration v^2 kappa in
                                     curves; ramps up at the start and slows down before the end.
Each returns {'p': position, 'v': feed-forward velocity, 'yaw': heading} from ``reference(t, p)``.
"""
import numpy as np


class Hover:
    def __init__(self, point, yaw=0.0):
        self.point = np.asarray(point, dtype=float)
        self.yaw = yaw
        self.done = False

    def reference(self, t, p):
        return {"p": self.point.copy(), "v": np.zeros(3), "yaw": self.yaw}

    def path_points(self):
        return self.point[None, :]


class Waypoints:
    def __init__(self, points, radius=0.5, yaw=0.0):
        self.points = np.asarray(points, dtype=float)
        self.radius = radius
        self.yaw = yaw
        self.i = 0
        self.done = False

    def reference(self, t, p):
        if np.linalg.norm(p - self.points[self.i]) < self.radius:
            if self.i < len(self.points) - 1:
                self.i += 1
            else:
                self.done = True
        return {"p": self.points[self.i].copy(), "v": np.zeros(3), "yaw": self.yaw}

    def path_points(self):
        return self.points


class PathFollower:
    def __init__(self, points, speed=4.0, brake_distance=None, yaw="path", samples_per_m=20, accel=2.0):
        pts = np.asarray(points, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(s[-1] * samples_per_m))
        si = np.linspace(0.0, s[-1], n)
        self.pts = np.column_stack([np.interp(si, s, pts[:, k]) for k in range(3)])
        self.s = si
        tan = np.gradient(self.pts, axis=0)
        self.tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        # curvature vector dT/ds: centripetal acceleration needed at speed v is v^2 * kappa
        ds = max(si[1] - si[0], 1e-9)
        self.kappa = np.gradient(self.tan, ds, axis=0)
        self.length = s[-1]
        self.speed = speed
        self.brake = brake_distance if brake_distance is not None else speed ** 2 / 2.0 + 0.5
        self.yaw_mode = yaw
        self.accel = accel            # speed ramps up at this rate [m/s^2] instead of jumping
        self.t0 = None
        self.k = 0
        self.done = False
        self.cross_track = 0.0

    def reference(self, t, p):
        # search the closest sample a little ahead of the current progress (progress never goes back)
        hi = min(len(self.pts), self.k + 200)
        d = np.linalg.norm(self.pts[self.k:hi] - p, axis=1)
        self.k += int(np.argmin(d))
        self.cross_track = float(d.min())
        remaining = self.length - self.s[self.k]
        if self.t0 is None:
            self.t0 = t
        ramp = min(1.0, self.accel * (t - self.t0) / self.speed)
        sp = self.speed * min(ramp, remaining / self.brake)
        v = sp * self.tan[self.k]
        a = sp * sp * self.kappa[self.k]                  # feed-forward: turn without lagging behind
        # look slightly ahead so the drone keeps moving while it corrects
        ahead = min(len(self.pts) - 1, self.k + 3)
        if remaining < 0.2 and np.linalg.norm(p - self.pts[-1]) < 0.5:
            self.done = True
        yaw = np.arctan2(self.tan[self.k, 1], self.tan[self.k, 0]) if self.yaw_mode == "path" else float(self.yaw_mode)
        return {"p": self.pts[ahead].copy(), "v": v, "a": a, "yaw": yaw}

    def path_points(self):
        return self.pts


def figure_eight(center=(0, 0, 10), a=15.0, b=8.0, n=400):
    t = np.linspace(0, 2 * np.pi, n)
    c = np.asarray(center, dtype=float)
    return np.column_stack([c[0] + a * np.sin(t), c[1] + b * np.sin(2 * t), c[2] + 0 * t])


def box_mission(size=20.0, h=10.0):
    return np.array([[0, 0, h], [size, 0, h], [size, size, h + 5], [0, size, h], [0, 0, h]], dtype=float)
