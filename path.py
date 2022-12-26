import numpy as np
from scipy.interpolate import splev, splprep
import math
import numpy as np
import os
import time
from functools import partial
from multiprocessing import Pool
# from plot import plot_path
from scipy.optimize import Bounds, minimize, minimize_scalar
# from track import Track
# from utils import define_corners, idx_modulo
from velocity import VelocityProfile
import scipy.constants

# # old code
# def distance(a, b):
#     """Euclidean distance between two points."""
#     return math.sqrt(np.sum(np.square(np.subtract(a, b))))
#
#
# def curvature(a, b, c):
#     """Computes the curvature between points a, b, and c."""
#     v1 = a - b
#     v2 = a - c
#     return 2 * math.sqrt((v1[1] * v2[2] - v1[2] * v2[1]) ** 2 +
#                          (v1[2] * v2[0] - v1[0] * v2[2]) ** 2 +
#                          (v1[0] * v2[1] - v1[1] * v2[0]) ** 2) / \
#            (distance(a, b) * distance(b, c) * distance(a, c))
#
#
# def max_velocity(curv, max_lat_acc):
#     """Maximum velocity achievable with a given curvature and acceleration limit."""
#     return math.sqrt(max_lat_acc * scipy.constants.g / curv)
#
#
# def speed_target(target_line, max_lat_acc):
#     """
#     Returns the maximum theoretical speed achievable by vehicle.
#     Uses maximum tire grip and vehicle acceleration profile in computation (friction circle).
#
#     :param target_line: the waypoints of desired line
#     :param max_lat_acc: maximum grip available to car
#     :return: maximum theoretical speed at each point of the target line in km/h (capped at 350)
#     """
#
#     length = target_line.shape[0]
#     target = []
#     curv = []  # store the curvature at each point in the target line
#
#     # compute the curvature and therefore the maximum velocity that can be taken at each set of three consecutive points
#     for i in range(length):
#         curv.append(curvature(target_line[(i - 1) % length], target_line[i], target_line[(i + 1) % length]))
#         target.append(max_velocity(curv[i], max_lat_acc))
#
#     # forward pass to limit acceleration profile
#     # TO-DO
#
#     # backward pass to limit deceleration profile
#     for i in range(length - 1, -1, -1):
#         if target[(i - 1) % length] > target[i % length]:
#             available_long_acc = max_lat_acc * scipy.constants.g - target[i % length] ** 2 * curv[(i - 1) % length]
#             angle = 2 * np.arcsin(
#                 distance(target_line[(i - 2) % length], target_line[i]) / 2 * curv[(i - 1) % length])
#             travel_dist = angle / curv[(i - 1) % length]
#             target[(i - 1) % length] = math.sqrt(target[i] ** 2 + available_long_acc * travel_dist)
#
#     return np.minimum(350, np.multiply(3.6, target))  # convert to km/h


class Path:
    """
    Wrapper for scipy.interpolate.BSpline.
    Taken from https://github.com/joedavison17/dissertation
    """

    def __init__(self, controls, closed=True):
        """Construct a spline through the given control points."""
        # self.controls = controls.transpose()
        self.controls = controls
        self.closed = closed
        self.dists = cumulative_distances(self.controls)
        # print(self.dists)
        self.spline, _ = splprep(self.controls, u=self.dists, k=3, s=0, per=self.closed)
        self.length = self.dists[-1]

    def position(self, s=None):
        """Returns x-y-z coordinates of sample points."""
        if s is None:
            return self.controls
        x, y, z = splev(s, self.spline)
        return np.array([x, y, z])

    def curvature(self, s=None):
        """Returns sample curvatures, Kappa."""
        if s is None:
            s = self.dists
        ddx, ddy, ddz = splev(s, self.spline, 2)
        return np.sqrt(ddx**2 + ddy**2 + ddz**2)

    def gamma2(self, s=None):
        """Returns the sum of the squares of sample curvatures, Gamma^2."""
        if s is None: s = self.dists
        ddx, ddy, ddz = splev(s, self.spline, 2)
        return np.sum(ddx**2 + ddy**2 + ddz**2)

    # def compute_v_desired(self, waypoints, vehicle):
    #     target = speed_target(waypoints, vehicle.tire_coefficient)  # 1.32 for BRZ, 1.70 for Zonda
    #     return target


class Trajectory:
    """
    Stores the geometry and dynamics of a path, handling optimisation of the
    racing line. Samples are taken every metre.
    """

    def __init__(self, track, vehicle=None):
        """Store track and vehicle and initialise a centerline path."""
        self.s = None
        self.velocity = None
        self.path = None
        self.track = track
        self.ns = math.ceil(track.length)
        self.alphas = np.full(track.size, 0.5)
        self.update(self.alphas)       # turn alphas into function param & pass on results for max curvature
        self.vehicle = vehicle

    def update(self, alphas):
        """Update control points and the resulting path."""
        self.alphas = alphas
        self.path = Path(self.track.control_points(alphas), self.track.closed)
        # Sample every metre
        self.s = np.linspace(0, self.path.length, self.ns)
        # self.s = np.linspace(0, self.path.length, math.ceil(self.track.size))

    def update_velocity(self):
        """Generate a new velocity profile for the current path."""
        s = self.s[:-1]
        s_max = self.path.length if self.track.closed else None
        k = self.path.curvature(s)
        self.velocity = VelocityProfile(self.vehicle, s, k, s_max)

    def lap_time(self):
        """Calculate lap time from the velocity profile."""
        return np.sum(np.diff(self.s) / self.velocity.v)

    def minimise_curvature(self):
        """Generate a path minimising curvature."""

        def objfun(alphas):
            self.update(alphas)
            return self.path.gamma2(self.s)

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=np.full(self.track.size, 0.5),
            method='L-BFGS-B',
            bounds=Bounds(0.0, 1.0),
            options={'maxfun': 1000000, 'maxiter': 1000000, 'gtol': 1e-09}
        )
        self.update(res.x)
        return time.time() - t0

    def minimise_compromise(self, eps):
        """
        Generate a path minimising a compromise between path curvature and path
        length. eps gives the weight for path length.
        """

        def objfun(alphas):
            self.update(alphas)
            k = self.path.gamma2(self.s)
            d = self.path.length
            return (1 - eps) * k + eps * d

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=np.full(self.track.size, 0.5),
            method='L-BFGS-B',
            bounds=Bounds(0.0, 1.0)
        )
        self.update(res.x)
        return time.time() - t0

    def minimise_optimal_compromise(self, eps_min=0, eps_max=0.2):
        """
        Determine the optimal compromise weight when using optimise_compromise to
        produce a path.
        """

        def objfun(eps):
            self.minimise_compromise(eps)
            self.update_velocity()
            t = self.lap_time()
            if self.epsilon_history.size > 0:
                self.epsilon_history = np.vstack((self.epsilon_history, [eps, t]))
            else:
                self.epsilon_history = np.array([eps, t])
            return t

        self.epsilon_history = np.array([])
        t0 = time.time()
        res = minimize_scalar(
            fun=objfun,
            method='bounded',
            bounds=(eps_min, eps_max)
        )
        self.epsilon = res.x
        self.minimise_compromise(self.epsilon)
        end = time.time()
        return end - t0

    def minimise_lap_time(self):
        """
        Generate a path that directly minimises lap time.
        """

        def objfun(alphas):
            self.update(alphas)
            self.update_velocity()
            return self.lap_time()

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=np.full(self.track.size, 0.5),
            method='L-BFGS-B',
            bounds=Bounds(0.0, 1.0),
            options={'maxfun': 100000, 'maxiter': 100000, 'maxls': 100, 'gtol': 1e-07}
        )
        self.update(res.x)
        return time.time() - t0

    # def optimise_sectors(self, k_min, proximity, length):
    #     """
    #     Generate a path that optimises the path through each sector, and merges
    #     the results along intervening straights.
    #     """
    #
    #     # Define sectors
    #     t0 = time.time()
    #     corners, _ = self.track.corners(self.s, k_min, proximity, length)
    #
    #     # Optimise path for each sector in parallel
    #     nc = corners.shape[0]
    #     pool = Pool(os.cpu_count() - 1)
    #     alphas = pool.map(
    #         partial(optimise_sector_compromise, corners=corners, traj=self),
    #         range(nc)
    #     )
    #     pool.close()
    #
    #     # Merge sectors and update trajectory
    #     alphas = np.sum(alphas, axis=0)
    #     self.update(alphas)
    #     return time.time() - t0

###############################################################################


def cumulative_distances(points):
    """Returns the cumulative linear distance at each point."""
    d = np.cumsum(np.linalg.norm(np.diff(points, axis=1), axis=0))
    return np.append(0, d)
