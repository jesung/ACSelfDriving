import math
import numpy as np
import time
from scipy.optimize import Bounds, minimize, minimize_scalar
from track import Track
from velocity import VelocityProfile
from path import Path, compute_long_angle
from vehicle import Vehicle
# from functools import partial
# from multiprocessing import Pool
# from plot import plot_path
# from utils import define_corners, idx_modulo


class Trajectory:
    """
    Stores the geometry and dynamics of a path, handling optimisation of the racing line.

    Samples are taken approximately every meter.

    Attributes:
        s (np.ndarray): sampled path interval
        velocity (VelocityProfile): Generate and store a velocity profile for a given path and vehicle.
        path (Path): Spline based on provided waypoints using scipy package.
        angle_long (np.ndarray): Longitudinal angle (pitch) of the track.
        angle_lat (np.ndarray): interpolated lateral angle and direction of turn
        track (Track): Represents track's boundaries and its associated attributes.
        alphas (np.ndarray): set of relative positions (0 corresponds to left boundary, 1 to the right) that defines
                             a path to be optimized.
        vehicle (Vehicle): Represents vehicle's static and dynamic attributes.
        bounds (scipy.optimize.Bounds): Optimization bounds for alphas. Useful if your track boundary is not 100%
                                        accurate.
    """

    def __init__(self, track: Track, vehicle: Vehicle) -> None:
        """Store track and vehicle and initialise a centerline path."""
        self.s = None
        self.velocity = None
        self.path = None
        self.angle_long = None
        self.angle_lat = None
        self.track = track
        self.ns = math.ceil(self.track.length)
        self.alphas = np.full(self.track.size, 0.5)
        self.update(self.alphas)       # turn alphas into function param & pass on results for max curvature
        self.vehicle = vehicle
        self.bounds = Bounds(0.0, 1.0)

    def update(self, alphas: np.ndarray) -> None:
        """Update control points and the resulting path."""
        self.alphas = alphas
        self.path = Path(self.track.control_points(alphas), closed=self.track.closed)
        self.s = np.linspace(0, self.path.length, self.ns)

    def update_angles(self, s: np.ndarray = None) -> None:
        """Compute track angle (lateral and longitudinal) as well as road camber directionality"""
        if s is None:
            s = self.s
        pos = self.path.position(s)
        self.angle_long = compute_long_angle(pos)
        self.angle_lat = self.track.compute_lat_angle(s)

    def update_velocity(self) -> None:
        """Generate a new velocity profile for the current path."""
        s = self.s[:-1]
        s_max = self.path.length if self.track.closed else None
        k = self.path.curvature(s)
        self.update_angles(s)
        self.velocity = VelocityProfile(self.vehicle, s, k, s_max, self.angle_lat, self.angle_long)

    def lap_time(self) -> np.array:
        """Calculate lap time from the velocity profile."""
        return np.sum(np.diff(self.s) / self.velocity.v)

    def minimise_curvature(self) -> float:
        """Generate a path minimising curvature."""
        def objfun(alphas):
            self.update(alphas)
            return self.path.gamma2(self.s)

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=np.full(self.track.size, 0.5),
            method='L-BFGS-B',
            bounds=self.bounds
        )
        self.update(res.x)
        return time.time() - t0

    def minimise_curvature_alphas(self, guess: np.ndarray, opt: dict = None) -> float:
        """
        Generate a path minimising curvature.

        Allows for user to define the initial guess and options. Some options are listed for convenience.
        """

        if opt is None:
            # opt = {'maxcor': 50, 'maxfun': 100000, 'maxiter': 100000, 'gtol': 1e-13, 'maxls': 100, 'ftol': 1e-12}
            # opt = {'maxcor': 100, 'maxfun': 300000, 'maxiter': 300000, 'gtol': 3e-14, 'maxls': 150, 'ftol': 3e-13}
            # opt = {'maxcor': 150, 'maxfun': 1000000, 'maxiter': 1000000, 'gtol': 1e-14, 'maxls': 300, 'ftol': 1e-13}
            # opt = {'maxcor': 150, 'maxfun': 1000000, 'maxiter': 1000000, 'gtol': 3e-15, 'maxls': 300, 'ftol': 3e-14}
            # opt = {'maxcor': 250, 'maxfun': 3000000, 'maxiter': 3000000, 'gtol': 1e-15, 'maxls': 500, 'ftol': 1e-14}
            opt = {'maxcor': 500, 'maxfun': 10000000, 'maxiter': 10000000, 'gtol': 1e-17, 'maxls': 1000, 'ftol': 1e-16}

        s = self.s

        def objfun(alphas):
            self.update(alphas)
            return self.path.gamma2(s)

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=guess,
            method='L-BFGS-B',
            bounds=self.bounds,
            options=opt
        )
        self.update(res.x)

        # update alphas with full track length
        return time.time() - t0

    def minimise_compromise(self, eps) -> float:
        """Generate a path minimising a compromise between path curvature and path length. eps gives the weight for
        path length."""

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
            bounds=self.bounds,
            options={'maxfun': 10000000, 'maxiter': 10000000, 'gtol': 1e-11}
        )
        self.update(res.x)
        return time.time() - t0

    def minimise_optimal_compromise(self, eps_min=0, eps_max=0.2) -> float:
        """Determine the optimal compromise weight when using optimise_compromise to produce a path."""

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
            bounds=(eps_min, eps_max),
            options={'xatol': 1e-11, 'maxiter': 1000000}
        )
        self.epsilon = res.x
        self.minimise_compromise(self.epsilon)
        end = time.time()
        return end - t0

    def minimise_lap_time(self) -> float:
        """Generate a path that directly minimises lap time. Much slower compared to minimizing curvature."""

        def objfun(alphas):
            self.update(alphas)
            self.update_velocity()
            return self.lap_time()

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=np.full(self.track.size, 0.5),
            method='L-BFGS-B',
            bounds=self.bounds
        )
        self.update(res.x)
        return time.time() - t0

    def minimise_lap_time_alphas(self, guess, opt=None) -> float:
        """
        Generate a path that directly minimises lap time. Much slower compared to minimizing curvature.

        Allows for user to define the initial guess and options.
        """

        if opt is None:
            opt = {}
            # opt = {'maxfun': 100000, 'maxiter': 1000000, 'gtol': 1e-14, 'ftol': 1e-12}

        def objfun(alphas):
            self.update(alphas)
            self.update_velocity()
            return self.lap_time()

        t0 = time.time()
        res = minimize(
            fun=objfun,
            x0=guess,
            method='L-BFGS-B',
            bounds=self.bounds,
            options=opt
        )
        self.update(res.x)
        return time.time() - t0

    # def optimise_sectors(self, k_min, proximity, length) -> float:
    #     """
    #     Generate a path that optimises the path through each sector, and merges the results along intervening straights.
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
    #         partial(self.optimise_sector_compromise, corners=corners, traj=self),
    #         range(nc)
    #     )
    #     pool.close()
    #
    #     # Merge sectors and update trajectory
    #     alphas = np.sum(alphas, axis=0)
    #     self.update(alphas)
    #     return time.time() - t0
    #
    # def optimise_sector_compromise(self, i, corners, traj) -> np.ndarray:
    #     """
    #     Builds a new Track for the given corner sequence, and optimises the path through it by the compromise method.
    #     """
    #
    #     # Represent sector as new Track
    #     nc = corners.shape[0]
    #     n = traj.track.size
    #     a = corners[(i - 1) % nc, 1]  # Sector start
    #     b = corners[i, 0]  # Corner entry
    #     c = corners[i, 1]  # Corner exit
    #     d = corners[(i + 1) % nc, 0]  # Sector end
    #     idxs = idx_modulo(a, d, n)
    #     sector = Trajectory(
    #         Track(left=traj.track.left[:, idxs], right=traj.track.right[:, idxs]),
    #         traj.vehicle
    #     )
    #
    #     # Optimise path through sector
    #     sector.minimise_optimal_compromise()
    #
    #     # Weight alphas for merging across straights
    #     weights = np.ones((d - a) % n)
    #     weights[:(b - a) % n] = np.linspace(0, 1, (b - a) % n)
    #     weights[(c - a) % n:] = np.linspace(1, 0, (d - c) % n)
    #     alphas = np.zeros(n)
    #     alphas[idxs] = sector.alphas * weights
    #
    #     # Report and plot sector results
    #     # print("  Sector {:d}: eps={:.4f}, run time={:.2f}s".format(
    #     #     i, sector.epsilon # , rt
    #     # ))
    #     # plot_path(
    #     #   "./plots/" + traj.track.name + "_sector" + str(i) + ".png",
    #     #   sector.track.left, sector.track.right, sector.path.position(sector.s)
    #     # )
    #
    #     return alphas

###############################################################################
