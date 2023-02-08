import numpy as np
from math import sqrt
import scipy.constants
from vehicle import Vehicle

GRAV = scipy.constants.g  # m/s^2


class VelocityProfile:
    """
    Generate and store a velocity profile for a given path and vehicle.

    Generate a velocity profile for the given vehicle and path parameters.
    :s: and :k: should NOT include the overlapping element for closed paths.
    The length of a closed path should be supplied in :s_max:

    Attributes:
        v (np.ndarray): computed theoretical maximum velocity
    """

    def __init__(self, vehicle: Vehicle, s: np.ndarray, k: np.ndarray, s_max: np.float64, angle_lat: np.ndarray,
                 angle_long: np.ndarray) -> None:
        """
        Parameters:
            vehicle (Vehicle): Represents vehicle's static and dynamic attributes.
            s (np.ndarray): sampled path interval
            k (np.ndarray): curvature at points sampled at interval s
            s_max (np.float64): length of path
            angle_lat (np.ndarray): interpolated lateral angle and direction of turn
            angle_long (np.ndarray): Longitudinal angle (pitch) of the track.
        """

        v_local = self.limit_local_velocities(k, angle_lat, vehicle.tire_coefficient)
        v_acc_lim = self.limit_acceleration(v_local, s, s_max, k, angle_long, vehicle)
        v_dec_lim = self.limit_deceleration(v_local, s, s_max, k, angle_long, vehicle)
        self.v = np.minimum(v_acc_lim, v_dec_lim)

    def limit_local_velocities(self, k, angle_lat, tire_coefficient) -> np.ndarray:
        """Calculate maximum speed given tire coefficient, road angle, and curvature"""

        # self.v_local = np.sqrt(self.vehicle.tire_coefficient * GRAV / k)      # 2-D formula
        lat = angle_lat[:, 0]
        direction = angle_lat[:, 1]

        return np.sqrt(
            (direction * np.sin(lat) + tire_coefficient * np.absolute(np.cos(lat))) /
            (np.absolute(np.cos(lat)) - tire_coefficient * direction * np.sin(lat)) *
            GRAV / k)

    def limit_acceleration(self, v_local: np.ndarray, s: np.ndarray, s_max: np.float64, k_in: np.ndarray,
                           angle_long: np.ndarray, vehicle: Vehicle) -> np.ndarray:
        """Move forwards through v_local to find maximum acceleration based on engine profile and available grip."""

        # Start at slowest point
        shift = -np.argmin(v_local)
        s = np.roll(s, shift)
        v = np.roll(v_local, shift)
        k = np.roll(k_in, shift)
        long = np.roll(angle_long, shift)

        # Limit according to acceleration
        for i in range(s.size):
            wrap = i == (shift % s.size)
            if wrap and s_max is None:
                continue

            if v[i] > v[i - 1]:
                traction = vehicle.traction(v[i - 1], k[i - 1])
                force = min(vehicle.engine_force(v[i - 1]), traction)
                # accel = max(force / self.vehicle.mass, 0)     # 2-D formula
                accel = max(force / vehicle.mass - GRAV * np.sin(long[i - 1]), 0)

                ds = s_max - s[i - 1] if wrap else s[i] - s[i - 1]
                v_lim = sqrt(v[i - 1] ** 2 + 2 * accel * ds)
                v[i] = min(v[i], v_lim)

        # Reset shift and return
        return np.roll(v, -shift)

    def limit_deceleration(self, v_local: np.ndarray, s: np.ndarray, s_max: np.float64, k_in: np.ndarray,
                           angle_long: np.ndarray, vehicle: Vehicle) -> np.ndarray:
        """Work backwards through v_local to find maximum deceleration based on available grip."""

        # Start at slowest point, move backwards
        shift = -np.argmin(v_local)
        s = np.flip(np.roll(s, shift), 0)
        k = np.flip(np.roll(k_in, shift), 0)
        v = np.flip(np.roll(v_local, shift), 0)
        long = np.flip(np.roll(angle_long, shift), 0)

        # Limit according to deceleration
        for i in range(s.size):
            wrap = i == (-shift)
            if wrap and s_max is None:
                continue

            if v[i] > v[i - 1]:
                # radius < 100-120 seems to correspond with corners
                radius = 1 / k[i - 1]

                traction = vehicle.traction(v[i - 1], k[i - 1])
                # decel = max(traction / self.vehicle.mass, 0)      # 2-D formula
                decel = max(traction / vehicle.mass + GRAV * np.sin(long[i - 1]), 0)

                # damping on total braking to account for load transfer
                if radius / (3.6 * v[i]) < 2 or radius < 150:
                    decel = min(decel, 1.25 * max(radius - 50, 0) / v[i - 1])  # 1.25 // -50

                ds = s_max - s[i] if wrap else s[i - 1] - s[i]
                v_lim = sqrt(v[i - 1] ** 2 + 2 * decel * ds)
                v[i] = min(v[i], v_lim)

        # Reset shift/flip and return
        return np.roll(np.flip(v, 0), -shift)
