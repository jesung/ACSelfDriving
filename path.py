import numpy as np
from scipy.interpolate import splev, splprep


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


def compute_long_angle(position):
    """
    Returns the longitudinal angle (pitch) of the track. Negative values
    means that the track is going downhill.
    """
    # if self.closed:
    #     position = np.concatenate((position, position[:, 0].reshape((3, 1))), axis=1)
    # note that y-axis is vertical and is flipped (negative is up)
    offset = np.roll(position, -1, axis=1)  # np.append(position[:, 1:], position[:, 0], axis=1) #
    diff = offset - position
    # calculate angle
    angle_long = np.arctan(np.divide(-diff[1, :], np.linalg.norm(diff[[0, 2], :], axis=0)))
    return angle_long


def cumulative_distances(points):
    """Returns the cumulative linear distance at each point."""
    d = np.cumsum(np.linalg.norm(np.diff(points, axis=1), axis=0))
    return np.append(0, d)


class Path:
    """
    Wrapper for scipy.interpolate.BSpline.
    Taken from https://github.com/joedavison17/dissertation
    """

    def __init__(self, controls, closed=True):
        """Construct a spline through the given control points."""
        self.controls = controls
        self.closed = closed
        self.dists = cumulative_distances(self.controls)
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
        if s is None:
            s = self.dists
        ddx, ddy, ddz = splev(s, self.spline, 2)
        return np.sum(ddx**2 + ddy**2 + ddz**2)

    # def compute_v_desired(self, waypoints, vehicle):
    #     target = speed_target(waypoints, vehicle.tire_coefficient)  # 1.32 for BRZ, 1.70 for Zonda
    #     return target

