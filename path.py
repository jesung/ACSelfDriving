import numpy as np
from scipy.interpolate import splev, splprep


def compute_long_angle(position: np.ndarray) -> np.ndarray:
    """
    Returns longitudinal angle (pitch) of the track. Negative values means that the track is going downhill.

    Attributes:
        position (np.ndarray): XYZ coordinates of sample points

    Returns:
        angle_long (np.ndarray): Longitudinal angle (pitch) of the track.
    """

    # note that y-axis is vertical and is flipped (negative is up)
    offset = np.roll(position, -1, axis=1)
    diff = offset - position

    angle_long = np.arctan(np.divide(-diff[1, :], np.linalg.norm(diff[[0, 2], :], axis=0)))
    return angle_long


def cumulative_distances(points):
    """Returns the cumulative linear distance at each point."""

    d = np.cumsum(np.linalg.norm(np.diff(points, axis=1), axis=0))
    return np.append(0, d)


class Path:
    """
    Spline based on provided waypoints using scipy package.

    Wrapper for scipy.interpolate.BSpline. Taken from https://github.com/joedavison17/dissertation

    Attributes:
        controls (np.ndarray): XYZ coordinates in which to construct path from
        closed (bool): whether the track loops or not
        dists (np.ndarray): cumulative linear distance at each point
        spline (list[np.ndarray, list, int): (t,c,k) a tuple containing the vector of knots, the B-spline
                                                coefficients, and the degree of the spline.
        length (np.float64): length of the path
    """

    def __init__(self, controls: np.ndarray, closed: bool = True) -> None:
        """Construct a spline through the given control points."""

        self.controls = controls
        self.closed = closed
        self.dists = cumulative_distances(self.controls)
        self.spline, _ = splprep(self.controls, u=self.dists, k=3, s=0, per=self.closed)
        self.length = self.dists[-1]

    def position(self, s=None) -> np.ndarray:
        """Returns XYZ coordinates of sample points."""

        if s is None:
            return self.controls
        x, y, z = splev(s, self.spline)
        return np.array([x, y, z])

    def curvature(self, s=None) -> np.array:
        """
        Returns sample curvatures, Kappa.

        Curvature is the inverse of the instantaneous radius of a turn (1/r).
        """

        if s is None:
            s = self.dists
        ddx, ddy, ddz = splev(s, self.spline, 2)
        return np.sqrt(ddx**2 + ddy**2 + ddz**2)

    def gamma2(self, s=None) -> np.array:
        """
        Returns the sum of the squares of sample curvatures, Gamma^2.

        Note that minimizing gamma^2 will also minimize gamma so there is no benefit to use additional CPU cycles
        to compute the square root.
        """

        if s is None:
            s = self.dists
        ddx, ddy, ddz = splev(s, self.spline, 2)
        return np.sum(ddx**2 + ddy**2 + ddz**2)
