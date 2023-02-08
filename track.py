import os
import numpy as np
from path import Path, cumulative_distances
import ac_utils as ac
# from utils import define_corners, idx_modulo, is_closed
from scipy.interpolate import splev, splprep


class Track:
    """
    Represents track's boundaries and its associated attributes.

    Attributes:
        closed (bool): whether the track loops or not
        left (np.ndarray): waypoints of left track boundary
        right (np.ndarray): waypoints of right track boundary
        diffs (np.ndarray): difference between right and left edges of track
        fast_ai (np.ndarray): waypoints for default AI line
        pit_lane (np.ndarray): waypoints for pit lane
        size (int): number of waypoints of left boundary
        mid (Path): Path along middle of track
        length (np.float64): length of mid
        angle_lat (np.ndarray): precomputed lateral angle and camber/off-camber values
        dir_name (str): directory for track folder
    """

    def __init__(self, track: str, closed: bool = True) -> None:
        """
        Parameters:
            track (str): name of the track
            closed (bool): whether the track loops or not
        """

        self.closed = closed
        self.left, self.right, self.fast_ai, self.pit_lane = ac.load_csv(track)
        self.normalize_boundary()
        self.diffs = self.right - self.left
        self.size = self.left[0, :].size - int(self.closed)
        self.mid = Path(self.control_points(np.full(self.size, 0.5)), self.closed)
        self.length = self.mid.length
        self.angle_lat = self.precompute_angle()

        # not used directly in the class but useful as reference when passing the class
        self.dir_name = os.path.join(os.path.dirname(__file__), "content", "tracks", track)

    def normalize_boundary(self) -> None:
        """
        Create standard-length left and right boundaries to define the track.

        Take left and right waypoints of arbitrary lengths, create an approximate path (spline), and output a new set
        of waypoints that are evenly spaced between left and right.
        """
        left_path = Path(np.hstack((self.left, self.left[:, :1])), self.closed)
        right_path = Path(np.hstack((self.right, self.right[:, :1])), self.closed)

        sample_length = 10      # sample every 10 meters
        length = int(np.ceil(left_path.length))
        s_left = np.linspace(0, length, int(np.floor(length / sample_length)))
        s_right = np.linspace(0, int(np.ceil(right_path.length)), int(np.floor(length / sample_length)))

        self.left = left_path.position(s_left)
        self.right = right_path.position(s_right)

    def control_points(self, alphas: np.ndarray) -> np.ndarray:
        """Translate alpha values into control point coordinates"""
        if self.closed:
            alphas = np.append(alphas, alphas[0])

        i = np.nonzero(alphas != -1)[0]
        return self.left[:, i] + (alphas[i] * self.diffs[:, i])

    # def corners(self, s, k_min, proximity, length):
    #     """Determine location of corners on this track."""
    #     return define_corners(self.mid, s, k_min, proximity, length)

    def precompute_angle(self) -> np.ndarray:
        """
        Sample track boundaries every meter and cache the lateral angle and direction of turn at each step.

        The output will be interpolated in compute_lat_angle to estimate the angle at each sampled point in the
        optimized path.

        Returns:
            Array of three numeric columns to be used for interpolation at runtime.
                s: cumulative linear distance along left boundary
                angles: the lateral angle corresponding to each point s on track
                camber: whether the road is cambered according to the direction of travel or not
        """

        angles = np.arctan(np.divide(-self.diffs[1, :], np.linalg.norm(self.diffs[[0, 2], :], axis=0)))

        # find direction of curvature. Note that y-axis is vertical and is flipped (negative is up)
        left_diff = np.roll(self.left, -1, axis=1)[[0, 2], :] - self.left[[0, 2], :]
        left_diff = left_diff / np.linalg.norm(left_diff, axis=0)   # normalize to unit vector

        s = cumulative_distances(self.left)
        left_spline, _ = splprep(self.left, u=s, k=3, s=0, per=self.closed)
        x, y, z = splev(x=s, tck=left_spline, der=1)    # 1st derivative of left spline
        left_der = np.array([x, z])
        left_der = left_der / np.linalg.norm(left_der, axis=0)  # normalize to unit vector

        # compute the determinant and check for its sign
        direction = np.multiply(left_diff[1, :], left_der[0, :]) - np.multiply(left_diff[0, :], left_der[1, :])

        camber = [1 if val > 0 else -1 for val in direction]
        return np.array([s, angles, camber])

    def compute_lat_angle(self, sample) -> np.ndarray:
        """
        Interpolate the lateral angle and direction of turn.

        Both lateral angle and direction of turn are pre-computed for performance reasons.
        Should be reasonable accurate unless there are significant changes in the local gradation.

        Parameters:
            sample (np.ndarray): cumulative linear distance along a path

        Returns:
            Two-column array of interpolated lateral angle and direction of travel for each point in sample.
        """

        return np.array([[np.interp(s, self.angle_lat[0, :], self.angle_lat[1, :]),
                          round(np.interp(s, self.angle_lat[0, :], self.angle_lat[2, :]))] for s in sample])
