import os
import numpy as np
from path import Path, cumulative_distances
import ac_utils as ac
# from utils import define_corners, idx_modulo, is_closed
from scipy.interpolate import splev, splprep
import sys
import random

class Track:
    left = None
    right = None
    middle = None
    fast_ai = None
    pit_lane = None
    v_desired = None
    closed = None   # does the track loop?
    diffs = None    # difference between left and right edges of track
    length = None
    angle_lat = None
    s = None

    def __init__(self, track=None, closed=True):
        self.closed = closed
        self.left, self.right, self.fast_ai, self.pit_lane = ac.load_csv(track)
        # s = np.sort(random.sample(range(self.left[0, :].size - int(self.closed)), 1000))
        # self.left = self.left[:, s]
        # self.right = self.right[:, s]
        self.diffs = self.right - self.left
        # print(self.diffs)         # why is the diff along y-axis 0??
        self.size = self.left[0, :].size - int(self.closed)
        self.mid = Path(self.control_points(np.full(self.size, 0.5)), self.closed)
        self.length = self.mid.length
        self.precompute_angle()
        dir_name = os.path.dirname(__file__)
        dir_name = os.path.join(dir_name, "content")
        dir_name = os.path.join(dir_name, "tracks")
        dir_name = os.path.join(dir_name, track)
        self.dir_name = dir_name

    def control_points(self, alphas, sample=None):
        """Translate alpha values into control point coordinates"""
        if self.closed:
            alphas = np.append(alphas, alphas[0])
        if sample is None:
            sample = range(alphas.shape[0])
        i = np.nonzero(alphas != -1)[0]
        return self.left[:, i] + (alphas[i] * self.diffs[:, i])

    # def corners(self, s, k_min, proximity, length):
    #     """Determine location of corners on this track."""
    #     return define_corners(self.mid, s, k_min, proximity, length)

    def precompute_angle(self):
        """Sample track boundaries every meter and cache the lateral angle and
        direction of turn at each step. This will be interpolated in compute_lat_angle"""
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
        self.angle_lat = np.array([s, angles, camber])
        # print(angles)

    def compute_lat_angle(self, sample):
        """Interpolate the lat angle and direction of turn. Both values are pre-computed
        for performance reasons. Should be fairly accurate if the optimization process is iterated."""
        return np.array([[np.interp(s, self.angle_lat[0, :], self.angle_lat[1, :]),
                          round(np.interp(s, self.angle_lat[0, :], self.angle_lat[2, :]))] for s in sample])

