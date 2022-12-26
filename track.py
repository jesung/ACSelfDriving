import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from path import Path, Trajectory, cumulative_distances
from vehicle import Vehicle
from scipy.interpolate import splev, splprep
import sys


class Track:
    left = None
    right = None
    middle = None
    fast_ai = None
    pit_lane = None
    v_desired = None
    closed = True   # does the track loop?
    diffs = None    # difference between left and right edges of track
    length = None

    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "content")

    def __init__(self, track):
        self.load_csv(track)
        self.diffs = self.right - self.left
        self.size = self.left[0, :].size - int(self.closed)
        self.mid = Path(self.control_points(np.full(self.size, 0.5)), self.closed)
        self.length = self.mid.length

    def load_csv(self, track):
        self.dir_name = os.path.join(self.dir_name, track)

        # only read the first three columns of AI file
        with open(os.path.join(self.dir_name, "fast_ai.csv"), "r") as f:
            reader = csv.reader(f)
            self.fast_ai = np.array([float(i) for row in reader for i in row[:3]]).reshape((-1, 3)).transpose()

        with open(os.path.join(self.dir_name, "pit_lane.csv"), "r") as f:
            reader = csv.reader(f)
            self.pit_lane = np.array([float(i) for row in reader for i in row[:3]]).reshape((-1, 3)).transpose()

        with open(os.path.join(self.dir_name, "left.csv")) as f:
            reader = csv.reader(f)
            self.left = np.array([float(i) for row in reader for i in row]).reshape((-1, 3)).transpose()

        with open(os.path.join(self.dir_name, "right.csv")) as f:
            reader = csv.reader(f)
            self.right = np.array([float(i) for row in reader for i in row]).reshape((-1, 3)).transpose()

    def control_points(self, alphas):
        """Translate alpha values into control point coordinates"""
        alphas = np.append(alphas, alphas[0])
        i = np.nonzero(alphas != -1)[0]
        return self.left[:, i] + (alphas[i] * self.diffs[:, i])


if __name__ == "__main__":
    track = Track('rbr_national')
    vehicle = Vehicle()
    trajectory = Trajectory(track, vehicle)

    # run_time = trajectory.minimise_lap_time()
    run_time = trajectory.minimise_curvature()

    print("[ Computing lap time ]")
    trajectory.update_velocity()
    lap_time = trajectory.lap_time()

    print()
    print("=== Results ==========================================================")
    print("Lap time = {:.3f}".format(lap_time))
    print("Run time = {:.3f}".format(run_time))
    print("======================================================================")
    print()

    dists_left = cumulative_distances(track.left)
    spline_left, _ = splprep(track.left, u=dists_left, k=3, s=0, per=True)
    x, y, z = splev(dists_left, spline_left)
    position_left = np.array([x, y, z]).transpose()

    dists_right = cumulative_distances(track.right)
    spline_right, _ = splprep(track.right, u=dists_right, k=3, s=0, per=True)
    x, y, z = splev(dists_right, spline_right)
    position_right = np.array([x, y, z]).transpose()

    dists_mid = cumulative_distances(trajectory.path.controls)
    spline_mid, _ = splprep(trajectory.path.controls, u=dists_mid, k=3, s=0, per=True)
    x, y, z = splev(dists_mid, spline_mid)
    position_mid = np.array([x, y, z]).transpose()

    # save results into CSV
    with open(os.path.join(track.dir_name, 'optimized.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(trajectory.alphas)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for color, var in [('red', position_left), ('blue', position_right), ('green', position_mid)]:
    # for color, var in [('red', position_left), ('blue', position_right)]:
        ax.scatter(var[:, 0], var[:, 1], var[:, 2], c=color)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
