import cv2 as cv
import numpy as np
import os
from time import time

from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt
import csv
import sys
import random

# from windowcapture import WindowCapture
from vehicle import Vehicle
from track import Track
from control import Controller
from path import cumulative_distances
from trajectory import Trajectory
import ac_utils as ac
from socket_class import ACSocket


# # get grayscale image
# def get_grayscale(image):
#     return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#
# # thresholding
# def thresholding(image):
#     return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def main():
    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # load track and compute speed target at each point
    # optional: compute car maximum g
    track = Track('rbr_national', closed=True)
    vehicle = Vehicle('pagani_zonda_r')
    trajectory = Trajectory(track, vehicle)

    mode = "optimize"                               # {"optimize", "race")
    file = os.path.join(track.dir_name, 'optimized_49944.csv')

    if mode == "optimize":
        guess = ac.load_alphas(file)
        # guess = np.full(trajectory.track.size, 0.5)
        prev_lap_time = np.inf
        prev_trajectory = None

        # optimize racing line
        while True:
            # print(s)
            run_time = trajectory.minimise_curvature_alphas(guess=guess)
            guess = trajectory.alphas
            print("[ Computing lap time ]")
            trajectory.update_velocity()
            lap_time = trajectory.lap_time()
            print("Lap time = {:.3f}".format(lap_time))
            print("Run time = {:.3f}".format(run_time))
            if lap_time > prev_lap_time:
                lap_time = prev_lap_time
                trajectory = prev_trajectory
                break
            else:
                prev_lap_time = lap_time
                prev_trajectory = trajectory

        # save results into CSV
        with open(os.path.join(track.dir_name, 'optimized.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(trajectory.alphas)

        # show optimized line in plot
        dists_left = cumulative_distances(track.left)
        spline_left, _ = splprep(track.left, u=dists_left, k=3, s=0, per=True)
        x, y, z = splev(dists_left, spline_left)
        position_left = np.array([x, y, z]).transpose()

        dists_right = cumulative_distances(track.right)
        spline_right, _ = splprep(track.right, u=dists_right, k=3, s=0, per=True)
        x, y, z = splev(dists_right, spline_right)
        position_right = np.array([x, y, z]).transpose()

        dists_mid = cumulative_distances(trajectory.path.controls)
        spline_mid = trajectory.path.spline
        x, y, z = splev(dists_mid, spline_mid)
        position_mid = trajectory.path.position(trajectory.s[:-1]).T

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for color, var in [('red', position_left), ('blue', position_right), ('green', position_mid)]:
            # for color, var in [('red', position_left), ('blue', position_right)]:
            ax.scatter(var[:, 0], var[:, 1], var[:, 2], c=color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    elif mode == "race":
        # variables for calculating fps
        loop_time = time()
        avg_fps = 0

        # initialize socket class and connect
        sock = ACSocket(host="127.0.0.1", port=65431)
        sock.connect()

        # initialize the WindowCapture class
        # win_cap = WindowCapture("Assetto Corsa")      # not needed currently

        trajectory.update(ac.load_alphas(file))
        trajectory.update_velocity()
        cont = Controller(path=trajectory.path.position(trajectory.s[:-1]), velocity=trajectory.velocity.v)

        print("Starting loop")

        while True:
            # debug the loop rate
            # avg_fps = avg_fps * 0.9 + 0.1 / (time() - loop_time)
            # print('FPS: {}'.format(avg_fps))
            # loop_time = time()

            # get an updated image of the game
            # screenshot = win_cap.get_screenshot_mss()
            # cv.imshow('Computer Vision', screenshot)

            # get game state from socket connection
            sock.update()
            vehicle.update(sock.data)  # pass on data from socket to update car's current state
            # update vehicle parameters here

            # compute target controls and update gamepad
            cont.update_target(vehicle)

            # press 'q' with the output window focused to exit.
            # waits 1 ms every loop to process key presses
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                sock.on_close()
                break

        print('Done.')


if __name__ == "__main__":
    main()
