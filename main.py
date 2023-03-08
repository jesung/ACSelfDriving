# import cv2 as cv
import numpy as np
import os
from time import time

from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt
import csv

# from windowcapture import WindowCapture
from vehicle import Vehicle
from track import Track
from control import Controller
from path import cumulative_distances
from trajectory import Trajectory
from socket_class import ACSocket
from replay import Replay
import ac_utils as ac


# # get grayscale image
# def get_grayscale(image):
#     return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#
# # thresholding
# def thresholding(image):
#     return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def show_plot(track: Track, trajectory: Trajectory, actual=None) -> None:
    """Plot results of optimized race line"""
    # show optimized line in plot
    dists_left = cumulative_distances(track.left)
    spline_left, _ = splprep(track.left, u=dists_left, k=3, s=0, per=True)
    x, y, z = splev(dists_left, spline_left)
    position_left = np.array([x, y, z]).transpose()

    dists_right = cumulative_distances(track.right)
    spline_right, _ = splprep(track.right, u=dists_right, k=3, s=0, per=True)
    x, y, z = splev(dists_right, spline_right)
    position_right = np.array([x, y, z]).transpose()

    position_mid = trajectory.path.position(trajectory.s[:-1]).T

    fig = plt.figure()

    if actual is None:
        ax = fig.add_subplot(projection='3d')
        for color, var in [('red', position_left), ('blue', position_right), ('green', position_mid)]:
            ax.scatter(var[:, 0], var[:, 1], var[:, 2], c=color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    else:
        ax = fig.add_subplot()

        for color, var in [('red', position_left), ('blue', position_right), ('green', position_mid), ('yellow', actual)]:
            ax.scatter(var[:, 0], var[:, 2], c=color, s=2.5)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

    plt.show()


def main():
    """
    Usage:
        * Choose track and vehicle from list available in comments
        * (optional) Specify file name of previously optimized path. Default value is "optimized.csv"
        * Comment/uncomment the mode desired
        * Run code
    """

    # Change the working directory to the folder this script is in.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # load track and compute speed target at each point
    track = Track('rbr_national', closed=True)      # 'rbr_national'
    vehicle = Vehicle('ks_toyota_gt86')     # 'pagani_zonda_r', 'ks_toyota_gt86', 'ks_nissan_gtr'
    trajectory = Trajectory(track, vehicle)
    file = os.path.join(track.dir_name, 'optimized.csv')
    actual = None

    # # comment this out if you just want to see the optimized line. Used for comparing reference and actual lines
    # with open(os.path.join(track.dir_name, "side_l.csv")) as f:
    #     reader = csv.reader(f)
    #     actual = np.array([float(i) for row in reader for i in row]).reshape((-1, 4))[:, :3]

    # mode = "optimize"
    mode = "race"

    if mode == "optimize":
        try:
            guess = ac.load_alphas(file)
        except:
            guess = np.full(trajectory.track.size, 0.5)     # create an initial guess with middle of track

        prev_lap_time = np.inf
        prev_trajectory = None

        # optimize racing line
        while True:
            run_time = trajectory.minimise_curvature_alphas(guess=guess)
            # run_time = trajectory.minimise_lap_time_alphas(guess=guess)

            guess = trajectory.alphas
            print("[ Computing lap time ]")
            trajectory.update_velocity()
            lap_time = trajectory.lap_time()
            print("Lap time = {:.3f}".format(lap_time))
            print("Run time = {:.3f}".format(run_time))
            if lap_time > prev_lap_time - 0.01:     # stop if improvement is less than 0.01 sec
                break
            elif lap_time > prev_lap_time:
                trajectory = prev_trajectory
                break
            else:
                prev_lap_time = lap_time
                prev_trajectory = trajectory

        # save results into CSV
        with open(os.path.join(track.dir_name, 'optimized.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(trajectory.alphas)

        trajectory.update(guess)
        show_plot(track, trajectory, actual)

    elif mode == "race":
        # # variables for calculating fps
        # loop_time = time()
        # avg_fps = 0

        try:
            trajectory.update(ac.load_alphas(file))
        except:
            alphas = np.full(trajectory.track.size, 0.5)
            trajectory.update(alphas)

        trajectory.update_velocity()
        cont = Controller(trajectory=trajectory)
        rep = Replay()
        # df = rep.load(os.path.join(track.dir_name, 'replay.csv'))
        # print(df)

        # initialize the WindowCapture class
        # win_cap = WindowCapture("Assetto Corsa")      # not needed currently

        # initialize socket class and connect
        print("Trying to connect to AC app...")
        sock = ACSocket()
        with sock.connect() as conn:
            print("Starting loop")

            while True:
                # # show the average loop rate
                # avg_fps = avg_fps * 0.9 + 0.1 / (time() - loop_time)
                # print('FPS: {}'.format(avg_fps))
                # loop_time = time()
                #
                # # get an updated image of the game
                # screenshot = win_cap.get_screenshot_mss()
                # cv.imshow('Computer Vision', screenshot)

                # get game state from socket connection. Exit the game first to ensure that the socket is closed
                try:
                    sock.update()
                    vehicle.update(sock.data)  # pass on data from socket to update car's current state
                    cont.update_target(vehicle)     # compute target controls and update gamepad
                    rep.update(vehicle)
                except:
                    sock.on_close()
                    rep.save(os.path.join(track.dir_name, 'replay.csv'))
                    break

                # # press 'q' with the output window focused to exit. Waits 1 ms every loop to process key presses
                # if cv.waitKey(1) == ord('q'):
                #     cv.destroyAllWindows()
                #     sock.on_close()
                #     break

        print('Done.')


if __name__ == "__main__":
    main()
