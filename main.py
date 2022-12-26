import cv2 as cv
import numpy as np
import os
from time import time
# from windowcapture import WindowCapture
from socket_class import ACSocket
from control import Controller
from vehicle import Vehicle
from track import Track
from path import Trajectory


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

    # variables for calculating fps
    loop_time = time()
    avg_fps = 0

    # initialize socket class and connect
    sock = ACSocket(host="127.0.0.1", port=65431)
    sock.connect()

    # initialize the WindowCapture class
    # win_cap = WindowCapture("Assetto Corsa")      # not needed currently

    # load track and compute speed target at each point
    # optional: compute car maximum g
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
    cont = Controller(path=trajectory.path.position(trajectory.s)[:, :-1], velocity=trajectory.velocity.v)

    print("Starting loop")

    while True:
        # debug the loop rate
        avg_fps = avg_fps * 0.9 + 0.1 / (time() - loop_time)
        # print('FPS: {}'.format(avg_fps))
        loop_time = time()

        # get an updated image of the game
        # screenshot = win_cap.get_screenshot_mss()
        # cv.imshow('Computer Vision', screenshot)

        # get game state from socket connection
        sock.update()
        vehicle.update(sock.data)  # pass on data from socket to update car's current state
        # print(vehicle)
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
