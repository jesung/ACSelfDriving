import cv2 as cv
import numpy as np
import vgamepad as vg
import os
from time import time
from windowcapture import WindowCapture
from socket_class import ACSocket
# import pytesseract


# get grayscale image
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# thresholding
def thresholding(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize socket class and connect
sock = ACSocket(host="127.0.0.1", port=65431)
sock.connect()

# initialize the WindowCapture class
win_cap = WindowCapture("Assetto Corsa")

# initialize the gamepad class
gamepad = vg.VX360Gamepad()

# load track and compute speed target at each point
# optional: compute car maximum g

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Jesung\AppData\Local\Tesseract-OCR\tesseract.exe'

print("Starting loop")
# variables for calculating fps
loop_time = time()
avg_fps = 0

while True:
    # debug the loop rate
    avg_fps = avg_fps * 0.9 + 0.1 / (time() - loop_time)
    print('FPS {}'.format(avg_fps))
    loop_time = time()

    # get an updated image of the game
    screenshot = win_cap.get_screenshot_mss()
    cv.imshow('Computer Vision', screenshot)

    # get game state from socket connection
    sock.update()
    print(sock.data)

    # compute target controls and update gamepad

    # controller.update(target)

    # pytesseract
    # world_coordinates = thresholding(get_grayscale(screenshot[110:220, 0:150]))
    # cv.imshow('Coordinates', world_coordinates)
    # print(reader.readtext(world_coordinates), allowlist='0123456789')
    # custom_config = 'digits'
    # print(pytesseract.image_to_string(world_coordinates, lang='eng', config=custom_config))

    # control gamepad
    # gamepad.right_trigger(value=255)
    # gamepad.update()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        sock.on_close()
        break

print('Done.')


