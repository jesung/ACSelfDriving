import csv
import os
import numpy as np
import vgamepad as vg
import time
import math
import scipy.constants


def get_time():
    return time.time_ns() / 1000000000


def find_closest_waypoint(waypoints, car_location):
    min_distance = float("inf")
    min_index = 0

    for i in range(waypoints.shape[0]):
        point_distance = np.linalg.norm(waypoints[i, :3] - car_location[:3])

        if point_distance < min_distance:
            min_distance = point_distance
            min_index = i

    return min_index, min_distance


def distance(a, b):
    return math.sqrt(np.sum(np.square(np.subtract(a, b))))


def curvature(a, b, c):
    v1 = a - b
    v2 = a - c
    return 2 * math.sqrt((v1[1]*v2[2] - v1[2]*v2[1])**2 +
             (v1[2]*v2[0] - v1[0]*v2[2])**2 +
             (v1[0]*v2[1] - v1[1]*v2[0])**2) / (distance(a, b) * distance(b, c) * distance(a, c))


def max_velocity(curv, max_lat_acc):
    return math.sqrt(max_lat_acc * scipy.constants.g / curv)


def speed_target(target_line, max_lat_acc):
    length = target_line.shape[0]
    target = []
    curv = []       # store the curvature at each point in the target line

    # compute the curvature and therefore the maximum velocity that can be taken at each set of three consecutive points
    for i in range(length):
        curv.append(curvature(target_line[(i - 1) % length], target_line[i], target_line[(i + 1) % length]))
        target.append(max_velocity(curv[i], max_lat_acc))

    # forward pass to limit acceleration profile
    # TO-DO

    # backward pass to limit deceleration profile
    for i in range(length - 1, -1, -1):
        if target[(i - 1) % length] > target[i % length]:
            available_long_acc = max_lat_acc * scipy.constants.g - target[i % length]**2 * curv[(i - 1) % length]
            angle = 2 * np.arcsin(distance(target_line[(i - 2) % length], target_line[i]) / 2 * curv[(i - 1) % length])
            travel_dist = angle / curv[(i - 1) % length]
            target[(i - 1) % length] = math.sqrt(target[i]**2 + available_long_acc * travel_dist)

    return np.minimum(350, np.multiply(3.6, target)) # convert to km/h


class Controller:
    left = None
    right = None
    fast_ai = None
    pit_lane = None
    waypoints = None
    v_desired = None
    index = 0
    gamepad = None
    mode = "pit_lane"
    target_throttle = 0
    target_steer = 0
    target_brake = 0
    t_previous = get_time()
    I_previous = 0
    throttle_previous = 0

    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "content")

    def __init__(self, track):
        self.load_csv(track)
        # self.waypoints = self.compute_v_desired(self.pit_lane)
        self.gamepad = vg.VX360Gamepad()

        # press and release a button to have the controller be recognized
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)

        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)

    def load_csv(self, track):
        self.dir_name = os.path.join(self.dir_name, track)

        # only read the first three columns of AI file
        with open(os.path.join(self.dir_name, "fast_ai.csv"), "r") as f:
            reader = csv.reader(f)
            self.fast_ai = np.array([float(i) for row in reader for i in row[:3]]).reshape((-1, 3))

        with open(os.path.join(self.dir_name, "pit_lane.csv"), "r") as f:
            reader = csv.reader(f)
            self.pit_lane = np.array([float(i) for row in reader for i in row[:3]]).reshape((-1, 3))

        with open(os.path.join(self.dir_name, "left.csv")) as f:
            reader = csv.reader(f)
            self.left = np.array([float(i) for row in reader for i in row]).reshape((-1, 3))

        with open(os.path.join(self.dir_name, "right.csv")) as f:
            reader = csv.reader(f)
            self.right = np.array([float(i) for row in reader for i in row]).reshape((-1, 3))

    def set_controller(self):
        """
        Sets and updates the controller values based on the target throttle, brake, and steer values
        """
        # print(f"Throttle: {self.target_throttle}\t Brake: {self.target_brake}\t Steer: {self.target_steer}")
        self.gamepad.right_trigger_float(value_float=self.target_throttle)  # value between 0 and 1
        self.gamepad.left_trigger_float(value_float=self.target_brake)  # value between 0 and 1
        self.gamepad.left_joystick_float(x_value_float=self.target_steer, y_value_float=0.0)  # range(-1, 1)  / np.pi

        self.gamepad.update()

    def update_target(self, car):
        # Longitudinal Controller
        epsilon = 0.03   # 0.03
        k_p = 0.7  # 0.7
        k_i = 0.00  # 0.05
        k_d = 0.0
        t = get_time()
        self.throttle_previous = self.target_throttle

        # code to switch reference waypoint from pit lane to track
        if self.mode == "pit_lane":
            lookahead_distance = 5
            target_velocity = 60
            waypoints = self.pit_lane
            # inefficient but only happens while in pit lane
            min_idx, min_dist = find_closest_waypoint(waypoints, car.location)
            _, dist = find_closest_waypoint(self.fast_ai, car.location)
            if min_dist > dist or min_idx > 880:
                self.mode = "track"
                self.v_desired = self.compute_v_desired(self.fast_ai)
                print("Changing mode to track")
        else:
            lookahead_distance = 16  # 17 for brz
            waypoints = self.fast_ai
            min_idx, _ = find_closest_waypoint(waypoints, car.location)
            target_velocity = self.v_desired[min_idx]  # update this to dynamic

        self.I_previous += (t - self.t_previous) * (target_velocity - car.speed)
        a = k_p * (target_velocity - car.speed) + k_i * self.I_previous + k_d * (target_velocity - car.speed) / (
                    t - self.t_previous)
        self.t_previous = t

        # if car.speed > 0:
        #     self.target_throttle = self.throttle_previous + epsilon * (target_velocity - car.speed) / car.speed
        # else:
        #     self.target_throttle = 10

        self.target_throttle = max(a, 0)
        self.target_brake = -min(a, 0)

        self.target_throttle = max(min(self.target_throttle, 1.0), 0.0)
        self.target_brake = max(min(self.target_brake, 1.0), 0.0)

        print("Target speed:", target_velocity, "\tActual speed:", car.speed, "\tThrottle:", self.target_throttle, "\tBrake:", self.target_brake)

        # Lateral Controller
        # PID controller parameters
        desired_heading = None
        e = 0  # cross-track error
        # lookahead_distance = 25   # this is declared above
        k = 0.1
        k_s = 2

        target_idx = (min_idx + lookahead_distance) % waypoints.shape[0]
        target_location = waypoints[target_idx, :]

        desired_heading = np.arctan((target_location[0] - car.location[0]) / (target_location[2] - car.location[2]))
        alpha = desired_heading - car.heading
        l_d = np.linalg.norm(np.array([waypoints[target_idx, :3] - car.location]))
        e = np.sin(alpha) * l_d

        if alpha > 1.5:
            alpha -= np.pi
        elif alpha < -1.5:
            alpha += np.pi

        # print(f"Current: {car.location}\tClosest: {waypoints[min_idx, :]}\tTarget: {waypoints[target_idx, :]}")
        # print(f"Min index: {min_idx}")
        # print(f"Current heading: {car.heading}\t\t\tDesired heading: {desired_heading}, Alpha: {alpha}")
        # print(f"Mode: {self.mode}")

        if car.speed > 0:
            steer_target = alpha  # + np.arctan(k*e/(k_s + car.speed))
            # steer_output = self.vars.steer_previous + (steer_target - self.vars.steer_previous) * 0.1
            steer_output = steer_target
        else:
            steer_output = 0

        # Change the steer output with the lateral controller.
        self.target_steer = steer_output  # max(min(steer_output, 1.22), -1.22)
        # print("steer_output",steer_output)

        self.set_controller()

    def compute_v_desired(self, waypoints):
        target = speed_target(waypoints, 1.72)  # 1.32 for BRZ, 1.70 for Zonda

        return target
