import csv
import os
import numpy as np
import vgamepad as vg
import time


def get_time():
    return time.time_ns() / 1000000000


class Controller:
    left = None
    right = None
    fast_ai = None
    pit_lane = None
    waypoints = None
    index = 0
    gamepad = None
    mode = "pit_lane"
    target_throttle = 0
    target_steer = 0
    target_brake = 0
    t_previous = get_time()
    I_previous = 0

    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "content")

    def __init__(self, track):
        self.load_csv(track)
        # self.waypoints = self.compute_v_desired(self.pit_lane)
        self.gamepad = vg.VX360Gamepad()

        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)

        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)

        # need a different function for initializing the index
        # self.find_closest_waypoint(waypoints=self.pit_lane, car_location=np.array([229.9498, -11.7186, -323.1104]))
        # print(self.index)

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

    def find_closest_waypoint(self, waypoints, car_location):
        # need to refactor
        min_distance = float("inf")
        min_index = 0

        for i in range(waypoints.shape[0]):
            distance = np.linalg.norm(waypoints[i, :3] - car_location[:3])

            if distance < min_distance:
                min_distance = distance
                min_index = i

        return min_index, min_distance

    def set_controller(self):
        # print(f"Throttle: {self.target_throttle}\t Brake: {self.target_brake}\t Steer: {self.target_steer}")
        self.gamepad.right_trigger_float(value_float=min(self.target_throttle, 0.5))  # value between 0 and 1
        self.gamepad.left_trigger_float(value_float=self.target_brake)  # value between 0 and 1
        self.gamepad.left_joystick_float(x_value_float=self.target_steer, y_value_float=0.0)  # range(-1, 1)  / np.pi

        self.gamepad.update()
        # time.sleep(0.1)

    def update_target(self, car):
        # Longitudinal Controller
        epsilon = 0.03
        k_p = 0.2  # 0.7
        k_i = 0.01  # 0.05
        k_d = 0.0
        a = 0
        t = get_time()

        if self.mode == "pit_lane":
            l = 5
            v_desired = 50  # update this to dynamic
            waypoints = self.pit_lane
            min_idx, min_dist = self.find_closest_waypoint(waypoints, car.location)
            _, dist = self.find_closest_waypoint(self.fast_ai, car.location)
            if min_dist > dist or min_idx > 880:
                self.mode = "track"
        else:
            l = 10
            v_desired = 80  # update this to dynamic
            waypoints = self.fast_ai
            min_idx, _ = self.find_closest_waypoint(waypoints, car.location)

        self.I_previous += (t - self.t_previous) * (v_desired - car.speed)
        a = k_p * (v_desired - car.speed) + k_i * self.I_previous + k_d * (v_desired - car.speed) / (
                    t - self.t_previous)
        self.t_previous = t

        # if car.speed > 0:
        #     throttle_output = self.vars.throttle_previous + epsilon * (v_desired - car.speed) / car.speed
        # else:
        #     throttle_output = 10

        if a > 0:
            self.target_throttle = a
            self.target_brake = 0
        else:
            self.target_throttle = 0
            self.target_brake = -a

        self.target_throttle = max(min(self.target_throttle, 1.0), 0.0)

        # Lateral Controller
        # PID controller parameters
        desired_heading = None
        e = 0  # cross-track error
        # l = 25  # lookahead distance; this is declared above
        k = 0.1
        k_s = 2

        target_idx = (min_idx + l) % waypoints.shape[0]
        target_location = waypoints[target_idx, :]

        desired_heading = np.arctan((target_location[0] - car.location[0]) / (target_location[2] - car.location[2]))
        alpha = desired_heading - car.heading
        l_d = np.linalg.norm(np.array([waypoints[target_idx, :3] - car.location]))
        e = np.sin(alpha) * l_d

        if alpha > 1.5:
            alpha -= np.pi
        elif alpha < -1.5:
            alpha += np.pi

        print(f"Current: {car.location}\tClosest: {waypoints[min_idx, :]}\tTarget: {waypoints[target_idx, :]}")
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
        new_waypoints = np.hstack((waypoints, np.ones((waypoints.shape[0], 1)) * 20))

        return new_waypoints
