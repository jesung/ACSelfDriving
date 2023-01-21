import numpy as np
import vgamepad as vg
import time
from track import Track


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


class Controller:
    waypoints = None
    v_desired = None
    index = 0
    gamepad = None
    # mode = "pit_lane"
    target_throttle = 0
    target_steer = 0
    target_brake = 0
    t_previous = get_time()
    I_previous = 0
    throttle_previous = 0

    def __init__(self, path, velocity):
        self.waypoints = path.transpose()
        self.v_desired = velocity * 3.6       # convert to km/h
        self.gamepad = vg.VX360Gamepad()

        # press and release a button to have the controller be recognized by the game
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)

    def set_controller(self):
        """
        Sets and updates the controller values based on the target throttle, brake, and steer values
        """
        # print(f"Throttle: {self.target_throttle}\t Brake: {self.target_brake}\t Steer: {self.target_steer}")
        self.gamepad.right_trigger_float(value_float=self.target_throttle)  # value between 0 and 1
        self.gamepad.left_trigger_float(value_float=self.target_brake)  # value between 0 and 1
        self.gamepad.left_joystick_float(x_value_float=self.target_steer, y_value_float=0.0)  # range(-1, 1)  / np.pi

        self.gamepad.update()

    def update_target(self, vehicle):
        # Longitudinal Controller
        epsilon = 0.03   # 0.03
        k_p = 0.7  # 0.7
        k_i = 0.00  # 0.05
        k_d = 0.0
        t = get_time()
        self.throttle_previous = self.target_throttle

        # code to switch reference waypoint from pit lane to track
        # if self.mode == "pit_lane":
        #     lookahead_distance = 5
        #     target_velocity = 60
        #     waypoints = self.track.pit_lane
        #     # inefficient but only happens while in pit lane
        #     min_idx, min_dist = find_closest_waypoint(waypoints, vehicle.location)
        #     _, dist = find_closest_waypoint(self.track.fast_ai, vehicle.location)
        #     if min_dist > dist or min_idx > 880:
        #         self.mode = "track"
        #         self.track.v_desired = self.track.compute_v_desired(self.track.fast_ai, vehicle)
        #         print("Changing mode to track")
        # else:
        lookahead_distance = 10 # 15  # 17 for brz
        # waypoints = self.track.fast_ai
        # min_idx, _ = find_closest_waypoint(self.waypoints, vehicle.location)
        # print(self.waypoints.shape)
        # print(vehicle.location.shape)
        min_idx, min_distance = find_closest_waypoint(self.waypoints, vehicle.location)
        # print("Min loc:", min_idx, "Min dist:", min_distance)
        target_velocity = self.v_desired[min_idx]

        self.I_previous += (t - self.t_previous) * (target_velocity - vehicle.speed)
        a = k_p * (target_velocity - vehicle.speed) + k_i * self.I_previous + k_d * (target_velocity - vehicle.speed) / (
                    t - self.t_previous)
        self.t_previous = t

        # if vehicle.speed > 0:
        #     self.target_throttle = self.throttle_previous + epsilon * (target_velocity - vehicle.speed) / vehicle.speed
        # else:
        #     self.target_throttle = 10

        self.target_throttle = max(a, 0)
        self.target_brake = -min(a, 0)

        self.target_throttle = max(min(self.target_throttle, 1.0), 0.0)
        self.target_brake = max(min(self.target_brake, 1.0), 0.0)

        print("Target speed:", target_velocity, "\tActual speed:", vehicle.speed, "\tThrottle:", self.target_throttle, "\tBrake:", self.target_brake)

        # Lateral Controller
        # PID controller parameters
        desired_heading = None
        e = 0  # cross-track error
        # lookahead_distance = 25   # this is declared above
        k = 0.1
        k_s = 2

        target_idx = (min_idx + lookahead_distance) % self.waypoints.shape[0]
        target_location = self.waypoints[target_idx, :]

        desired_heading = np.arctan((target_location[0] - vehicle.location[0]) / (target_location[2] - vehicle.location[2]))
        # print(desired_heading)
        alpha = desired_heading - vehicle.heading
        l_d = np.linalg.norm(np.array([self.waypoints[target_idx, :3] - vehicle.location]))
        e = np.sin(alpha) * l_d

        if alpha > 1.5:
            alpha -= np.pi
        elif alpha < -1.5:
            alpha += np.pi

        # print(f"Current: {vehicle.location}\tClosest: {waypoints[min_idx, :]}\tTarget: {waypoints[target_idx, :]}")
        # print(f"Min index: {min_idx}")
        # print(f"Current heading: {vehicle.heading}\t\t\tDesired heading: {desired_heading}, Alpha: {alpha}")
        # print(f"Mode: {self.mode}")

        if vehicle.speed > 0:
            steer_target = alpha  # + np.arctan(k*e/(k_s + vehicle.speed))
            # steer_output = self.vars.steer_previous + (steer_target - self.vars.steer_previous) * 0.1
            steer_output = steer_target
        else:
            steer_output = 0

        # Change the steer output with the lateral controller.
        self.target_steer = steer_output  # max(min(steer_output, 1.22), -1.22)
        # print("steer_output",steer_output)

        self.set_controller()

