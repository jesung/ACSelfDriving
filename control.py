import numpy as np
import vgamepad as vg
import time
from trajectory import Trajectory
from vehicle import Vehicle
from typing import Tuple        # , Dict, Any


def find_closest_waypoint(waypoints: np.ndarray, car_location: np.ndarray) -> Tuple[int, float]:
    """Return the index and distance of the closest waypoint in relation to car's current location."""
    min_distance = float("inf")
    min_index = 0

    for i in range(waypoints.shape[0]):
        point_distance = np.linalg.norm(waypoints[i, :3] - car_location[:3])

        if point_distance < min_distance:
            min_distance = point_distance
            min_index = i

    return min_index, min_distance


class Controller:
    """
    Virtual controller to interface with the game. Attempts to follow the target waypoints and velocity.

    Attributes:
        mode (str): current mode between ["pit_lane", "track"]. Default is "pit_lane"
        target_steer (float): target steer value to follow waypoints
        target_throttle (float): target throttle value to achieve desired velocity
        target_brake (float): target brake value to achieve desired velocity
        t_previous (float): timestamp of the previous update cycle
        I_previous (float): integral of the difference between target and actual velocity
        throttle_previous (float): previous value of target_throttle
        steer_previous (float): previous value of target_steer
        reference_line (np.ndarray): XYZ coordinates of the reference line
        pit_lane (np.ndarray): XYZ coordinates of pit lane line
        v_desired (np.ndarray): target velocity at each index corresponding to the reference line
        gamepad (vg.VX360Gamepad): virtual controller used as input for the game
        waypoints (np.ndarray): XYZ coordinates for line that the controller will try to follow. Default to pit lane
    """

    def __init__(self, trajectory: Trajectory) -> None:
        """
        Parameters:
            trajectory (Trajectory): Stores the geometry and dynamics of a path, handling optimisation of the racing line.
        """

        self.mode = "pit_lane"
        self.target_steer = 0.0
        self.target_throttle = 0.0
        self.target_brake = 0.0
        self.t_previous = time.time()
        self.I_previous = 0.0
        self.throttle_previous = 0.0
        self.steer_previous = 0.0

        self.reference_line = trajectory.path.position(trajectory.s[:-1]).transpose()
        self.pit_lane = trajectory.track.pit_lane.T
        self.v_desired = trajectory.velocity.v * 3.6       # convert from m/s to km/h
        self.gamepad = vg.VX360Gamepad()
        self.waypoints = self.pit_lane

        # press and release a button to have the controller be recognized by the game
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)

    def set_controller(self) -> None:
        """Sets and updates the controller values based on the target throttle, brake, and steer values."""

        self.gamepad.right_trigger_float(value_float=self.target_throttle)  # value between 0 and 1
        self.gamepad.left_trigger_float(value_float=self.target_brake)  # value between 0 and 1
        self.gamepad.left_joystick_float(x_value_float=self.target_steer, y_value_float=0.0)  # range(-1, 1)  / np.pi

        self.gamepad.update()

    def update_target(self, vehicle: Vehicle) -> None:
        """
        Computes target steer, throttle, and brake values based on target line & velocity, and vehicle position &
        speed.

        Parameters
            vehicle (Vehicle):
        """
        # Longitudinal Controller. PID controller parameters
        # epsilon = 0.7   # 0.03
        k_p = 0.3           # 0.7 default
        k_i = 0.00          # 0.05 default. Zero this out to not drift the target throttle/brake over time.
        k_d = 0.0

        lookahead_distance = 21         # 15~30 seems to work
        t = time.time()

        self.throttle_previous = self.target_throttle
        self.steer_previous = self.target_steer

        min_idx, min_dist = find_closest_waypoint(self.waypoints, vehicle.location)

        # code to switch reference waypoint from pit lane to track
        if self.mode == "pit_lane":
            lookahead_distance = 5
            target_velocity = 60
            _, dist = find_closest_waypoint(self.reference_line, vehicle.location)

            if min_dist > dist or min_idx > 880:
                self.mode = "track"
                self.waypoints = self.reference_line
                print("Changing mode to track")
        else:
            target_velocity = self.v_desired[min_idx]

        self.I_previous += (t - self.t_previous) * (target_velocity - vehicle.speed)
        a = k_p * (target_velocity - vehicle.speed) + k_i * self.I_previous + k_d * (target_velocity - vehicle.speed) \
            / (t - self.t_previous)
        self.t_previous = t

        self.target_throttle = max(a, 0)
        self.target_throttle = max(min(self.target_throttle, 1.0), 0.0)

        self.target_brake = -min(a, 0)
        self.target_brake = max(min(self.target_brake, 1.0), 0.0)

        # # Lateral Controller
        # k = 0.1     # 0.1
        # k_s = 2.0

        target_idx = (min_idx + lookahead_distance) % self.waypoints.shape[0]
        target_location = self.waypoints[target_idx, :]

        desired_heading = np.arctan((target_location[0] - vehicle.location[0]) / (target_location[2] -
                                                                                  vehicle.location[2]))
        alpha = desired_heading - vehicle.heading
        # l_d = np.linalg.norm(np.array([self.waypoints[target_idx, :3] - vehicle.location]))
        # e = np.sin(alpha) * l_d     # cross-track error

        if alpha > 1.5:
            alpha -= np.pi
        elif alpha < -1.5:
            alpha += np.pi

        if vehicle.speed > 0:
            steer_target = alpha        # + np.arctan(k*e/(k_s + vehicle.speed))
            # steer_output = self.steer_previous + (steer_target - self.steer_previous) * 0.67
            steer_output = steer_target
        else:
            steer_output = 0

        # Change the steer output with the lateral controller.
        self.target_steer = steer_output
        self.set_controller()
