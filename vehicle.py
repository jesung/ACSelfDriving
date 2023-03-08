import numpy as np
import scipy.constants
import ac_utils as ac


class Vehicle:
    """
    Represents vehicle's static and dynamic attributes.

    Static attributes such as mass, engine profile, and tire coefficient are assigned during initialization. Dynamic
    attributes such as vehicle's position, heading, and speed are refreshed using self.update().

    Attributes:
        location (tuple): XYZ
        speed (float): current speed of vehicle (m/s)
        position (float): position of the car on the track in normalized [0,1]
        mass (float): mass of the vehicle in kg
        engine_profile (np.ndarray): mapping of vehicle speed (m/s) to wheel force (N)
        tire_coefficient (float): coefficient of friction of the driven wheel
    """

    location = [0.0, 0.0, 0.0]
    heading = 0.0
    speed = 0.0
    position = 0.0
    lap_time = 0
    throttle = 0.0
    brake = 0.0
    steer = 0.0

    def __init__(self, name: str) -> None:
        self.mass, self.tire_coefficient, self.engine_profile = ac.load_vehicle(name)
        self.tire_coefficient *= 0.95       # allows car to follow the reference line better

    def __repr__(self) -> str:
        """Overwrites print(vehicle) function."""
        return f"[X, Y, Z]: {self.location}\tHeading: {self.heading}  \tSpeed: {self.speed} \tPosition: {self.position}"

    def update(self, socket_data: bytes) -> None:
        # cutting off at five columns allows for overflow of data
        x, y, z, self.heading, self.speed, self.position, self.lap_time, self.throttle, self.brake, self.steer = \
            [float(i) if i != '' else 0 for i in socket_data.decode('utf8').split(',')[:10]]

        # update location with flipped z-axis. Not sure why Kunos uses such odd axes between game and track map.
        self.location = np.array([x, y, -z])

    def engine_force(self, velocity: np.float64) -> np.float64:
        """Interpolate current velocity to force output by the engine."""

        return np.interp(velocity, self.engine_profile[:, 0], self.engine_profile[:, 1])

    def traction(self, velocity: np.float64, curvature: np.float64) -> np.float64:
        """
        Determine remaining traction when negotiating a corner.

        This does not take into account for road camber which is accounted for during the velocity profile calculation.

        Parameters:
            velocity (np.float64): Instantaneous velocity of vehicle at a given position
            curvature (np.float64): Instantaneous curvature of path at a given position

        Returns:
            Available longitudinal traction force given total available grip and instantaneous velocity and curvature.
        """

        # TO-DO: update to account for different lateral & longitudinal mu
        f = self.tire_coefficient * scipy.constants.g * self.mass
        f_lat = velocity ** 2 * curvature * self.mass
        if f <= f_lat:
            return np.float64(0.0)
        return np.sqrt(f**2 - f_lat**2)
