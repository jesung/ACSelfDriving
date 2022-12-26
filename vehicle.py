import numpy as np
import scipy.constants
from math import sqrt


class Vehicle:
    location = [0, 0, 0]
    speed = 0
    heading = 0
    mass = 1000
    engine_profile = None
    tire_coefficient = 1.2

    def __init__(self):
        self.engine_profile = [[5.0,   10.0,   15.0,   20.0,   25.0,   30.0,   35.0], [5000.0, 4700.0, 3500.0, 2800.0, 2300.0, 1900.0, 1600.0]]

    def __repr__(self):
        return f"[X, Y, Z]: {self.location} \tHeading: {self.heading} \tSpeed: {self.speed}"

    def update(self, socket_data):
        ##############################################################
        # TO-DO: update tire coefficient and engine profile
        ##############################################################

        # cutting off at five columns allows for overflow of data
        x, y, z, self.heading, self.speed = [float(i) if i != '' else 0 for i in socket_data.decode('utf8').split(',')[:5]]

        # to-do: include code to pull other car and environment state data (e.g., hp & torque, mass, tire, weather)

        # flip z-axis
        self.location = np.array([x, y, -z])

    def engine_force(self, velocity, gear=None):
        """Map current velocity to force output by the engine."""
        return np.interp(velocity, self.engine_profile[0], self.engine_profile[1])

    def traction(self, velocity, curvature):
        """Determine remaining traction when negotiating a corner."""
        ##############################################################
        # TO-DO: update to 3-axis calculation based on surface slope #
        ##############################################################

        f = self.tire_coefficient * scipy.constants.g * self.mass
        f_lat = velocity ** 2 * curvature * self.mass
        if f <= f_lat:
            ##############################################################
            # TO-DO: update tire coefficient; PID controller?
            ##############################################################
            return 0
        return sqrt(f**2 - f_lat**2)
