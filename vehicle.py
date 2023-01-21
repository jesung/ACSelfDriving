import numpy as np
import scipy.constants
from math import sqrt
import ac_utils as ac


class Vehicle:
    location = [0, 0, 0]
    speed = 0
    heading = 0
    mass = None
    engine_profile = None
    tire_coefficient = None

    def __init__(self, name):
        self.mass, self.tire_coefficient, self.engine_profile = ac.load_vehicle(name)

    def __repr__(self):
        return f"[X, Y, Z]: {self.location} \tHeading: {self.heading} \tSpeed: {self.speed}"

    def update(self, socket_data):
        ##############################################################
        # TO-DO: update tire coefficient and engine profile
        ##############################################################

        # cutting off at five columns allows for overflow of data
        x, y, z, self.heading, self.speed = [float(i) if i != '' else 0 for i in socket_data.decode('utf8').split(',')[:5]]

        # to-do: include code to pull other car and environment state data (e.g., tire, weather)

        # update location with flipped z-axis
        self.location = np.array([x, y, -z])

    def engine_force(self, velocity, gear=None):
        """Map current velocity to force output by the engine."""
        return np.interp(velocity, self.engine_profile[:, 0], self.engine_profile[:, 1])

    def traction(self, velocity, curvature):
        """Determine remaining traction when negotiating a corner."""
        ##############################################################
        # TO-DO: update to account for different lateral & longitudinal mu
        ##############################################################
        f = self.tire_coefficient * scipy.constants.g * self.mass
        f_lat = velocity ** 2 * curvature * self.mass
        if f <= f_lat:
            ##############################################################
            # TO-DO: update tire coefficient with changing track conditions
            ##############################################################
            return 0
        return sqrt(f**2 - f_lat**2)
