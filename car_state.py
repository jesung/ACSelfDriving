import numpy as np


class CarState:
    location = [0, 0, 0]
    speed = 0
    heading = 0

    # def __init__(self):
    #     None

    def __repr__(self):
        return f"[X, Y, Z]: {self.location} \tHeading: {self.heading} \tSpeed: {self.speed}"

    def update(self, socket_data):
        # cutting off at five columns allows for overflow of data
        x, y, z, self.heading, self.speed = [float(i) for i in socket_data.decode('utf8').split(',')[:5]]

        # to-do: include code to pull other car and environment state data (e.g., hp & torque, tire, weather)

        # flip z-axis
        self.location = np.array([x, y, -z])
