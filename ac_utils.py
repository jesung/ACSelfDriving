import os
import numpy as np
import csv
from typing import Tuple        # , Dict, Any


def load_alphas(file: str) -> np.ndarray:
    """Load precomputed values of alphas"""
    with open(file, 'r') as f:
        reader = csv.reader(f)
        alphas = np.array([float(i) for row in reader for i in row]).transpose()
    return alphas


def load_csv(track: str) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Read waypoint/boundary files.

    Note that the default AC format is 3 columns (XYZ) whereas the output from AI Line Helper app
    (https://www.racedepartment.com/downloads/ai-line-helper.16016/) has 4 columns (XYZ+relative position). It is
    possible to use the extract game's default AI line and track boundaries from fast_lane.ai using Blender
    (https://github.com/leBluem/io_import_accsv)

    Parameters:
        track (str): name of the track folder in ./content/tracks/

    Returns:
        left (np.ndarray):waypoints for left track boundary
        right (np.ndarray): waypoints for right track boundary
        fast_ai (np.ndarray): waypoints for default AI line
        pit_lane (np.ndarray): waypoints for pit lane
    """

    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "content")
    dir_name = os.path.join(dir_name, "tracks")
    dir_name = os.path.join(dir_name, track)

    # only read the first three columns of AI file
    with open(os.path.join(dir_name, "fast_ai.csv"), "r") as f:
        reader = csv.reader(f)
        fast_ai = np.array([float(i) for row in reader for i in row[:3]]).reshape((-1, 3)).transpose()

    with open(os.path.join(dir_name, "pit_lane.csv"), "r") as f:
        reader = csv.reader(f)
        pit_lane = np.array([float(i) for row in reader for i in row[:3]]).reshape((-1, 3)).transpose()

    with open(os.path.join(dir_name, "left.csv")) as f:
        reader = csv.reader(f)
        # left = np.array([float(i) for row in reader for i in row]).reshape((-1, 3)).transpose()
        left = np.array([float(i) for row in reader for i in row]).reshape((-1, 4)).transpose()

    with open(os.path.join(dir_name, "right.csv")) as f:
        reader = csv.reader(f)
        # right = np.array([float(i) for row in reader for i in row]).reshape((-1, 3)).transpose()
        right = np.array([float(i) for row in reader for i in row]).reshape((-1, 4)).transpose()

    return left[:3, :], right[:3, :], fast_ai, pit_lane


def load_ini(file: str) -> dict[str, dict[str, str]]:
    """
    Load files with extension .ini from the game data.

    Example from tyres.ini:
        [FRONT]
        NAME=ECO
        RADIUS=0.3126				; tyre radius in meters
    Note that input after semicolon ';' is a comment and thus stripped out during extraction. You can access radius
    value with data["FRONT"]["RADIUS"]. This function stores all variables as string and thus appropriate conversions
    will need to be applied downstream.

    Parameters:
        file (str): name of the .ini file in ./content/tracks/track_name

    Returns:
        A nested dictionary of headers and values in the .ini file
    """

    with open(file) as f:
        data = {}
        reader = csv.reader(f, delimiter='=')
        for row in reader:
            if not row:
                continue
            if row[0][0] == '[':
                header = row[0].strip(" []")
                data[header] = {}
            elif row[0][0] == ';':
                pass
            else:
                data[header][row[0]] = row[1].split(";", 1)[0]
    return data


def load_lut(file: str) -> np.ndarray:
    """
    Load files with extension .lut from the game data

    All values are separated with the pipe delimiter '|' and are numeric.

    Parameters:
        file (str): Name of the .lut file in ./content/tracks/track_name

    Returns:
        A two-column numpy array of two numeric variables (e.g., engine rpm and engine torque).
    """
    with open(file) as f:
        data = []
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if not row:
                continue
            data.append([float(row[0]), float(row[1])])
    return np.array(data)


def load_vehicle(vehicle: str):
    """
    Load vehicle parameters mass, tire coefficient of friction, and engine force map.

    Compute vehicle's force production profile [speed (m/s), force (N)] across the gear range. Note that the wheel
    radius is from the driven wheel.

    Formulae used:
    speed (m/s) = rpm(engine rotation) / 60 / gear ratio / final ratio * 2 * pi * radius
    force applied to ground by wheel = engine torque * gear ratio / wheel radius

    Parameters:
        vehicle (str): name of the vehicle folder in ./content/cars/

    Returns:
        mass (float): mass of the vehicle in kg
        tire_coefficient (float): coefficient of friction of the driven wheel
        engine_profile (np.ndarray): mapping of vehicle speed (m/s) to wheel force (N)
    """

    dir_name = os.path.join(os.path.dirname(__file__), "content", "cars", vehicle)

    tyres = load_ini(os.path.join(dir_name, "tyres.ini"))
    car = load_ini(os.path.join(dir_name, "car.ini"))
    drivetrain = load_ini(os.path.join(dir_name, "drivetrain.ini"))
    power = load_lut(os.path.join(dir_name, "power.lut"))

    if drivetrain["TRACTION"]["TYPE"] == "FWD":
        driven_wheel = "FRONT"
    else:
        driven_wheel = "REAR"

    tire_radius = float(tyres[driven_wheel]["RADIUS"])

    if int(tyres["COMPOUND_DEFAULT"]["INDEX"]) == 0:
        tyre = driven_wheel
    else:
        tyre = driven_wheel + "_" + tyres["COMPOUND_DEFAULT"]["INDEX"]

    mass = float(car["BASIC"]["TOTALMASS"])
    tire_coefficient = float(tyres[tyre]["DY_REF"])     # return DY_REF only for now. DX_REF tends to be slightly higher
    engine_profile = compute_engine_profile(drivetrain, tire_radius, power)

    return mass, tire_coefficient, engine_profile


def compute_engine_profile(drivetrain: dict[str, dict[str, str]], tire_radius: float, power: np.ndarray) -> np.ndarray:
    """
    Helper function for load_vehicle(). Computes speed -> wheel force mapping for the car specified in parent function.

    Parameters:
        drivetrain (dict[str, dict[str, str]]): stored data from drivetrain.ini
        tire_radius (float): radius of driven wheel's tires
        power (np.ndarray): stored data from power.lut. Maps engine rpm to engine torque (Nm).

    Returns:
        Mapping of speed (m/s) to force applied by the driven wheel at that speed. Assumes that shifting occurs at
        the default upshift rpm.
    """
    tire_circumference = 2 * np.pi * tire_radius
    num_gears = int(drivetrain["GEARS"]["COUNT"])
    final_drive = float(drivetrain["GEARS"]["FINAL"])
    up_shift = int(drivetrain["AUTO_SHIFTER"]["UP"])
    # down_shift = int(drivetrain["AUTO_SHIFTER"]["DOWN"])

    gears = []
    engine_profile = []

    for i in range(num_gears):
        gear = "GEAR_" + str(i + 1)
        gears.append(float(drivetrain["GEARS"][gear]))

    speed = 0
    epsilon = 1e-02

    for i in range(num_gears):
        ratio = gears[i] * final_drive
        speed += epsilon
        multiplier = 60 * ratio / tire_circumference
        rpm = speed * multiplier

        while rpm < up_shift:
            engine_profile.append([rpm / multiplier, np.interp(rpm, power[:, 0], power[:, 1]) * ratio / tire_radius])
            rpm += 250      # increment by 250 rpm until the up-shift threshold

        speed = up_shift / multiplier
        engine_profile.append([speed, np.interp(up_shift, power[:, 0], power[:, 1]) * ratio / tire_radius])

    return np.array(engine_profile)
