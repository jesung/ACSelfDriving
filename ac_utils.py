import os
import numpy as np
import csv


def load_alphas(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        alphas = np.array([float(i) for row in reader for i in row]).transpose()
    return alphas


def load_csv(track):
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
        left = np.array([float(i) for row in reader for i in row]).reshape((-1, 3)).transpose()

    with open(os.path.join(dir_name, "right.csv")) as f:
        reader = csv.reader(f)
        right = np.array([float(i) for row in reader for i in row]).reshape((-1, 3)).transpose()

    return left, right, fast_ai, pit_lane


def load_ini(file):
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


def load_lut(file):
    with open(file) as f:
        data = []
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if not row:
                continue
            data.append([float(row[0]), float(row[1])])
    return np.array(data)


def load_vehicle(vehicle):
    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "content")
    dir_name = os.path.join(dir_name, "cars")
    dir_name = os.path.join(dir_name, vehicle)

    tyres = load_ini(os.path.join(dir_name, "tyres.ini"))
    car = load_ini(os.path.join(dir_name, "car.ini"))
    drivetrain = load_ini(os.path.join(dir_name, "drivetrain.ini"))
    power = load_lut(os.path.join(dir_name, "power.lut"))

    mass = float(car["BASIC"]["TOTALMASS"])
    tire_coefficient = float(tyres["FRONT"]["DY_REF"])  # return DY REF only for now

    if drivetrain["TRACTION"]["TYPE"] == "FWD":
        tire_circumference = 2 * np.pi * float(tyres["FRONT"]["RADIUS"])
    else:
        tire_circumference = 2 * np.pi * float(tyres["REAR"]["RADIUS"])

    num_gears = int(drivetrain["GEARS"]["COUNT"])
    final_drive = float(drivetrain["GEARS"]["FINAL"])
    up_shift = int(drivetrain["AUTO_SHIFTER"]["UP"])
    down_shift = int(drivetrain["AUTO_SHIFTER"]["DOWN"])
    gears = []
    engine_profile = []

    for i in range(num_gears):
        gear = "GEAR_" + str(i + 1)
        gears.append(float(drivetrain["GEARS"][gear]))

    speed = 0
    epsilon = 1e-02

    # compute vehicle's force production profile [speed (m/s), force (Nm)] across the gear range
    for i in range(num_gears):
        ratio = gears[i] * final_drive
        speed += epsilon
        multiplier = 60 * ratio / tire_circumference
        rpm = speed * multiplier

        # multiplying computed torque values as the formula seems to be underestimating available torque at higher speeds
        # NEED TO FIX!!! #
        while rpm < up_shift:
            engine_profile.append([rpm / multiplier, np.interp(rpm, power[:, 0], power[:, 1]) * ratio * 2])
            rpm += 250

        speed = up_shift / multiplier
        engine_profile.append([speed, np.interp(up_shift, power[:, 0], power[:, 1]) * ratio * 2])

    # rpm(engine rotation) / gear ratio / final ratio * 2pi*radius
    # engine torque * gear ratio = transmission output torque
    # engine_profile = [[5.0,   10.0,   15.0,   20.0,   25.0,   30.0,   35.0], [5000.0, 4700.0, 3500.0, 2080.0, 2300.0, 1900.0, 1600.0]]
    # print(np.array(engine_profile))

    return mass, tire_coefficient, np.array(engine_profile)
