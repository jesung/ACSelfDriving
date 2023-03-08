import csv
import pandas as pd


class Replay:
    history = []
    prev_position = 0.0

    def __init__(self):
        self.history.append(["X", "Y", "Z", "throttle", "brake", "steer", "speed", "position", "lap_time"])

    def save(self, file) -> None:
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            for r in self.history:
                writer.writerow(r)
        print("Saved replay to", file)

    def update(self, vehicle) -> None:
        if vehicle.position != self.prev_position:
            self.history.append([vehicle.location[0], vehicle.location[1], vehicle.location[2],
                                 round(vehicle.throttle, 3), round(vehicle.brake, 3),
                                 round(vehicle.steer, 3), vehicle.speed, vehicle.position, vehicle.lap_time])

        self.prev_position = vehicle.position

    def load(self, file, driver="default"):
        df = pd.read_csv(file)
        df = df.reset_index()
        lap_counter = -1
        lap = []
        label = []
        prev_position = 0.0

        for index, row in df.iterrows():
            if row['position'] < prev_position:
                lap_counter += 1
            prev_position = row['position']
            lap.append(lap_counter)
            label.append(driver)
            if row["position"] <= 0.001:
                df.loc[index, "lap_time"] = 0

        df['lap'] = lap
        df["driver"] = label

        print("Loaded replay", file)

        return df
