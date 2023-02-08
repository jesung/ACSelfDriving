# ACSelfDriving
## Description
Python code to interact with Assetto Corsa app [AI Driver](https://github.com/jesung/aidriver). This repository largely has two functions: 
1. Compute optimized racing line by minimizing curvature or lap time
2. Take the computed racing line and drive a car within the game Assetto Corsa

## Versions
### 1.0
* Communicate with [AI Driver](https://github.com/jesung/aidriver) app over local socket.
* Helper functions to
  * Load arbitrary track boundary waypoints and split waypoints every 10 meters
  * Read car details such as mass and tire coefficient of friction
    * Tire coefficient of friction is for the default tire
  * Calculate car engine mapping (speed -> force applied to ground) across full forward gear range
* Updated velocity calculation to account for change in slope on track (3D)
* Navigate out of pit lane
* Brake modulation to account for loss of total grip available due to load transfer
* Allow for loading of previously computed optimal lines in the optimization process
