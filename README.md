# ACSelfDriving
## Description
Python code to interact with Assetto Corsa app [AI Driver](https://github.com/jesung/aidriver). This repository largely has two functions: 
1. Compute optimized racing line by minimizing curvature or lap time
2. Take the computed racing line and drive a car within the game Assetto Corsa


## How to get started
<ul>
					<li>Install Assetto Corsa (<a href="https://store.steampowered.com/app/244210/Assetto_Corsa/" target="_blank">steam</a>)</li>
					<li>Download <a href="https://github.com/jesung/aidriver" target="_blank">AI Driver app</a> for Assetto Corsa</li>
					<li>(optional) Download <a href="https://assettocorsa.club/content-manager.html" target="_blank">Content Manager</a></li>
						<ul>
							<li>Select car and track of choice from what's available</li>
							<li>Confirm tires are the default option</li>
							<li>Practice mode with ideal conditions</li>
						</ul>
					<li>(optional) If the track you want is not available, get <a href="https://www.racedepartment.com/downloads/ai-line-helper.16016/" target="_blank">AI Line Helper</a></li>
						<ul>
							<li>Use the “Track Left” button to map the left boundary</li>
							<li>Rename side_l.csv in the game installation directory to left.csv</li>
							<li>Use the “Track Left” button to map the right boundary</li>
							<li>Rename side_l.csv in the game installation directory to right.csv</li>
							<li>Copy into the content/tracks/track_name folder under the AC Self Driving repo below</li>
						</ul>
					<li>(optional) If the car you want is not available (<a href="https://www.youtube.com/watch?v=H-Fji4-boME&ab_channel=UnleashedDrivers" target="_blank">YouTube instructions</a>></li>
						<ul>
							<li>Open Content Manager and go to the About section</li>
							<li>Click "Version" multiple times to enable developer mode</li>
							<li>Go to Content -> Cars and select the car you want</li>
							<li>Click "Unpack data" in the bottom row</li>
							<li>Copy into the content/cars/car_name folder under the AC Self Driving repo below</li>
						</ul>
					<li>Download the <a href="https://github.com/jesung/ACSelfDriving" target="_blank">AC Self Driving</a> repo</li>
						<ul>
						<li>pip install -r requirements.txt</li>
						<li>Run main.py</li>
						</ul>
				</ul>

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

## Acknowledgements
This repository is based on https://github.com/joedavison17/dissertation
