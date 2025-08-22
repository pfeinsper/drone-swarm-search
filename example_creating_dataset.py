from DSSE import H5DatasetBuilder
from datetime import datetime, timedelta, timezone
import time

""" 
This script demonstrates how to create a dataset using the H5DatasetBuilder class from the DSSE package.
It is used to create the dataset that is used in the DSSE package under the `DSSE/environment/TrajectoryDatasets` directory.
"""

# Example of creating a dataset using H5DatasetBuilder
# Initialize the H5DatasetBuilder with the path to the dataset and folders for netCDF and image files
dataset_builder = H5DatasetBuilder(
    "DSSE/environment/TrajectoryDatasets/sample_dataset.h5",
    "DSSE/environment/TrajectoryDatasets/sample_dataset_nc_files",
    "DSSE/environment/imgs/sample_dataset_imgs",
    use_env=True,
)

# Create the dataset under DSSE/environment/TrajectoryDatasets/sample_dataset.h5
dataset_builder.create_or_check_h5_dataset()

# Creates the necessary folders for netCDF and image files
# netCDF files will be stored in DSSE/environment/TrajectoryDatasets/sample_dataset_nc_files
# Images will be stored in DSSE/environment/imgs/sample_dataset_imgs
dataset_builder.create_or_check_nc_folder()
dataset_builder.create_or_check_img_folder()

# I compiled some latitudes, longitudes, and times of events in order to ensure the dataset
# encompasses a variety of conditions.
# Sources: https://www.nhc.noaa.gov/data/tcr/AL082023_Franklin.pdf, https://www.nhc.noaa.gov/data/tcr/AL142024_Milton.pdf,
# https://www.nhc.noaa.gov/data/tcr/AL022024_Beryl.pdf
events = [
    ("HurricaneFranklinTropicalStorm", 14, -65.2, datetime(2023, 8, 20, 12)),
    ("HurricaneFranklinCat1", 23.2, -66.5, datetime(2023, 8, 26, 12)),
    ("HurricaneFranklinCat4", 28.9, -71.1, datetime(2023, 8, 29, 0)),
    ("HurricaneMiltonTropicalDepression", 22.0, -95.5, datetime(2024, 10, 5, 12)),
    ("HurricaneMiltonTropicalStorm", 22.5, -95.5, datetime(2024, 10, 5, 18)),
    ("HurricaneMiltonCat3", 22.1, -92.9, datetime(2024, 10, 7, 6)),
    ("HurricaneMiltonCat5", 21.8, -90.9, datetime(2024, 10, 7, 20)),
    ("HurricaneBerylTropicalStorm", 9.2, -43.1, datetime(2024, 6, 29, 0)),
    ("HurricaneBerylCat1", 10.1, -50.5, datetime(2024, 6, 30, 0)),
    ("HurricaneBerylCat5", 13.5, -64.1, datetime(2024, 7, 2, 0)),
    ("AtlanticOceanRandom", 36.316592, -42.708463, datetime(2025, 6, 5, 0)),
    ("PacificOceanRandom", -11.739559, -159.901656, datetime(2025, 6, 5, 0)),
]

# Add multiple examples to the dataset using the predefined events
for i, (name, lat, long, event_time) in enumerate(events):
    print(
        f"Adding example {i + 1}: {name} at ({lat}, {long}) on {event_time.isoformat()}"
    )
    start_time = time.time()  # Record the start time for performance measurement
    dataset_builder.add_one_example(
        disaster_lat=lat,
        disaster_long=long,
        start_time=event_time,
        duration_hours=15,  # The simulation will run for 15 hours from the event time
        loglevel=0,  # Set log level to 0 for no logging. Higher levels will produce more logs.
        particle_amount=17000,  # Number of particles to simulate per object. So if there are 3 object types, this will create 51000 particles.
        particle_radius=1000,  # The area in which a particle can be placed around the disaster location. This is kinda of like variance in a a gaussian distribution.
        object_types=[
            1,
            13,
            55,
        ],  # The object types to simulate found in the Leeway drift model. 1 is a person in water, 13 is a life raft, and 55 is a fishing vessel.
        time_step=timedelta(
            minutes=30
        ),  # This is the time between calculations in the computer model.
        time_step_output=timedelta(
            minutes=30
        ),  # This is how often an output is written to a file. For example, the model can calculate with 15-minute timesteps, but only report every hour
        wind_uncertainity=0.2,  # Measure of inherent uncertainty in the wind obseration data
        current_uncertainity=0.2,  # Measure of inherent uncertainty in the current obseration data
        horizontal_diffusion=10,  # measure how how particles spread over time due to inherent uncertainty in the model
        save_img=i % 2
        == 0,  # Save images for every second example. Note that this take a lot of time, even more that just the simulation.
        name=name,
    )
    print(f"Example {i + 1} added in {time.time() - start_time:.2f} seconds")

dataset_builder.list_examples()  # List all examples in the dataset
