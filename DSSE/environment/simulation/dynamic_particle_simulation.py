import math
import numpy as np
from datetime import datetime, timedelta
from typing import List
import os
import dotenv
from datetime import datetime, timedelta
import copernicusmarine
from opendrift.models.leeway import Leeway
from opendrift.readers.reader_netCDF_CF_generic import Reader
from copy import deepcopy

EARTH_MEAN_RADIUS = 6373.0


class DynamicParticleSimulation:
    def __init__(
        self,
        disaster_lat: float,
        disaster_long: float,
        start_time: datetime,
        duration_hours: int = 10,
        loglevel: int = 20,
        particle_amount: int = 50_000,
        particle_radius: int = 1000,  # in meters
        object_types: List[int] = [1],
        time_step=timedelta(hours=1),
        time_step_output=timedelta(minutes=60),
        wind_uncertainity: float = 0.1,
        current_uncertainity: float = 0.1,
        horizontal_diffusion: float = 10,
    ) -> None:
        """
        Creates a particle simulation and .nc file given latititude, longitude, and time.

        :param disaster_lat: Lattitude of the disaster. North is positive, south is negative
        :type disaster_lat: float
        :param disaster_long: Longitude of the disaster. East is positive, West is negative
        :type disaster_long: float
        :param start_time: datetime variable that is the time when the disaster starts
        :type start_time: datetime
        :param duration_hours: How many hours to simulate
        :type duration_hours: int
        :param loglevel: How much the OpenDrift model should log. 0 is most, higher is less logging
        :type loglevel: int
        :param particle_amount: How maparticle_amount: int
        :param particle_radius: What radius away from the starting position particles should be spread
        :type particle_radius: int
        :param object_types: What leeway model object types should be spawned in. Each object will have "particle amount" of particles spawned in
        :type object_types: List[int]
        :param time_step: How often the simulation should update meteorological and oceanic parameters to simulate particles
        :type time_step: timedelta
        :param time_step_output: How often the particles should be outputted in the .nc file. Must be greater than or equal to time_step
        :type time_step_output: timedelta
        :param wind_uncertainity: Wind uncertainity in m/s
        :type wind_uncertainity: float
        :param current_uncertainity: Current uncertainity in m/s
        :type current_uncertainity: float
        :param horizontal_diffusion: Horizontal diffusion in horizontal diffusion units
        :type horizontal_diffusion: float
        """

        self.disaster_lat = disaster_lat
        self.disaster_long = disaster_long
        self.start_time = start_time

        if start_time.year < 2022:
            raise ValueError(
                "Start time must be greater than or equal to 2022, as the "
                "NCEP Global Atmospheric Model/Best Time Series dataset only has data from 2022 onwards"
            )

        self.loglevel = loglevel
        self.duration_hours = duration_hours
        self.particle_amount = particle_amount
        self.particle_radius = particle_radius
        self.object_types = object_types
        self.time_step = time_step
        self.time_step_output = time_step_output
        self.current_uncertainity = current_uncertainity
        self.wind_uncertainity = wind_uncertainity
        self.horizontal_diffusion = horizontal_diffusion

    def init(self, username=None, password=None, use_env=True):
        if use_env:
            if username is not None or password is not None:
                raise ValueError(
                    "Username and password must not be provided if use_env is True"
                )
            # Load environment variables from .env file
            dotenv.load_dotenv()
            username = os.getenv("USERNAME")
            password = os.getenv("PASSWORD")
        else:
            if username is None or password is None:
                raise ValueError(
                    "Username and password must be provided if use_env is False"
                )

        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
            chunk_size_limit=0,
            username=username,
            password=password,
        )
        print(ds)  # Default Xarray output
        print(ds.cf)  # Output from cf-xarray

        # Creating and summing readers
        reader_total = Reader(
            ds,
            standard_name_mapping={
                "utotal": "x_sea_water_velocity",
                "vtotal": "y_sea_water_velocity",
            },
            name="Total current",
        )

        self.reader_currents = reader_total

        # Works from 2022-12-01T12:00:00Z to 2025-06-18T12:00:00Z
        self.reader_winds = Reader(
            "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ncep_global/NCEP_Global_Atmospheric_Model_best.ncd",
            name="Wind Reader",
        )

    def simulate(
        self,
        outfile_img=None,
        plot=False,
        outfile_nc=os.path.abspath("DSSE/environment/TrajectoryDatasets/sar_sim.nc"),
    ):
        l = Leeway(loglevel=100)
        print(Leeway.required_variables)
        try:
            l.add_reader(
                self.reader_currents,
                variables=["x_sea_water_velocity", "y_sea_water_velocity"],
            )
            l.add_reader(self.reader_winds, variables=["x_wind", "y_wind"])
        except NameError:
            print("You must initialize the simulation")
            raise AssertionError

        for object in self.object_types:
            print(self.disaster_long, self.disaster_lat)
            l.seed_elements(
                lon=self.disaster_long,
                lat=self.disaster_lat,
                time=self.start_time,
                object_type=object,
                **{"number": self.particle_amount, "radius": self.particle_radius},
            )

        l.set_config("drift:current_uncertainty", self.current_uncertainity)
        l.set_config("drift:wind_uncertainty", self.wind_uncertainity)
        l.set_config("drift:horizontal_diffusivity", self.horizontal_diffusion)

        l.run(
            duration=timedelta(hours=self.duration_hours),
            time_step=self.time_step,
            time_step_output=self.time_step_output,
            outfile=outfile_nc,
            export_variables=["object_type"],
        )  # outfile records required variables like lat, long, id, and status
        print("Simulation finished. Output file:", outfile_nc)

        if plot:
            l.plot(filename=outfile_img)
            print("Plot saved to:", outfile_img)


if __name__ == "__main__":
    import random

    # Create ten .nc files for testing
    outfile_nc_folder = "DSSE/environment/TrajectoryDatasets/temp_nc_files/"
    outfile_img_folder = "DSSE/environment/TrajectoryDatasets/sample_plot_imgs/"

    for i in range(4, 5):
        outfile_nc = os.path.abspath(
            outfile_nc_folder + "test_trj_one_particle" + str(i) + ".nc"
        )
        outfile_img = os.path.abspath(
            outfile_img_folder + "test_trj_one_particle" + str(i) + ".jpg"
        )
        # latitude betwen 32N and 14N
        # longitude between -180, -146 E
        dis_time = datetime(year=2024, month=10, day=9, hour=20)
        lat = 27 + 11 / 60 + 45 / 3600
        long = -(82 + 52 / 60 + 40 / 3600)
        sim = DynamicParticleSimulation(
            lat,
            long,
            dis_time,
            particle_amount=100,
            object_types=[1],
            particle_radius=0,
        )
        sim.init()
        sim.simulate(outfile_nc=outfile_nc, outfile_img=outfile_img, plot=True)
