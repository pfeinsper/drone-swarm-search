from DSSE import DynamicParticleSimulation
import random
import h5py
import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np


class H5DatasetBuilder:
    def __init__(
        self,
        dataset_path: str,
        outfile_nc_folder: str,
        outfile_img_folder: str = None,
        username: str = None,
        password: str = None,
        use_env: str = True,
    ):
        """Creates a dataset builder that can create .h5 datasets from the Copernicus Marine Service data.
        This class is used to create datasets for the DSSE environment. An important thing to note is that you need to register to the
        Copernicus Marine Service in order to run your own particle simulations. Registration is free and can be done at
        https://data.marine.copernicus.eu/register.

        :param dataset_path: The path to the dataset file, which is a .h5 file. Ex: "DSSE/environment/TrajectoryDatasets/my_dataset.h5"
        :type dataset_path: str
        :param outfile_nc_folder: The name of the folder where the .nc files will be saved. All of the .nc. files saved in this folder
        are temporary and will be deleted after the dataset is created. This folder should not be used for anything else.
        Ex: "DSSE/environment/TrajectoryDatasets/temp_nc_files"
        :type outfile_nc_folder: str
        :param outfile_img_folder: If you want to save images of the simulation, this is the folder where they will be saved.
        Ex: "DSSE/environment/TrajectoryDatasets/temp_img_files". Defaults to None, which means no images can be saved.
        :type outfile_img_folder: str
        :param username: Your usename for the Copernicus Marine Service, which you can register at
        https://data.marine.copernicus.eu/register. If you chose to modify usernameIf you leave it as none, then make sure you set use_env to true. Defaults to None
        :type username: str, optional
        :param password: Your password for the Copernicus Marine Service, which you can register at
        https://data.marine.copernicus.eu/register. If you leave it as none, then make sure you set use_env to true. Defaults to None
        :type password: str, optional
        :param use_env: If you have a .env file in the same folder where you are running this file. If this is set
        to True, username and password must be None. If this is set to False, you must set a username and password, defaults to True
        :type use_env: bool, optional
        """
        self.dataset_path = dataset_path
        self.outfile_nc_folder = outfile_nc_folder
        self.outfile_img_folder = outfile_img_folder
        # Checks to make sure that the outfile folders end in a slash, otherwise it adds one
        if not self.outfile_nc_folder.endswith("/"):
            self.outfile_nc_folder += "/"
        if (
            self.outfile_img_folder is not None
            and not self.outfile_img_folder.endswith("/")
        ):
            self.outfile_img_folder += "/"
        self.established = False
        self.username = username
        self.password = password
        self.use_env = use_env

        # Check if the dataset path exists and if the num_examples key exists.
        if os.path.exists(self.dataset_path):
            try:
                with h5py.File(self.dataset_path, "r+") as f:
                    self.cur_index: int = f["num_examples"][()]
            except KeyError:
                with h5py.File(self.dataset_path, "a") as f:
                    i = 0
                    for key in f.keys():
                        i += 1
                    f.create_dataset("num_examples", data=i)
                self.cur_index = i

        print("Initialized H5DatasetBuilder")

    def establish_connection(self) -> None:
        """Establishes a connection to the Copernicus Marine Service. This is used to check if the connection is established before running the simulation.
        If the connection is not established, it will raise an error.
        """
        if not self.established:
            sim = DynamicParticleSimulation(
                random.uniform(14, 32),
                random.uniform(-180, -146),
                datetime.now(),
                particle_amount=500,
                object_types=[1, 13, 55],
            )
            # Established connection to Copernicus Marine Service
            sim.init(
                username=self.username, password=self.password, use_env=self.use_env
            )
            self.simulation = sim
            self.established = True
            print("Connection established successfully.")
        else:
            print("Connection already established.")

    def create_or_check_h5_dataset(self) -> bool:
        """This function will create a .h5 dataset if it does not exist, or check if it exists.
        If the dataset already exists, it will return True. If the dataset does not exist, it will create it and return False.

        :return: If the dataset already exists, it will return True. If the dataset does not exist, it will create it and return False.
        :rtype: bool
        """

        if os.path.exists(self.dataset_path):
            print(f"Dataset {self.dataset_path} already exists.")
            return True

        print(f"Creating dataset {self.dataset_path}...")
        with h5py.File(self.dataset_path, "w") as f:
            f.create_dataset("num_examples", data=0)
            self.cur_index = 0

        return False

    def check_h5_dataset(self) -> bool:
        return os.path.exists(self.dataset_path)

    def create_or_check_nc_folder(self) -> bool:
        """This function will create the .nc folder if it does not exist, or check if it exists.
        If the folder already exists, it will return True. If the folder does not exist, it will create it and return False.

        :return: If the folder already exists, it will return True. If the folder does not exist, it will create it and return False.
        :rtype: bool
        """
        if os.path.exists(self.outfile_nc_folder):
            print(f"Output folder {self.outfile_nc_folder} already exists.")
            return True

        print(f"Creating output folder {self.outfile_nc_folder}...")
        os.makedirs(self.outfile_nc_folder)

        return False

    def check_nc_folder(self) -> bool:
        return os.path.exists(self.outfile_nc_folder)

    def create_or_check_img_folder(self) -> bool:
        """This function will create the .img folder if it does not exist, or check if it exists.
        If the folder already exists, it will return True. If the folder does not exist, it will create it and return False.

        :return: If the folder already exists, it will return True. If the folder does not exist, it will create it and return False.
        :rtype: bool
        """
        if self.outfile_img_folder is None:
            print("No output image folder specified. Skipping image folder creation.")
            return True

        if os.path.exists(self.outfile_img_folder):
            print(f"Output image folder {self.outfile_img_folder} already exists.")
            return True

        print(f"Creating output image folder {self.outfile_img_folder}...")
        os.makedirs(self.outfile_img_folder)

        return False

    def check_img_folder(self) -> bool:
        return (
            os.path.exists(self.outfile_img_folder)
            if self.outfile_img_folder is not None
            else True
        )

    def add_one_example(
        self,
        disaster_lat: float,
        disaster_long: float,
        start_time: datetime,
        duration_hours: int = 10,
        loglevel: int = 20,
        particle_amount: int = 50_000,
        particle_radius: int = 1000,
        object_types: list = [1],
        time_step=timedelta(hours=1),
        time_step_output=timedelta(hours=1),
        wind_uncertainity: float = 0.1,
        current_uncertainity: float = 0.1,
        horizontal_diffusion: float = 10,
        save_img=True,
        name="",
    ) -> None:
        """Adds one example to the dataset.

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
        :param particle_amount: How many particles to spawn in each object type
        :type particle_amount: int
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
        :param save_img: If True, the simulation will save images of the particles in the outfile_img_folder. Defaults to True
        :type save_img: bool

        """
        # Check if the dataset exists, if not, raise an error
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset {self.dataset_path} does not exist. Please create it first."
            )

        # Check if the output folders exist, if not, raise an error
        if not os.path.exists(self.outfile_nc_folder):
            raise FileNotFoundError(
                f"Output folder {self.outfile_nc_folder} does not exist. Please create it first."
            )

        # Raising an error if save-img is True but outfile_img_folder is None
        if save_img and self.outfile_img_folder is None:
            raise ValueError(
                "You cannot save images if outfile_img_folder is None. Set save_img to False or set outfile_img_folder to a valid folder path."
            )
        if self.outfile_img_folder is not None and not os.path.exists(
            self.outfile_img_folder
        ):
            raise FileNotFoundError(
                f"Output folder {self.outfile_img_folder} does not exist. Please create it first."
            )

        outfile_nc = os.path.abspath(
            f"{self.outfile_nc_folder}trj_{name}_{self.cur_index}.nc"
        )
        # Exiting this function if the outfile_nc already exists
        if os.path.exists(outfile_nc):
            print(
                f"File {outfile_nc} already exists. Cannot simulate because it would overwrite the file."
            )
            print("Please delete the file or change the name of this example.")
            return

        # Check if the example already exists in the dataset
        if self.check_example_exists(
            disaster_lat,
            disaster_long,
            start_time,
            duration_hours,
            particle_amount,
            particle_radius,
            object_types,
            time_step,
            time_step_output,
            wind_uncertainity,
            current_uncertainity,
            horizontal_diffusion,
        ):
            print(
                f"Example with disaster_lat={disaster_lat}, disaster_long={disaster_long}, start_time={start_time}, duration_hours={duration_hours}, particle_amount={particle_amount}, particle_radius={particle_radius}, object_types={object_types}, time_step={time_step}, time_step_output={time_step_output}, wind_uncertainity={wind_uncertainity}, current_uncertainity={current_uncertainity}, horizontal_diffusion={horizontal_diffusion} already exists in the dataset. Skipping..."
            )
            return

        # Check if the simulation is established, if not, establish it
        if not self.established:
            print("Establishing connection to Copernicus Marine Service...")
            self.establish_connection()
            print("Connection to Copernicus Marine Service established successfully.")
        else:
            print("Connection to Copernicus Marine Service already established.")

        outfile_img = (
            os.path.abspath(f"{self.outfile_img_folder}trj_{name}_{self.cur_index}.jpg")
            if save_img
            else None
        )
        self.simulation.__init__(
            disaster_lat=disaster_lat,
            disaster_long=disaster_long,
            start_time=start_time,
            duration_hours=duration_hours,
            loglevel=loglevel,
            particle_amount=particle_amount,
            particle_radius=particle_radius,
            object_types=object_types,
            time_step=time_step,
            time_step_output=time_step_output,
            wind_uncertainity=wind_uncertainity,
            current_uncertainity=current_uncertainity,
            horizontal_diffusion=horizontal_diffusion,
        )
        self.simulation.simulate(
            outfile_nc=outfile_nc, outfile_img=outfile_img, plot=save_img
        )

        # Load the NetCDF file
        ds = xr.open_dataset(outfile_nc)

        lon = ds["lon"]
        lat = ds["lat"]

        dt = h5py.string_dtype(encoding="utf-8")

        # data conversion
        try:
            with h5py.File(
                self.dataset_path, "r+"
            ) as f:  # Opens existing file and raises an error if it does not exist
                if f"example_{name}_{self.cur_index}" not in f:  # Check if group exists
                    grp = f.create_group(f"example_{name}_{self.cur_index}")
                    grp.create_dataset("lat", data=lat.values)
                    grp.create_dataset("long", data=lon.values)
                    probs = np.ones_like(lon.values[:, 0])
                    probs /= lon.shape[0]
                    grp.create_dataset("prob", data=probs)
                    grp.create_dataset(
                        "start time",
                        data=np.datetime_as_string(
                            ds["status"]["time"][0].values, unit="s"
                        ),
                        dtype=dt,
                    )
                    grp.create_dataset(
                        "time delta",
                        data=str(
                            ds["status"]["time"][1].values
                            - ds["status"]["time"][0].values
                        ),
                        dtype=dt,
                    )

                    # Add all the simulation variables to the group, so we can check that no
                    # two simualtions have the same variables
                    grp.create_dataset("disaster_lat", data=disaster_lat)
                    grp.create_dataset("disaster_long", data=disaster_long)
                    grp.create_dataset("duration_hours", data=duration_hours)
                    grp.create_dataset("particle_amount", data=particle_amount)
                    grp.create_dataset("particle_radius", data=particle_radius)
                    grp.create_dataset("object_types", data=np.array(object_types))
                    grp.create_dataset("time_step", data=str(time_step), dtype=dt)
                    grp.create_dataset(
                        "time_step_output", data=str(time_step_output), dtype=dt
                    )
                    grp.create_dataset("wind_uncertainity", data=wind_uncertainity)
                    grp.create_dataset(
                        "current_uncertainity", data=current_uncertainity
                    )
                    grp.create_dataset(
                        "horizontal_diffusion", data=horizontal_diffusion
                    )

                    # Increment the num_examples dataset
                    f["num_examples"][()] += 1

                    ds.close()
                    # Delete the NetCDF file after adding it to the HDF5 dataset
                    os.remove(outfile_nc)
                    print(
                        f"Added example_{name}_{self.cur_index} to {self.dataset_path} & deleted {outfile_nc}"
                    )
                else:
                    print(f"example_{name}_{self.cur_index}" + " already exists")
                    print("Change the name of the .nc file and run clear_nc_folder().")
        except FileNotFoundError:
            print(f"File {self.dataset_path} does not exist. Please create it first.")

        self.cur_index += 1

    def check_example_exists(
        self,
        disaster_lat: float,
        disaster_long: float,
        start_time: datetime,
        duration_hours: int,
        particle_amount: int,
        particle_radius: int,
        object_types: list,
        time_step,
        time_step_output,
        wind_uncertainity: float,
        current_uncertainity: float,
        horizontal_diffusion: float,
    ) -> bool:
        """Checks if an example with the given parameters already exists in the dataset."""

        with h5py.File(self.dataset_path, "r") as f:
            for key in f.keys():
                if not key.startswith("example_"):
                    continue
                grp = f[key]
                try:
                    if (
                        grp["disaster_lat"][()] == disaster_lat
                        and grp["disaster_long"][()] == disaster_long
                        and grp["start time"][()].decode()
                        == np.datetime_as_string(np.datetime64(start_time), unit="s")
                        and grp["duration_hours"][()] == duration_hours
                        and grp["particle_amount"][()] == particle_amount
                        and grp["particle_radius"][()] == particle_radius
                        and np.array_equal(
                            grp["object_types"][()], np.array(object_types)
                        )
                        and grp["time_step"][()].decode() == str(time_step)
                        and grp["time_step_output"][()].decode()
                        == str(time_step_output)
                        and grp["wind_uncertainity"][()] == wind_uncertainity
                        and grp["current_uncertainity"][()] == current_uncertainity
                        and grp["horizontal_diffusion"][()] == horizontal_diffusion
                    ):
                        return True
                except KeyError:
                    continue
        return False

    def clear_nc_folder(self) -> None:
        """Clears the .nc folder, deleting all .nc files in it and adding them to the dataset as examples.
        This should only be used if there is an error in add_one_example() but the .nc file still generates. This function is not recommended to be used.
        The best option is always to use add_one_example() which will automatically delete the .nc file after adding it to the dataset.
        """
        if not os.path.exists(self.outfile_nc_folder):
            print(
                f"Output folder {self.outfile_nc_folder} does not exist. Nothing to clear."
            )
            return

        if not os.listdir(self.outfile_nc_folder):
            print(
                f"Output folder {self.outfile_nc_folder} is already empty. Nothing to clear."
            )
            return

        for file in os.listdir(self.outfile_nc_folder):
            if file.endswith(".nc"):
                outfile_nc = os.path.join(self.outfile_nc_folder, file)
                # Get the name of the example from the file name
                name = "_".join(file.split("_")[2:])[:-3] if "_" in file else "unknown"
                if name == "unknown":
                    raise ValueError(
                        "File name does not contain an example name. Please use a valid file name."
                    )

                # Load the NetCDF file
                ds = xr.open_dataset(outfile_nc)

                lon = ds["lon"]
                lat = ds["lat"]

                dt = h5py.string_dtype(encoding="utf-8")

                # data conversion
                with h5py.File(
                    self.dataset_path, "r+"
                ) as f:  # Opens existing file and raises an error if it does not exist
                    if f"example_{name}" not in f:  # Check if group exists
                        grp = f.create_group(f"example_{name}")
                        grp.create_dataset("lat", data=lat.values)
                        grp.create_dataset("long", data=lon.values)
                        probs = np.ones_like(lon.values[:, 0])
                        probs /= lon.shape[0]
                        grp.create_dataset("prob", data=probs)
                        grp.create_dataset(
                            "start time",
                            data=np.datetime_as_string(
                                ds["status"]["time"][0].values, unit="s"
                            ),
                            dtype=dt,
                        )
                        grp.create_dataset(
                            "time delta",
                            data=str(
                                ds["status"]["time"][1].values
                                - ds["status"]["time"][0].values
                            ),
                            dtype=dt,
                        )

                        # Increment the num_examples dataset
                        f["num_examples"][()] += 1

                        ds.close()
                        # Delete the NetCDF file after adding it to the HDF5 dataset
                        os.remove(outfile_nc)
                        print(
                            f"Deleted {file} from {self.outfile_nc_folder} and moved to {self.dataset_path} as example_{name}"
                        )
                    else:
                        print(f"example_{name}" + " already exists")

    def list_examples(self, print_example=True) -> list:
        """Lists all examples in the dataset.

        :return: A list of example names in the dataset.
        :rtype: list
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset {self.dataset_path} does not exist. Please create it first."
            )

        with h5py.File(self.dataset_path, "r") as f:
            if print_example:
                print("Examples in the dataset:")
                for key in f.keys():
                    if key.startswith("num_examples"):
                        continue
                    print(f" - {key}")

            return list(f.keys())

    def delete_example(self, example_name) -> bool:
        """Deletes an example for the dataset. This is actually not too useful by itself because
        it only deletes the key, not the actual data under the key. To do that you have to run
        h5repack original.h5 original_tmp.h5 && mv original_tmp.h5 original.h5 in the terminal

        :return: If the deletion was successful or not
        :rtype: bool
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset {self.dataset_path} does not exist. Please create it first."
            )

        with h5py.File(self.dataset_path, "r+") as f:
            try:
                del f[example_name]
                print(f"Successfully deleted {example_name}")
                return True
            except KeyError:
                return False
