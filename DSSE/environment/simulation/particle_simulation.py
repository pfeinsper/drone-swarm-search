import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


EARTH_MEAN_RADIUS = 6373.0


class ParticleSimulation:
    def __init__(
        self,
        disaster_lat: float,
        disaster_long: float,
        duration_hours: int = 10,
        loglevel: int = 20,
        animate: bool = False,
        cell_size: int = 130,
    ) -> None:
        try:
            from opendrift.models.oceandrift import OceanDrift
            self.ocean_drift = OceanDrift
        except ImportError:
            raise ImportError("OpenDrift not installed. Install the environment with the 'coverage' extra: pip install DSSE[coverage]")
        
        self.disaster_lat = disaster_lat
        self.disaster_long = disaster_long
        self.loglevel = loglevel
        self.animate = animate
        self.duration_hours = duration_hours
        self.cell_size = cell_size

        # Internal variables
        self.map_size = 0
        self.original_map = None
        self.probability_map = None

    def run_or_get_simulation(self):
        if self.probability_map is None:
            self.run_simulation()
        self.probability_map = self.original_map.copy()

    def run_simulation(self):
        duration = timedelta(hours=self.duration_hours)
        start_time = datetime.now() - duration
        number = 50_000
        radius = 1000

        coordinates = self.simulate(start_time, number, radius, duration)
        self.map_size = self.calculate_map_size(coordinates)
        cartesian = self.convert_lat_lon_to_xy(coordinates)
        self.probability_map = self.create_probability_map(cartesian)
        # Maintain always a copy of the original map
        self.original_map = self.probability_map.copy()

    def simulate(
        self,
        time: datetime,
        number: int,
        radius: int,
        duration: timedelta,
    ) -> List[Tuple[float, float]]:
        o = self.ocean_drift(loglevel=self.loglevel)
        o.add_readers_from_list(
            ["https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"]
        )
        o.seed_elements(
            lat=self.disaster_lat,
            lon=self.disaster_long,
            time=time,
            number=number,
            radius=radius,
        )

        o.run(duration=duration, time_step=1800)
        if self.animate:
            o.animation(filename="animation.mp4")

        lat_final = o.elements.lat
        lon_final = o.elements.lon

        return list(zip(lat_final, lon_final))

    def lat_long_to_global_xy(
        self, lat: float, lon: float, ratio: float
    ) -> Tuple[int, int]:
        lat = math.radians(lat)
        lon = math.radians(lon)

        x = EARTH_MEAN_RADIUS * lon * ratio
        y = EARTH_MEAN_RADIUS * lat
        return (x, y)

    def convert_lat_lon_to_xy(
        self, coordinates: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Using equirectangular projection to convert latitudes and longitudes to Cartesian coordinates.
        """
        min_lat, max_lat, min_lon, max_lon = self.calculate_bounding_rectangle(
            coordinates
        )

        ratio = math.cos((math.radians(min_lat) + math.radians(max_lat)) / 2)
        p0_x, p0_y = self.lat_long_to_global_xy(min_lat, min_lon, ratio)
        p1_x, p1_y = self.lat_long_to_global_xy(max_lat, max_lon, ratio)

        cartesian_coordinates = []
        for lat, lon in coordinates:
            x, y = self.lat_long_to_global_xy(lat, lon, ratio)
            percent_x = (x - p0_x) / (p1_x - p0_x)
            percent_y = (y - p0_y) / (p1_y - p0_y)
            actual_x = (self.map_size - 1) * percent_x
            actual_y = (self.map_size - 1) * percent_y
            cartesian_coordinates.append((round(actual_x), round(actual_y)))
        return cartesian_coordinates

    def calculate_map_size(self, coordinates) -> int:

        min_lat, max_lat, min_lon, max_lon = self.calculate_bounding_rectangle(
            coordinates
        )

        width = self.distance_in_meters(min_lat, min_lon, min_lat, max_lon)
        height = self.distance_in_meters(min_lat, min_lon, max_lat, min_lon)

        map_width = round(width / self.cell_size)
        map_height = round(height / self.cell_size)

        return max(map_width, map_height)

    def calculate_bounding_rectangle(
        self, coordinates: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        latitudes, longitudes = zip(*coordinates)

        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)

        return min_lat, max_lat, min_lon, max_lon

    def distance_in_meters(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Uses the Haversine formula to calculate the distance between two points on Earth, given their latitudes and longitudes.

        Returns the distance in meters.
        """
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return EARTH_MEAN_RADIUS * c * 1000

    def create_probability_map(
        self, cartesian_coords: List[Tuple[float, float]]
    ) -> List[List[int]]:
        """
        Creates a probability map based on the coordinates of the particles.
        """
        prob_map = np.zeros((self.map_size, self.map_size))

        for x, y in cartesian_coords:
            prob_map[y][x] += 1

        particle_sum = max(np.sum(prob_map), 1)

        probability_map = prob_map / particle_sum
        return probability_map

    def get_matrix(self):
        return self.probability_map

    def get_map_size(self):
        return self.map_size
    
    def save_state(self, output_path: str):
        with open(output_path, "wb") as f:
            np.save(f, self.original_map)
        
    
    def load_state(self, input_path: str):
        with open(input_path, "rb") as f:
            self.probability_map = np.load(f)
            self.original_map = self.probability_map.copy()
            self.map_size = len(self.probability_map)