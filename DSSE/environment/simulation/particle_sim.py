from datetime import datetime, timedelta
from typing import List, Tuple, str
from opendrift.models.oceandrift import OceanDrift


def open_drift(lat: float, lon: float, time: datetime, number: int, radius: int, duration: timedelta, outfile: str) -> List[Tuple[float, float]]:
    o = OceanDrift()
    o.add_readers_from_list(
        ["https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"]
    )
    o.seed_elements(lat=lat, lon=lon, time=time, number=number, radius=radius)

    o.run(duration=duration, outfile=outfile)

    lat_final = o.elements.lat
    lon_final = o.elements.lon

    return list(zip(lat_final, lon_final))

def convert_lat_lon_to_xy(coordinates: List[Tuple[float, float]], map_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    min_lat, max_lat, min_lon, max_lon = calculate_bounding_rectangle(coordinates)
    width = max_lon - min_lon
    height = max_lat - min_lat
    map_width, map_height = map_size

    cartesian_coordinates = []
    for lat, lon in coordinates:
        x = round((lon - min_lon) / width * map_width)
        y = round((lat - min_lat) / height * map_height)
        cartesian_coordinates.append((x, y))

    return cartesian_coordinates

def calculate_bounding_rectangle(coordinates: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    latitudes, longitudes = zip(*coordinates)

    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)

    return min_lat, max_lat, min_lon, max_lon
