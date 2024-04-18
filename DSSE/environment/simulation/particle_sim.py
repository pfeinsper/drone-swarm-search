from opendrift.models.oceandrift import OceanDrift


def open_drift(lat, lon, time, number, radius, duration, outfile):
    o = OceanDrift()
    o.add_readers_from_list(
        ["https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"]
    )
    o.seed_elements(lat=lat, lon=lon, time=time, number=number, radius=radius)

    o.run(duration=duration, outfile=outfile)

    lat_final = o.elements.lat
    lon_final = o.elements.lon

    return list(zip(lat_final, lon_final))

def calculate_bounding_rectangle(coordinates):
    min_lat = max_lat = coordinates[0][0]
    min_lon = max_lon = coordinates[0][1]

    for lat, lon in coordinates:
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)

    return min_lat, max_lat, min_lon, max_lon

def convert_lat_lon_to_xy(coordinates, map_size):
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
