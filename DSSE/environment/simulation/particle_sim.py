from datetime import datetime, timedelta
from opendrift.models.oceandrift import OceanDrift


def open_drift(lat, lon, time, number, radius, duration, outfile):
    o = OceanDrift()
    o.add_readers_from_list(
        ["https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"]
    )
    o.seed_elements(lat=lat, lon=lon, time=time, number=number, radius=radius)

    # Running the model
    o.run(duration=duration, outfile=outfile)

    lat_final = o.elements.lat
    lon_final = o.elements.lon

    # Return final positions as list of tuples (latitude, longitude)
    return list(zip(lat_final, lon_final))
