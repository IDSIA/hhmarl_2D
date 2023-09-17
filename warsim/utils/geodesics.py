"""
    Geodesics computations
"""

from typing import Tuple

from geographiclib.geodesic import Geodesic

from utils.angles import normalize_angle


def geodetic_distance_km(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.DISTANCE)
    return r["s12"] / 1000.0


def geodetic_bearing_deg(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    r = Geodesic.WGS84.Inverse(lat_1, lon_1, lat_2, lon_2, outmask=Geodesic.AZIMUTH)
    return normalize_angle(r["azi1"])


def geodetic_direct(lat: float, lon: float, heading: float, distance: float) -> Tuple[float, float]:
    d = Geodesic.WGS84.Direct(lat, lon, heading, distance, outmask=Geodesic.LATITUDE | Geodesic.LONGITUDE)
    return d["lat2"], d["lon2"]
