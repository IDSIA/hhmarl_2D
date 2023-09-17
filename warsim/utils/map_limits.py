"""
    Implements a latitude-longitude rectangle that defines the allowable region
    for a simulation.
"""

from geographiclib.geodesic import Geodesic
import numpy as np


class MapLimits:
    def __init__(self, left_lon, bottom_lat, right_lon, top_lat):
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat

    def latitude_extent(self):
        return self.top_lat - self.bottom_lat

    def longitude_extent(self):
        return self.right_lon - self.left_lon

    def max_latitude_extent_km(self):
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.top_lat, self.left_lon,
                                    outmask=Geodesic.DISTANCE)
        d2 = Geodesic.WGS84.Inverse(self.bottom_lat, self.right_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def max_longitude_extent_km(self):
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.bottom_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        d2 = Geodesic.WGS84.Inverse(self.top_lat, self.left_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def relative_position(self, lat, lon):
        lat_rel = (lat - self.bottom_lat) / self.latitude_extent()
        lon_rel = (lon - self.left_lon) / self.longitude_extent()
        return np.clip(lat_rel, 0, 1), np.clip(lon_rel, 0, 1)

    def absolute_position(self, lat_rel, lon_rel):
        lat = lat_rel * self.latitude_extent() + self.bottom_lat
        lon = lon_rel * self.longitude_extent() + self.left_lon
        return lat, lon

    def in_boundary(self, lat, lon):
        return self.left_lon <= lon <= self.right_lon and self.bottom_lat <= lat <= self.top_lat
