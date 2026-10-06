"""Geometry and kinematics (owner: agent A).  Spec §1.

Positions are (lat, lon) degree OFFSETS from the map origin; all functions are elementwise and
broadcastable.  Weapon geometry is "geodesic" (here: local ellipsoidal flat-earth, spec §1 [PORT]);
observation/reward geometry is PLANAR degree space (lon = x, lat = y, no cos(lat)).
"""
from __future__ import annotations

import jax

Array = jax.Array


def direct(lat: Array, lon: Array, heading_deg: Array, dist_m: Array) -> tuple[Array, Array]:
    """Move dist_m metres along heading (flat-earth WGS84, mid-latitude radii). Returns (lat, lon)."""
    raise NotImplementedError


def distance_km(lat1: Array, lon1: Array, lat2: Array, lon2: Array) -> Array:
    """Approximate geodesic distance in km (error <= 0.11 m on the map)."""
    raise NotImplementedError


def bearing_deg(lat1: Array, lon1: Array, lat2: Array, lon2: Array) -> Array:
    """Initial azimuth 1->2 in [0, 360)."""
    raise NotImplementedError


def signed_heading_diff(actual: Array, desired: Array) -> Array:
    """desired - actual wrapped into [-180, 180] (warsim/utils/angles.py)."""
    raise NotImplementedError


def planar_distance(lat1: Array, lon1: Array, lat2: Array, lon2: Array) -> Array:
    """hypot(dlon, dlat) in degrees (env_base._distance, unnormalised)."""
    raise NotImplementedError


def focus_angle(lat1: Array, lon1: Array, hdg1: Array, lat2: Array, lon2: Array) -> Array:
    """ATA of unit 1 towards unit 2 in degrees [0, 180] (env_base._focus_angle, 1e-10 eps)."""
    raise NotImplementedError


def heading_diff_norm(h1: Array, h2: Array) -> Array:
    """Angle between heading vectors / 180 (env_base._heading_diff)."""
    raise NotImplementedError
