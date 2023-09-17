"""
    CmanoSimulator is a simulator of the essential characteristic of CMANO.
    It is used to pre-train AI agent with basic capabilities that will be
    further refined with a real CMANO simulation.
"""

from __future__ import annotations

import random
from abc import ABC
from datetime import datetime, timedelta
from typing import Callable, List, Dict

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname((__file__))))

from utils.geodesics import geodetic_direct, geodetic_distance_km, geodetic_bearing_deg

# --- Constants
knots_to_ms = 0.514444


# --- Classes
class Position:
    def __init__(self, lat: float, lon: float, alt: float):
        self.lat = lat  # Latitude [-90; +90]
        self.lon = lon  # Longitude [0; 180]
        self.alt = alt  # Altitude (meters)

    def copy(self) -> Position:
        return Position(self.lat, self.lon, self.alt)


class Event(ABC):
    def __init__(self, name, origin: Unit):
        self.name = name
        self.origin = origin

    def __str__(self):
        return f"{self.origin.type}[{self.origin.id}].{self.name}"


class UnitDestroyedEvent(Event):
    def __init__(self, origin: Unit, unit_killer: Unit, unit_destroyed: Unit):
        super().__init__("UnitDestroyedEvent", origin)
        self.unit_killer = unit_killer
        self.unit_destroyed = unit_destroyed

    def __str__(self):
        return super().__str__() + f"({self.unit_killer.type}{[self.unit_killer.id]} ->" \
                                   f"{self.unit_destroyed.type}{[self.unit_destroyed.id]})"


class Unit(ABC):
    def __init__(self, type: str, position: Position, heading: float, speed_knots: float):
        if heading >= 360 or heading < 0:
            raise Exception(f"Unit.__init__: bad heading {heading}")
        self.type = type
        self.position = position
        self.heading = heading  # bearing, in degrees [0, 360)
        self.speed = speed_knots  # scalar, in knots
        self.id = None

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Basic position update based on speed
        if self.speed > 0:
            d = geodetic_direct(self.position.lat, self.position.lon, self.heading,
                                self.speed * knots_to_ms * tick_secs)
            self.position.lat = d[0]
            self.position.lon = d[1]
        return []

    def to_string(self) -> str:
        return f"{self.type}[{self.id}]: p=({self.position.lat:.4f}, {self.position.lon:.4f}, " \
               f"{self.position.alt:.4f}) h=({self.heading:.4f}) s=({self.speed:.4f})"


class CmanoSimulator:
    def __init__(self, utc_time=datetime.now(), tick_secs=1, random_seed=None, num_units=0, num_opp_units=0):
        self.active_units: Dict[int, Unit] = {}  # id -> Unit
        self.trace_record_units = {}  # id ->[(time, position, heading, speed), ...]
        self.utc_time = utc_time
        self.utc_time_initial = utc_time
        self.tick_secs = tick_secs
        self._tick_callbacks: List[Callable[[datetime], None]] = []  # registered tick callbacks fn: datetime -> ()
        self.random_seed = random_seed
        self.rnd_gen = random.Random(random_seed)
        self._next_unit_id = 1
        self.status_text = None

        self.num_units = num_units
        self.num_opp_units = num_opp_units

    def reset_sim(self, units):
        self.utc_time = datetime.now()
        self._tick_callbacks = []
        self.status_text = None
        
        self.active_units = units
        for i in units.keys():
            self.record_unit_trace(i)

    def add_unit(self, unit: Unit) -> int:
        self.active_units[self._next_unit_id] = unit
        unit.id = self._next_unit_id
        self._next_unit_id += 1
        return self._next_unit_id - 1

    def remove_unit(self, unit_id: int):
        # self.active_units[unit_id].id = None  # Remove this line after some tests

        #if self.unit_exists(unit_id):
        del self.active_units[unit_id]

        # if unit_id in self.trace_record_units:
        #     del self.trace_record_units[unit_id]

    def get_unit(self, unit_id: int) -> Unit:
        return self.active_units[unit_id]

    def unit_exists(self, unit_id: int) -> bool:
        return unit_id in self.active_units

    def record_unit_trace(self, unit_id: int):
        if unit_id not in self.active_units:
            raise Exception(f"Unit.record_unit_trace(): unknown unit {unit_id}")
        if unit_id not in self.trace_record_units:
            self.trace_record_units[unit_id] = []
            self._store_unit_state(unit_id)

    def set_status_text(self, text: str):
        self.status_text = text

    def add_tick_callback(self, cb_fn: Callable[[datetime], None]):
        self._tick_callbacks.append(cb_fn)

    def do_tick(self) -> List[Event]:
        # Update the state of all units
        events = []
        #kills = {}
        for unit in list(self.active_units.values()):
            event = unit.update(self.tick_secs, self)
            events.extend(event)
            #kills.update(kill)
        self.utc_time += timedelta(seconds=self.tick_secs)

        # Save trace
        for _ in self.trace_record_units.keys():
            self._store_unit_state(_)

        # Notify clients
        for fn in self._tick_callbacks:
            fn(self.utc_time)

        #return events, kills
        return events

    def _store_unit_state(self, unit_id):
        if self.unit_exists(unit_id):
            unit = self.active_units[unit_id]
            self.trace_record_units[unit_id].append((self.utc_time, unit.position.copy(), unit.heading, unit.speed))


# --- General purpose utilities

def units_distance_km(unit_a: Unit, unit_b: Unit) -> float:
    return geodetic_distance_km(unit_a.position.lat, unit_a.position.lon,
                                unit_b.position.lat, unit_b.position.lon)


def units_bearing(unit_from: Unit, unit_to: Unit) -> float:
    return geodetic_bearing_deg(unit_from.position.lat, unit_from.position.lon,
                                unit_to.position.lat, unit_to.position.lon)
