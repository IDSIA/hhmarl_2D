"""
    A modified Rafale airplane unit
"""

from typing import List
import numpy as np
import random

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_bearing, UnitDestroyedEvent, \
    units_distance_km
from utils.angles import signed_heading_diff, sum_angles

class UnitDetected(Event):
    def __init__(self, origin: Unit, detected_unit: Unit):
        super().__init__("UnitDetected", origin)
        self.detected_unit = detected_unit

    def __str__(self):
        return super().__str__() + f"({self.detected_unit.type}{[self.detected_unit.id]})"


class RafaleLong(Unit):
    max_deg_sec = 3.5
    min_speed_knots = 0
    max_speed_knots = 600
    max_knots_sec = 28
    cannon_range_km = 4.5
    cannon_width_deg = 7
    cannon_max_time_sec = 200
    cannon_burst_time_sec = 3
    cannon_hit_prob = 0.9
    aircraft_type = 2

    def __init__(self, position: Position, heading: float, speed: float, group:str, friendly_check: bool = True):
        super().__init__("RafaleLong", position, heading, speed)
        self.new_heading = heading
        self.new_speed = speed
        self.max_speed = RafaleLong.max_speed_knots

        # Cannon
        self.cannon_remain_secs = RafaleLong.cannon_max_time_sec
        self.cannon_current_burst_secs = 0
        self.cannon_max = RafaleLong.cannon_max_time_sec

        self.actual_missile = None
        self.missile_remain = 0
        self.rocket_max = 0

        # group type and if friendly aircrafts considered
        self.friendly_check = friendly_check
        self.group = group
        self.ac_type = RafaleLong.aircraft_type

    def set_heading(self, new_heading: float):
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"RafaleLong.set_heading Heading must be in [0, 360), got {new_heading}")
        self.new_heading = new_heading

    def set_speed(self, new_speed: float):
        if new_speed > RafaleLong.max_speed_knots or new_speed < RafaleLong.min_speed_knots:
            raise Exception(f"RafaleLong.set_speed Speed must be in [{RafaleLong.min_speed_knots}, {RafaleLong.max_speed_knots}] "
                            f"knots, got {new_speed}")
        self.new_speed = new_speed

    def fire_cannon(self):
        self.cannon_current_burst_secs = min(self.cannon_remain_secs, RafaleLong.cannon_burst_time_sec)

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = RafaleLong.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg
                self.heading %= 360

        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = RafaleLong.max_knots_sec * tick_secs
            if abs(delta) <= max_delta:
                self.speed = self.new_speed
            else:
                self.speed += max_delta if delta >= 0 else -max_delta

        # Update cannon
        events = []
        if self.cannon_current_burst_secs > 0:
            self.cannon_current_burst_secs = max(self.cannon_current_burst_secs - tick_secs, 0.0)
            self.cannon_remain_secs = max(self.cannon_remain_secs - tick_secs, 0.0)
            for unit in list(sim.active_units.values()):
                if unit.id != self.id:
                    if unit.id <= sim.num_units + sim.num_opp_units: # consider only aircrafts, no rockets
                        if self.friendly_check or (self.group == "agent" and unit.id >= sim.num_units+1) or (self.group == "opp" and unit.id <= sim.num_units):
                            if unit.type in ["RafaleLong", "Rafale"]:
                                if self._unit_in_cannon_range(unit):
                                    if sim.rnd_gen.random() < \
                                            (RafaleLong.cannon_hit_prob / (RafaleLong.cannon_burst_time_sec / tick_secs)):
                                        sim.remove_unit(unit.id)
                                        events.append(UnitDestroyedEvent(self, self, unit))
        
        # Update position
        events.extend(super().update(tick_secs, sim))

        return events

    def _unit_in_cannon_range(self, u: Unit) -> bool:
        distance = units_distance_km(self, u)
        if distance < RafaleLong.cannon_range_km:
            bearing = units_bearing(self, u)
            delta = abs(signed_heading_diff(self.heading, bearing))
            return delta <= RafaleLong.cannon_width_deg / 2.0
        else:
            return False
