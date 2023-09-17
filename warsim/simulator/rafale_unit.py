"""
    A Rafale airplane unit
"""

from typing import List

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_bearing, UnitDestroyedEvent, \
    units_distance_km
from utils.angles import signed_heading_diff


class Rafale(Unit):
    max_deg_sec = 5
    min_speed_knots = 0
    max_speed_knots = 900
    max_knots_sec = 34
    cannon_range_km = 1.8
    cannon_width_deg = 10
    cannon_max_time_sec = 500
    cannon_burst_time_sec = 5
    cannon_hit_prob = 0.75

    def __init__(self, position: Position, heading: float, speed: float):
        super().__init__("Rafale", position, heading, speed)
        self.new_heading = heading
        self.new_speed = speed

        # Cannon
        self.cannon_remain_secs = Rafale.cannon_max_time_sec
        self.cannon_current_burst_secs = 0

    def set_heading(self, new_heading: float) -> ():
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"Rafale.set_heading Heading must be in [0, 360), got {new_heading}")
        self.new_heading = new_heading

    def set_speed(self, new_speed: float) -> ():
        if new_speed > Rafale.max_speed_knots or new_speed < Rafale.min_speed_knots:
            raise Exception(f"Rafale.set_speed Speed must be in [{Rafale.min_speed_knots}, {Rafale.max_speed_knots}] "
                            f"knots, got {new_speed}")
        self.new_speed = new_speed

    def fire_cannon(self) -> ():
        self.cannon_current_burst_secs = min(self.cannon_remain_secs, Rafale.cannon_burst_time_sec)

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Rafale.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg
                self.heading %= 360

        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = Rafale.max_knots_sec * tick_secs
            if abs(delta) <= max_delta:
                self.speed = self.new_speed
            else:
                self.speed += max_delta if delta >= 0 else -max_delta

        # Update cannon
        events = []
        kill_ids = {}
        if self.cannon_current_burst_secs > 0:
            self.cannon_current_burst_secs = max(self.cannon_current_burst_secs - tick_secs, 0.0)
            self.cannon_remain_secs = max(self.cannon_remain_secs - tick_secs, 0.0)
            for unit in list(sim.active_units.values()):
                if unit.id != self.id:
                    if unit.type == "Rafale":
                        if self._unit_in_cannon_range(unit):
                            if sim.rnd_gen.random() < \
                                    (Rafale.cannon_hit_prob / (Rafale.cannon_burst_time_sec / tick_secs)):
                                sim.remove_unit(unit.id)
                                kill_ids[self.id] = unit.id
                                events.append(UnitDestroyedEvent(self, self, unit))

        # Update position
        events.extend(super().update(tick_secs, sim))
        return events, kill_ids

    def _unit_in_cannon_range(self, u: Unit) -> bool:
        distance = units_distance_km(self, u)
        if distance < Rafale.cannon_range_km:
            bearing = units_bearing(self, u)
            delta = abs(signed_heading_diff(self.heading, bearing))
            return delta <= Rafale.cannon_width_deg / 2.0
        else:
            return False
