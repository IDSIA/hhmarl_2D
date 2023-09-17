"""
    A Patriot SAM unit
"""

from typing import List, Dict

from simulator.cmano_simulator import Unit, Position, CmanoSimulator, Event, units_distance_km, units_bearing
from simulator.pac3_missile_unit import Pac3Missile
from utils.angles import sum_angles, signed_heading_diff


class UnitDetected(Event):
    def __init__(self, origin: Unit, detected_unit: Unit):
        super().__init__("UnitDetected", origin)
        self.detected_unit = detected_unit

    def __str__(self):
        return super().__str__() + f"({self.detected_unit.type}{[self.detected_unit.id]})"


class MissileFired(Event):
    def __init__(self, origin: Unit, source: Unit, missile: Unit, target: Unit):
        super().__init__("MissileFired", origin)
        self.source = source
        self.missile_unit = missile
        self.target_unit = target

    def __str__(self):
        return super().__str__() + f"({self.missile_unit.type}{[self.missile_unit.id]} ->" \
                                   f"{self.target_unit.type}{[self.target_unit.id]})"


class Patriot(Unit):
    radar_width_deg = 140
    aircraft_detection_range_km = 140
    missile_range_km = 111

    def __init__(self, position: Position, heading: float, speed: float):
        super().__init__("Patriot", position, heading, speed)
        self.detected_aircrafts: List[Unit] = []
        self.flying_missiles: Dict[Unit, Unit] = {}  # missile -> aircraft

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Units detection
        events = []
        detected = []
        for unit in list(sim.active_units.values()):  # Copy values to avoid "dictionary changed size during iteration"
            # Check for radar detection
            if unit.type == "Rafale":
                unit_distance = units_distance_km(self, unit)
                if unit_distance < Patriot.aircraft_detection_range_km and \
                        self._angle_in_radar_range(units_bearing(self, unit)):
                    detected.append(unit)
                    if unit not in self.detected_aircrafts:
                        events.append(UnitDetected(self, unit))
                    # Decide if firing
                    if unit_distance <= Patriot.missile_range_km and unit not in self.flying_missiles.values():
                        pac3 = Pac3Missile(self.position.copy(), self.heading, sim.utc_time, unit)
                        sim.add_unit(pac3)
                        events.append(MissileFired(self, self, pac3, unit))
                        self.flying_missiles[pac3] = unit
        self.detected_aircrafts = detected

        # Remove dead missiles
        for m in list(self.flying_missiles):
            if m.id is None:
                del self.flying_missiles[m]

        # Missile guidance
        for missile, aircraft in self.flying_missiles.items():
            if aircraft in self.detected_aircrafts:
                missile.set_heading(units_bearing(missile, aircraft))

        return events

    def _angle_in_radar_range(self, angle: float) -> bool:
        delta = abs(signed_heading_diff(sum_angles(self.heading, Patriot.radar_width_deg / 2), angle))
        return delta <= Patriot.radar_width_deg / 2.0
