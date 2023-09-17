"""
    A PAC-3 Missile Unit
"""

from datetime import datetime
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_distance_km, UnitDestroyedEvent
from utils.angles import signed_heading_diff

class Rocket(Unit):
    max_deg_sec = 10
    speed_profile_time = np.array([0, 10, 20, 30])
    #speed_profile_time = np.array([0, 10, 20, 30, 40])
    speed_profile_knots = np.array([500, 2000, 1400, 600])
    #speed_profile_knots = np.array([400, 2000, 1800, 1200, 500]) #fill_value = (400,500)
    speed_profile = interp1d(speed_profile_time, speed_profile_knots, kind='quadratic', assume_sorted=True,
                             bounds_error=False, fill_value=(500, 600))

    def __init__(self, position: Position, heading: float, firing_time: datetime, target: Unit, source: Unit, friendly_check: bool = False):
        self.speed = Rocket.speed_profile(0)
        super().__init__("Rocket", position, heading, self.speed)
        self.new_heading = heading
        self.firing_time = firing_time
        self.target = target
        self.source = source
        self.friendly_check = friendly_check

    def set_heading(self, new_heading: float):
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"Rocket.set_heading Heading must be in [0, 360), got {new_heading}")
        self.new_heading = new_heading

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Check if the target has been hit
        if units_distance_km(self, self.target) < 1 and sim.unit_exists(self.target.id):
            sim.remove_unit(self.id)
            sim.remove_unit(self.target.id)
            return [UnitDestroyedEvent(self, self.source, self.target)]
        
        if self.friendly_check:
            # check if friendly aircraft has been hit
            friendly_id = 1 if self.source.id == 2 else 2
            if sim.unit_exists(friendly_id):
                friendly_unit = sim.get_unit(friendly_id)
                if units_distance_km(self, friendly_unit) < 1:
                    sim.remove_unit(self.id)
                    sim.remove_unit(friendly_id)
                    return [UnitDestroyedEvent(self, self.source, friendly_unit)]

        # Check if eol is arrived
        life_time = (sim.utc_time - self.firing_time).seconds
        if life_time > Rocket.speed_profile_time[1]:
            sim.remove_unit(self.id)
            return []

        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Rocket.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg

        # Update speed
        self.speed = Rocket.speed_profile(life_time)

        # Update position
        return super().update(tick_secs, sim)
