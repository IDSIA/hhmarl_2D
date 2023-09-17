"""
    A static Waypoint unit
"""

from typing import List

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator


class Waypoint(Unit):

    def __init__(self, position: Position, heading: float, text=None):
        super().__init__("Waypoint", position, heading, 0)
        self.text = text

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        return []
