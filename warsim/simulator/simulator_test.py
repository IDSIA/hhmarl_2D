"""
    Test class for the CmanoSimulator
"""

from datetime import datetime

from cmano_simulator import Position, CmanoSimulator
from simulator.patriot_unit import Patriot, MissileFired
from simulator.rafale_unit import Rafale
from simulator.waypoint_unit import Waypoint
from utils.map_limits import MapLimits


def print_units(sim: CmanoSimulator):
    for id, unit in sim.active_units.items():
        print(f"[{id}] {unit.to_string()}")


ml = MapLimits(4, 45, 12, 48.5)
tgt_unit = Waypoint(Position(47.374167, 8.648056, 400), heading=0, text="Target")
pat_unit = Patriot(Position(47.1335734412755, 7.06062200381671, 400), heading=245, speed=0)
raf_unit = Rafale(Position(47.3099374631706, 5.2946515109602, 10017.82), heading=86, speed=480)
sim = CmanoSimulator(utc_time=datetime(2021, 7, 24, 15, 00, 00), tick_secs=1)
raf_id = sim.add_unit(raf_unit)
sim.add_unit(pat_unit)
sim.add_unit(tgt_unit)
sim.record_unit_trace(raf_id)

print(f"Time: {sim.utc_time}")
print_units(sim)

actions = [(86, 140, 480), (260, 200, 920), (50, 200, 480), (100, 800, 480), (40, 150, 480)]
for heading, ticks, speed in actions:
    raf_unit.set_heading(heading)
    raf_unit.set_speed(speed)
    for i in range(ticks):
        ev = sim.do_tick()
        for e in ev:
            print(e)
            sim.set_status_text(f"{sim.utc_time}: {e}")
            if isinstance(e, MissileFired):
                sim.record_unit_trace(e.missile_unit.id)

print(f"Time: {sim.utc_time}")
print_units(sim)
