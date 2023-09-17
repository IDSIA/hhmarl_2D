"""
    Test of the ScenarioPlotter class
"""

from datetime import datetime

from scenario_plotter import ScenarioPlotter, PlotConfig, Airplane, ColorRGBA, StatusMessage, \
    TopLeftMessage, Waypoint, SamBattery, Missile, PolyLine, Arc
from utils.map_limits import MapLimits

map_extents = MapLimits(4, 45, 12, 48.5)

plt_cfg = PlotConfig()
plt = ScenarioPlotter(map_extents, dpi=200, config=plt_cfg)

colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1),
    'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1),
    'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1),
    'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}

objects = [
    StatusMessage("Time to go"),
    TopLeftMessage(datetime.now().strftime("%Y %b %d %H:%M:%S")),
    Airplane(47.3099374631706, 5.2946515109602, 86, edge_color=colors['red_outline'], fill_color=colors['red_fill'],
             info_text="Rafale", zorder=0),
    SamBattery(47.1335734412755, 7.06062200381671, 245, missile_range_km=111, radar_range_km=140,
               radar_amplitude_deg=140, edge_color=colors['blue_outline'], fill_color=colors['blue_fill'],
               info_text="Patriot", zorder=0),
    Missile(47.0, 6.0, 300, edge_color=colors['blue_outline'], fill_color=colors['blue_fill'],
            info_text="missile", zorder=0),
    Waypoint(47.374167, 8.648056, edge_color=colors['waypoint_outline'], fill_color=colors['waypoint_fill'],
             info_text="Target", zorder=0),
    PolyLine([(47, 8), (47.5, 9), (47.6, 10)], line_width=2, dash=(2, 2), edge_color=colors['red_outline'],
             zorder=0),
    Arc(47.3099374631706, 5.2946515109602, 14_000, 60, 114, line_width=1, dash=None, edge_color=colors['red_outline'],
        fill_color=None, zorder=0),
    Arc(47.3099374631706, 5.2946515109602, 14_000, -50, 50, line_width=1, dash=None, edge_color=None,
        fill_color=colors['red_outline'], zorder=0)
]
plt.to_png("sample_out.png", objects)
