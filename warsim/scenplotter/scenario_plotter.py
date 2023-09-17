"""
    ScenarioPlotter produces a graphical representation of a war scenario
"""

import io
import math
from collections import namedtuple
from typing import List, Tuple, Optional

import cairo
import cartopy
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits

ColorRGBA = namedtuple('ColorRGBA', ['red', 'green', 'blue', 'alpha'])


class PlotConfig:
    def __init__(self):
        self.show_grid = True
        self.units_scale = 35
        self.background_color = '#191b24'
        self.borders_color = '#ffffff'
        self.sprites_info_font = "sans-serif"
        self.sprites_info_font_style = cairo.FONT_SLANT_NORMAL
        self.sprites_info_font_size = 12
        self.sprites_info_spacing = 26
        self.status_message_font = "sans-serif"
        self.status_message_font_style = cairo.FONT_SLANT_NORMAL
        self.status_message_font_size = 14


class Drawable:
    def __init__(self, zorder):
        """ Base class for anything that can be drawn on a map

        :param zorder: z-ordering position for drawing
        """
        self.zorder = zorder


class StatusMessage(Drawable):
    def __init__(self, text, text_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        """ Shows a message in the bottom left position """
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class TopLeftMessage(Drawable):
    def __init__(self, text, text_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        """ Shows a message in the top left position """
        super().__init__(zorder)
        self.text = text
        self.text_color = text_color


class PolyLine(Drawable):
    def __init__(self, points: List[Tuple[float, float]], line_width: float = 1.0,
                 dash: Optional[Tuple[float, float]] = None, edge_color=ColorRGBA(1, 1, 1, 1), zorder: int = 0):
        """ Draw a series of lines """
        super().__init__(zorder)
        self.points = points
        self.line_width = line_width
        self.dash = dash
        self.edge_color = edge_color


class Rect(Drawable):
    def __init__(self, left_lon: float, bottom_lat: float, right_lon: float, top_lat: float, line_width: float = 1.0,
                 edge_color=ColorRGBA(1, 1, 1, 1), fill_color=ColorRGBA(1, 1, 1, 0), zorder: int = 0):
        """ Draw a square """
        super().__init__(zorder)
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat
        self.line_width = line_width
        self.edge_color = edge_color
        self.fill_color = fill_color


class Arc(Drawable):
    def __init__(self, center_lat: float, center_lon: float, radius: float, angle1: float, angle2: float,
                 line_width: float = 1.0, dash: Optional[Tuple[float, float]] = None, edge_color=None,
                 fill_color=None, zorder: int = 0):
        """ Draw a square """
        super().__init__(zorder)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.angle1 = angle1
        self.angle2 = angle2
        self.line_width = line_width
        self.dash = dash
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.zorder = zorder


class Sprite(Drawable):
    def __init__(self, lat: float, lon: float, heading: float, edge_color=ColorRGBA(1, 1, 1, 1),
                 fill_color=ColorRGBA(.5, .5, .5, 1), info_text: Optional[str] = None, zorder: int = 0):
        """ Base class for all the drawable shapes

        :param lat: latitude [-90, +90]
        :param lon: longitude [-180, +180]
        :param heading: [0, 360)
        :param edge_color: the outline color
        :param fill_color: the fill color
        :param info_text: if present, the text to be displayed under the shape
        """
        super().__init__(zorder)
        self.lat = lat
        self.lon = lon
        self.heading = heading
        self.edge_color = edge_color
        self.fill_color = fill_color
        self.info_text = info_text


class Airplane(Sprite):
    def __init__(self, lat: float, lon: float, heading: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """ An airplane shaped sprite """
        super().__init__(lat, lon, heading, edge_color, fill_color, info_text, zorder)


class SamBattery(Sprite):
    def __init__(self, lat: float, lon: float, heading: float, missile_range_km: float, radar_range_km: float,
                 radar_amplitude_deg: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """ An SAM battery shaped sprite """
        super().__init__(lat, lon, heading, edge_color, fill_color, info_text, zorder)
        self.missile_range_km = missile_range_km
        self.radar_range_km = radar_range_km
        self.radar_amplitude_deg = radar_amplitude_deg


class Missile(Sprite):
    def __init__(self, lat: float, lon: float, heading: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """ An missile shaped sprite """
        super().__init__(lat, lon, heading, edge_color, fill_color, info_text, zorder)


class Waypoint(Sprite):
    def __init__(self, lat: float, lon: float, edge_color='#ffffff', fill_color='#888888',
                 info_text: Optional[str] = None, zorder: int = 0):
        """ An waypoint shaped sprite """
        super().__init__(lat, lon, 0, edge_color, fill_color, info_text, zorder)


class BackgroundMesh:
    def __init__(self, lons, lats, vals, cmap: str, vmin: float = None, vmax: float = None):
        """ Class to define a background mesh """
        self.lons = lons
        self.lats = lats
        self.vals = vals
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax


class ScenarioPlotter:
    def __init__(self, map_extents: MapLimits, dpi=200, background_mesh: Optional[BackgroundMesh] = None,
                 config=PlotConfig()):
        self.map_extents = map_extents
        self.dpi = dpi
        self.bg_mesh = background_mesh
        self.cfg = config
        self.projection = cartopy.crs.Mercator(central_longitude=(map_extents.left_lon + map_extents.right_lon) / 2)
        self._background = self._build_background_image()
        self.img_width = self._background.get_width()
        self.img_height = self._background.get_height()
        self.pixels_km = self.img_width / map_extents.max_longitude_extent_km()

    def _build_background_image(self):
        # Produce a background map
        plt.figure()
        ax = plt.axes(projection=self.projection)
        ax.set_extent((self.map_extents.left_lon, self.map_extents.right_lon,
                       self.map_extents.bottom_lat, self.map_extents.top_lat))
        if self.bg_mesh is not None:
            ax.pcolormesh(self.bg_mesh.lons, self.bg_mesh.lats, self.bg_mesh.vals,
                          vmin=self.bg_mesh.vmin, vmax=self.bg_mesh.vmax,
                          cmap=self.bg_mesh.cmap, shading='nearest', transform=cartopy.crs.PlateCarree())
        ax.patch.set_facecolor(self.cfg.background_color)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor=self.cfg.borders_color, linewidth=0.2, linestyle='-',
                       alpha=1)
        ax.coastlines(resolution='110m')
        # ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.5, 0.5, 0.5))
        if self.cfg.show_grid:
            ax.gridlines(linewidth=0.2)

        # Make it a cairo image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return cairo.ImageSurface.create_from_png(buf)

    def to_png(self, filename: str, objects: List[Drawable]):
        # Setup Cairo surface
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.img_width, self.img_height)
        ctx = cairo.Context(surface)

        # Draw the background
        ctx.set_source_surface(self._background)
        ctx.paint()

        # Put the origin to left-bottom as in a Cartesian axis
        ctx.translate(0, surface.get_height())
        ctx.scale(1, -1)

        # Draw objects
        for o in objects:
            if isinstance(o, Airplane):
                self._draw_airplane(ctx, o)
            elif isinstance(o, SamBattery):
                self._draw_sam_battery(ctx, o)
            elif isinstance(o, Waypoint):
                self._draw_waypoint(ctx, o)
            elif isinstance(o, Missile):
                self._draw_missile(ctx, o)
            elif isinstance(o, StatusMessage):
                self._draw_status_message(ctx, o)
            elif isinstance(o, TopLeftMessage):
                self._draw_top_left_message(ctx, o)
            elif isinstance(o, PolyLine):
                self._draw_poly_line(ctx, o)
            elif isinstance(o, Arc):
                self._draw_arc(ctx, o)
            elif isinstance(o, Rect):
                self._draw_rect(ctx, o)
            else:
                raise RuntimeError(f"Can't draw object of type {type(o)}")

        # Write the png
        surface.write_to_png(filename)

    @staticmethod
    def _get_image_angle(heading: float):
        return -heading / 180 * math.pi

    def _get_image_xya(self, lat: float, lon: float, heading: float):
        lat_rel = (lat - self.map_extents.bottom_lat) / (self.map_extents.top_lat - self.map_extents.bottom_lat)
        lon_rel = (lon - self.map_extents.left_lon) / (self.map_extents.right_lon - self.map_extents.left_lon)
        a = -heading / 180 * math.pi
        return lon_rel * self.img_width, lat_rel * self.img_height, a

    def _get_image_distance(self, dst_meters: float):
        d1 = geodetic_direct(self.map_extents.bottom_lat, self.map_extents.left_lon, 0, dst_meters)
        lat_rel = (d1[0] - self.map_extents.bottom_lat) / (self.map_extents.top_lat - self.map_extents.bottom_lat)
        return lat_rel * self.img_height

    def _draw_status_message(self, ctx, o: StatusMessage):
        msg = f"> {o.text}"
        ctx.save()
        ctx.set_source_rgba(*o.text_color)
        ctx.select_font_face(self.cfg.status_message_font, self.cfg.status_message_font_style)
        ctx.set_font_matrix(
            cairo.Matrix(xx=self.cfg.status_message_font_size, yy=-self.cfg.status_message_font_size))
        # x_bearing, y_bearing, width, height = ctx.text_extents(msg)[:4]
        ctx.move_to(10, 8)
        ctx.show_text(msg)
        ctx.new_path()
        ctx.restore()

    def _draw_top_left_message(self, ctx, o: TopLeftMessage):
        msg = f"{o.text}"
        ctx.save()
        ctx.set_source_rgba(*o.text_color)
        ctx.select_font_face(self.cfg.status_message_font, self.cfg.status_message_font_style)
        ctx.set_font_matrix(
            cairo.Matrix(xx=self.cfg.status_message_font_size, yy=-self.cfg.status_message_font_size))
        x_bearing, y_bearing, width, height = ctx.text_extents(msg)[:4]
        ctx.move_to(self.img_width - width - 10, self.img_height - height - 5)
        ctx.show_text(msg)
        ctx.new_path()
        ctx.restore()

    # Draw status text (not rotated)
    def _draw_text(self, ctx, x, y, text):
        ctx.select_font_face(self.cfg.sprites_info_font, self.cfg.sprites_info_font_style)
        ctx.set_font_matrix(cairo.Matrix(xx=self.cfg.sprites_info_font_size, yy=-self.cfg.sprites_info_font_size))
        x_bearing, y_bearing, width, height = ctx.text_extents(text)[:4]
        ctx.move_to(x - width / 2 - x_bearing, y - height / 2 - y_bearing)
        ctx.show_text(text)

    def _draw_airplane(self, ctx, o: Airplane):
        # Translate to airplane position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        # Rotate to airplane heading
        ctx.rotate(angle)

        ctx.set_source_rgba(*o.fill_color)
        ctx.set_line_width(1)
        # First half
        ctx.move_to(0.00 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.line_to(0.06 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.line_to(0.08 * self.cfg.units_scale, -0.31 * self.cfg.units_scale)
        ctx.line_to(0.28 * self.cfg.units_scale, -0.29 * self.cfg.units_scale)
        ctx.line_to(0.28 * self.cfg.units_scale, -0.19 * self.cfg.units_scale)
        ctx.line_to(0.09 * self.cfg.units_scale, 0.03 * self.cfg.units_scale)
        ctx.line_to(0.09 * self.cfg.units_scale, 0.05 * self.cfg.units_scale)
        ctx.line_to(0.13 * self.cfg.units_scale, 0.04 * self.cfg.units_scale)
        ctx.line_to(0.13 * self.cfg.units_scale, 0.08 * self.cfg.units_scale)
        ctx.line_to(0.05 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.44 * self.cfg.units_scale)
        # Symmetric half
        ctx.line_to(-0.05 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.13 * self.cfg.units_scale, 0.08 * self.cfg.units_scale)
        ctx.line_to(-0.13 * self.cfg.units_scale, 0.04 * self.cfg.units_scale)
        ctx.line_to(-0.09 * self.cfg.units_scale, 0.05 * self.cfg.units_scale)
        ctx.line_to(-0.09 * self.cfg.units_scale, 0.03 * self.cfg.units_scale)
        ctx.line_to(-0.28 * self.cfg.units_scale, -0.19 * self.cfg.units_scale)
        ctx.line_to(-0.28 * self.cfg.units_scale, -0.29 * self.cfg.units_scale)
        ctx.line_to(-0.08 * self.cfg.units_scale, -0.31 * self.cfg.units_scale)
        ctx.line_to(-0.06 * self.cfg.units_scale, -0.38 * self.cfg.units_scale)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # Ref. point
        ctx.move_to(0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.line_to(-0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.move_to(0.0 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, -0.1 * self.cfg.units_scale)
        ctx.stroke()

        ctx.restore()

    def _draw_sam_battery(self, ctx, o: SamBattery):
        # Translate to SAM battery position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        ctx.rotate(angle)
        ctx.set_source_rgba(*o.fill_color)
        ctx.set_line_width(1)
        # Square
        ctx.move_to(0.15 * self.cfg.units_scale, -0.15 * self.cfg.units_scale)
        ctx.line_to(0.15 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.15 * self.cfg.units_scale, 0.15 * self.cfg.units_scale)
        ctx.line_to(-0.15 * self.cfg.units_scale, -0.15 * self.cfg.units_scale)
        ctx.close_path()
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # Radar
        ctx.arc(0.0, 0.0, 0.4 * self.cfg.units_scale, -1 + math.pi / 2.0, 1 + math.pi / 2.0)
        ctx.new_sub_path()
        ctx.arc(0.0, 0.0, 0.6 * self.cfg.units_scale, -1 + math.pi / 2.0, 1 + math.pi / 2.0)
        ctx.stroke()
        # Ref. point
        ctx.move_to(0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.line_to(-0.1 * self.cfg.units_scale, 0.0 * self.cfg.units_scale)
        ctx.move_to(0.0 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, -0.1 * self.cfg.units_scale)
        ctx.stroke()
        # Radar limit
        ctx.set_dash([5, 5])
        ctx.move_to(0, 0)
        ctx.arc(0.0, 0.0, self.pixels_km * o.radar_range_km,
                math.pi / 2.0 - o.radar_amplitude_deg / 180 * math.pi, math.pi / 2.0)
        ctx.close_path()
        ctx.stroke()
        # Missile range
        ctx.arc(0.0, 0.0, self.pixels_km * o.missile_range_km, 0, 2.0 * math.pi)
        ctx.stroke()

        ctx.restore()

    def _draw_waypoint(self, ctx, o: Waypoint):
        # Translate to waypoint position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        ctx.rotate(angle)
        ctx.set_line_width(1)
        # Circle
        ctx.new_path()
        ctx.arc(0.0, 0.0, 0.1 * self.cfg.units_scale, 0, 2 * math.pi)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        # Flag
        ctx.move_to(0.0, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0, 0.25 * self.cfg.units_scale)
        ctx.stroke()
        ctx.move_to(0.0, 0.25 * self.cfg.units_scale)
        ctx.line_to(0.0, 0.45 * self.cfg.units_scale)
        ctx.line_to(0.25 * self.cfg.units_scale, 0.35 * self.cfg.units_scale)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()

        ctx.restore()

    def _draw_missile(self, ctx, o: Missile):
        # Translate to waypoint position
        x, y, angle = self._get_image_xya(o.lat, o.lon, o.heading)
        ctx.save()
        ctx.translate(x, y)

        # Draw status text (not rotated)
        if o.info_text:
            ctx.set_source_rgba(*o.edge_color)
            self._draw_text(ctx, 0, -self.cfg.sprites_info_spacing, o.info_text)

        ctx.rotate(angle)
        ctx.set_line_width(1)
        # Body
        ctx.move_to(0.05 * self.cfg.units_scale, -0.3 * self.cfg.units_scale)
        ctx.line_to(0.07 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(0.0 * self.cfg.units_scale, 0.4 * self.cfg.units_scale)
        # Sym
        ctx.line_to(-0.07 * self.cfg.units_scale, 0.1 * self.cfg.units_scale)
        ctx.line_to(-0.05 * self.cfg.units_scale, -0.3 * self.cfg.units_scale)
        ctx.close_path()
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()

        ctx.restore()

    def _draw_poly_line(self, ctx, o: PolyLine):
        if len(o.points) <= 1:
            return

        ctx.save()
        ctx.set_line_width(o.line_width)
        if o.dash:
            ctx.set_dash(o.dash)
        ctx.set_source_rgba(*o.edge_color)
        ctx.new_path()
        x, y, a = self._get_image_xya(o.points[0][0], o.points[0][1], 0)
        ctx.move_to(x, y)
        for i in range(1, len(o.points)):
            x, y, a = self._get_image_xya(o.points[i][0], o.points[i][1], 0)
            ctx.line_to(x, y)
        ctx.stroke()
        ctx.restore()

    def _draw_rect(self, ctx, o: Rect):
        ctx.save()

        left, top, _ = self._get_image_xya(o.top_lat, o.left_lon, 0)
        right, bottom, _ = self._get_image_xya(o.bottom_lat, o.right_lon, 0)
        ctx.rectangle(left, top, right - left, bottom - top)

        ctx.set_line_width(o.line_width)
        ctx.set_source_rgba(*o.fill_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*o.edge_color)
        ctx.stroke()
        ctx.restore()

    def _draw_arc(self, ctx, o: Arc):
        ctx.save()

        x, y, a1 = self._get_image_xya(o.center_lat, o.center_lon, o.angle1 - 90)
        a2 = self._get_image_angle(o.angle2 - 90)
        ctx.new_path()
        ctx.arc(x, y, self._get_image_distance(o.radius), min(a1, a2), max(a1, a2))

        ctx.set_line_width(o.line_width)
        if o.dash:
            ctx.set_dash(o.dash)
        if o.fill_color:
            ctx.set_source_rgba(*o.fill_color)
            ctx.fill_preserve()
        if o.edge_color:
            ctx.set_source_rgba(*o.edge_color)
            ctx.stroke()
        ctx.restore()
