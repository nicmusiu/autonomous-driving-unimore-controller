from math import hypot, nextafter

from .cubic_spline_planner import Spline2D


class Trajectory:
    """Spline queries and projection progress for one controller."""

    def __init__(self, track):
        points = list(zip(track.x, track.y))
        distinct = [points[0]]
        for point in points[1:]:
            if hypot(point[0] - distinct[-1][0], point[1] - distinct[-1][1]) > 1e-9:
                distinct.append(point)
        if len(distinct) < 2:
            raise ValueError("Trajectory requires at least two distinct waypoints")
        self.closed = hypot(distinct[0][0] - distinct[-1][0],
                            distinct[0][1] - distinct[-1][1]) <= 1e-6
        if self.closed:
            distinct[-1] = distinct[0]
            if len(distinct) < 4:
                raise ValueError("A closed track requires at least three distinct waypoints")
        self.spline = Spline2D(*zip(*distinct))
        self.length = self.spline.s[-1]

    def _query_s(self, s):
        if self.closed:
            return s % self.length
        # The legacy spline wraps position at the exact final knot and its
        # derivative routines cannot evaluate that knot directly.
        return min(max(s, 0.0), nextafter(self.length, 0.0))

    def reference_pose(self, s):
        s = self._query_s(s)
        return (*self.spline.calc_position(s), self.spline.calc_yaw(s))

    def curvature(self, s):
        return self.spline.calc_curvature(self._query_s(s))

    def project(self, x, y):
        if self.closed:
            self.spline.update_current_s((x, y))
            s = self._query_s(self.spline.cur_s)
            self.spline.cur_s = s
            return s
        # The legacy projection assumes a loop. For open tracks use a
        # bounded search on the spline, including the final waypoint.
        s = min(self.spline.s, key=lambda knot: self._distance(knot, x, y))
        index = self.spline.s.index(s)
        lower = self.spline.s[max(index - 1, 0)]
        upper = self.spline.s[min(index + 1, len(self.spline.s) - 1)]
        count = max(1, int((upper - lower) / 0.05))
        return min((lower + (upper - lower) * i / count for i in range(count + 1)),
                   key=lambda knot: self._distance(knot, x, y))

    def _distance(self, s, x, y):
        px, py = self.spline.calc_position(self._query_s(s))
        return hypot(px - x, py - y)
