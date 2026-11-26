from math import cos, hypot, inf, isfinite, sin
from numbers import Real

from numpy import errstate

from controllers.factory import CombinedController
from models import ModelData
from scenarios import TrackData
from simulator.simulation_engine import VehicleState

from control_algorithms.purepursuit import PurePursuitController
from control_algorithms.stanley import StanleyController
from utilities.trajectory import Trajectory


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


class _SteeringWheelController:
    def __init__(self, lateral, steering, forward_time):
        self._lateral = lateral
        self._steering = steering
        self._forward_time = forward_time

    def compute(self, sim_time, vehicle_state):
        if not _finite(sim_time):
            return None
        if self._forward_time is not None:
            self._forward_time(sim_time)
        if vehicle_state is None or not _finite(vehicle_state.vx):
            return None
        road_wheel = self._lateral.compute(sim_time, vehicle_state)
        if not _finite(road_wheel):
            return None
        steering_wheel = self._steering.ratio * road_wheel
        return min(max(steering_wheel, self._steering.command_min), self._steering.command_max)


def compose_controller(lateral, longitudinal, steering, *, forward_time=None):
    """Connect road-wheel steering and pedal control to the simulator."""
    return CombinedController(_SteeringWheelController(lateral, steering, forward_time), longitudinal)


class AdaptivePurePursuitController:
    def __init__(
        self, model: ModelData, track: TrackData, *,
        base_lookahead, speed_gain, curvature_gain, curvature_floor, min_lookahead,
    ):
        self._steering_law = PurePursuitController(model.geometry.wheelbase, max_steer=inf)
        self.base_lookahead = base_lookahead
        self.speed_gain = speed_gain
        self.curvature_gain = curvature_gain
        self.curvature_floor = curvature_floor
        self.min_lookahead = min_lookahead

        self.trajectory = Trajectory(track)

    def lookahead(self, speed, curvature):
        distance = (self.base_lookahead + self.speed_gain * speed
                    + self.curvature_gain / max(self.curvature_floor, abs(curvature)))
        return max(self.min_lookahead, distance)

    def compute(self, sim_time: float, vehicle_state: VehicleState) -> float | None:
        """Return a road-wheel angle in radians or None for an invalid target."""
        x, y, yaw, speed = (vehicle_state.rear_axle_x, vehicle_state.rear_axle_y,
                            vehicle_state.yaw, vehicle_state.vx)
        if any(value is None or not isfinite(value) for value in (x, y, yaw, speed)):
            return None
        s = self.trajectory.project(x, y)
        with errstate(divide="ignore", invalid="ignore"):
            curvature = self.trajectory.curvature(s)
        if not isfinite(curvature):
            return None
        tx, ty, _ = self.trajectory.reference_pose(s + self.lookahead(speed, curvature))
        dx, dy = tx - x, ty - y
        distance = hypot(dx, dy)
        if not isfinite(distance) or distance <= 1e-9:
            return None
        # Keep the relative displacement in world axes. The course law subtracts yaw.
        return self._steering_law.compute_steering_angle((dx, dy), yaw, distance)


class StanleyLateralController:
    def __init__(self, model: ModelData, track: TrackData, *, k,
                 heading_error_weight=1.0, cross_track_error_weight=1.0):
        self._steering_law = StanleyController(
            k=k, l_f=model.geometry.lf, max_steer=inf,
            heading_error_weight=heading_error_weight,
            cross_track_error_weigth=cross_track_error_weight,
        )
        self.trajectory = Trajectory(track)

    def compute(self, sim_time: float, vehicle_state: VehicleState) -> float | None:
        """Return road-wheel radians using the projected front axle."""
        pose = (vehicle_state.com_x, vehicle_state.com_y, vehicle_state.yaw)
        speed = vehicle_state.vx
        if any(value is None or not isfinite(value) for value in (*pose, speed)):
            return None
        x, y, yaw = pose
        l_f = self._steering_law.l_f
        front_axle = (x + l_f * cos(yaw), y + l_f * sin(yaw))
        if not all(isfinite(value) for value in front_axle):
            return None
        target = self.trajectory.reference_pose(self.trajectory.project(*front_axle))
        if any(value is None or not isfinite(value) for value in target):
            return None
        angle = self._steering_law.compute_steering_angle(pose, target, speed)
        return angle if isfinite(angle) else None
