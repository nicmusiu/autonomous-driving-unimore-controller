import logging
from math import cos, hypot, inf, isclose, isfinite, remainder, sin, tau
from numbers import Real
from types import SimpleNamespace

from numpy import errstate
from numpy.linalg import eigvals

from controllers.factory import CombinedController
from models import ModelData
from scenarios import TrackData
from simulator.simulation_engine import VehicleState

from control_algorithms.purepursuit import PurePursuitController
from control_algorithms.stanley import StanleyController
from utilities.trajectory import Trajectory


logger = logging.getLogger(__name__)


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


class MPCLateralController:
    def __init__(self, model, track, *, horizon, timestep, min_speed, fallback):
        if any(not _finite(value) or value <= 0 for value in (horizon, timestep, min_speed)):
            raise ValueError("MPC horizon, timestep and min_speed must be finite and positive")
        steps = horizon / timestep
        if not isfinite(steps) or steps < 1 or not isclose(steps, round(steps), rel_tol=0, abs_tol=1e-9):
            raise ValueError("lateral.mpc.horizon must be a positive integer multiple of timestep")
        self.steps = round(steps)
        self.timestep = timestep
        self.min_speed = min_speed
        self.fallback = fallback
        cf, cr = model.tires.Cf, model.tires.Cr
        if any(not _finite(value) or value <= 0 for value in (cf, cr)):
            raise ValueError("MPC requires finite positive model tires.Cf and tires.Cr (1/rad)")
        geometry, mass = model.geometry, model.mass_properties
        self.parameters = SimpleNamespace(
            mass=mass.mass, I_z=mass.Izz, l_f=geometry.lf, l_r=geometry.lr,
            l_wb=geometry.wheelbase, Cf=cf, Cr=cr,
        )
        self._check_euler_activation()
        steering = model.steering
        self.steering_bounds = (steering.command_min / steering.ratio, steering.command_max / steering.ratio)
        points = model.reference_points
        self.cog_dx = points.center_of_mass[0] - points.chassis_origin[0]
        self.cog_dy = points.center_of_mass[1] - points.chassis_origin[1]
        try:
            from casadi import has_nlpsol
            from control_algorithms.mpc import casadi_model, opt_step

            if not has_nlpsol("ipopt"):
                raise RuntimeError("IPOPT plugin unavailable")
        except (ImportError, OSError, RuntimeError) as error:
            raise ValueError("MPC requires CasADi with IPOPT. Install it with python -m pip install casadi.") from error
        self.prediction = casadi_model(self.parameters, timestep)
        self._solve = opt_step
        self.trajectory = Trajectory(track)
        self.low_speed_fallbacks = 0
        self.invalid_feedback_fallbacks = 0
        self.solver_failures = 0
        self.solver_failure_fallbacks = 0

    def _check_euler_activation(self):
        # Linearize lateral velocity and yaw rate about straight motion.
        p = self.parameters
        front = p.Cf * p.mass * 9.81 * p.l_r / p.l_wb
        rear = p.Cr * p.mass * 9.81 * p.l_f / p.l_wb
        moment = rear * p.l_r - front * p.l_f
        v = self.min_speed
        dt = self.timestep
        poles = eigvals([
            [1 - dt * (front + rear) / (p.mass * v), dt * (moment / (p.mass * v) - v)],
            [dt * moment / (p.I_z * v), 1 - dt * (front * p.l_f**2 + rear * p.l_r**2) / (p.I_z * v)],
        ])
        if max(abs(poles)) >= 1:
            raise ValueError("MPC Euler model is unstable at min_speed; increase min_speed or reduce timestep")

    def _state(self, feedback):
        x, y, yaw, vx, vy, r = feedback.com_x, feedback.com_y, feedback.yaw, feedback.vx, feedback.vy, feedback.yaw_rate
        if not all(_finite(value) for value in (x, y, yaw, vx, vy, r)):
            return None
        vx -= r * self.cog_dy
        vy += r * self.cog_dx
        if not all(_finite(value) for value in (vx, vy)):
            return None
        return SimpleNamespace(x=x, y=y, theta=yaw, vx=vx, vy=vy, r=r)

    def _targets(self, state):
        s = self.trajectory.project(state.x, state.y)
        yaw = state.theta
        targets = []
        for step in range(1, self.steps + 1):
            target_s = s + step * state.vx * self.timestep
            if not _finite(target_s):
                return None
            x, y, heading = self.trajectory.reference_pose(target_s)
            if not all(_finite(value) for value in (x, y, heading)):
                return None
            yaw += remainder(heading - yaw, tau)
            targets.append([x, y, yaw])
        return targets

    def compute(self, sim_time, vehicle_state):
        state = self._state(vehicle_state)
        if state is None:
            self.invalid_feedback_fallbacks += 1
            return self.fallback.compute(sim_time, vehicle_state)
        if state.vx < self.min_speed:
            self.low_speed_fallbacks += 1
            return self.fallback.compute(sim_time, vehicle_state)
        targets = self._targets(state)
        if targets is None:
            self.invalid_feedback_fallbacks += 1
            return self.fallback.compute(sim_time, vehicle_state)
        try:
            angle = self._solve(targets, state, self.prediction, *self.steering_bounds)
            if not _finite(angle):
                raise ValueError("non-finite MPC steering")
            return float(angle)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            self.solver_failures += 1
            self.solver_failure_fallbacks += 1
            if self.solver_failures == 1 or self.solver_failures % 100 == 0:
                logger.warning("MPC solve failed at t=%.3f (%d failures): %s",
                               sim_time, self.solver_failures, error)
            return self.fallback.compute(sim_time, vehicle_state)
