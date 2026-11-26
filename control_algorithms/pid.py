from math import isfinite
from numbers import Real


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


class PIDController:
    """Compute bounded PID demand with conditional integration."""

    def __init__(self, kp: float, ki: float, kd: float, output_limits=(None, None)):
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
            if not _finite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self._lower, self._upper = output_limits
        if any(limit is not None and not _finite(limit) for limit in (self._lower, self._upper)):
            raise ValueError("Output limits must be finite or None")
        if self._lower is not None and self._upper is not None and self._lower > self._upper:
            raise ValueError("The lower output limit must not exceed the upper limit")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self._previous_error = None
        self._integral = 0.0

    @property
    def initialized(self) -> bool:
        """Whether a valid sample has established the derivative history."""
        return self._previous_error is not None

    def _clamp(self, demand):
        if self._lower is not None:
            demand = max(demand, self._lower)
        if self._upper is not None:
            demand = min(demand, self._upper)
        return demand

    def _invalid_sample(self):
        self.reset()
        return self._clamp(0.0)

    def compute(self, setpoint: float, measured_value: float, dt: float | None) -> float:
        """Use proportional action on the first sample, when dt may be None."""
        if not _finite(setpoint) or not _finite(measured_value):
            return self._invalid_sample()
        if dt is None:
            if self._previous_error is not None:
                return self._invalid_sample()
        elif not _finite(dt) or dt <= 0.0:
            return self._invalid_sample()

        error = setpoint - measured_value
        demand = self.kp * error + self._integral
        if self._previous_error is not None:
            if self.kd:
                demand += self.kd * (error - self._previous_error) / dt
            increment = self.ki * error * dt
            if not _finite(demand) or not _finite(increment):
                return self._invalid_sample()

            # Use only the available actuator headroom, including the D term.
            if increment > 0.0 and self._upper is not None:
                increment = min(increment, max(0.0, self._upper - demand),
                                max(0.0, self._upper) - self._integral)
            elif increment < 0.0 and self._lower is not None:
                increment = max(increment, min(0.0, self._lower - demand),
                                min(0.0, self._lower) - self._integral)
            self._integral += increment
            demand += increment

        if not _finite(demand):
            return self._invalid_sample()
        self._previous_error = error
        return self._clamp(demand)


class PIDSpeedController:
    """Supply simulation intervals and map signed PID demand to pedals."""

    def __init__(self, target_speed: float, kp: float, ki: float, kd: float):
        if not _finite(target_speed) or target_speed < 0:
            raise ValueError("target_speed must be finite and nonnegative")
        self.target_speed = target_speed
        self._pid = PIDController(kp, ki, kd, output_limits=(-1.0, 1.0))
        self.reset()

    def reset(self):
        self.sim_time = None
        self._previous_time = None
        self._pid.reset()

    def compute(self, vx: float) -> tuple[float, float]:
        time = self.sim_time
        if not _finite(time) or not _finite(vx):
            self.reset()
            return 0.0, 0.0
        dt = None if self._previous_time is None else time - self._previous_time
        if dt is not None and (not _finite(dt) or dt <= 0.0):
            self.reset()
            return 0.0, 0.0
        demand = self._pid.compute(self.target_speed, vx, dt)
        if not self._pid.initialized:
            self.reset()
            return 0.0, 0.0
        self._previous_time = time
        if demand >= 0.0:
            return demand, 0.0
        return 0.0, -demand
