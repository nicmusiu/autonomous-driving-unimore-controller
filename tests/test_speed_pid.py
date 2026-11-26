from unittest import TestCase

from control_algorithms.pid import PIDController, PIDSpeedController


class PIDControllerTests(TestCase):
    def test_p_pi_pd_and_pid_use_the_same_interface(self):
        for ki, kd, expected in ((0, 0, 2), (0.5, 0, 2.5), (0, 0.25, 2.5), (0.5, 0.25, 3)):
            with self.subTest(ki=ki, kd=kd):
                pid = PIDController(kp=1, ki=ki, kd=kd)
                self.assertEqual(pid.compute(10, 9, 0.25), 1)
                self.assertEqual(pid.compute(10, 8, 0.5), expected)

    def test_asymmetric_and_one_sided_output_limits(self):
        pid = PIDController(2, 0, 0, output_limits=(-3, 1))
        self.assertEqual(pid.compute(10, 8, None), 1)
        self.assertEqual(pid.compute(10, 12, 0.5), -3)
        pid = PIDController(2, 0, 0, output_limits=(None, 1))
        self.assertEqual(pid.compute(10, 12, None), -4)
        self.assertEqual(pid.compute(10, 8, 0.5), 1)
        pid = PIDController(2, 0, 0, output_limits=(-3, None))
        self.assertEqual(pid.compute(10, 8, None), 4)
        self.assertEqual(pid.compute(10, 12, 0.5), -3)

    def test_asymmetric_anti_windup_considers_derivative(self):
        pid = PIDController(0, 1, 1, output_limits=(-3, 1))
        self.assertEqual(pid.compute(0, 0, None), 0)
        self.assertEqual(pid.compute(1, 0, 0.25), 1)
        self.assertEqual(pid.compute(0, 0, 0.25), -3)
        self.assertEqual(pid.compute(0, 0, 0.25), 0)

    def test_invalid_sample_is_bounded_and_resets_history(self):
        for sample in ((float("nan"), 9, 1), (10, float("inf"), 1), (10, 9, -1),
                       (10, 9, 0), (10, 9, None), (10, 9, float("inf"))):
            with self.subTest(sample=sample):
                pid = PIDController(1, 1, 1)
                pid.compute(10, 9, None)
                pid.compute(10, 9, 1)
                self.assertEqual(pid.compute(*sample), 0)
                self.assertEqual(pid.compute(10, 8, None), 2)
        pid = PIDController(1, 0, 0, output_limits=(2, 4))
        self.assertEqual(pid.compute(0, float("nan"), None), 2)

    def test_bad_limits_are_rejected(self):
        for limits in ((2, 1), (float("nan"), 1), (None, float("inf"))):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    PIDController(1, 0, 0, output_limits=limits)

    def test_reset_restores_first_sample_behavior(self):
        pid = PIDController(1, 1, 1)
        pid.compute(10, 9, None)
        self.assertEqual(pid.compute(10, 9, 1), 2)
        pid.reset()
        self.assertEqual(pid.compute(10, 8, None), 2)


class PIDSpeedTests(TestCase):
    def sample(self, pid, time, speed):
        pid.sim_time = time
        throttle, brake = pid.compute(speed)
        self.assertTrue(0 <= throttle <= 1 and 0 <= brake <= 1)
        self.assertFalse(throttle and brake)
        return throttle, brake

    def test_proportional_action_with_both_error_signs(self):
        pid = PIDSpeedController(10, kp=0.2, ki=0, kd=0)
        self.assertEqual(self.sample(pid, 0, 9), (0.2, 0.0))
        self.assertEqual(self.sample(pid, 1, 11), (0.0, 0.2))
        self.assertEqual(self.sample(pid, 2, 10), (0.0, 0.0))
        self.assertEqual(self.sample(pid, 3, 0), (1.0, 0.0))
        self.assertEqual(self.sample(pid, 4, 20), (0.0, 1.0))

    def test_first_call_is_proportional_only(self):
        pid = PIDSpeedController(10, kp=0.2, ki=0.3, kd=0.4)
        self.assertEqual(self.sample(pid, 12, 9), (0.2, 0.0))

    def test_integral_action_uses_variable_simulation_intervals(self):
        pid = PIDSpeedController(10, kp=0, ki=0.5, kd=0)
        self.assertEqual(self.sample(pid, 0, 9), (0.0, 0.0))
        self.assertEqual(self.sample(pid, 0.25, 9), (0.125, 0.0))
        self.assertEqual(self.sample(pid, 1, 9), (0.5, 0.0))

    def test_derivative_action_uses_variable_simulation_intervals(self):
        pid = PIDSpeedController(10, kp=0, ki=0, kd=0.25)
        self.assertEqual(self.sample(pid, 0, 10), (0.0, 0.0))
        self.assertEqual(self.sample(pid, 0.5, 9), (0.5, 0.0))
        self.assertEqual(self.sample(pid, 1.5, 10), (0.0, 0.25))

    def test_saturation_and_anti_windup_in_both_directions(self):
        for speed, saturated in ((0, (1.0, 0.0)), (20, (0.0, 1.0))):
            with self.subTest(speed=speed):
                pid = PIDSpeedController(10, kp=0.5, ki=0.5, kd=0)
                for time in range(100):
                    self.assertEqual(self.sample(pid, time, speed), saturated)
                self.assertEqual(self.sample(pid, 100, 10), (0.0, 0.0))

    def test_integrator_uses_available_headroom_and_can_unwind(self):
        pid = PIDSpeedController(10, kp=0, ki=0.5, kd=0)
        self.sample(pid, 0, 8)
        self.assertEqual(self.sample(pid, 4, 8), (1.0, 0.0))
        self.assertEqual(self.sample(pid, 5, 11), (0.5, 0.0))
        self.assertEqual(self.sample(pid, 6, 11), (0.0, 0.0))

    def test_derivative_saturation_blocks_integral_growth(self):
        pid = PIDSpeedController(10, kp=0, ki=0.5, kd=1)
        self.sample(pid, 0, 10)
        self.assertEqual(self.sample(pid, 0.25, 9), (1.0, 0.0))
        self.assertEqual(self.sample(pid, 0.5, 10), (0.0, 1.0))
        self.assertEqual(self.sample(pid, 1, 10), (0.0, 0.0))

    def test_reset_clears_integral_time_and_derivative_history(self):
        pid = PIDSpeedController(10, kp=0.125, ki=0.5, kd=0.25)
        self.sample(pid, 0, 9)
        self.assertEqual(self.sample(pid, 1, 9), (0.625, 0.0))
        pid.reset()
        self.assertEqual(self.sample(pid, 0, 8), (0.25, 0.0))

    def test_invalid_feedback_or_time_returns_zero_and_discards_history(self):
        for time, speed in ((1, None), (1, float("nan")), (1, float("inf")),
                            (float("nan"), 9), (0, 9), (-1, 9)):
            with self.subTest(time=time, speed=speed):
                pid = PIDSpeedController(10, kp=0.2, ki=0.5, kd=0.5)
                self.sample(pid, 0, 9)
                self.assertEqual(self.sample(pid, time, speed), (0.0, 0.0))
                self.assertEqual(self.sample(pid, 2, 9), (0.2, 0.0))

    def test_nonfinite_demand_resets_the_adapter_timestamp(self):
        pid = PIDSpeedController(10, kp=2, ki=0, kd=0)
        self.sample(pid, 0, 10)
        self.assertEqual(self.sample(pid, 1, -1e308), (0.0, 0.0))
        self.assertEqual(self.sample(pid, 1, 9.75), (0.5, 0.0))
