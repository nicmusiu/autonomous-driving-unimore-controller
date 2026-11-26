from math import atan2, inf, pi
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from models.model import SteeringParameters
from simulator.simulation_engine import VehicleState

from control_algorithms.stanley import StanleyController
from controller import StanleyLateralController, compose_controller
from utilities.trajectory import Trajectory


class StanleyLawTests(TestCase):
    def setUp(self):
        self.law = StanleyController(1.0, 2.0, inf)
        self.target = (2.0, 0.0, 0.0)

    def test_zero_error_and_both_cross_track_signs(self):
        self.assertEqual(self.law.compute_steering_angle((0, 0, 0), self.target, 2), 0)
        for offset in (-1.0, 1.0):
            self.assertEqual(self.law.compute_steering_angle((0, offset, 0), self.target, 2),
                             atan2(-offset, 2))

    def test_heading_sign_and_wrapping(self):
        law = StanleyController(1.0, 2.0, inf, cross_track_error_weigth=0.0)
        for error in (-0.2, 0.2):
            self.assertAlmostEqual(law.compute_steering_angle((0, 0, 0), (0, 0, error), 2), error)
            self.assertAlmostEqual(
                law.compute_steering_angle((0, 0, pi - error / 2), (0, 0, -pi + error / 2), 2), error,
            )

    def test_low_speed_and_standstill_use_one_meter_per_second_floor(self):
        for speed in (-1.0, 0.0, 0.2, 1.0):
            self.assertEqual(self.law.compute_steering_angle((0, 1, 0), self.target, speed), atan2(-1, 1))

    def test_independent_weights_and_symmetric_generic_limit(self):
        pose = (0.0, 1.0, 0.0)
        target = (2.0, 0.0, 0.2)
        heading = StanleyController(1.0, 2.0, inf, 2.0, 0.0)
        cross = StanleyController(1.0, 2.0, inf, 0.0, 3.0)
        both = StanleyController(1.0, 2.0, inf, 2.0, 3.0)
        self.assertAlmostEqual(heading.compute_steering_angle(pose, target, 2), 0.4)
        self.assertAlmostEqual(both.compute_steering_angle(pose, target, 2),
                               0.4 + cross.compute_steering_angle(pose, target, 2))
        limited = StanleyController(1.0, 2.0, 0.1)
        self.assertEqual(limited.compute_steering_angle(pose, target, 2), -0.1)


class StanleyIntegrationTests(TestCase):
    def setUp(self):
        self.model = SimpleNamespace(geometry=SimpleNamespace(lf=2.0))
        self.track = SimpleNamespace(x=(0.0, 5.0, 10.0), y=(0.0, 0.0, 0.0))
        self.stanley = StanleyLateralController(self.model, self.track, k=1.0)

    @staticmethod
    def state(x=4.0, y=0.0, yaw=0.0, speed=2.0):
        return VehicleState(0.0, speed, 0.0, 1, (0.0,) * 4, position_x=x, position_y=y, yaw=yaw)

    def test_uses_the_same_front_axle_for_projection_and_steering(self):
        trajectory = self.stanley.trajectory
        trajectory.project = Mock(return_value=3.0)
        trajectory.reference_pose = Mock(return_value=(4.0, 2.0, pi / 2))
        self.assertAlmostEqual(self.stanley.compute(0, self.state(yaw=pi / 2)), 0.0)
        trajectory.project.assert_called_once_with(4.0, 2.0)
        trajectory.reference_pose.assert_called_once_with(3.0)

    def test_open_path_steering_and_shared_bridge_saturation(self):
        pedals = Mock()
        pedals.compute.return_value = (0.1, 0.0)
        limits = SteeringParameters(ratio=4.0, command_min=-0.3, command_max=0.6)
        combined = compose_controller(self.stanley, pedals, limits)
        self.assertEqual(self.stanley.compute(0, self.state()), 0.0)
        self.assertEqual(combined.compute(0, self.state(y=1.0)).steering, -0.3)
        self.assertEqual(combined.compute(0, self.state(y=-1.0)).steering, 0.6)

    def test_invalid_state_or_target(self):
        for state in (self.state(x=None), self.state(y=inf), self.state(yaw=None), self.state(speed=float("nan"))):
            with self.subTest(state=state):
                self.assertIsNone(self.stanley.compute(0, state))
        self.stanley.trajectory.reference_pose = Mock(return_value=(0.0, 0.0, float("nan")))
        self.assertIsNone(self.stanley.compute(0, self.state()))

    def test_closed_trajectory_wraps_with_independent_projection_progress(self):
        track = SimpleNamespace(x=(0, 3, 3, 0, 0), y=(0, 0, 3, 3, 0))
        first = StanleyLateralController(self.model, track, k=1.0)
        second = StanleyLateralController(self.model, track, k=1.0)
        self.assertTrue(first.trajectory.closed)
        self.assertEqual(first.trajectory.reference_pose(first.trajectory.length),
                         first.trajectory.reference_pose(0))
        self.assertGreater(first.trajectory.project(1, 0), 0)
        self.assertEqual(second.trajectory.spline.cur_s, 0)
        self.assertIsNotNone(first.compute(0, self.state(x=-1.0)))

    def test_open_trajectory_drops_consecutive_duplicates_and_clamps_at_end(self):
        track = SimpleNamespace(x=(0, 3, 3, 6), y=(0, 0, 0, 0))
        trajectory = Trajectory(track)
        self.assertFalse(trajectory.closed)
        s = trajectory.project(7, 0)
        self.assertAlmostEqual(s, 6)
        self.assertAlmostEqual(trajectory.reference_pose(s)[0], 6)
