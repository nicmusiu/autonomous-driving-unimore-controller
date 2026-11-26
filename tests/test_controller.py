from functools import partial
from math import atan, inf, pi, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import yaml

from controllers import VehicleCommand
from models.model import SteeringParameters
from simulator.simulation_engine import VehicleState

from control_algorithms.pid import PIDSpeedController
from control_algorithms.purepursuit import PurePursuitController
from controller import compose_controller
from main import load_controller


def state(time, speed=9.0):
    return VehicleState(time, speed, 0.0, 1, (0.0,) * 4)


class ControllerCompositionTests(TestCase):
    def setUp(self):
        self.lateral = Mock()
        self.longitudinal = Mock()
        self.longitudinal.compute.return_value = (0.25, 0.0)
        self.limits = SteeringParameters(ratio=4.0, command_min=-0.3, command_max=0.6)
        self.controller = compose_controller(self.lateral, self.longitudinal, self.limits)

    def test_converts_road_wheel_to_steering_wheel_radians(self):
        feedback = state(2.0)
        self.lateral.compute.return_value = 0.1
        self.assertEqual(self.controller.compute(2.0, feedback), VehicleCommand(0.25, 0.0, 0.4))
        self.lateral.compute.assert_called_once_with(2.0, feedback)
        self.longitudinal.compute.assert_called_once_with(9.0)

    def test_applies_asymmetric_limits_after_conversion(self):
        for angle, expected in ((0.2, 0.6), (-0.1, -0.3)):
            with self.subTest(angle=angle):
                self.lateral.compute.return_value = angle
                self.assertEqual(self.controller.compute(0.0, state(0.0)).steering, expected)

    def test_invalid_lateral_output_skips_longitudinal_control(self):
        for invalid in (None, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                self.lateral.compute.return_value = invalid
                self.assertEqual(self.controller.compute(0.0, state(0.0)), VehicleCommand(0.0, 0.0, 0.0))
        self.longitudinal.compute.assert_not_called()

    def test_valid_zero_steering_still_computes_longitudinal_control(self):
        self.lateral.compute.return_value = 0.0
        self.assertEqual(self.controller.compute(0.0, state(0.0)), VehicleCommand(0.25, 0.0, 0.0))
        self.longitudinal.compute.assert_called_once_with(9.0)

    def test_invalid_shared_feedback_skips_both_algorithms(self):
        for time, feedback in ((0.0, None), (0.0, state(0.0, float("nan"))), (float("inf"), state(0.0))):
            with self.subTest(time=time, feedback=feedback):
                self.assertEqual(self.controller.compute(time, feedback), VehicleCommand(0.0, 0.0, 0.0))
        self.lateral.compute.assert_not_called()
        self.longitudinal.compute.assert_not_called()

    def test_forwards_simulation_time_before_both_algorithms(self):
        calls = Mock()
        calls.lateral.compute.return_value = 0.0
        calls.longitudinal.compute.return_value = (0.25, 0.0)
        controller = compose_controller(
            calls.lateral, calls.longitudinal, self.limits, forward_time=calls.forward_time,
        )
        feedback = state(3.0)
        controller.compute(3.0, feedback)
        self.assertEqual([entry[0] for entry in calls.mock_calls],
                         ["forward_time", "lateral.compute", "longitudinal.compute"])
        calls.forward_time.assert_called_once_with(3.0)

    def test_invalid_lateral_output_does_not_advance_pid_state(self):
        pid = PIDSpeedController(target_speed=10.0, kp=0.0, ki=0.5, kd=0.0)
        self.lateral.compute.side_effect = [0.0, None, 0.0]
        controller = compose_controller(
            self.lateral, pid, self.limits, forward_time=partial(setattr, pid, "sim_time"),
        )
        self.assertEqual(controller.compute(0.0, state(0.0)).throttle, 0.0)
        self.assertEqual(controller.compute(0.5, state(0.5, 6.0)), VehicleCommand(0.0, 0.0, 0.0))
        self.assertEqual(pid.sim_time, 0.5)
        self.assertEqual(controller.compute(1.0, state(1.0)).throttle, 0.5)


class PurePursuitLawTests(TestCase):
    def test_world_axis_displacement_and_euclidean_chord(self):
        law = PurePursuitController(wheelbase=4.0, max_steer=inf)
        self.assertEqual(law.compute_steering_angle((0.0, 2.0), pi / 2, 2.0), 0.0)
        self.assertAlmostEqual(law.compute_steering_angle((1.0, 1.0), 0.0, sqrt(2.0)), atan(4.0))

    def test_generic_law_can_apply_a_symmetric_limit(self):
        law = PurePursuitController(wheelbase=4.0, max_steer=0.2)
        self.assertEqual(law.compute_steering_angle((0.0, 2.0), 0.0, 2.0), 0.2)
        self.assertEqual(law.compute_steering_angle((0.0, -2.0), 0.0, 2.0), -0.2)


class ControllerConfigurationTests(TestCase):
    def setUp(self):
        self.config = {
            "endpoint": "tcp://127.0.0.1:5555",
            "model_file": "model.yaml",
            "track_file": "track.json",
            "lateral": {
                "selected": "pure_pursuit",
                "pure_pursuit": {"base_lookahead": 10.0, "speed_gain": 0.05, "curvature_gain": 0.0001,
                                 "curvature_floor": 0.01, "min_lookahead": 1.4},
                "stanley": {"k": 1.0, "heading_error_weight": 1.0, "cross_track_error_weight": 1.0},
            },
            "longitudinal": {"target_speed": 33.3333, "kp": 0.2, "ki": 0.04, "kd": 0.02},
        }
        self.path = Path(self.enterContext(TemporaryDirectory())) / "controller.yaml"
        model = SimpleNamespace(geometry=SimpleNamespace(wheelbase=4.0, lf=2.0),
                                steering=SteeringParameters(1.0, -1.0, 1.0))
        track = SimpleNamespace(x=(0.0, 25.0, 50.0), y=(0.0, 0.0, 0.0))
        self.enterContext(patch("main.load_model", return_value=model))
        self.enterContext(patch("main.load_track", return_value=track))

    def load(self):
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        return load_controller(self.path)[1]

    def commands(self):
        controller = self.load()
        return [controller.compute(time, VehicleState(
            time, 33.0, 0.0, 1, (0.0,) * 4, position_x=4.0, position_y=1.0,
            yaw=0.0, rear_axle_x=2.0, rear_axle_y=1.0,
        )) for time in (0.0, 0.02)]

    def test_switches_lateral_controller_without_changing_pid(self):
        pure_pursuit = self.commands()
        self.config["lateral"]["selected"] = "stanley"
        stanley = self.commands()
        for pure_pursuit_command, stanley_command in zip(pure_pursuit, stanley):
            self.assertNotEqual(pure_pursuit_command.steering, stanley_command.steering)
            self.assertEqual(pure_pursuit_command.throttle, stanley_command.throttle)
        self.assertGreater(stanley[1].throttle, stanley[0].throttle)

    def test_inactive_blocks_are_ignored(self):
        expected = self.commands()
        self.config["lateral"]["stanley"] = "invalid block"
        self.assertEqual(self.commands(), expected)

    def test_unknown_selection_and_missing_selected_block(self):
        lateral = self.config["lateral"]
        lateral["selected"] = "unknown"
        with self.assertRaisesRegex(ValueError, r"lateral\.selected.*unknown"):
            self.load()
        lateral["selected"] = "stanley"
        del lateral["stanley"]
        with self.assertRaisesRegex(ValueError, r"Missing selected controller block lateral\.stanley"):
            self.load()

    def test_missing_and_invalid_selected_parameters(self):
        pid = self.config["longitudinal"]
        del pid["ki"]
        with self.assertRaisesRegex(ValueError, r"Missing required parameter longitudinal\.ki"):
            self.load()
        pid["ki"] = True
        with self.assertRaisesRegex(ValueError, r"longitudinal\.ki must be finite and nonnegative"):
            self.load()
        pid["ki"] = 0.04
        self.config["lateral"]["pure_pursuit"]["min_lookahead"] = 0
        with self.assertRaisesRegex(ValueError, r"lateral\.pure_pursuit\.min_lookahead must be finite and positive"):
            self.load()
