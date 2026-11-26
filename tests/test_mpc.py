import sys
from math import inf, nan, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, call, patch

import yaml

from controller import MPCLateralController
from models.model import SteeringParameters
from simulator.simulation_engine import VehicleState
from utilities.model_loader import load_model


class ControllerModelTests(TestCase):
    def setUp(self):
        asset = Path(__file__).resolve().parents[1] / "assets" / "model.yaml"
        self.data = yaml.safe_load(asset.read_text(encoding="utf-8"))
        self.path = Path(self.enterContext(TemporaryDirectory())) / "model.yaml"

    def load(self):
        self.path.write_text(yaml.safe_dump(self.data), encoding="utf-8")
        return load_model(self.path)

    def test_reads_independent_controller_tire_metadata(self):
        self.data["tires"].update(Cf=16, Cr=14, future_parameter=3)
        model = self.load()
        self.assertEqual((model.tires.Cf, model.tires.Cr, model.tires.future_parameter), (16.0, 14.0, 3))

    def test_validates_optional_stiffness(self):
        self.data["tires"].pop("Cf", None)
        self.assertIsNone(self.load().tires.Cf)
        for value in (False, 0, -1, nan, inf):
            with self.subTest(value=value):
                self.data["tires"]["Cf"] = value
                with self.assertRaisesRegex(ValueError, r"tires\.Cf must be finite and positive"):
                    self.load()


class MPCLateralTests(TestCase):
    def setUp(self):
        self.casadi = SimpleNamespace(has_nlpsol=Mock(return_value=True))
        self.algorithm = SimpleNamespace(casadi_model=Mock(), opt_step=Mock(return_value=0.1))
        self.enterContext(patch.dict(sys.modules, {"casadi": self.casadi,
                                                  "control_algorithms.mpc": self.algorithm}))
        self.model = SimpleNamespace(
            geometry=SimpleNamespace(lf=1.3, lr=1.4, wheelbase=2.7),
            mass_properties=SimpleNamespace(mass=1500, Izz=1600),
            tires=SimpleNamespace(Cf=15.0, Cr=14.0),
            steering=SteeringParameters(20.0, -4.0, 6.0),
            reference_points=SimpleNamespace(center_of_mass=(-1.3, -0.1, 0), chassis_origin=(0, 0, 0)),
        )
        self.track = SimpleNamespace(x=(0.0, 10.0, 20.0), y=(0.0, 0.0, 0.0))
        self.fallback = Mock()
        self.fallback.compute.return_value = -0.02
        self.controller = self.make_controller()

    def make_controller(self, **settings):
        return MPCLateralController(self.model, self.track, fallback=self.fallback,
                                    **(dict(horizon=0.1, timestep=0.05, min_speed=8.0) | settings))

    @staticmethod
    def feedback(speed=10.0, yaw=0.0, yaw_rate=0.2):
        return VehicleState(0.0, speed, 0.0, 1, (0.0,) * 4, position_x=2.0, position_y=0.0,
                            yaw=yaw, lateral_velocity=0.3, yaw_rate=yaw_rate)

    def test_model_state_bounds_and_finite_steering(self):
        parameters, timestep = self.algorithm.casadi_model.call_args.args
        self.assertEqual(vars(parameters), dict(mass=1500, I_z=1600, l_f=1.3, l_r=1.4,
                                               l_wb=2.7, Cf=15.0, Cr=14.0))
        self.assertEqual(timestep, 0.05)
        result = self.controller.compute(0.0, self.feedback())
        self.assertIs(type(result), float)
        self.assertEqual(result, 0.1)
        targets, state, _, lower, upper = self.algorithm.opt_step.call_args.args
        self.assertEqual((lower, upper), (-0.2, 0.3))
        self.assertEqual((state.x, state.y, state.theta, state.vx, state.r), (2.0, 0.0, 0.0, 10.02, 0.2))
        self.assertAlmostEqual(state.vy, 0.04)
        self.assertAlmostEqual(targets[0][0], 2.0 + 10.02 * 0.05)
        self.assertAlmostEqual(targets[1][0], 2.0 + 2 * 10.02 * 0.05)

    def test_unwraps_reference_from_current_yaw_and_projects_cog(self):
        trajectory = self.controller.trajectory = Mock()
        trajectory.project.return_value = 5
        trajectory.reference_pose.side_effect = [(1, 2, -pi + 0.1), (2, 3, -pi + 0.2)]
        self.controller.compute(0, self.feedback(yaw=pi - 0.1, yaw_rate=0))
        trajectory.project.assert_called_once_with(2.0, 0.0)
        self.assertEqual(trajectory.reference_pose.call_args_list, [call(5.5), call(6.0)])
        targets = self.algorithm.opt_step.call_args.args[0]
        self.assertAlmostEqual(targets[0][2], pi + 0.1)
        self.assertAlmostEqual(targets[1][2], pi + 0.2)

    def test_open_target_clamps_and_closed_target_wraps(self):
        state = self.controller._state(self.feedback(yaw_rate=0))
        self.controller.trajectory.project = Mock(return_value=19.9)
        self.assertAlmostEqual(self.controller._targets(state)[0][0], 20.0)
        self.track = SimpleNamespace(x=(0, 3, 3, 0, 0), y=(0, 0, 3, 3, 0))
        closed = self.make_controller()
        closed.trajectory.project = Mock(return_value=closed.trajectory.length - 0.25)
        target = closed._targets(state)[0]
        self.assertEqual(tuple(target[:2]), closed.trajectory.reference_pose(0.25)[:2])

    def test_fallback_on_low_speed_invalid_feedback_and_solver_errors(self):
        self.assertEqual(self.controller.compute(0.0, self.feedback(speed=0)), -0.02)
        self.assertEqual(self.controller.compute(0.02, self.feedback(yaw_rate=None)), -0.02)
        self.algorithm.opt_step.assert_not_called()
        self.algorithm.opt_step.side_effect = [RuntimeError("IPOPT failed"), nan]
        with self.assertLogs("controller", level="WARNING"):
            self.assertEqual(self.controller.compute(0.04, self.feedback()), -0.02)
        self.assertEqual(self.controller.compute(0.06, self.feedback()), -0.02)
        self.assertEqual(self.controller.low_speed_fallbacks, 1)
        self.assertEqual(self.controller.invalid_feedback_fallbacks, 1)
        self.assertEqual(self.controller.solver_failures, 2)
        self.assertEqual(self.controller.solver_failure_fallbacks, 2)

    def test_requires_integer_horizon_optional_tire_data_and_ipopt(self):
        for settings in (dict(horizon=0.11), dict(timestep=0), dict(min_speed=inf)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.make_controller(**settings)
        self.model.tires.Cr = None
        with self.assertRaisesRegex(ValueError, "tires.Cf and tires.Cr"):
            self.make_controller()
        self.model.tires.Cr = 15
        with self.assertRaisesRegex(ValueError, "Euler model is unstable"):
            self.make_controller(min_speed=3.0)
        self.casadi.has_nlpsol.return_value = False
        with self.assertRaisesRegex(ValueError, "pip install casadi"):
            self.make_controller()
