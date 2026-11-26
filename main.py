import argparse
from functools import partial
from math import isfinite
from pathlib import Path

import yaml

from communication.controller_server import ControllerServer
from scenarios import load_track

from control_algorithms.pid import PIDSpeedController
from controller import AdaptivePurePursuitController, MPCLateralController, StanleyLateralController, compose_controller
from utilities.model_loader import load_model


PROJECT = Path(__file__).resolve().parent
PURE_PURSUIT_PARAMETERS = ("base_lookahead", "speed_gain", "curvature_gain", "curvature_floor", "min_lookahead")
PURE_PURSUIT_POSITIVE = ("curvature_floor", "min_lookahead")


def _mapping(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _number(value, path, *, positive=False):
    try:
        valid = type(value) in (int, float) and isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{path} must be finite and {bound}")
    return value


def _parameters(block, path, required, *, optional=(), positive=()):
    block = _mapping(block, path)
    for key in required:
        if key not in block:
            raise ValueError(f"Missing required parameter {path}.{key}")
    for key in block:
        if key not in required and key not in optional:
            raise ValueError(f"Unknown parameter {path}.{key}")
    return {key: _number(value, f"{path}.{key}", positive=key in positive)
            for key, value in block.items()}


def load_controller(path):
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        config = _mapping(yaml.safe_load(stream), "Controller configuration")
    endpoint = config["endpoint"]
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
        raise ValueError("endpoint must be a tcp:// address")
    lateral = _mapping(config.get("lateral"), "lateral")
    longitudinal = _mapping(config.get("longitudinal"), "longitudinal")
    lateral_type = lateral.get("selected")
    optional = ()
    positive = ()
    match lateral_type:
        case "pure_pursuit":
            lateral_class = AdaptivePurePursuitController
            required = PURE_PURSUIT_PARAMETERS
            positive = PURE_PURSUIT_POSITIVE
        case "stanley":
            lateral_class = StanleyLateralController
            required = ("k",)
            optional = ("heading_error_weight", "cross_track_error_weight")
        case "mpc":
            lateral_class = MPCLateralController
            required = positive = ("horizon", "timestep", "min_speed")
        case _:
            raise ValueError(f"lateral.selected must be pure_pursuit, stanley or mpc, got {lateral_type!r}")
    if lateral_type not in lateral:
        raise ValueError(f"Missing selected controller block lateral.{lateral_type}")
    lateral_parameters = _parameters(
        lateral[lateral_type], f"lateral.{lateral_type}", required, optional=optional, positive=positive,
    )
    longitudinal_parameters = _parameters(longitudinal, "longitudinal", ("target_speed", "kp", "ki", "kd"))
    longitudinal_controller = PIDSpeedController(**longitudinal_parameters)
    model = load_model(path.parent / config["model_file"])
    track = load_track(path.parent / config["track_file"], yaw_unit=config.get("yaw_unit"))
    if lateral_type == "mpc":
        fallback_parameters = _parameters(
            lateral.get("pure_pursuit"), "lateral.pure_pursuit", PURE_PURSUIT_PARAMETERS,
            positive=PURE_PURSUIT_POSITIVE,
        )
        lateral_parameters["fallback"] = AdaptivePurePursuitController(model, track, **fallback_parameters)
    lateral_controller = lateral_class(model, track, **lateral_parameters)
    controller = compose_controller(
        lateral_controller, longitudinal_controller, model.steering,
        forward_time=partial(setattr, longitudinal_controller, "sim_time"),
    )
    return ControllerServer, controller, endpoint


def main():
    parser = argparse.ArgumentParser(description="Serve external vehicle commands over ZeroMQ.")
    parser.add_argument("--config", type=Path, default=PROJECT / "config/controller.yaml")
    args = parser.parse_args()
    try:
        server_type, controller, endpoint = load_controller(args.config)
        server = server_type(controller, endpoint)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")
    try:
        print(f"External controller listening on {server.endpoint}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
