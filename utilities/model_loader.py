from dataclasses import fields
from math import isclose, isfinite
from types import SimpleNamespace

from models.model import MassProperties, ReferencePoints, SteeringParameters, VehicleGeometry
from yaml_config import read_yaml


def _number(value, label, *, positive=False):
    try:
        valid = type(value) in (int, float) and isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        restriction = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {restriction}")
    return float(value)


def _section(data, name):
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    return section


def _numbers(data, name, schema, *, signed=()):
    section = _section(data, name)
    values = {}
    for field in fields(schema):
        key = field.name
        values[key] = _number(section.get(key), f"{name}.{key}", positive=key not in signed)
    return schema(**values)


def load_model(path):
    """Read the controller's calibration and extensible tire metadata."""
    try:
        data = read_yaml(path)
        if not isinstance(data, dict):
            raise ValueError("model must be a mapping")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model.name must be a nonempty string")
        geometry = _numbers(data, "geometry", VehicleGeometry)
        mass = _numbers(data, "mass_properties", MassProperties)
        if mass.chassis_mass > mass.mass:
            raise ValueError("mass_properties.chassis_mass must not exceed total mass")
        reference = _section(data, "reference_points")
        offsets = {}
        for field in fields(ReferencePoints):
            label = f"reference_points.{field.name}"
            point = reference.get(field.name)
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"{label} must contain exactly three coordinates")
            offsets[field.name] = tuple(_number(value, f"{label}[{i}]") for i, value in enumerate(point))
        points = ReferencePoints(**offsets)
        front, rear, cog = (point[0] for point in (points.front_axle, points.rear_axle, points.center_of_mass))
        for expected, actual in ((geometry.wheelbase, geometry.lf + geometry.lr),
                                 (geometry.wheelbase, front - rear),
                                 (geometry.lf, front - cog), (geometry.lr, cog - rear)):
            if not isclose(expected, actual, rel_tol=0, abs_tol=0.001):
                raise ValueError("Inconsistent axle geometry and reference points (tolerance 1 mm)")
        steering = _numbers(data, "steering", SteeringParameters, signed=("command_min", "command_max"))
        if steering.command_min >= steering.command_max or not steering.command_min <= 0 <= steering.command_max:
            raise ValueError("steering command limits must be increasing and include zero")
        tires = dict(_section(data, "tires"))
        tires["unloaded_radius"] = _number(tires.get("unloaded_radius"), "tires.unloaded_radius", positive=True)
        for key in ("Cf", "Cr"):
            value = tires.get(key)
            tires[key] = None if value is None else _number(value, f"tires.{key}", positive=True)
        return SimpleNamespace(name=name, geometry=geometry, mass_properties=mass,
                               reference_points=points, steering=steering, tires=SimpleNamespace(**tires))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid controller model {path}: {error}") from error
