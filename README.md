# External controller

Use this project to develop and tune autonomous driving algorithms. The simulator (`autonomous-driving-unimore-sim`) runs the vehicle FMU and sends the vehicle state over ZeroMQ. The controller responds with throttle, brake, and steering commands.

Choose Pure Pursuit, Stanley, or lateral MPC. All three use the same longitudinal PID.

## Install

Open the simulator and controller repositories in separate VS Code workspaces.
Follow the
[simulator installation guide](../autonomous-driving-unimore-sim/README.md#install)
to set up the shared Python environment. Activate it from the controller root.

```sh
source ../autonomous-driving-unimore-sim/.venv/bin/activate
```

MPC also needs CasADi with IPOPT. With the same environment active, run

```sh
python -m pip install casadi
python -c 'import casadi; assert casadi.has_nlpsol("ipopt"), "IPOPT unavailable"'
```

## Configure

In the simulator workspace, edit config/simulation.yaml:
- Set `controller.mode: remote`,

In the controller workspace, edit config/controller.yaml:
- Set model_file to `../assets/model.yaml`.
- Set track_file to the track JSON used by the simulator.
- Match the controller endpoint with the simulator configuration.
- Select the lateral controller and configure its parameters.
- Configure the longitudinal target speed and PID gains.

Vehicle parameters used by the controllers can be independently modified in assets/model.yaml.
The supplied track path points to the simulator's local IMS track. If you select a different track in the simulator, update the controller path explicitly.

Relative paths are resolved from the configuration file's directory.

## Run

Assuming both repositories are located in the same parent directory, run the following commands from the controller workspace:

```sh
source ../autonomous-driving-unimore-sim/.venv/bin/activate
python main.py
```

Launch the simulator and optionally the GUI following the simulator README.
Stop the controller with Ctrl+C and restart it before a new run to reset its projection and PID state.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py`, `controller.py` | Configuration, controller integration, and steering conversion |
| `control_algorithms/` | Pure Pursuit, Stanley, MPC, and PID |
| `utilities/` | Vehicle model loading, trajectory projection, and spline |
| `config/controller.yaml`, `assets/model.yaml` | Controller settings and vehicle parameters |
| `tests/` | Controller regression tests |

## Documentation

- [Controller development](docs/controller_development.md)
- [Architecture and validation](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Simulator logging and plotting](../autonomous-driving-unimore-sim/README.md#plot-csv-logs)

## Test

From the controller root, with the shared environment active, run

```sh
python -B -m unittest discover -s tests -v
```
