# External controller

Use this project to develop and tune driving algorithms. The simulator runs the vehicle FMU and sends the vehicle state over ZeroMQ. The controller responds with throttle, brake, and steering commands.

## 1. Open two VS Code workspaces

Open each folder in a separate VS Code window.

- Simulator workspace — `~/Documents/unimore_autonomous_driving/autonomous-driving-unimore-sim`
- Controller workspace — `~/Documents/unimore_autonomous_driving/autonomous-driving-unimore-control`

Both projects use the simulator's Python environment. If it needs setup, run these commands once. Python 3.12 or newer is required. The extras install the optional dashboard and plotter.

```sh
cd ~/Documents/unimore_autonomous_driving/autonomous-driving-unimore-sim
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gui,plot]"
```

## 2. Configure each workspace

In the **simulator workspace**, edit `config/simulation.yaml`.

1. Set `simulation.fmu_path` to the compatible FMI 2.0 Co-Simulation FMU. Set `model.model_file` and `scenario.track_file` to its vehicle model YAML and track JSON. Set `scenario.yaw_unit` to match the track if needed.
2. Set `initial_pose.reference_index` to a zero-based track point to start from, or use explicit `x`, `y`, and `yaw` in radians.
3. Set `simulation.duration` and `controller.period` in seconds. Both must be integer multiples of `simulation.fmu_step` (default 0.001 s). Set `simulation.real_time` to `true` for wall-clock pacing or `false` for an unpaced run.
4. Select `controller.mode: remote`. Set `controller.endpoint` to the same address as the external controller, such as `tcp://127.0.0.1:5555`. Enable `telemetry.enabled` to use the GUI.

In the **controller workspace**, edit `config/controller.yaml`.

1. Keep the controller model and reference trajectory in `assets/`. The default paths in `config/controller.yaml` are `../assets/model.yaml` and `../assets/track.json`. Paths resolve from the configuration file even when the controller starts from another directory. Supply your own authorized local track JSON at that path and keep private track files out of version control. Use a reference and yaw units compatible with the simulator road, and match the two ZeroMQ controller endpoints.
2. Edit the controller's model independently of the simulator's plant model. The controller's loader reads vehicle geometry, steering calibration, mass, inertia, CoG position, and tire data from its own model. MPC uses the controller-only `Kf` and `Kr` coefficients. The installed simulator package supplies shared interfaces and track loading. Its model loader and plant model require no prediction metadata.
3. Set `lateral.selected` to `pure_pursuit`, `stanley`, or `mpc`. The launcher reads the selected block at startup. MPC also reads the Pure Pursuit block for fallback. Set the PID target speed in m/s under `longitudinal.target_speed`.

Keep controller code and tuning in the controller workspace. You can edit the simulator YAML to set up your run without editing simulator source code.

### Stanley selection

Set `lateral.selected: stanley` and tune the settings under `lateral.stanley`. For a first Stanley run, set `longitudinal.target_speed: 10.0`.

Stanley uses the model's CoG-to-front-axle distance to project the front axle onto the spline. It tracks the pose at that projection without lookahead. The two weights scale heading and cross-track corrections independently. Cross-track correction uses a speed floor of 1 m/s. The shared steering bridge applies the vehicle's steering-wheel limits.

### Lateral MPC

With the shared environment active, install optional CasADi and verify its IPOPT plugin.

```sh
python -m pip install casadi
python -c 'import casadi; assert casadi.has_nlpsol("ipopt"), "IPOPT unavailable"'
```

Set `lateral.selected: mpc` and start with `longitudinal.target_speed: 10.0`. The existing `lateral.mpc` settings use a 1 s horizon, 0.05 s prediction timestep, and 8 m/s minimum activation speed. The horizon must be an integer multiple of the timestep. The controller's `assets/model.yaml` supplies the positive linear coefficients `tires.Kf` and `tires.Kr`. The provisional values of 15 1/rad scale slip angle and normal load to produce tire force. These coefficients are calibrated independently of the optional Pacejka coefficients, which the current controllers do not use. MPC derives its road-wheel bounds from this same file's steering-wheel limits and ratio.

The predictor uses the CoG pose and shifts chassis-origin velocity to the CoG using yaw rate and the model offset. In the FMU validation, that shift reduced the RMS lateral-velocity discrepancy against CoG position derivatives from 0.0312 to 0.0008 m/s. The forward position derivative was about 5% lower than the reported speed during steady motion. The controller uses the exported velocity without rescaling. Reference sampling starts one Euler step ahead and unwraps yaw relative to the vehicle. The 3 m/s provisional threshold gives an unstable straight-line Euler linearization for this model. The 8 m/s setting provides margin above the approximately 6.82 m/s stability boundary. The startup check rejects unstable combinations of activation speed and timestep at the straight-line operating point.

Pure Pursuit steers while the vehicle is below the activation speed, or when MPC has invalid feedback, an unsuccessful solve, or non-finite steering. PID remains responsible for the pedals. The controller keeps separate fallback counters and logs the first solver failure and every 100th thereafter. IPOPT is constructed anew at each update, so plan initial runs in unpaced mode. The FMU comparison measured occasional calls longer than the 20 ms control period.

### Longitudinal PID tuning

Set `target_speed`, `kp`, `ki`, and `kd` directly under `longitudinal`. Set `ki: 0` and `kd: 0` for pure proportional action through the same PID.

For PID, `kp` acts on speed error in m/s, `ki` on accumulated error in metres, and `kd` on the rate of change of error in m/s². Start with `ki` and `kd` at zero. Increase `kp` for response, add `ki` to reduce sustained error, and add a small `kd` if damping is needed. Derivative action is unfiltered and can amplify noise. Inspect the speed and both pedal traces before reusing gains with another speed, vehicle, or control period.

The first valid call uses proportional action. Later calls derive elapsed time from consecutive simulation timestamps. Positive demand commands throttle and negative demand commands brake. Demand is bounded to `[-1, 1]` and the integral grows only within available actuator headroom. Invalid speed or non-increasing time presented to the PID resets it and produces zero pedals. Restart the controller between runs to reset its state.

## 3. Run the controller, GUI, and simulator

The controller and simulator can start in either order. If the simulator starts first, it waits at simulation time zero for the first valid command. The `controller.startup_timeout` setting in the simulator YAML defaults to 30 wall-clock seconds.

**Controller workspace · Controller terminal**

```sh
cd ~/Documents/unimore_autonomous_driving/autonomous-driving-unimore-control
source ../autonomous-driving-unimore-sim/.venv/bin/activate
python main.py --config config/controller.yaml
```

The controller prints `External controller listening on ...` when ready.

**Simulator workspace · Optional GUI terminal**

```sh
cd ~/Documents/unimore_autonomous_driving/autonomous-driving-unimore-sim
source .venv/bin/activate
python run_gui.py --config config/simulation.yaml
```

The GUI can display the initial state while the simulator is waiting.

**Simulator workspace · Simulation terminal**

```sh
cd ~/Documents/unimore_autonomous_driving/autonomous-driving-unimore-sim
source .venv/bin/activate
python run_simulator.py --config config/simulation.yaml
```

The simulator exits at the configured duration. For an early stop, press Ctrl+C in the simulation terminal. Stop the controller with Ctrl+C in its terminal and restart it before the next run to reset path-tracking state. You can leave the GUI open.

## 4. Results

The default output is `simulation_results.csv` in the simulator workspace. Set `simulation.csv` in the simulator YAML to choose a different output. With the environment active, plot the default output from the simulator workspace.

```sh
python plotter.py simulation_results.csv
```

## 5. Controller development

1. Add your lateral controller under `control_algorithms/`. See the instructor's steering laws there and the simulator-facing integrations in `controller.py`.
2. Implement `compute(self, sim_time, vehicle_state)`. Simulation time is in seconds. Feedback includes `vx` and `vy` in m/s, `yaw` in radians, `yaw_rate` in rad/s, and global XY positions in metres for the chassis, rear axle, and centre of mass. The state also carries gear, engine speed, and wheel speeds. Check that the fields your algorithm needs are available and finite.
3. Return a **road-wheel angle in radians**. Positive steering turns left. Return `None` if there is no valid steering command. The shared bridge in `controller.py` checks the timestamp and speed, applies the model's steering ratio, and clamps to its steering-wheel limits. The simulator's `CombinedController` supplies normalized throttle and brake in `[0, 1]` for valid steering, including a valid zero angle. On invalid steering it sends a zero command and skips the longitudinal calculation.
4. Add your controller to `load_controller()` in the controller project's `main.py`. Add its named parameter block under `lateral`, set `lateral.selected` to that name, and restart the controller after changes.

The simulator's composition calls the PID's `compute(vx)`, which returns `(throttle, brake)`. A hook in `controller.py` forwards simulation time. A future integrated controller can return `VehicleCommand` directly to `ControllerServer`. Run the focused tests from this workspace with the shared environment active.

```sh
python -B -m unittest discover -s tests -v
```

## 6. Troubleshooting

| Symptom | Check |
| --- | --- |
| Asset file missing or rejected | Confirm the path exists under the controller's `assets/` directory and resolves from the YAML's directory. Check that the reference track is compatible with the simulator road and uses the configured yaw unit. |
| `ModuleNotFoundError` or GUI import error | Activate the simulator's `.venv` in that terminal. Run `python -m pip show unimore-fmu-vehicle-simulation` to verify the installation. Include the `gui` and `plot` extras for those tools. |
| Startup timeout or address already in use | Check that the controller started within the startup timeout and that both controller endpoints match. If the port is busy, stop the old controller or choose a free port in both files. The GUI uses a separate telemetry endpoint. |
| Simulator loads the wrong assets or runs locally | Run from the simulator workspace and check that the launch command includes `--config config/simulation.yaml`. Confirm `controller.mode: remote` in that file. |

## Technical notes

The course spline in `utilities/cubic_spline_planner.py` retains Atsushi Sakai's attribution. Its dense fit can be slow on large tracks and can overshoot between sparse waypoints. Its curvature formula assumes approximately unit parameter speed.

A repeated endpoint marks a closed track. The controller wraps position and yaw queries together and clamps open-track queries just below the final knot, where the spline derivative routine can fail. The spline is nonperiodic, so heading and curvature can still jump at the seam. Closed-track projection searches forward and can pick the wrong branch near crossings or after off-track motion.

`utilities/trajectory.py` shares the spline queries. Each controller has its own projection state. Pure Pursuit projects the measured rear axle and then selects a lookahead target. Stanley projects the front axle reconstructed from CoG and yaw. In the 10 m/s IMS comparison, Stanley reduced lateral error and produced faster steering changes than Pure Pursuit.

### Original course algorithms

`control_algorithms/purepursuit.py` and `control_algorithms/stanley.py` are the instructor's original files. The lateral classes in `controller.py` prepare vehicle feedback and targets, then call those laws directly. The steering bridge in the same file handles conversion, limits, and command composition for both algorithms.

Adaptive Pure Pursuit passes the rear-axle-to-target displacement in world axes, the world-frame heading, and the Euclidean distance to `PurePursuitController.compute_steering_angle(target, actual_heading, Lf)`. Stanley passes the CoG pose, the spline pose projected from the reconstructed front axle, and speed to the original law. The YAML key `cross_track_error_weight` is passed to the original Stanley constructor as `cross_track_error_weigth`. Both integrations supply an infinite symmetric road-wheel limit and use the shared steering-wheel limits.

### Runtime PID

The external project uses its validated pedal PID. It handles the first sample, checks finite values, and limits integral growth against the full P+I+D demand. Its gains and limits use normalized pedal units.

`control_algorithms/pid.py` contains the shared `PIDController` and the speed adapter. Call `PIDController.compute(setpoint, measured_value, dt)` with elapsed time in seconds. The first call uses proportional action and can pass `dt=None`. Later calls require a positive, finite interval. Each side of `output_limits` can be a finite bound or `None`. Invalid samples reset the controller and return zero clipped to those limits. Use `reset()` to clear history explicitly.

The speed adapter supplies the interval from simulation timestamps and maps the shared PID's demand in `[-1, 1]` to throttle or brake. The shared PID limits integral growth against the complete P+I+D demand and supports zero integral gain.
