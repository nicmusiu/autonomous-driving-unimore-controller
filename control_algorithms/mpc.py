from math import isfinite

import numpy as np
from casadi import Function, MX, cos, mtimes, nlpsol, sin, vertcat, vertsplit


def casadi_model(sim, dt):
    # Control
    # Create 1r-1c matrix containing control inputs.
    # Set steer as the only input element
    u = MX.sym("u", 1)
    steer = u[0]

    # State
    x = MX.sym("x", 6)
    sx, sy, yaw, speed_long, speed_lat, yaw_rate = vertsplit(x)

    Fz = sim.mass * 9.81      # Normal force

    alpha_f = steer - (speed_lat + sim.l_f * yaw_rate) / speed_long
    alpha_r = - (speed_lat - sim.l_r * yaw_rate) / speed_long
    Fzf = Fz * sim.l_r / sim.l_wb
    Fzr = Fz * sim.l_f / sim.l_wb
    Fyf = alpha_f * sim.Cf * Fzf
    Fyr = alpha_r * sim.Cr * Fzr

    # ODE right hand side
    sxdot    = speed_long*cos(yaw) - speed_lat*sin(yaw)  # vx
    sydot    = speed_long*sin(yaw) + speed_lat*cos(yaw)  # vy
    yawdot   = yaw_rate
    speeddot_long = 0  # Since it acts just as lateral controller, long control provided by someone else
    speeddot_lat = (Fyr + Fyf * cos(steer)) / sim.mass - speed_long * yaw_rate
    yaw_rate_dot = (Fyf * sim.l_f * cos(steer) - Fyr * sim.l_r) / sim.I_z

    # Concatenate vertically the expressions creating a row vector
    xdot = vertcat(sxdot, sydot, yawdot, speeddot_long, speeddot_lat, yaw_rate_dot)

    # ODE right hand side function
    # as input are used the state and control inputs,
    # and as output the row vector containing the model expressions
    f = Function('f', [x,u],[xdot])

    # Integrate the step with Explicit Euler
    IN = 1
    xj = x
    for _ in range(IN):
        fj = f(xj,u)
        xj += dt*fj/IN

    # Discrete time dynamics function
    return Function('F', [x,u],[xj])

def opt_step(target, state, F, min_steer, max_steer):

    # Control for all segments
    nu = len(target) # Number of steering inputs
    Us = MX.sym("U",nu) #steer control input

    # Initial conditions
    x0 = [state.x, state.y, state.theta, state.vx, state.vy, state.r]
    X0 = MX(x0) # vector containing the initial state

    J = 0 # objective function that should be minimized by the nlp solver

    # build graph
    X=X0

    # For every temporal step
    for k in range(nu):
        X = F(X, vertcat(Us[k]))  # Integrate the step
        gain_mult = 1

        # Give more importance to the last step using a bigger gain
        if(k == nu-1):
            gain_mult=2 # You can use this multiplier as terminal cost

        J += 100.0*gain_mult*(X[0]-target[k][0])**2  # x error cost
        J += 100.0*gain_mult*(X[1]-target[k][1])**2  # y error cost
        J += 10.00*gain_mult*(X[2]-target[k][2])**2  # heading error cost

    # Objective function and constraints
    J += mtimes(Us.T,Us) * (1000.0 + 100 * state.vx)    # Speed-dependent

    # NLP
    nlp = {'x':vertcat(Us), 'f':J}

    # Allocate an NLP solver
    opts = {
    "ipopt.tol": 1e-3,
    "ipopt.acceptable_tol": 1e-2,
    "ipopt.acceptable_iter": 5,
    "ipopt.max_iter": 20,
    "ipopt.print_level": 1,
    "expand": True,
    }

    solver = nlpsol("solver", "ipopt", nlp, opts)
    arg = {}

    # Bounds on u and initial condition
    arg["lbx"] =  vertcat(min_steer*np.ones(nu)) # lower bound for steer
    arg["ubx"] =  vertcat(max_steer*np.ones(nu)) # upper bound for steer
    arg["x0"] =    0.0 # first guess

    # Build and solve a fresh single-shooting NLP on every call.
    res = solver(**arg)
    stats = solver.stats()
    if not stats.get("success"):
        raise RuntimeError(f"IPOPT returned {stats.get('return_status', 'unknown status')}")
    controls = res["x"].full().ravel()
    if not all(isfinite(value) for value in controls):
        raise RuntimeError("IPOPT returned non-finite steering")
    return float(controls[0])
