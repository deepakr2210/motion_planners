"""
run_go2.py – Demo: iLQR on Unitree GO2 (MuJoCo Menagerie model)

Two demo modes (choose via --task flag):
  standing  : stabilise the robot in a neutral standing pose
  velocity  : track a forward velocity command (0.5 m/s)

Usage
-----
  python run_go2.py --task standing
  python run_go2.py --task velocity --horizon 50 --max_iter 30
  python run_go2.py --task standing --render        # closed-loop feedback replay, Ctrl-C to stop
  python run_go2.py --task standing --simulate      # online MPC loop, Ctrl-C to stop

Model
-----
  Uses robots/mujoco_menagerie/unitree_go2/scene.xml
  Override with:  GO2_MODEL=/path/to/scene.xml python run_go2.py ...

Dependencies
------------
  pip install mujoco numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import signal

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ilqr import ILQR, ILQRConfig
from go2_task import (
    GO2_DEFAULT_QPOS,
    GO2_STANDING_HEIGHT,
    StandingTask,
    StandingTaskConfig,
    VelocityTask,
    VelocityTaskConfig,
)

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_MENAGERIE_GO2 = os.path.join(
    _SCRIPT_DIR, "..", "..", "assets",
    "mujoco_menagerie", "unitree_go2", "scene.xml"
)

def get_model_path() -> str:
    path = os.environ.get("GO2_MODEL", _MENAGERIE_GO2)
    path = os.path.realpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"GO2 model not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Initial state from model keyframe
# ---------------------------------------------------------------------------

def make_default_x0(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Return the default standing state from the model's first keyframe.
    The menagerie go2.xml ships a keyframe with height=0.27 and nominal joints.
    """
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_trajectory_stats(
    model: mujoco.MjModel,
    X: np.ndarray,
    U: np.ndarray,
    task_name: str,
) -> None:
    T  = U.shape[0]
    nq = model.nq

    base_z    = X[:, 2]
    joint_pos = X[:, 7:19]
    linvel    = X[:-1, nq:nq+3]
    ctrl_norm = np.linalg.norm(U, axis=1)

    print(f"\n{'='*55}")
    print(f" Trajectory statistics: {task_name}")
    print(f"{'='*55}")
    print(f"  Horizon         : {T} steps  ({T * model.opt.timestep * 1000:.0f} ms)")
    print(f"  Base height     : mean={base_z.mean():.4f}  "
          f"std={base_z.std():.4f}  "
          f"[{base_z.min():.3f}, {base_z.max():.3f}] m")
    print(f"  |linear vel|    : mean={np.linalg.norm(linvel, axis=1).mean():.4f} m/s")
    print(f"  Max joint dev.  : "
          f"{np.abs(joint_pos - GO2_DEFAULT_QPOS).max():.4f} rad")
    print(f"  Ctrl norm       : "
          f"mean={ctrl_norm.mean():.2f}  max={ctrl_norm.max():.2f} Nm")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Viewer helper
# ---------------------------------------------------------------------------

def _get_viewer():
    try:
        import mujoco.viewer as mjv
        return mjv
    except ImportError:
        print("mujoco.viewer unavailable – install mujoco[extras] or upgrade.")
        return None


# ---------------------------------------------------------------------------
# Closed-loop replay  (--render)
# Applies  u = U[t] + k[t] + K[t]·dx  at every physics step.
# Loops until the viewer window is closed OR Ctrl-C is pressed.
# ---------------------------------------------------------------------------

def render_trajectory(
    model: mujoco.MjModel,
    solver: ILQR,
    x0: np.ndarray,
    render_fps: int = 60,
) -> None:
    """
    Closed-loop policy replay.

    Applies  u = U[t] + k[t] + K[t]·dx  each physics step, cycling
    t = 0 → T-1 → 0 → …  WITHOUT ever resetting the physics state.
    External forces applied via the GUI therefore persist and the
    feedback gains try to compensate — the sim never "restarts".

    Viewer is synced at `render_fps` Hz regardless of physics rate.
    """
    mjv = _get_viewer()
    if mjv is None:
        return

    data  = mujoco.MjData(model)
    T     = solver.T
    nq    = model.nq
    dt_s  = model.opt.timestep
    # How many physics steps between viewer syncs to hit render_fps
    sync_every = max(1, int(round(1.0 / (render_fps * dt_s))))

    # Load initial state
    data.qpos[:] = x0[:nq]
    data.qvel[:] = x0[nq:]
    mujoco.mj_forward(model, data)

    print(f"\nClosed-loop replay  (physics {1/dt_s:.0f} Hz, render {render_fps} Hz)"
          f"  –  Ctrl-C or close window to stop")

    t    = 0   # trajectory index, cycles 0..T-1 forever
    step = 0

    try:
        with mjv.launch_passive(model, data) as viewer:
            viewer.sync()
            time.sleep(0.4)   # let the user orient the camera before motion starts

            while viewer.is_running():
                x_curr = np.concatenate([data.qpos.copy(), data.qvel.copy()])

                # iLQR feedback policy
                dx = solver.state_diff(x_curr, solver.X[t])
                u  = solver.U[t] + solver.k[t] + solver.K[t] @ dx
                u  = np.clip(u, -solver.cfg.ctrl_limit, solver.cfg.ctrl_limit)

                data.ctrl[:] = u
                mujoco.mj_step(model, data)

                # Sync viewer at render_fps, not every physics step
                if step % sync_every == 0:
                    viewer.sync()

                t    = (t + 1) % T   # cycle — no physics reset
                step += 1

    except KeyboardInterrupt:
        pass

    print("Viewer closed.")


# ---------------------------------------------------------------------------
# Online MPC loop  (--simulate)
# Re-solves iLQR every `replan_every` steps from the current sim state.
# Runs until the viewer window is closed OR Ctrl-C is pressed.
# ---------------------------------------------------------------------------

def simulate_mpc(
    model: mujoco.MjModel,
    task,
    x0: np.ndarray,
    base_cfg: ILQRConfig,
    replan_every: int = 20,
    mpc_max_iter: int  = 5,
    render_fps:   int  = 60,
) -> None:
    """
    Online MPC: run physics at full speed, replan every `replan_every` steps,
    sync viewer at `render_fps` Hz.  Runs until Ctrl-C or window close.

    The ILQR solver object is reused across replans (warm-start: previous U
    is already set in solver.U so the next solve converges faster).
    """
    mjv = _get_viewer()
    if mjv is None:
        return

    data  = mujoco.MjData(model)
    nq    = model.nq
    dt_s  = model.opt.timestep
    sync_every = max(1, int(round(1.0 / (render_fps * dt_s))))

    mpc_cfg = ILQRConfig(
        horizon    = base_cfg.horizon,
        mu_init    = base_cfg.mu_init,
        mu_min     = base_cfg.mu_min,
        mu_max     = base_cfg.mu_max,
        delta_0    = base_cfg.delta_0,
        alpha_min  = base_cfg.alpha_min,
        tol        = base_cfg.tol,
        max_iter   = mpc_max_iter,
        fd_eps     = base_cfg.fd_eps,
        ctrl_limit = base_cfg.ctrl_limit,
        verbose    = False,
    )

    # Warm-start initial solve
    print(f"\nMPC warm-start  (horizon={mpc_cfg.horizon}, iter={mpc_max_iter})…")
    data.qpos[:] = x0[:nq]
    data.qvel[:] = x0[nq:]
    mujoco.mj_forward(model, data)

    solver_mpc = ILQR(model, task, mpc_cfg)
    solver_mpc.solve(x0)

    print(f"MPC running  (physics {1/dt_s:.0f} Hz, render {render_fps} Hz, "
          f"replan every {replan_every} steps)  –  Ctrl-C or close window to stop\n")

    step = 0
    try:
        with mjv.launch_passive(model, data) as viewer:
            viewer.sync()
            time.sleep(0.3)

            while viewer.is_running():
                x_curr = np.concatenate([data.qpos.copy(), data.qvel.copy()])

                # Replan: warm-start reuses solver.U from previous plan
                if step % replan_every == 0:
                    t0 = time.perf_counter()
                    solver_mpc.solve(x_curr)
                    dt_plan = (time.perf_counter() - t0) * 1e3
                    print(f"  step={step:6d}  z={x_curr[2]:.3f}m  "
                          f"vx={x_curr[nq]:.3f}m/s  plan={dt_plan:.0f}ms",
                          end="\r", flush=True)

                # Policy: first-step feedforward + feedback
                dx = solver_mpc.state_diff(x_curr, solver_mpc.X[0])
                u  = (solver_mpc.U[0]
                      + solver_mpc.k[0]
                      + solver_mpc.K[0] @ dx)
                u  = np.clip(u, -base_cfg.ctrl_limit, base_cfg.ctrl_limit)

                data.ctrl[:] = u
                mujoco.mj_step(model, data)

                if step % sync_every == 0:
                    viewer.sync()

                step += 1

    except KeyboardInterrupt:
        pass

    print("\nMPC viewer closed.")


# ---------------------------------------------------------------------------
# Standing demo
# ---------------------------------------------------------------------------

def run_standing(args: argparse.Namespace) -> None:
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    print(f"\nModel : {model_path}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  "
          f"nx={model.nq+model.nv}  ndx={2*model.nv+model.na}")

    x0 = make_default_x0(model, data)
    print(f"  Keyframe z = {x0[2]:.4f} m")

    task_cfg = StandingTaskConfig(
        w_height=200.0, w_orient=100.0, w_linvel=10.0, w_angvel=10.0,
        w_joint_pos=5.0, w_joint_vel=1.0, w_ctrl=0.001, w_terminal=10.0,
    )
    task = StandingTask(model, task_cfg)

    cfg = ILQRConfig(
        horizon=args.horizon, mu_init=1.0, mu_min=1e-6, mu_max=1e8,
        delta_0=2.0, alpha_min=1e-8, tol=1e-5,
        max_iter=args.max_iter, fd_eps=1e-6, ctrl_limit=33.5, verbose=True,
    )

    solver = ILQR(model, task, cfg)

    print("\n── iLQR Optimisation (Standing) ──")
    U_opt, X_opt, cost = solver.solve(x0)
    print_trajectory_stats(model, X_opt, U_opt, "Standing")

    if args.render:
        render_trajectory(model, solver, x0)
    if args.simulate:
        simulate_mpc(model, task, x0, cfg,
                     replan_every=args.replan_every,
                     mpc_max_iter=args.mpc_iter)


# ---------------------------------------------------------------------------
# Velocity tracking demo
# ---------------------------------------------------------------------------

def run_velocity(args: argparse.Namespace) -> None:
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    print(f"\nModel : {model_path}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}")

    x0 = make_default_x0(model, data)

    task_cfg = VelocityTaskConfig(
        target_vx=args.target_vx, target_vy=0.0, target_yaw=0.0,
        w_height=150.0, w_orient=50.0, w_vel_xy=100.0, w_yaw=50.0,
        w_vz=20.0, w_angvel_xy=5.0, w_joint_pos=2.0,
        w_joint_vel=0.5, w_ctrl=0.001, w_terminal=5.0,
    )
    task = VelocityTask(model, task_cfg)

    cfg = ILQRConfig(
        horizon=args.horizon, mu_init=1.0, mu_min=1e-6, mu_max=1e8,
        delta_0=2.0, alpha_min=1e-8, tol=1e-5,
        max_iter=args.max_iter, fd_eps=1e-6, ctrl_limit=33.5, verbose=True,
    )

    solver = ILQR(model, task, cfg)

    print(f"\n── iLQR Optimisation (Velocity  vx={task_cfg.target_vx} m/s) ──")
    U_opt, X_opt, cost = solver.solve(x0)
    print_trajectory_stats(model, X_opt, U_opt, "Velocity")

    nq = model.nq
    print(f"  Target vx={task_cfg.target_vx:.2f}  achieved={X_opt[-1, nq]:.4f} m/s\n")

    if args.render:
        render_trajectory(model, solver, x0)
    if args.simulate:
        simulate_mpc(model, task, x0, cfg,
                     replan_every=args.replan_every,
                     mpc_max_iter=args.mpc_iter)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GO2 iLQR demo (MuJoCo Menagerie model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",      choices=["standing", "velocity"],
                        default="standing", help="Task to solve")
    parser.add_argument("--horizon",   type=int,   default=50,
                        help="Planning horizon (steps)")
    parser.add_argument("--max_iter",  type=int,   default=30,
                        help="iLQR max iterations (initial solve)")
    parser.add_argument("--render",    action="store_true",
                        help="Replay optimised policy in viewer (loops until Ctrl-C)")
    parser.add_argument("--simulate",  action="store_true",
                        help="Online MPC: live simulation with replanning (loops until Ctrl-C)")
    parser.add_argument("--replan_every", type=int, default=20,
                        help="[--simulate] replan every N physics steps")
    parser.add_argument("--mpc_iter",  type=int,   default=5,
                        help="[--simulate] iLQR iterations per MPC replan")
    parser.add_argument("--target_vx", type=float, default=0.5,
                        help="[velocity task] target forward velocity (m/s)")
    args = parser.parse_args()

    if args.task == "standing":
        run_standing(args)
    else:
        run_velocity(args)


if __name__ == "__main__":
    main()
