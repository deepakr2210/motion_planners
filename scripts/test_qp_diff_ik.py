"""
Franka Panda — QP Diff IK circle-drawing test.

Generates a horizontal (XY-plane) circle trajectory internally.
No external twist publisher needed.

Run order:
    Terminal 1: uv run python -m sim.mujoco_sim
    Terminal 2: uv run python scripts/test_qp_diff_ik.py

Options (edit constants below):
    RADIUS      circle radius [m]
    PERIOD      one full revolution [s]
    ALPHA       task-error gain
    LAM         damping coefficient λ
    W_POS       position-axis weights (x, y, z)
    W_ROT       rotation-axis weights (rx, ry, rz)
    SOFT_NULL   whether to add a soft nullspace (home-posture) constraint
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.python.qp_diff_ik_control import QPDiffIKControl
from messages.types    import CommandMsg, MODE_POSITION
from messages.topics   import STATE
from messages.protocol import decode_state, encode_cmd

# ── Trajectory parameters ─────────────────────────────────────────────────────

RADIUS    = 0.08          # circle radius [m]
PERIOD    = 8.0           # seconds per revolution
ALPHA     = 1.0           # task-error gain (α)
LAM       = 0.01          # damping λ
W_POS     = (10., 10., 5.)   # task-space position weights
W_ROT     = (0.5, 0.5, 0.5) # task-space rotation weights
SOFT_NULL = True          # add soft nullspace constraint towards home posture

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rot_err(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Rotation error as a 3-vector (axis-angle, small-angle approximation)."""
    R_e = R_des @ R_cur.T
    return 0.5 * np.array([R_e[2, 1] - R_e[1, 2],
                            R_e[0, 2] - R_e[2, 0],
                            R_e[1, 0] - R_e[0, 1]])


def _circle_target(
    center:   np.ndarray,
    radius:   float,
    period:   float,
    t:        float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (target_pos, target_vel) for a horizontal circle in the XY plane.
    """
    omega = 2 * np.pi / period
    pos = center.copy()
    pos[0] += radius * np.cos(omega * t)
    pos[1] += radius * np.sin(omega * t)
    vel = np.array([-radius * omega * np.sin(omega * t),
                     radius * omega * np.cos(omega * t),
                     0.0])
    return pos, vel


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(ROOT / "config/sim_config.yaml") as f:
        cfg = yaml.safe_load(f)

    robot_cfg   = cfg["robot"]
    ik_cfg      = cfg["diff_ik_control"]
    zmq_cfg     = cfg["zmq"]

    # ── Model & controller ────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(ROOT / robot_cfg["robot_path"]))
    data  = mujoco.MjData(model)
    ndof  = robot_cfg["ndof"]

    W = np.diag([*W_POS, *W_ROT])

    ctrl = QPDiffIKControl(
        model, data,
        ee_site = "ee_site",
        dt      = ik_cfg["dt"],
        W       = W,
        lam     = LAM,
        alpha   = ALPHA,
        solver  = "osqp",
    )

    # hard: per-joint velocity limits (Franka hardware limits, conservative)
    v_max = np.array([2.1, 2.1, 2.1, 2.1, 2.6, 2.6, 2.6])
    ctrl.add_hard_velocity_limits(-v_max, v_max)

    # hard: joint position limits from model
    q_lo = model.jnt_range[:ndof, 0].copy()
    q_hi = model.jnt_range[:ndof, 1].copy()
    ctrl.add_hard_position_limits(q_lo, q_hi)

    # soft: stay near home posture in the nullspace
    if SOFT_NULL:
        q_home = np.array(robot_cfg["home_q"], dtype=float)
        margin = 0.3  # [rad] — soft band around home
        ctrl.add_soft_constraint(
            A      = np.eye(ndof),
            lb     = q_home - margin,
            ub     = q_home + margin,
            weight = 2.0,
        )

    # ── ZMQ ──────────────────────────────────────────────────────────────────
    ctx = zmq.Context()

    state_sub = ctx.socket(zmq.SUB)
    state_sub.connect(zmq_cfg["state_sub_addr"])
    state_sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)

    cmd_push = ctx.socket(zmq.PUSH)
    cmd_push.connect(zmq_cfg["cmd_push_addr"])

    poller = zmq.Poller()
    poller.register(state_sub, zmq.POLLIN)

    # ── Control loop ─────────────────────────────────────────────────────────
    rate_hz = cfg["controller"]["rate_hz"]
    rate_dt = 1.0 / rate_hz

    ee_id   = ctrl.ee_id
    q_cmd:          np.ndarray | None = None
    circle_center:  np.ndarray | None = None
    R_des:          np.ndarray | None = None
    t_start:        float | None      = None

    print(f"[test_qp_diff_ik] waiting for first state message …")

    t_next = time.monotonic()
    while True:
        socks = dict(poller.poll(timeout=0))

        if state_sub in socks:
            _, raw = state_sub.recv_multipart()
            state  = decode_state(raw)

            if q_cmd is None:
                q_cmd = np.array(state.q[:ndof], dtype=float)

                # snapshot EE pose as circle centre & desired orientation
                data.qpos[:ndof] = q_cmd
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)

                circle_center = data.site_xpos[ee_id].copy()
                R_des         = data.site_xmat[ee_id].reshape(3, 3).copy()
                t_start       = time.monotonic()

                print(f"[test_qp_diff_ik] centre = {circle_center.round(3)}, "
                      f"r = {RADIUS} m, T = {PERIOD} s")
                print(f"[test_qp_diff_ik] running at {rate_hz} Hz  "
                      f"(solver = {ctrl.solver})")

        now = time.monotonic()
        if q_cmd is not None and now >= t_next:
            t_next = now + rate_dt
            t_rel  = now - t_start

            # ── forward kinematics at q_cmd (for error computation) ───────
            data.qpos[:ndof] = q_cmd
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            p_cur = data.site_xpos[ee_id].copy()
            R_cur = data.site_xmat[ee_id].reshape(3, 3).copy()

            # ── desired pose on circle ────────────────────────────────────
            p_des, v_ff = _circle_target(circle_center, RADIUS, PERIOD, t_rel)

            # ── task error  e = [pos_err; rot_err]  (sign: e = current − desired)
            # The QP drives  J q̇ → −α e, i.e. towards the target.
            # We fold the feedforward velocity in by subtracting v_ff/α so that
            # −α e = α(p_des − p_cur) + v_ff.
            pos_err = (p_cur - p_des) - v_ff / ALPHA
            rot_err = _rot_err(R_cur, R_des)
            e       = np.concatenate([pos_err, rot_err])

            # ── solve QP ─────────────────────────────────────────────────
            q_cmd, _ = ctrl.execute(q_cmd, e)

            # ── send command ─────────────────────────────────────────────
            cmd = CommandMsg(values=q_cmd.tolist(), mode=MODE_POSITION)
            cmd_push.send(encode_cmd(cmd)[1])
        else:
            time.sleep(0.0005)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[test_qp_diff_ik] stopped")
