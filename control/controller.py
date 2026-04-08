"""
Real-time controller node.

Reads the control mode from the active trajectory and acts accordingly:

  "torque"    PD + feedforward gravity compensation (default)
                τ = τ_bias + Kp*(q_des - q) + Kd*(qd_des - qd)
              sends CommandMsg(mode="torque", values=torques)

  "position"  Forwards desired joint positions directly to the sim.
              The sim runs its own internal PD servo at full physics rate.
              sends CommandMsg(mode="position", values=q_des)

Subscribes:
  STATE on port 5555  (from sim)
  TRAJ  on port 5557  (from planner)
Pushes:
  CMD   on port 5556  (to sim)

Launch
------
  uv run python -m control.controller
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from messages.types  import StateMsg, TrajectoryMsg, CommandMsg, Waypoint, MODE_TORQUE, MODE_POSITION
from messages.topics import STATE, TRAJ
from messages.protocol import decode_state, decode_traj, encode_cmd


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def interpolate_trajectory(
    traj: TrajectoryMsg,
    wall_now: float,
    ndof: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_des, qd_des) by linear interpolation into the trajectory."""
    t_rel = wall_now - traj.start_time
    wps   = traj.waypoints

    if t_rel <= wps[0].t:
        return np.array(wps[0].q), np.array(wps[0].qd)
    if t_rel >= wps[-1].t:
        return np.array(wps[-1].q), np.array(wps[-1].qd)

    lo, hi = 0, len(wps) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if wps[mid].t <= t_rel:
            lo = mid
        else:
            hi = mid

    alpha  = (t_rel - wps[lo].t) / max(wps[hi].t - wps[lo].t, 1e-9)
    q_des  = (1 - alpha) * np.array(wps[lo].q)  + alpha * np.array(wps[hi].q)
    qd_des = (1 - alpha) * np.array(wps[lo].qd) + alpha * np.array(wps[hi].qd)
    return q_des, qd_des


def main() -> None:
    cfg   = load_config(ROOT / "config/sim_config.yaml")
    ndof  = cfg["robot"]["ndof"]
    Kp    = np.array(cfg["controller"]["kp"])
    Kd    = np.array(cfg["controller"]["kd"])
    rate  = cfg["controller"]["rate_hz"]
    dt    = 1.0 / rate

    ctx = zmq.Context()

    state_sub = ctx.socket(zmq.SUB)
    state_sub.connect(cfg["zmq"]["state_sub_addr"])
    state_sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)
    state_sub.setsockopt(zmq.RCVTIMEO, 0)

    traj_sub = ctx.socket(zmq.SUB)
    traj_sub.connect(cfg["zmq"]["traj_sub_addr"])
    traj_sub.setsockopt(zmq.SUBSCRIBE, TRAJ.bytes)
    traj_sub.setsockopt(zmq.RCVTIMEO, 0)

    cmd_push = ctx.socket(zmq.PUSH)
    cmd_push.connect(cfg["zmq"]["cmd_push_addr"])

    print(f"[ctrl] SUB state {cfg['zmq']['state_sub_addr']}")
    print(f"[ctrl] SUB traj  {cfg['zmq']['traj_sub_addr']}")
    print(f"[ctrl] PUSH cmd  {cfg['zmq']['cmd_push_addr']}")
    print(f"[ctrl] rate={rate}Hz")

    state:  StateMsg     | None = None
    traj:   TrajectoryMsg | None = None
    home_q = np.array(cfg["robot"]["home_q"])

    # Torque limits for clipping in torque mode
    limits = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    print("[ctrl] running — waiting for state...")
    t_next = time.monotonic()

    while True:
        t_next += dt

        # -- Drain state queue --
        try:
            while True:
                topic, raw = state_sub.recv_multipart()
                state = decode_state(raw)
        except zmq.Again:
            pass

        # -- Drain trajectory queue --
        try:
            while True:
                topic, raw = traj_sub.recv_multipart()
                traj = decode_traj(raw)
                print(f"[ctrl] new trajectory: "
                      f"{len(traj.waypoints)} waypoints  mode={traj.mode}")
        except zmq.Again:
            pass

        if state is None:
            sleep = t_next - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            continue

        q  = np.array(state.q)
        qd = np.array(state.qd)

        # -- Desired setpoint from trajectory --
        if traj is not None:
            q_des, qd_des = interpolate_trajectory(traj, time.time(), ndof)
            mode = traj.mode
        else:
            q_des  = home_q
            qd_des = np.zeros(ndof)
            mode   = MODE_TORQUE   # hold home with torque control until a traj arrives

        # -- Build command based on mode --
        if mode == MODE_TORQUE:
            # PD + gravity compensation — compute torques here
            qfrc_bias = np.array(state.qfrc_bias)
            torques   = qfrc_bias + Kp * (q_des - q) + Kd * (qd_des - qd)
            torques   = np.clip(torques, -limits, limits)
            cmd = CommandMsg(values=torques.tolist(), mode=MODE_TORQUE)

        elif mode == MODE_POSITION:
            # Forward desired positions — sim runs its own PD at physics rate
            cmd = CommandMsg(values=q_des.tolist(), mode=MODE_POSITION)

        else:
            # Unknown mode — default to torque hold
            qfrc_bias = np.array(state.qfrc_bias)
            torques   = qfrc_bias + Kp * (home_q - q) + Kd * (0.0 - qd)
            torques   = np.clip(torques, -limits, limits)
            cmd = CommandMsg(values=torques.tolist(), mode=MODE_TORQUE)

        cmd_push.send(encode_cmd(cmd)[1])

        sleep = t_next - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ctrl] stopped")
