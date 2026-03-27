"""
Python trajectory planner node.

Algorithm: cubic polynomial interpolation in joint space.

  q(t)  = a0 + a1*t + a2*t^2 + a3*t^3
  qd(t) = a1 + 2*a2*t + 3*a3*t^2

Boundary conditions (zero velocity at start and end):
  q(0)=q0,  qd(0)=0
  q(T)=qf,  qd(T)=0

  → a0=q0, a1=0,
    a2 =  3/T^2 * (qf-q0)
    a3 = -2/T^3 * (qf-q0)

Publishes: TrajectoryMsg  on ZMQ port 5557
Subscribes: StateMsg      on ZMQ port 5555

Launch
------
  uv run python -m planners.python.planner
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from messages.types   import StateMsg, TrajectoryMsg, Waypoint, MODE_TORQUE
from messages.topics  import STATE, TRAJ
from messages.protocol import decode_state, encode_traj


# ── Predefined goal configurations (home → goal1 → goal2 → ...) ──────────

GOALS = [
    np.array([ 0.4,  -0.3,  0.2, -1.8,  0.3,  2.0,  1.0]),   # reach out
    np.array([-0.4,  -0.3, -0.2, -1.8, -0.3,  2.0, -1.0]),   # other side
    np.array([ 0.0,  -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), # home
]


def cubic_trajectory(
    q0: np.ndarray,
    qf: np.ndarray,
    duration: float,
    dt: float,
) -> list[Waypoint]:
    """Generate a cubic-spline trajectory in joint space."""
    T   = max(duration, 0.1)
    dq  = qf - q0
    a2  =  3.0 / T**2 * dq
    a3  = -2.0 / T**3 * dq

    waypoints = []
    t = 0.0
    while t <= T + 1e-9:
        q  = q0 + a2 * t**2 + a3 * t**3
        qd = 2.0 * a2 * t + 3.0 * a3 * t**2
        waypoints.append(Waypoint(t=float(t),
                                   q=q.tolist(),
                                   qd=qd.tolist()))
        t += dt

    return waypoints


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg  = load_config(ROOT / "config/sim_config.yaml")
    ndof = cfg["robot"]["ndof"]
    dt   = cfg["planner"]["trajectory_dt"]
    dur  = cfg["planner"]["default_duration"]

    ctx = zmq.Context()

    # Subscribe to state from sim
    sub = ctx.socket(zmq.SUB)
    sub.connect(cfg["zmq"]["state_sub_addr"])
    sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)
    sub.setsockopt(zmq.RCVTIMEO, 2000)

    # Publish trajectory for controller
    pub = ctx.socket(zmq.PUB)
    pub.bind(cfg["zmq"]["traj_pub_addr"])

    print(f"[planner-py] SUB {cfg['zmq']['state_sub_addr']}")
    print(f"[planner-py] PUB {cfg['zmq']['traj_pub_addr']}")
    print("[planner-py] waiting for first state...")

    # ── Wait for first state ──────────────────────────────────────────────
    while True:
        try:
            topic, raw = sub.recv_multipart()
            state = decode_state(raw)
            break
        except zmq.Again:
            print("[planner-py] no state yet, retrying...")

    print(f"[planner-py] got state: q={np.round(state.q, 3)}")

    goal_idx  = 0
    q_current = np.array(state.q)

    while True:
        # ── Plan trajectory to next goal ─────────────────────────────────
        q_goal = GOALS[goal_idx % len(GOALS)]
        goal_idx += 1

        print(f"[planner-py] planning to goal {goal_idx}: "
              f"{np.round(q_goal, 3)}")

        waypoints  = cubic_trajectory(q_current, q_goal, dur, dt)
        start_time = time.time()
        traj_msg   = TrajectoryMsg(waypoints=waypoints, start_time=start_time)
        pub.send_multipart(encode_traj(traj_msg))

        print(f"[planner-py] trajectory published: "
              f"{len(waypoints)} waypoints over {dur:.1f}s")

        # ── Wait for the trajectory to finish, then get current state ─────
        time.sleep(dur + 0.5)

        # Update q_current from latest state
        try:
            topic, raw = sub.recv_multipart()
            state = decode_state(raw)
            q_current = np.array(state.q)
        except zmq.Again:
            q_current = q_goal   # fallback: assume we reached the goal

        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[planner-py] stopped")
