"""
Franka Panda — Diff IK node.

Subscribes to STATE (5555) and TWIST (5558).

Strategy: open-loop integration.
  - Snapshot q from the first state message received.
  - Each cycle: q_cmd += J(q_cmd)⁺ · v_twist · dt
  - Push q_cmd as a MODE_POSITION command to the sim.

The twist queue is drained every cycle so v_cmd is always the freshest
sample. The state socket is only used for the initial snapshot.

Run order:
    Terminal 1: uv run python -m sim.mujoco_sim
    Terminal 2: uv run python scripts/test_diff_ik.py
    Terminal 3: uv run python scripts/teleop.py        # interactive
                uv run python tasks/draw_circle.py     # scripted
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

from control.python.diff_ik_control import DiffIKControl
from messages.types          import CommandMsg, MODE_POSITION
from messages.topics         import STATE, TWIST
from messages.protocol       import decode_state, decode_twist, encode_cmd


def main() -> None:
    with open(ROOT / "config/sim_config.yaml") as f:
        cfg = yaml.safe_load(f)

    ik_cfg  = cfg["diff_ik_control"]
    rate_hz = ik_cfg.get("rate_hz", 100)
    rate_dt = 1.0 / rate_hz

    # ── Model for Jacobian computation ────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg["robot"]["robot_path"]))
    data  = mujoco.MjData(model)
    controller = DiffIKControl(
        model, data,
        ee_site = "ee_site",
        dt      = ik_cfg["dt"],
        lam     = 0.1,
    )
    ndof = cfg["robot"]["ndof"]

    # ── ZMQ sockets ───────────────────────────────────────────────────────────
    ctx = zmq.Context()

    state_sub = ctx.socket(zmq.SUB)
    state_sub.connect(cfg["zmq"]["state_sub_addr"])
    state_sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)
    state_sub.setsockopt(zmq.RCVTIMEO, 0)

    twist_sub = ctx.socket(zmq.SUB)
    twist_sub.connect(cfg["zmq"]["twist_sub_addr"])
    twist_sub.setsockopt(zmq.SUBSCRIBE, TWIST.bytes)
    twist_sub.setsockopt(zmq.RCVTIMEO, 0)

    cmd_push = ctx.socket(zmq.PUSH)
    cmd_push.connect(cfg["zmq"]["cmd_push_addr"])

    print(f"[diff_ik] SUB state {cfg['zmq']['state_sub_addr']}")
    print(f"[diff_ik] SUB twist {cfg['zmq']['twist_sub_addr']}")
    print(f"[diff_ik] PUSH cmd  {cfg['zmq']['cmd_push_addr']}")
    print(f"[diff_ik] rate={rate_hz}Hz  dt={ik_cfg['dt']}s")
    print("[diff_ik] waiting for first state snapshot...")

    # ── Wait for the initial state snapshot ───────────────────────────────────
    state_sub.setsockopt(zmq.RCVTIMEO, 5000)   # block up to 5 s for first msg
    q_cmd: np.ndarray | None = None
    while q_cmd is None:
        try:
            _, raw = state_sub.recv_multipart()
            q_cmd = np.array(decode_state(raw).q[:ndof], dtype=float)
            print(f"[diff_ik] snapshot: q={q_cmd.round(3)}")
        except zmq.Again:
            print("[diff_ik] no state received yet, retrying...")
    state_sub.setsockopt(zmq.RCVTIMEO, 0)      # back to non-blocking

    v_cmd  = np.zeros(6)
    t_next = time.monotonic()

    print("[diff_ik] running")
    while True:
        # ── Drain twist queue — always use the freshest v_cmd ─────────────────
        try:
            while True:
                _, raw = twist_sub.recv_multipart()
                v_cmd = np.array(decode_twist(raw).twist)
        except zmq.Again:
            pass

        now = time.monotonic()
        if now < t_next:
            time.sleep(t_next - now)
            continue

        t_next += rate_dt

        # ── Open-loop integration: q_cmd is never re-seeded from state ────────
        q_cmd, dq_cmd = controller.execute(q_cmd, v_cmd)

        cmd = CommandMsg(values=q_cmd.tolist(), mode=MODE_POSITION)
        cmd_push.send(encode_cmd(cmd)[1])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[diff_ik] stopped")
