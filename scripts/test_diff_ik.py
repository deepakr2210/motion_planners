"""
Franka Panda — Diff IK test.

Run order:
    Terminal 1: uv run python -m sim.mujoco_sim
    Terminal 2: uv run python scripts/test_diff_ik.py
    Terminal 3: uv run python scripts/pub_twist.py
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

    # ── Franka Panda setup ────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg["robot"]["robot_path"]))
    data  = mujoco.MjData(model)
    controller = DiffIKControl(model, data, ee_site="ee_site",
                               dt=cfg["diff_ik_control"]["dt"], lam=0.1)

    # ── ZMQ sockets ───────────────────────────────────────────────────────────
    ctx = zmq.Context()

    state_sub = ctx.socket(zmq.SUB)
    state_sub.connect(cfg["zmq"]["state_sub_addr"])
    state_sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)

    twist_sub = ctx.socket(zmq.SUB)
    twist_sub.connect(cfg["zmq"]["twist_sub_addr"])
    twist_sub.setsockopt(zmq.SUBSCRIBE, TWIST.bytes)

    cmd_push = ctx.socket(zmq.PUSH)
    cmd_push.connect(cfg["zmq"]["cmd_push_addr"])

    poller = zmq.Poller()
    poller.register(state_sub, zmq.POLLIN)
    poller.register(twist_sub, zmq.POLLIN)

    # ── Control loop ──────────────────────────────────────────────────────────
    rate_hz = cfg["diff_ik_control"].get("rate_hz", 100)
    rate_dt = 1.0 / rate_hz

    # q_cmd is snapshotted from the first state message, then integrated open-loop.
    q_cmd: np.ndarray | None = None
    v_cmd = np.zeros(6)

    print(f"[test_diff_ik] running at {rate_hz} Hz")

    t_next = time.monotonic()
    while True:
        socks = dict(poller.poll(timeout=0))

        if state_sub in socks:
            _, raw = state_sub.recv_multipart()
            state = decode_state(raw)
            if q_cmd is None:
                q_cmd = np.array(state.q, dtype=float)
                print("[test_diff_ik] initialised q_cmd from state snapshot")

        if twist_sub in socks:
            _, raw = twist_sub.recv_multipart()
            v_cmd = np.array(decode_twist(raw).twist)

        now = time.monotonic()
        if q_cmd is not None and now >= t_next:
            t_next = now + rate_dt
            q_cmd, _ = controller.execute(q_cmd, v_cmd)
            cmd   = CommandMsg(values=q_cmd.tolist(), mode=MODE_POSITION)
            cmd_push.send(encode_cmd(cmd)[1])
        else:
            time.sleep(0.0005)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[test_diff_ik] stopped")
