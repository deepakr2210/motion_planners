"""
Send a desired joint-position command directly to the sim.

The sim must already be running (uv run python -m sim.mujoco_sim).
Connects to the CMD PULL socket (port 5556) and sends MODE_POSITION
commands at 100 Hz until Ctrl-C.

Usage:
    # move to home pose (from sim_config.yaml)
    uv run python scripts/cmd_qpos.py

    # move to a specific pose (7 values for Franka)
    uv run python scripts/cmd_qpos.py 0 -0.785 0 -2.356 0 1.571 0.785

    # teleport without physics (kinematic mode)
    uv run python scripts/cmd_qpos.py --mode kinematic 0 0 0 -1.57 0 1.57 0.785
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from messages.types    import CommandMsg, MODE_POSITION, MODE_KINEMATIC
from messages.protocol import encode_cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct joint-position commander",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "q", nargs="*", type=float,
        help="Joint positions [rad]. Defaults to home_q from config if omitted.",
    )
    parser.add_argument(
        "--mode", choices=["position", "kinematic"], default="position",
        help="'position': physical PD servo (default).  'kinematic': teleport, no dynamics.",
    )
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--config", default=str(ROOT / "config/sim_config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ndof   = cfg["robot"]["ndof"]
    home_q = cfg["robot"]["home_q"]

    if args.q:
        if len(args.q) != ndof:
            print(f"[cmd_qpos] ERROR: need exactly {ndof} values, got {len(args.q)}")
            sys.exit(1)
        q_target = list(args.q)
    else:
        q_target = home_q
        print(f"[cmd_qpos] no q given — using home_q")

    mode = MODE_KINEMATIC if args.mode == "kinematic" else MODE_POSITION

    ctx  = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.connect(cfg["zmq"]["cmd_push_addr"])

    dt  = 1.0 / args.rate_hz
    cmd = CommandMsg(values=q_target, mode=mode)

    fmt = "  ".join(f"{v:+.4f}" for v in q_target)
    print(f"[cmd_qpos] q      : [{fmt}]")
    print(f"[cmd_qpos] mode   : {mode}  @ {args.rate_hz:.0f} Hz")
    print("[cmd_qpos] Ctrl-C to stop")

    try:
        while True:
            push.send(encode_cmd(cmd)[1])
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[cmd_qpos] stopped")
    finally:
        push.close()
        ctx.term()


if __name__ == "__main__":
    main()
