"""
Twist command publisher.

Default mode: trace a horizontal circle (constant z) by publishing the
tangential velocity at each time step.

  At angle θ, the tangential velocity is:
      vx = -r·ω·sin(θ)
      vy =  r·ω·cos(θ)

  This traces a circle of radius r in the world XY plane at whatever
  height the EE starts at.

Usage:
    uv run python scripts/pub_twist.py                         # circle
    uv run python scripts/pub_twist.py --radius 0.05 --omega 0.3
    uv run python scripts/pub_twist.py --vz 0.05 --mode linear # move up
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import yaml
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from messages.types    import Twist
from messages.protocol import encode_twist


def main() -> None:
    parser = argparse.ArgumentParser(description="Twist command publisher")
    parser.add_argument("--config",   default=str(ROOT / "config/sim_config.yaml"))
    parser.add_argument("--mode",     default="circle", choices=["circle", "linear"],
                        help="circle: trace XY circle | linear: constant twist")
    # Circle options
    parser.add_argument("--radius",   type=float, default=0.08,  help="Circle radius [m]")
    parser.add_argument("--omega",    type=float, default=0.4,   help="Angular speed [rad/s]")
    parser.add_argument("--revs",     type=float, default=2.0,   help="Number of full revolutions")
    # Linear options
    parser.add_argument("--vx",       type=float, default=0.0)
    parser.add_argument("--vy",       type=float, default=0.0)
    parser.add_argument("--vz",       type=float, default=0.05)
    parser.add_argument("--wx",       type=float, default=0.0)
    parser.add_argument("--wy",       type=float, default=0.0)
    parser.add_argument("--wz",       type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=5.0,   help="Duration for linear mode [s]")
    parser.add_argument("--rate-hz",  type=float, default=50.0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(cfg["zmq"]["twist_pub_addr"])

    time.sleep(0.3)  # let subscribers connect

    dt = 1.0 / args.rate_hz
    print(f"[pub_twist] addr : {cfg['zmq']['twist_pub_addr']}  mode: {args.mode}")

    if args.mode == "circle":
        duration = (2 * math.pi * args.revs) / args.omega
        print(f"[pub_twist] circle  r={args.radius}m  ω={args.omega}rad/s  "
              f"{args.revs} rev → {duration:.1f}s")
        theta   = 0.0
        t_start = time.monotonic()
        while time.monotonic() - t_start < duration:
            vx = -args.radius * args.omega * math.sin(theta)
            vy =  args.radius * args.omega * math.cos(theta)
            pub.send_multipart(encode_twist(Twist(twist=[vx, vy, 0.0, 0.0, 0.0, 0.0])))
            theta += args.omega * dt
            time.sleep(dt)

    else:  # linear
        twist = [args.vx, args.vy, args.vz, args.wx, args.wy, args.wz]
        print(f"[pub_twist] linear  twist={twist}  duration={args.duration}s")
        t_start = time.monotonic()
        while time.monotonic() - t_start < args.duration:
            pub.send_multipart(encode_twist(Twist(twist=twist)))
            time.sleep(dt)

    # send zero twist to stop
    zero = Twist(twist=[0.0] * 6)
    for _ in range(20):
        pub.send_multipart(encode_twist(zero))
        time.sleep(dt)

    print("[pub_twist] done")
    pub.close()


if __name__ == "__main__":
    main()
