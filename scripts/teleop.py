"""
Keyboard teleop — publishes EE twist to port 5558.

Key bindings (velocity is latched until Space or Ctrl-C):
  ↑ / ↓    vz  +/-   (EE up / down)
  ← / →    vy  -/+   (EE left / right)
  W / S    vx  +/-   (EE fwd / back)
  Q / E    wz  +/-   (yaw CCW / CW)
  A / D    wy  -/+   (pitch up / down)
  Z / X    wx  +/-   (roll L / R)
  Space    stop (zero all)
  ] / [    speed × 2 / ÷ 2
  Ctrl-C   quit

Run:
    uv run python scripts/teleop.py
"""

from __future__ import annotations

import curses
import sys
import time
from pathlib import Path

import yaml
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from messages.types    import Twist
from messages.protocol import encode_twist

_HELP = """\
=== Franka EE Teleop ===

  ↑ / ↓    vz  +/-   (up / down)
  ← / →    vy  -/+   (left / right)
  W / S    vx  +/-   (fwd / back)
  Q / E    wz  +/-   (yaw CCW / CW)
  A / D    wy  -/+   (pitch up / dn)
  Z / X    wx  +/-   (roll L / R)

  Space    stop
  ] / [    speed × 2 / ÷ 2
  Ctrl-C   quit
"""


def _run(stdscr: "curses._CursesWindow", cfg: dict) -> None:
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(cfg["zmq"]["twist_pub_addr"])
    time.sleep(0.2)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    speed = 0.05          # m/s (linear) and rad/s (angular)
    twist = [0.0] * 6    # [vx, vy, vz, wx, wy, wz]
    rate_hz = 50.0
    dt      = 1.0 / rate_hz

    help_lines = _HELP.splitlines()

    def draw() -> None:
        stdscr.erase()
        for i, line in enumerate(help_lines):
            try:
                stdscr.addstr(i, 0, line)
            except curses.error:
                pass
        row = len(help_lines) + 1
        try:
            stdscr.addstr(row,     0, f"  speed : {speed:.4f}  m/s | rad/s")
            stdscr.addstr(row + 1, 0,
                f"  twist : vx={twist[0]:+.3f}  vy={twist[1]:+.3f}  vz={twist[2]:+.3f}"
                f"  wx={twist[3]:+.3f}  wy={twist[4]:+.3f}  wz={twist[5]:+.3f}"
            )
        except curses.error:
            pass
        stdscr.refresh()

    draw()

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:         twist = [0, 0, +speed, 0, 0, 0]
        elif key == curses.KEY_DOWN:     twist = [0, 0, -speed, 0, 0, 0]
        elif key == curses.KEY_LEFT:     twist = [0, -speed, 0, 0, 0, 0]
        elif key == curses.KEY_RIGHT:    twist = [0, +speed, 0, 0, 0, 0]
        elif key == ord('w'):            twist = [+speed, 0, 0, 0, 0, 0]
        elif key == ord('s'):            twist = [-speed, 0, 0, 0, 0, 0]
        elif key == ord('q'):            twist = [0, 0, 0, 0, 0, +speed]
        elif key == ord('e'):            twist = [0, 0, 0, 0, 0, -speed]
        elif key == ord('a'):            twist = [0, 0, 0, 0, -speed, 0]
        elif key == ord('d'):            twist = [0, 0, 0, 0, +speed, 0]
        elif key == ord('z'):            twist = [0, 0, 0, +speed, 0, 0]
        elif key == ord('x'):            twist = [0, 0, 0, -speed, 0, 0]
        elif key == ord(' '):            twist = [0.0] * 6
        elif key == ord(']'):            speed = min(speed * 2.0, 2.0)
        elif key == ord('['):            speed = max(speed / 2.0, 0.001)
        elif key == 3:                   break   # Ctrl-C

        draw()
        pub.send_multipart(encode_twist(Twist(twist=list(twist))))
        time.sleep(dt)

    # send zero twist before exiting so the robot stops
    zero = Twist(twist=[0.0] * 6)
    for _ in range(10):
        pub.send_multipart(encode_twist(zero))
        time.sleep(0.02)

    pub.close()
    ctx.term()


def main() -> None:
    with open(ROOT / "config/sim_config.yaml") as f:
        cfg = yaml.safe_load(f)
    try:
        curses.wrapper(_run, cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
