"""
MuJoCo simulation node.

Responsibilities
----------------
  - Load the Franka Panda model and step physics.
  - PUB joint state (q, qd, qfrc_bias) on ZMQ port 5555.
  - PULL CommandMsg from the controller on ZMQ port 5556.
  - Dispatch to one of three control modes based on cmd.mode:

      "torque"    values = joint torques [Nm]
                  → written directly to data.ctrl
                  physics integrates normally

      "position"  values = desired joint positions [rad]
                  → sim runs internal PD servo at full physics rate
                  → resulting torques written to data.ctrl
                  fully physical: inertia, contacts, gravity all active

      "kinematic" values = desired joint positions [rad]
                  → written directly to data.qpos; data.qvel zeroed
                  → mj_forward() called instead of mj_step()
                  bypasses dynamics — useful for plan visualisation only

Launch
------
  uv run python -m sim.mujoco_sim
  uv run python -m sim.mujoco_sim --no-render
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

import mujoco
import mujoco.viewer
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from messages.types  import StateMsg, CommandMsg, MODE_TORQUE, MODE_POSITION, MODE_KINEMATIC
from messages.topics import CMD
from messages.protocol import decode_cmd, encode_state


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_sockets(cfg: dict) -> tuple[zmq.Socket, zmq.Socket]:
    ctx = zmq.Context()

    pub = ctx.socket(zmq.PUB)
    pub.bind(cfg["zmq"]["state_pub_addr"])

    pull = ctx.socket(zmq.PULL)
    pull.bind(cfg["zmq"]["cmd_pull_addr"])
    pull.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking

    return pub, pull


def run(render: bool, cfg: dict) -> None:
    model_path = ROOT / cfg["sim"]["model_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data  = mujoco.MjData(model)

    # Initialise to home keyframe
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    pub, pull = build_sockets(cfg)

    ndof       = cfg["robot"]["ndof"]
    state_dt   = 1.0 / cfg["sim"]["state_hz"]
    physics_dt = cfg["sim"]["timestep"]
    realtime   = cfg["sim"]["realtime"]

    # Internal PD gains for position-servo mode (run at full physics rate)
    servo_cfg = cfg["sim"]["position_servo"]
    servo_kp  = np.array(servo_cfg["kp"][:ndof])
    servo_kd  = np.array(servo_cfg["kd"][:ndof])

    print(f"[sim] model: {model_path.name}  ndof={ndof}")
    print(f"[sim] physics dt={physics_dt*1e3:.1f}ms  "
          f"state pub @ {cfg['sim']['state_hz']}Hz")
    print(f"[sim] PUB {cfg['zmq']['state_pub_addr']}  "
          f"PULL {cfg['zmq']['cmd_pull_addr']}")
    print(f"[sim] modes: torque | position (internal PD kp={servo_kp}) | kinematic")

    # Last received command — persists across physics steps
    last_cmd: CommandMsg = CommandMsg(values=[0.0] * ndof, mode=MODE_TORQUE)

    def _apply_command() -> None:
        """Apply last_cmd to the simulation."""
        vals = np.array(last_cmd.values[:ndof])
        mode = last_cmd.mode

        if mode == MODE_TORQUE:
            # Direct torque: clip to actuator limits and write to ctrl
            np.clip(vals, model.actuator_ctrlrange[:ndof, 0],
                    model.actuator_ctrlrange[:ndof, 1], out=vals)
            data.ctrl[:ndof] = vals

        elif mode == MODE_POSITION:
            # Internal PD servo running at physics rate
            # τ = Kp*(q_des - q) + Kd*(0 - qd)  — zero desired velocity
            q  = data.qpos[:ndof]
            qd = data.qvel[:ndof]
            torques = servo_kp * (vals - q) + servo_kd * (0.0 - qd)
            np.clip(torques, model.actuator_ctrlrange[:ndof, 0],
                    model.actuator_ctrlrange[:ndof, 1], out=torques)
            data.ctrl[:ndof] = torques

        elif mode == MODE_KINEMATIC:
            # Bypass physics: teleport joints, zero velocity, forward kinematics
            data.qpos[:ndof] = vals
            data.qvel[:ndof] = 0.0
            # Skip mj_step below — caller checks this flag
        else:
            print(f"[sim] unknown mode '{mode}', ignoring")

    def _step_and_publish(viewer=None) -> None:
        nonlocal last_cmd, step_count, next_publish

        # --- drain command queue (non-blocking) ---
        while True:
            try:
                raw = pull.recv()
                last_cmd = decode_cmd(raw)
            except zmq.Again:
                break

        # --- apply command and advance simulation ---
        _apply_command()

        if last_cmd.mode == MODE_KINEMATIC:
            mujoco.mj_forward(model, data)   # kinematics only, no integration
        else:
            mujoco.mj_step(model, data)      # full physics step

        step_count += 1

        # --- publish state at reduced rate ---
        now = time.monotonic()
        if now >= next_publish:
            next_publish = now + state_dt
            state = StateMsg(
                q=data.qpos[:ndof].tolist(),
                qd=data.qvel[:ndof].tolist(),
                qfrc_bias=data.qfrc_bias[:ndof].tolist(),
                sim_time=float(data.time),
            )
            pub.send_multipart(encode_state(state))

        if viewer is not None:
            viewer.sync()

    step_count   = 0
    next_publish = time.monotonic()

    if render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth  = 135
            viewer.cam.elevation = -20
            print("[sim] viewer launched — close window to stop")
            t_wall = time.monotonic()
            while viewer.is_running():
                _step_and_publish(viewer)
                if realtime:
                    t_wall += physics_dt
                    sleep_t = t_wall - time.monotonic()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
    else:
        print("[sim] headless mode — Ctrl-C to stop")
        t_wall = time.monotonic()
        while True:
            _step_and_publish()
            if realtime:
                t_wall += physics_dt
                sleep_t = t_wall - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

    pub.close()
    pull.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo simulation node")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless (no viewer)")
    parser.add_argument("--config", default=str(ROOT / "config/sim_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    render = cfg["sim"]["render"] and not args.no_render

    try:
        run(render=render, cfg=cfg)
    except KeyboardInterrupt:
        print("\n[sim] stopped")


if __name__ == "__main__":
    main()
