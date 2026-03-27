# TrajOpt Planning Framework

A multi-process trajectory optimization framework built around **MuJoCo** (simulation), **ZeroMQ** (IPC), and support for planners written in both **C++** and **Python**.

---

## Architecture

Three independent processes communicate over ZeroMQ:

```
┌──────────────────────────────────────────────────────────────┐
│                      Process topology                        │
│                                                              │
│   ┌─────────────┐  STATE (PUB:5555)   ┌──────────────────┐  │
│   │  MuJoCo Sim │ ───────────────────►│   Planner        │  │
│   │             │ ───────────────────►│   (C++ or Python)│  │
│   │  scene.xml  │                     └────────┬─────────┘  │
│   │             │                  TRAJ        │            │
│   │             │            (PUB:5557)        ▼            │
│   │             │  CMD        ┌──────────────────┐          │
│   │             │◄────────────│   Controller     │          │
│   └─────────────┘  (PULL:5556)│                  │          │
│                               └──────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

| Process | Role | Rate |
|---------|------|------|
| **Sim** | Steps MuJoCo physics, renders viewer, publishes state | 500 Hz physics / 100 Hz pub |
| **Planner** | Receives state, computes joint trajectory, publishes waypoints | ~10 Hz |
| **Controller** | Interpolates trajectory, computes command, sends to sim | 100 Hz |

---

## Directory layout

```
planning/
├── pyproject.toml              # uv project — Python dependencies
├── .python-version             # Python 3.11 (managed by uv)
├── CMakeLists.txt              # Root CMake — adds C++ planner subdirectory
├── launch.sh                   # One-shot launcher for all three processes
│
├── config/
│   └── sim_config.yaml         # All tuneable parameters (rates, ZMQ ports, PD gains)
│
├── models/
│   └── franka_panda/           # Official mujoco_menagerie Franka Panda model
│       ├── scene.xml           # Entry point: floor + lights + includes panda.xml
│       ├── panda.xml           # 7-DoF arm with torque actuators + home keyframe
│       └── assets/             # 67 OBJ/STL mesh files
│
├── messages/
│   ├── types.py                # Dataclasses: StateMsg, TrajectoryMsg, CommandMsg, Waypoint
│   │                           # Mode constants: MODE_TORQUE, MODE_POSITION, MODE_KINEMATIC
│   ├── topics.py               # Topic registry: TopicSpec + TOPICS, STATE, TRAJ, CMD
│   ├── protocol.py             # Encode / decode only — imports from types.py and topics.py
│   └── __init__.py             # Re-exports the full public API
│
├── sim/
│   └── mujoco_sim.py           # Simulation process
│
├── planners/
│   ├── python/
│   │   └── planner.py          # Python cubic-spline planner
│   └── cpp/
│       ├── include/
│       │   └── planner.hpp     # CubicPlanner class (header-only)
│       ├── src/
│       │   └── main.cpp        # C++ ZMQ planner node
│       └── CMakeLists.txt      # Fetches cppzmq + nlohmann/json via FetchContent
│
├── control/
│   └── controller.py           # Controller process
│
└── scripts/
    └── setup_cpp_deps.sh       # Install libzmq3-dev and build tools (apt)
```

---

## Quick start

### 1. Install uv (if not present)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

### 2. Install Python dependencies

```bash
uv sync
```

### 3a. Run with the Python planner

```bash
bash launch.sh
# or headless:
bash launch.sh python --no-render
```

### 3b. Run with the C++ planner

Build once:

```bash
bash scripts/setup_cpp_deps.sh          # installs libzmq3-dev, cmake (apt)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

Then launch:

```bash
bash launch.sh cpp
```

`launch.sh` starts the sim, waits for its sockets to bind, starts the controller, then starts the chosen planner. `Ctrl-C` tears everything down cleanly.

---

## Components in detail

### Sim — `sim/mujoco_sim.py`

- Loads `models/franka_panda/scene.xml` and resets to the `home` keyframe.
- Steps physics at 500 Hz (2 ms timestep); publishes `STATE` at 100 Hz.
- **PUB** `STATE` frames on port **5555** containing:
  - `q` — joint positions [rad]
  - `qd` — joint velocities [rad/s]
  - `qfrc_bias` — gravity + Coriolis forces [Nm]
  - `sim_time`, `wall_time`
- **PULL** `CMD` frames on port **5556** and dispatches to the appropriate control mode (see [Control modes](#control-modes)).
- Launches the MuJoCo passive viewer by default; pass `--no-render` for headless.

### Planner (Python) — `planners/python/planner.py`

Implements **cubic polynomial** joint-space interpolation:

```
q(t)  = q0 + a2·t² + a3·t³       a2 =  3/T² · (qf − q0)
qd(t) = 2·a2·t + 3·a3·t²         a3 = −2/T³ · (qf − q0)
```

Zero-velocity boundary conditions at start and end. Cycles through three predefined goal configurations every ~4 s. The planner also sets the `mode` field on the trajectory it publishes, which propagates through the controller to the sim.

### Planner (C++) — `planners/cpp/`

Identical cubic-spline algorithm, implemented in C++17.

- **`include/planner.hpp`** — `CubicPlanner` class; header-only, STL only.
- **`src/main.cpp`** — ZMQ subscriber/publisher loop using [cppzmq](https://github.com/zeromq/cppzmq) and [nlohmann/json](https://github.com/nlohmann/json).

Both planners use the same ZMQ topic and JSON wire format and are **drop-in replacements** for each other.

### Controller — `control/controller.py`

Runs at 100 Hz. Reads the `mode` from the active `TrajectoryMsg`, interpolates the desired setpoint, and sends the appropriate `CommandMsg` to the sim.

See [Control modes](#control-modes) for per-mode behaviour.

### ZMQ message layer — `messages/`

The `messages/` package is split into three files with distinct responsibilities:

```
messages/
├── types.py     what data looks like   dataclasses, mode constants — no ZMQ
├── topics.py    what goes on which     TopicSpec registry, wire bytes, pub/sub map
│               topic
└── protocol.py  how data is sent       encode() / decode() only, imports the above
```

**`types.py`** — the only place message fields are defined. No serialisation, no ZMQ imports.

| Type | Fields |
|------|--------|
| `StateMsg` | `q`, `qd`, `qfrc_bias`, `sim_time`, `wall_time` |
| `TrajectoryMsg` | `waypoints`, `start_time`, `mode`, `wall_time` |
| `CommandMsg` | `values`, `mode`, `wall_time` |
| `Waypoint` | `t`, `q`, `qd` |

Mode constants (also in `types.py`): `MODE_TORQUE`, `MODE_POSITION`, `MODE_KINEMATIC`.

**`topics.py`** — the single place to add, rename, or document a topic. Each entry is a `TopicSpec`:

```python
@dataclass(frozen=True)
class TopicSpec:
    name:        str     # human-readable key
    bytes:       bytes   # ZMQ wire bytes  e.g. b"STATE"
    msg_type:    Type    # dataclass from types.py
    publisher:   str     # which process publishes
    subscribers: tuple   # which processes subscribe
    description: str
```

| Name | Wire bytes | `msg_type` | Publisher | Subscribers |
|------|-----------|-----------|-----------|-------------|
| `STATE` | `b"STATE"` | `StateMsg` | Sim | Planner, Controller |
| `TRAJ` | `b"TRAJ"` | `TrajectoryMsg` | Planner | Controller |
| `CMD` | `b"CMD"` | `CommandMsg` | Controller | Sim |

Look up a topic at runtime by its wire bytes: `TOPICS.by_bytes[b"STATE"]`.

**`protocol.py`** — thin serialisation layer. All messages travel as two-frame ZMQ multipart messages:

```
frame 0 : topic bytes   e.g. b"STATE"
frame 1 : JSON payload  UTF-8
```

```python
# Generic API (works for any topic)
encode(spec, msg)   →  [bytes, bytes]
decode(spec, raw)   →  msg

# Convenience helpers
encode_state(msg)  /  decode_state(raw)
encode_traj(msg)   /  decode_traj(raw)
encode_cmd(msg)    /  decode_cmd(raw)
```

### Robot model — `models/franka_panda/`

The official [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) Franka Panda model with one modification: the original position-servo `general` actuators are replaced with direct `motor` actuators for torque control.

- Hardware torque limits: ±87 Nm (joints 1–4), ±12 Nm (joints 5–7).
- Full per-link inertial properties and collision meshes.
- Named keyframes: `home` (standard ready pose), `zero`.
- Entry point is `scene.xml` (floor + lighting); it `<include>`s `panda.xml`.

### Configuration — `config/sim_config.yaml`

All parameters in one YAML file. Loaded by every Python process at startup.

```yaml
sim:
  model_path: ...       # path to scene.xml
  timestep: 0.002       # physics step [s]
  render: true
  state_hz: 100         # state publish rate [Hz]
  position_servo:
    kp: [...]           # internal PD gains (used in position mode, runs at 500 Hz)
    kd: [...]

zmq:                    # bind/connect addresses for all three sockets

robot:
  ndof: 7
  home_q: [...]         # home configuration [rad]

controller:
  rate_hz: 100
  kp: [...]             # PD gains for torque-mode tracking (runs at 100 Hz)
  kd: [...]

planner:
  trajectory_dt: 0.02   # waypoint spacing [s]
  default_duration: 4.0 # move duration [s]
```

---

## Control modes

The planner sets a `mode` field on every `TrajectoryMsg` it publishes. The controller reads it, and the sim dispatches accordingly. All three modes use the same ZMQ topics — only the semantics of `CommandMsg.values` change.

### `"torque"` (default)

The controller computes torques and sends them directly.

```
τ = τ_bias + Kp·(q_des − q) + Kd·(qd_des − qd)
```

- `τ_bias` (`qfrc_bias` from the sim) provides gravity + Coriolis feedforward — no second model copy needed.
- Gains from `controller.kp / kd` in the config (run at 100 Hz).
- The sim writes received values straight to `data.ctrl`.
- Best for trajopt: full access to the torque-level dynamics.

### `"position"` — physically correct servo

The controller forwards desired joint positions (`q_des`); the **sim** runs its own PD loop at the full **500 Hz** physics rate:

```
τ = Kp_servo·(q_des − q) + Kd_servo·(0 − qd)   [inside the sim, every physics step]
```

- Gains from `sim.position_servo.kp / kd` in the config. Because they run at 500 Hz they can be much stiffer than the external controller gains.
- Fully physical: inertia, contacts, and gravity still apply.
- Good for high-level waypoint following where you don't need to tune torque-level control.

### `"kinematic"` — bypass physics

The sim writes desired positions directly to `data.qpos`, zeros `data.qvel`, and calls `mj_forward` (no integration step, simulation clock does not advance).

- Contacts, gravity, and inertia are **ignored**.
- Use for: scrubbing through a planned trajectory to visually verify it, debugging IK solutions, or recording reference poses.
- **Not suitable for real control.**

### How mode flows through the system

```
Planner:     TrajectoryMsg(waypoints=[...], mode="position")
                 ↓  ZMQ TRAJ
Controller:  reads mode, interpolates q_des, builds CommandMsg(values=q_des, mode="position")
                 ↓  ZMQ CMD
Sim:         branches on cmd.mode → internal PD servo → data.ctrl → mj_step
```

To use a specific mode from your planner:

```python
from messages.types   import TrajectoryMsg, MODE_POSITION
from messages.topics  import TRAJ
from messages.protocol import encode_traj

traj = TrajectoryMsg(waypoints=waypoints, start_time=time.time(), mode=MODE_POSITION)
pub.send_multipart(encode_traj(traj))
```

---

## Writing your own planner

### Python

1. Copy `planners/python/planner.py` as a starting point.
2. Import your types and topic specs:
   ```python
   from messages.types    import StateMsg, TrajectoryMsg, Waypoint, MODE_TORQUE
   from messages.topics   import STATE, TRAJ
   from messages.protocol import decode_state, encode_traj
   ```
3. Subscribe with `sub.setsockopt(zmq.SUBSCRIBE, STATE.bytes)`.
4. Replace `cubic_trajectory` with your optimizer.
5. Set `mode` on the `TrajectoryMsg` (`MODE_TORQUE`, `MODE_POSITION`, or `MODE_KINEMATIC`).
6. Publish with `pub.send_multipart(encode_traj(traj))`.

**To add a new topic**, add a `TopicSpec` to `messages/topics.py` and a dataclass to `messages/types.py`. Nothing else needs to change.

### C++

1. Copy `planners/cpp/` as a starting point.
2. Replace the `planner.plan(...)` call in `main.cpp` with your solver.
3. Add a `"mode"` field to the JSON payload (default `"torque"` if omitted).
4. Rebuild: `cd build && make -j$(nproc)`.

The controller and sim are **planner-agnostic** — only the ZMQ wire format needs to match.

---

## Dependencies

### Python

Managed by uv via `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `mujoco >= 3.1` | Physics simulation + viewer |
| `pyzmq >= 25.1` | ZeroMQ bindings |
| `numpy >= 1.26` | Array math |
| `scipy >= 1.11` | Optimisation utilities (available for your planners) |
| `pyyaml >= 6.0` | YAML config loading |

### C++

| Library | Source |
|---------|--------|
| `libzmq` | System (`apt install libzmq3-dev`) |
| `cppzmq` | Downloaded by CMake FetchContent |
| `nlohmann/json` | Downloaded by CMake FetchContent |
