# Whole-Body Trajectory Optimization in the SE(3) Tangent Space

## Overview

Agile whole-body motion generation for legged and humanoid robots remains a fundamental challenge in robotics. This repository implements a trajectory optimization framework that formulates the robot's floating-base dynamics in the tangent space of SE(3), enabling efficient optimization using standard off the self NLP solvers (IPOPT) without requiring manifold-specific techniques.

The implementation supports whole-body dynamics, contact constraints and terrain modeling while leveraging analytical derivatives via the Pinocchio library. 

See more at [https://lar.upatras.gr/projects/ibrics.html](https://lar.upatras.gr/projects/ibrics.html).


## Results

You can find a video summarizing the approach and results at [https://www.youtube.com/watch?v=zBJSsiUExCw](https://www.youtube.com/watch?v=zBJSsiUExCw).

The videos below showcase various motions that were generated using the implemented trajectory optimization solver. The code for each particular motion can be found under `src/examples/agile_exps/`.

## Maintainers

- Evangelos Tsiatsianas (University of Patras) - etsiatsianas@ac.upatras.gr
- Konstantinos Chatzilygeroudis (University of Patras) - costashatz@upatras.gr

## Publication & Citation

This trajectory optimization solver was developed as part of our research on floating-base space parameterizations for agile whole-body motion planning. The work has been published in (also available on [arXiv](https://arxiv.org/abs/2508.11520)):

**A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning**  
*Evangelos Tsiatsianas, Chairi Kiourt, Konstantinos Chatzilygeroudis*  
IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2025

If you use this code in a scientific publication, please use the following citation:

```bibtex
@inproceedings{tsiatsianas2025comparative,
      title={{A Comparative Study of Floating-Base Space Parameterizations for Agile Whole-Body Motion Planning}},
      author={Tsiatsianas, Evangelos and Kiourt, Chairi and Chatzilygeroudis, Konstantinos},
      booktitle={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
      year={2025}
    }
```

## Install on your system

### Installation
We recommend using [Conda](https://docs.conda.io/) (or Mamba) with the [conda-forge](https://conda-forge.org/) channel. Use **Python 3.10+** (3.11 or 3.13 works well). **Python 3.8 is not supported** by recent `cyipopt` on PyPI; prefer conda-forge for `cyipopt` and IPOPT.

#### Create and activate the environment
From the repository root:

```bash
conda create -n se3traj python=3.11 -y
conda activate se3traj
```

You can also use a local prefix, for example `conda create -p ./.conda python=3.13 -y` and `conda activate ./.conda`.

#### Install dependencies
Install the runtime stack in one step:

```bash
conda install -c conda-forge pinocchio cyipopt meshcat-python matplotlib numpy -y
```

Optional: `example-robot-data` (not required for the bundled robot URDFs under `src/nltrajopt/robots/` and `src/robots/`).

### Environment variable: `PYTHONPATH`
Examples import modules as `trajectory_optimization`, `node`, … which live under `src/nltrajopt/`, and `robots`, `visualiser`, `terrain` under `src/`. From the **repository root**, set:

```bash
export PYTHONPATH="$(pwd)/src/nltrajopt:$(pwd)/src"
```

Add this to your shell profile if you use the project often. **`PYTHONPATH=$(pwd)/src` alone is not enough** and will raise `ModuleNotFoundError: trajectory_optimization`. For **`datasets/`** scripts (e.g. `viz_go2_amp_trajectory.py`), prepend **`$(pwd):`** so the repo root is on the path (see the visualization block below).

### Running examples
Always run commands from the repository root with `conda` activated and `PYTHONPATH` set as above.

**Visualization:** pass `--vis` to open [MeshCat](https://github.com/rdeits/meshcat) in the browser and play back the optimized trajectory.

**Simple demos** (`src/examples/simple/`):

```bash
python src/examples/simple/go2_trajopt.py --vis
```

**Agile motions** (`src/examples/agile_exps/`):

```bash
python src/examples/agile_exps/quad_walk_forward.py --vis
python src/examples/agile_exps/quad_walk_backward_ramp.py --vis
python src/examples/agile_exps/quad_strafe_left_ramp.py --vis
python src/examples/agile_exps/quad_strafe_right_ramp.py --vis
python src/examples/agile_exps/quad_spin_inplace.py --vis
python src/examples/agile_exps/quad_spin_inplace_reverse.py --vis
python src/examples/agile_exps/quad_sideflip.py --vis
python src/examples/agile_exps/quad_backflip.py --vis
python src/examples/agile_exps/quad_frontflip.py --vis
python src/examples/agile_exps/quad_jump_forward.py --vis
python src/examples/agile_exps/go2_trajopt.py --vis
```

| 键名 | 必选 | 形状 | 含义 |
|------|:----:|------|------|
| `fps` | 是 | 标量或可 `float(...)` 的标量 | 轨迹采样帧率（Hz）。帧间隔 `dt = 1/fps` |
| `root_pos` | 是 | `(N, 3)` | 根连杆在世界系下的位置 `[x,y,z]`（米） |
| `root_rot` | 是 | `(N, 4)` | 根连杆在世界系下四元数 **`xyzw`** |
| `dof_pos` | 是 | `(N, num_dof)` | 关节位置（弧度），列顺序同 `dof_names` |
| `root_lin_vel` | DeepMimic 可选；AMP 必选 | `(N, 3)` | 根连杆线速度（m/s），**机体系（基座系）** |
| `root_ang_vel` | 同上 | `(N, 3)` | 根连杆角速度（rad/s），**机体系（基座系）** |
| `dof_vel` | 同上 | `(N, num_dof)` | 关节速度（rad/s），列顺序同 `dof_pos` |
| `key_body_pos_relative_to_base` | DeepMimic 可选；AMP 必选 | `(N, K, 3)` | 每个关键身体点在**世界系**下相对根位置的位移，即近似 \(\mathbf{p}_\text{key} - \mathbf{p}_\text{root}\)（与 `process_reference_motion` 中 `cur_key_body_pos - cur_base_pos` 一致），`K` 为关键身体数 |

### Go2 export data formats

Successful agile / Go2 examples write datasets through `src/examples/agile_exps/_export_go2_datasets.py` → `datasets/go2_pin_trajectory.py` (NPZ/CSV) and `datasets/go2_amp_export.py` (Isaac-style mocap JSON). Implementation details also live in `datasets/go2_amp_export.py`.

#### `datasets/go2/trajectories/<run_name>/`

| File | Contents |
|------|----------|
| `trajectory.npz` | `q` `(T, 19)`, `v` `(T, 18)`, `dt` `(T,)`, `joint_names`, `actuated_q_start` — Pinocchio / URDF order. |
| `meta.json` | `q_pinocchio_layout`, `v_pinocchio_layout`, `actuated_joint_names`, `run_name`, `base_z_offset_applied_m`, etc. |
| `joints_only.csv` | Actuated joint angles only (header = URDF joint names). |
| `trajectory_full.csv` | Time `t` + full `q` (header documents base + joints). |

**`q` (19 floats per row):** `q[0:3]` world position of the floating base (m); `q[3:7]` base quaternion **xyzw**; `q[7:19]` leg joint positions (rad), same order as `actuated_joint_names` in `meta.json`.

**`v` (18 floats per row) — Go2 `JointModelFreeFlyer`:** `v[0:3]` base **linear** velocity expressed in the **base** frame (m/s); `v[3:6]` base **angular** velocity in the **base** frame (rad/s); `v[6:18]` joint velocities (rad/s), aligned with `q[7:]`.

#### `datasets/go2/mocap_motions_go2/<run>_50hz.txt`

JSON envelope: `LoopMode`, `FrameDuration` (= `1/fps`), `Frames` = list of flat numeric rows. A sidecar **`<stem>.meta.json`** records `format`, `frame_dim`, `frame_layout` (`"default"`), `tail_kind`, `foot_frame_names` (same order as **key_body** below), and a `layout` map of column semantics.

**50 Hz row (field order, 49 floats):**

`pos(3)` · `quat(4)` xyzw · **机体系线速度 (3)** · **机体系角速度 (3)** · `dof_pos(12)` · `dof_vel(12)` · **`key_body(12)`**

- **`pos` / `quat`:** root in **world** (m) and quaternion **xyzw**.  
- **机体系线速度 / 机体系角速度:** base linear / angular rate in the **base** frame (m/s, rad/s) = Pinocchio `v[0:3]` and `v[3:6]`.  
- **`dof_pos` / `dof_vel`:** `q[7:19]` and `v[6:18]` (URDF / Pinocchio joint order).  
- **`key_body`:** four triples in **world** frame, each \((\mathbf{p}_\text{key}-\mathbf{p}_\text{base})\) (m). **Fixed order** (same as `foot_frame_names` in meta):

```json
["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
```

| Column index | Length | Meaning |
|:-------------|:------:|---------|
| `0:3` | 3 | `pos` — root in **world** (m) |
| `3:7` | 4 | `quat` — **xyzw** (world) |
| `7:10` | 3 | 机体系线速度 = `v[0:3]` (m/s) |
| `10:13` | 3 | 机体系角速度 = `v[3:6]` (rad/s) |
| `13:25` | 12 | `dof_pos` = `q[7:19]` (rad) |
| `25:37` | 12 | `dof_vel` = `v[6:18]` (rad/s) |
| `37:49` | 12 | `key_body` — \((\mathbf{p}_\text{key}-\mathbf{p}_\text{base})_w` for `FL_foot` … `RR_foot` |

**Export options:** Set **`GO2_NO_DATASET=1`** (or `QUAD_SPIN_NO_DATASET=1`) to skip writing these files. Optional **`GO2_EXPORT_BASE_Z_OFFSET`** (metres, default **0.022**) is added to each knot `q[2]` before export (NPZ and mocap root height); set to **`0`** for raw solver height.

**Legacy:** Older **85**-column `Frames` used sixteen key-body triples; re-export for the current four-foot layout. Legacy **49-D** `*_25hz.txt` AMP rows (`export_amp_mocap_txt`, `format: amp49_legacy` in meta) store full `q` in columns `0:18`, then foot positions in the **base** frame, base twist, and joint rates — see `datasets/go2_amp_export.py`.

**Go2 flat-ground ramps (3 s, 0→1 m/s translation or 0→1 rad/s yaw):** forward/back use **12 cm** swing clearance; **strafe** uses **8 cm** and a **pure ±y** displacement (no forward-axis motion in the task). Spin in place: **12 cm** swing. Export names include `_ramp_3s` where applicable (e.g. `quad_walk_forward_ramp_3s`).

**Visualize exported trajectories (MeshCat)** — after a successful solve, replay **50 Hz** mocap JSON or NPZ. Put the **repository root** first in `PYTHONPATH` when using `datasets/viz_go2_amp_trajectory.py` (so `import datasets` works):

```bash
export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
# Default export is 50 Hz Isaac-style mocap; legacy 49-D *_25hz.txt still works with --amp.
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_frontflip_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_sideflip_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_jump_forward_1m_50hz.txt
# If you change ``JUMP_FORWARD_M`` in ``quad_jump_forward.py``, the run name uses that distance (e.g. ``quad_jump_forward_0p5m_50hz.txt``).
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_walk_forward_ramp_3s_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_walk_backward_ramp_3s_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_strafe_left_ramp_3s_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_strafe_right_ramp_3s_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/spin_inplace_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/go2_trajopt_50hz.txt
```

Replay from **NPZ** (same URDF stack; root in `PYTHONPATH` is optional if you only need `src` + `nltrajopt`):

```bash
export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
python datasets/viz_go2_amp_trajectory.py --npz datasets/go2/trajectories/quad_backflip/trajectory.npz
python datasets/viz_go2_amp_trajectory.py --npz datasets/go2/trajectories/quad_walk_forward_ramp_3s/trajectory.npz
```

After a solve, the scripts print **planning time** (IPOPT wall time) and iteration count. Trajectories may be saved under `trajopt_solutions_batch/` depending on the script.

### Troubleshooting
- **`No module named 'datasets'`** (when running `datasets/viz_go2_amp_trajectory.py`) — Prepend the repo root: `export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"`.
- **`No module named 'trajectory_optimization'`** — Set `PYTHONPATH` to both `src/nltrajopt` and `src` as shown above.
- **`No module named 'cyipopt'`** — Install with `conda install -c conda-forge cyipopt` (recommended). Building `cyipopt` with `pip` on old Python or without matching NumPy often fails.
- **`libhsl.so: cannot open shared object file`** — This repository configures IPOPT to use the **MUMPS** linear solver (not HSL). Use the conda-forge `cyipopt`/`ipopt` stack; avoid IPOPT builds that require proprietary HSL libraries.

## Projects using SE3_TrajOpt

- [AHMP](https://github.com/hucebot/ahmp): Motion Planning and Contact Discovery based on Mixed-Distribution Cross-Entropy Method.

## Acknowledgments

This work has been partially supported by project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.

<p align="center">
<img src="https://archimedesai.gr/images/logo_en.svg" alt="logo_archimedes" width="50%"/>
<p/>

This work was conducted within the [Laboratory of Automation and Robotics](https://lar.ece.upatras.gr/) (LAR), Department of Electrical and Computer Engineering, and [Archimedes Research Unit](https://archimedesai.gr/en/), RC Athena, Greece.

<p align="center">
<img src="http://lar.ece.upatras.gr/wp-content/uploads/sites/147/2022/10/lar_profile_alpha.png" alt="logo_lar" width="20%"/><br/>
<img src="https://www.upatras.gr/wp-content/uploads/up_2017_logo_en.png" alt="logo_upatras" width="50%"/>
</p>

