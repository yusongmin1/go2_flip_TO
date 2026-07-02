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
- yusongmin - 1902219511@qq.com

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

Forward / backward walk (3 s diagonal trot, **foot-end Bezier swing trajectory optimization**):

```bash
python src/examples/agile_exps/quad_walk_forward.py --vis
python src/examples/agile_exps/quad_walk_backward_ramp.py --vis
```

Other agile demos:

```bash
python src/examples/agile_exps/quad_sideflip.py --vis
python src/examples/agile_exps/quad_backflip.py --vis
python src/examples/agile_exps/quad_frontflip.py --vis
python src/examples/agile_exps/quad_jump_forward.py --vis
```

### Go2 agile dataset format

Agile scripts write **only** `datasets/go2/mocap_motions_go2/<run>_50hz.txt` (JSON, 50 Hz). Each row is **49** floats:

| 列 | 长度 | 含义 |
|----|:----:|------|
| `root_pos` | 3 | 世界系根位置（m） |
| `root_rot` | 4 | 四元数 **xyzw** |
| `dof_pos` | 12 | 关节角（rad），URDF 顺序见下 |
| `key_body_pos_relative_to_base` | 12 | 四足在**机体系**下的位置（FL/FR/RL/RR × xyz） |
| `root_lin_vel` | 3 | 基座线速度（m/s，**机体系**） |
| `root_ang_vel` | 3 | 基座角速度（rad/s，**机体系**） |
| `dof_vel` | 12 | 关节速度（rad/s），顺序同 `dof_pos` |

`dof_pos` / `dof_vel` 关节顺序：

`FL_hip`, `FL_thigh`, `FL_calf`, `FR_hip`, …, `RR_calf`

文件头：`LoopMode`, `FrameDuration` (= `1/50`), `Frames` = 上述行的数组。

实现：`src/examples/agile_exps/_export_go2_datasets.py` → `datasets/go2_amp_export.py`。

**Export options:** `GO2_NO_DATASET=1` 跳过写文件。`GO2_EXPORT_BASE_Z_OFFSET`（默认 **0.022** m）加在根高度 `q[2]` 上；设为 **`0`** 则用求解器原始高度。

**Go2 flat-ground ramps (3 s):** 见各 `quad_*_ramp.py` 脚本；导出名为 `<run>_50hz.txt`。

**Visualize (MuJoCo)** — 默认加载 `datasets/go2/go2/scene.xml`（机器人 OBJ 网格 + 地面 + 天空），回放 txt：

```bash
export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_frontflip_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_sideflip_50hz.txt
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_jump_forward_1m_50hz.txt
# 默认循环播放；只播一遍：加 --no-loop
```

若交互窗口 **段错误 / GLX 报错**（Linux 显卡驱动常见），用离屏导出 MP4：

```bash
export MUJOCO_GL=egl
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt --video /tmp/quad_backflip.mp4
```

Wayland 下可再试：`export GLFW_USE_WAYLAND=0` 或 `--gl glfw`。

Solver **`--vis`** 仍用 MeshCat（优化结果）；**数据集回放**用 MuJoCo 脚本 above.

After a solve, the scripts print **planning time** (IPOPT wall time) and iteration count, and write **`datasets/go2/mocap_motions_go2/<run>_50hz.txt`** only.

### Troubleshooting
- **`No module named 'datasets'`** (when running `datasets/viz_go2_amp_trajectory.py`) — Prepend the repo root: `export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"`.
- **`No module named 'trajectory_optimization'`** — Set `PYTHONPATH` to both `src/nltrajopt` and `src` as shown above.
- **`No module named 'cyipopt'`** — Install with `conda install -c conda-forge cyipopt` (recommended). Building `cyipopt` with `pip` on old Python or without matching NumPy often fails.
- **`No module named 'mujoco'`** — Install for dataset playback: `pip install mujoco` or `conda install -c conda-forge mujoco`.
- **MuJoCo viewer segfault / GLXBadContext** — Use offscreen export: `MUJOCO_GL=egl python datasets/viz_go2_amp_trajectory.py --amp ... --video out.mp4` (needs `pip install imageio imageio-ffmpeg`). On Wayland try `export GLFW_USE_WAYLAND=0`.
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

