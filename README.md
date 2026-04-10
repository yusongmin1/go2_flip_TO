# Whole-Body Trajectory Optimization in the SE(3) Tangent Space

## Overview

Agile whole-body motion generation for legged and humanoid robots remains a fundamental challenge in robotics. This repository implements a trajectory optimization framework that formulates the robot's floating-base dynamics in the tangent space of SE(3), enabling efficient optimization using standard off the self NLP solvers (IPOPT) without requiring manifold-specific techniques.

The implementation supports whole-body dynamics, contact constraints and terrain modeling while leveraging analytical derivatives via the Pinocchio library. 

See more at [https://lar.upatras.gr/projects/ibrics.html](https://lar.upatras.gr/projects/ibrics.html).


## Results

You can find a video summarizing the approach and results at [https://www.youtube.com/watch?v=zBJSsiUExCw](https://www.youtube.com/watch?v=zBJSsiUExCw).

The videos below showcase various motions that were generated using the implemented trajectory optimization solver. The code for each particular motion can be found under `src/examples/agile_exps/`.

### Jump
https://github.com/user-attachments/assets/47698c0f-a3dd-43d8-beb0-040ad7d6c012

### Backflip
https://github.com/user-attachments/assets/be5b7db3-2185-4c06-8226-fc28839872ab

### Handstand
https://github.com/user-attachments/assets/fab747f8-0eb0-4b04-819b-5b10ca46a669

### Hopscotch
https://github.com/user-attachments/assets/f579ce10-0d2a-4a95-9d9b-9fb9b3147b29

### Sideflip
https://github.com/user-attachments/assets/31e9e250-8758-4190-8677-22f8fd361e7c

### Walk
https://github.com/user-attachments/assets/eb647602-af8b-4d3d-903f-c3fedc6d0f09


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

Add this to your shell profile if you use the project often. **`PYTHONPATH=$(pwd)/src` alone is not enough** and will raise `ModuleNotFoundError: trajectory_optimization`.

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
python src/examples/agile_exps/quad_sideflip.py --vis
python src/examples/agile_exps/quad_backflip.py --vis
python src/examples/agile_exps/quad_frontflip.py --vis
python src/examples/agile_exps/quad_jump_forward.py --vis
python src/examples/agile_exps/quad_spin_inplace.py --vis
```

All **`src/examples/agile_exps/quad_*.py`** Go2 demos export the same way after a successful solve: `datasets/go2/trajectories/<run_name>/` (NPZ/CSV) and `datasets/go2/mocap_motions_go2/<run>_25hz.txt` at **25 Hz** (AMP JSON like `datasets/ai.py`). Set **`GO2_NO_DATASET=1`** to skip (legacy: `QUAD_SPIN_NO_DATASET=1`). Use **`--vis`** for MeshCat playback. Example AMP viz: `python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_25hz.txt` (add repo root to `PYTHONPATH` for `datasets.*`).

After a solve, the scripts print **planning time** (IPOPT wall time) and iteration count. Trajectories may be saved under `trajopt_solutions_batch/` depending on the script.

### Troubleshooting
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

