# 数据采集与处理

## Go2 agile 轨迹（`src/examples/agile_exps/quad_*.py`）

任一 **`quad_*.py`** 在 **IPOPT 求解成功** 后都会通过 ``src/examples/agile_exps/_export_go2_datasets.py`` 写出：

- **`datasets/go2/trajectories/<run_name>/`**：`trajectory.npz`、`meta.json`、`joints_only.csv`、`trajectory_full.csv`（**关节顺序 = URDF / Pinocchio `q` / `v`**）
- **`datasets/go2/mocap_motions_go2/<run>_25hz.txt`**：与 **`datasets/ai.py`** 中 ``save_as_txt_with_metadata`` 相同结构的 JSON（**25 Hz**，每帧 **49 维**）；原地转圈正向仍用 **`spin_inplace_25hz.txt`**，反向用 **`spin_inplace_ccw_25hz.txt`**（``quad_spin_inplace_reverse.py``）

不生成数据：`GO2_NO_DATASET=1`（兼容旧变量 `QUAD_SPIN_NO_DATASET=1`）。

播放 MeshCat：各脚本加 **`--vis`**；或单独用下方 `viz_go2_amp_trajectory.py` 播 AMP / NPZ。

**可视化（MeshCat）**（仓库根目录、已安装 meshcat-python）：

```bash
export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/spin_inplace_25hz.txt
# 或播放 NPZ：
python datasets/viz_go2_amp_trajectory.py --npz datasets/go2/trajectories/quad_spin_inplace_ramp_3s/trajectory.npz
```

---

legged control go2 这是一个基于legged_control 的开源代码 采集的数据，
具体操作把我提供的代码进行替换
报错某某变量是受保护的，
直接把protect改为 public
采集的数据保存在目录下 原始数据1000hz ,我使用
```bash
awk 'NR%40==1' input.txt > output.txt
```
降采样至25hz，注意原始数据的关节顺序和isaacgym的关节顺序不同，
所以我在 replay_pin 和 ai两个程序进行处理了，
output_go2.txt 为我原始数据的降采样版本，
mocap_motions_go2，a1 是我制作的数据集