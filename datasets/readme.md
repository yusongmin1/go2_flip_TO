# 数据采集与处理

## Go2 agile 轨迹（`src/examples/agile_exps/quad_*.py`）

求解成功后 **只写** ``datasets/go2/mocap_motions_go2/<run>_50hz.txt``（JSON，50 Hz，每行 49 维）：

``root_pos(3)`` · ``root_rot`` xyzw ``(4)`` · ``dof_pos(12)`` · 四足机体系位置 ``(12)`` · 基座线速度 ``(3)`` · 基座角速度 ``(3)`` · ``dof_vel(12)``

不生成 NPZ / CSV / ``trajectories/`` 目录。跳过导出：`GO2_NO_DATASET=1`。

**MuJoCo 回放**（需 ``pip install mujoco``）：

```bash
export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt
# 默认循环播放，关闭窗口退出；只播一遍加 --no-loop
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
