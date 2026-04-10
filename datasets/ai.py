import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import os, time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial.transform import Rotation as R, Slerp
# import meshcat.geometry as g
# import meshcat.transformations as tf

import numpy as np
np.set_printoptions(precision=6, suppress=True)   # 6 位小数，不科学计数
class play2Robot:
    def __init__(self):
        description_path_y = os.path.abspath("./datasets/go2/")
        urdf_file_y = "/go2.urdf"
        urdf_path_y = os.path.abspath(description_path_y + urdf_file_y)
        self.rrobot_y = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path_y,
            package_dirs=description_path_y,
            root_joint=pin.JointModelFreeFlyer())
        model_y = self.rrobot_y.model
        print(f"Model name: {model_y.name}")
        print(f"共 {model_y.nq} 维配置向量（含浮动 base）")
        self.collision_model_y = pin.GeometryModel()
        self.visual_model_y = pin.GeometryModel()
        mesh_dir_y = description_path_y
        pin.buildGeomFromUrdf(model_y, urdf_path_y, pin.GeometryType.COLLISION, self.collision_model_y, mesh_dir_y)
        pin.buildGeomFromUrdf(model_y, urdf_path_y, pin.GeometryType.VISUAL, self.visual_model_y, mesh_dir_y)
        viz_y = MeshcatVisualizer(model_y, self.collision_model_y, self.visual_model_y)
        viz_y.initViewer(open=True)
        viz_y.loadViewerModel(rootNodeName="Robot_y")
        ground_plane = g.Box([10.0, 10.0, 0.1])
        transform = tf.translation_matrix([0, 0, -0.05])
        viz_y.viewer["/Ground"].set_object(ground_plane)
        viz_y.viewer["/Ground"].set_transform(transform)
        q_y = np.zeros(model_y.nq)
        q_y[1] = -0.5
        viz_y.display(q_y)
        self.viz_y = viz_y
        self.model_y = model_y
        viz_y.viewer["trunk_0"].set_property("color", [0.7, 0.7, 0.7, 1.0])

        # 足端 frame 名称表（A1 四足）
        self.ankle_names = ['FL_foot','FR_foot','RL_foot', 'RR_foot']

    # ================ 先算局部坐标，再手动转到 world ================
    def ankle_world_from_local(self, q):
        data = self.model_y.createData()
        pin.forwardKinematics(self.model_y, data, q)
        pin.updateFramePlacements(self.model_y, data)

        base_id = self.model_y.getFrameId('base')
        oMbase = data.oMf[base_id]
        R_base = oMbase.rotation
        p_base = oMbase.translation

        world = {}
        for name in self.ankle_names:
            fid = self.model_y.getFrameId(name)
            oMankle = data.oMf[fid]
            local_vec = (oMbase.inverse() * oMankle).translation
            world_pos = R_base @ local_vec + p_base
            world[name] = world_pos
        return world

    # ================ 播放并实时画小红球验证 world 位置 ================
    def robotPlay(self, y1_data):
        dt = 0.025
        num = len(y1_data)
        radius = 0.025
        sphere = g.Sphere(radius)
        material = g.MeshLambertMaterial(color=0xff0000)
        for i in range(num):
            q = y1_data[i]
            self.viz_y.display(q)
            world_pos = self.ankle_world_from_local(q)   # ← 改用新函数
            for idx, (name, pos) in enumerate(world_pos.items()):
                node = f"/ankle_world_{idx}"
                self.viz_y.viewer[node].set_object(sphere, material)
                self.viz_y.viewer[node].set_transform(tf.translation_matrix(pos))
            time.sleep(dt)
def save_as_txt_with_metadata(frames, fps, output_path):
    """
    保存为txt文件，包含指定的元数据，数据横向排列
    """
    # 计算帧持续时间
    frame_duration = 1.0 / fps
    
    # 保存为JSON格式的txt文件，确保数据横向排列
    with open(output_path, 'w') as f:
        f.write('{\n')
        f.write('  "LoopMode": "Wrap",\n')
        f.write(f'  "FrameDuration": {frame_duration},\n')
        f.write('  "EnableCycleOffsetPosition": true,\n')
        f.write('  "EnableCycleOffsetRotation": true,\n')
        f.write('  "MotionWeight": 0.5,\n')
        f.write('  "Frames": [\n')
        
        # 逐行写入帧数据，确保横向排列
        for i, frame in enumerate(frames):
            # 将每个帧的数据转换为字符串，确保横向排列
            frame_str = ', '.join([f'{x:.6f}' for x in frame])
            if i < len(frames) - 1:
                f.write(f'    [{frame_str}],\n')
            else:
                f.write(f'    [{frame_str}]\n')
        
        f.write('  ]\n')
        f.write('}\n')
    
    print(f"数据已保存到: {output_path}")
    print(f"帧数: {frames.shape[0]}")
    print(f"每帧数据维度: {frames.shape[1]}")
    print(f"帧持续时间: {frame_duration:.6f}秒")
    print(f"数据顺序: [线速度, 角速度, 关节位置, 关节速度,位置(x,y,z), 四元数(w,x,y,z), ]")

# -------------------- 原主程序（丝毫未动） --------------------
if __name__ == '__main__':
    vizPlay = play2Robot()

    txt_file =  '/home/zju/amp/rl_amp-main-yu/datasets/output_go2.txt'
    raw_data = np.loadtxt(txt_file)
    assert raw_data.shape[1] == 37, "每行必须是 37 维"

    root_pos  = raw_data[:, :3]
    print(root_pos[:10])
    root_quat = raw_data[:, 3:7]
    dof_pos   = raw_data[:, 13:25]

    num_frames = len(dof_pos)
    data = np.zeros((num_frames, 19))
    data[:, :3]  = root_pos
    data[:, 3:7] = root_quat
    dof = dof_pos.copy()
    dof[:, 6:9]  = dof_pos[:, 3:6]
    dof[:, 3:6]  = dof_pos[:, 6:9]
    data[:, 7:]  = dof
    start_range=range(4500,4700) #800 900 原地站立， 1200,1500 前进 1600,1800 后退 2300:2700转圈 3050:3200 右平移 3650:3900 前左 4000:4180 前右 4500:4700 右转 4800:5000坐转
    data=data[start_range,:] 
    num_frames = list(start_range)   # [2420, 2421, ..., 2519]
    # 播放并实时画 world 足端红球（已改用局部→world手动转换）
    vizPlay.robotPlay(data)
    dof_vel=raw_data[:, 25:37]

    vel=dof_vel.copy()
    vel[:, 6:9]  = dof_vel[:, 3:6]
    vel[:, 3:6]  = dof_vel[:, 6:9] ##原来的长度为3000的数据 




    # ------------------------------------------------------------------
    # 以下是你需要的 43 列 base-系文件生成（全新分支，零改动上方）
    # ------------------------------------------------------------------
    print('\n>>> 正在生成 43 列 base-系文件 ...')
    out43 = np.zeros((len(start_range), 49))
    for k in range(len(start_range)):
        print("写入帧号:", start_range[k], "raw 索引:", k+start_range[0])
        q = data[k].copy()
        # base 系速度
        R_wb = pin.Quaternion(q[3:7]).toRotationMatrix() #????
        lin_b = R_wb.T @ raw_data[k+start_range[0], 7:10]
        print(lin_b,raw_data[k+start_range[0], 7:10])
        ang_b = R_wb.T @ raw_data[k+start_range[0], 10:13]

        # base 系足端局部向量（复用你类里逻辑）
        world_dict = vizPlay.ankle_world_from_local(q)
        base_id  = vizPlay.model_y.getFrameId('base')
        data_tmp = vizPlay.model_y.createData()
        pin.forwardKinematics(vizPlay.model_y, data_tmp, q)
        pin.updateFramePlacements(vizPlay.model_y, data_tmp)
        oMbase = data_tmp.oMf[base_id]
        locals = []
        for name in vizPlay.ankle_names:
            fid = vizPlay.model_y.getFrameId(name)
            local_vec = (oMbase.inverse() * data_tmp.oMf[fid]).translation
            locals.append(local_vec)
        foot_local_b = np.concatenate(locals)

        out43[k] = np.concatenate([
            q[:7],               # 0:7  root_pos + root_rot
            q[7:],               # 7:19 joint_pos
            foot_local_b,        # 19:31 base-系足端局部
            lin_b, ang_b,        # 31:37 base-系线速度/角速度
            vel[k+start_range[0]]  # 37:49 joint_vel
        ])
    save_as_txt_with_metadata(out43, 25, '/home/zju/amp/rl_amp-main-yu/datasets/mocap_motions_go2/turn_right.txt')

    # root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel, joint_vel, foot_vel