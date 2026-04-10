import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import os, time
# pin 安装：：  conda install pinocchio=3.1.0 -c conda-forge
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
 
# import  joblib
from scipy.spatial.transform import Rotation as R, Slerp
import meshcat.geometry as g
import meshcat.transformations as tf


class play2Robot:
    def __init__(self ):
        description_path_y =  os.path.abspath("./datasets/go2/")# os.path.dirname(urdf_path)
        urdf_file_y = "/go2.urdf"
        # Step 1: 设置URDF文件路径
        urdf_path_y = os.path.abspath(description_path_y +urdf_file_y)  
        self.rrobot_y = pin.RobotWrapper.BuildFromURDF(
                    filename=urdf_path_y,  # URDF 文件路径
                    package_dirs=description_path_y,                  # 资源目录
                    root_joint=pin.JointModelFreeFlyer())                   # 根关节模型
           
        model_y = self.rrobot_y.model
        data_y = model_y.createData() 
        print(f"Model name: {model_y.name}")
        model = model_y          # 或直接 model = model_y / model_g

        print(f"共 {model.nq} 维配置向量（含浮动 base）")
        print("关节索引 -> 关节名称")
        for idx, name in enumerate(model.names):
            print(f"{idx:2d} : {name}")
        # 创建空的几何模型（碰撞和视觉）
        self.collision_model_y = pin.GeometryModel()
        self.visual_model_y = pin.GeometryModel() 
        # 几何模型目录（假设网格文件与URDF文件在同一目录下）
        mesh_dir_y = description_path_y  
        # 使用 buildGeomFromUrdf 填充几何模型
        pin.buildGeomFromUrdf(model_y, urdf_path_y, pin.GeometryType.COLLISION, self.collision_model_y, mesh_dir_y)
        pin.buildGeomFromUrdf(model_y, urdf_path_y, pin.GeometryType.VISUAL, self.visual_model_y, mesh_dir_y)
        # Step 3: 初始化可视化器
        # 创建 MeshCat 可视化器
        viz_y = MeshcatVisualizer(model_y, self.collision_model_y, self.visual_model_y) 
        # 启动可视化器
        viz_y.initViewer(open=True)  # open=True 会在浏览器中自动打开 MeshCat 界面 
        viz_y.loadViewerModel(rootNodeName="Robot_y")      # 加载模型到可视化器 
        ground_plane = g.Box([10.0, 10.0, 0.1])

        # 设置平面的位置和方向（如果需要的话）
        # 这里我们将它平移到z轴-0.1处，模拟地面效果
        transform = tf.translation_matrix([0, 0, -0.05])

        # 将地面添加到 MeshCat 场景中，给它一个唯一的名称
        viz_y.viewer["/Ground"].set_object(ground_plane)
        viz_y.viewer["/Ground"].set_transform(transform)

###############################################
       
        q_y = np.zeros(model_y.nq)  # 所有关节角度为零  
        q_y[1] = -0.5
        viz_y.display(q_y) 
        self.viz_y = viz_y 
        self.model_y = model_y
        # print(1)
        viz_y.loadViewerModel(rootNodeName="Robot_y")

        # ---- 新增：把黑色改成银灰色 ----

        viz_y.viewer["trunk_0"].set_property("color", [0.7, 0.7, 0.7, 1.0])

    def robotPlay(self, y1_data): 
        dt = 0.04  # 时间步长
        num = len(y1_data) 
        for i in range(num):
            self.viz_y.display(y1_data[i])
            time.sleep(dt)  # 等待一段时间以模拟动态效果
            # print(i)
    def computeFootPos(self, y1_data):
        foot_pos_y = [] 
        num = len(y1_data)
        foot_frame_name = ['left_ankle_roll_link',
                           'right_ankle_roll_link']
        
        data_y = self.model_y.createData() 

        for i in range(num): 
            pin.forwardKinematics(self.model_y, data_y, y1_data[i])
            pin.updateFramePlacements(self.model_y, data_y)
            pos_y = np.zeros((2,3))
            for j in range(2) :
                id_y = self.model_y.getFrameId( foot_frame_name[j])
                pos_y[j,:] = data_y.oMf[id_y].translation 
            foot_pos_y.append(pos_y) 
        foot_pos_y = np.array(foot_pos_y)
   
        return foot_pos_y
    
    def computeFootForwardHindPos(self, y1_data, g1_data):
        foot_pos_y = []
        foot_pos_g = []
        num = len(y1_data)
        # q_y =q.copy()
        foot_frame_name = ['left_foot_forward_link','left_foot_hind_link',
                           'right_foot_forward_link','right_foot_hind_link']
        
        data_y = self.model_y.createData()
        data_g = self.model_g.createData()
        for i in range(num): 
            pin.forwardKinematics(self.model_y, data_y, y1_data[i])
            pin.updateFramePlacements(self.model_y, data_y)
            pos_y = np.zeros((4,3))
            for j in range(4) :
                id_y = self.model_y.getFrameId( foot_frame_name[j])
                pos_y[j,:] = data_y.oMf[id_y].translation 
            foot_pos_y.append(pos_y)

            pin.forwardKinematics(self.model_g, data_g, g1_data[i])
            pin.updateFramePlacements(self.model_g, data_g)
            pos_g = np.zeros((4,3))
            for j in range(4) :
                id_g = self.model_g.getFrameId( foot_frame_name[j])
                pos_g[j,:] = data_g.oMf[id_g].translation 
            foot_pos_g.append(pos_g)
    
        foot_pos_g = np.array(foot_pos_g)
        foot_pos_y = np.array(foot_pos_y)
        # 假设 foot_pos_g 和 foot_pos_y 是 (N, M, 3) 形状的数组
        flag_contact = foot_pos_g[:, :, 2]<0.06
        aa = np.sum(np.abs((foot_pos_g[:, :, 2] - foot_pos_y[:, :, 2]))*flag_contact, axis=(0, 1))
        aa += 0.3*np.sum(np.abs((foot_pos_g[:, :, 1]*0.95 - foot_pos_y[:, :, 1]))*flag_contact, axis=(0, 1))
  
        aa += 0.3*np.sum(np.abs((foot_pos_g[:, :, 0]*0.95 - foot_pos_y[:, :, 0]))*flag_contact, axis=(0, 1))
      
        return aa
  
    def miniSlip_scale(self, y1_data, g1_data): 
      
        x  = 0.95
        def fk(x):
            y1_scale = y1_data.copy()
            y1_scale[:,:3] = y1_data[:,:3] * x 
            aa = self.computeFootForwardHindPos( y1_scale, g1_data )
            return aa
        
        a = minimize(fk, x)
        return a.x
    
    def miniGround(self, y1_data, g1_data, scale): 
        num = len(y1_data)
        x = np.zeros(7)  
        y1_data[:,:3] = y1_data[:,:3] *scale
        y1_ans = y1_data.copy() 
        last_q = y1_data[0,:].copy()
        for i in range(num):
            print(i)
            y1 = y1_data[i,:].copy()
            g1 = g1_data[i,:].copy()
            # y1[:3]  = y1[:3] + x[:3]  
            def fk(x):
                y1_scale = y1.copy()
                y1_scale[:3] = y1[:3]   + x[:3] * 0.1
                y1_scale[7:9] = y1[7:9] + x[3:5] 
                y1_scale[13:15] = y1[13:15] + x[5:7]
                aa = self.computeFootForwardHindPos(np.expand_dims(y1_scale, axis=0) , np.expand_dims(g1, axis=0 ))
                aa += np.sum(np.abs(y1_scale[:3] - last_q[:3] )) * 0.3
                aa += np.sum(np.abs(x )) * 0.02
                return aa
            a = minimize(fk, x)
            x = a.x
            y1_ans[i,:3]  = y1[:3] + x[:3]* 0.1
            y1_ans[i,7:9] = y1[7:9] + x[3:5] 
            y1_ans[i,13:15] = y1[13:15] + x[5:7]
            last_q = y1_ans[i, :]
        return y1_ans
if __name__ == '__main__':
    import copy
    vizPlay = play2Robot()

    # 替换为你的 txt 文件路径
    txt_file = '/home/zju/YU/amp_vae/datasets/output_go2.txt'

    # 加载 txt 文件
    raw_data = np.loadtxt(txt_file)

    # 校验维度
    assert raw_data.shape[1] ==37, "每一行必须是 31 维：3+4+3+3+12+12"

    # 提取你需要的部分：位置、四元数、关节位置（忽略速度）
    root_pos = raw_data[:, :3]          # shape (N, 3)
    root_quat = raw_data[:, 3:7]        # shape (N, 4)
    dof_pos = raw_data[:, 13:25]        # shape (N, 12)

    # 构造 data 矩阵
    num_frames = len(dof_pos)
    data = np.zeros((num_frames, 19))   # 3+4+22=29
    data[:, :3] = root_pos
    data[:, 3:7] = root_quat
    dof=np.zeros((num_frames, 12))
    dof=dof_pos.copy()
    dof[:,6:9]=dof_pos[:,3:6]

    dof[:,3:6]=dof_pos[:,6:9]    
    # dof[:,6]=-

    # dof=dof_pos


    data[:, 7:] = dof
    # data=data[4800:5000,:]  #800 900 原地站立， 1200,1500 前进 1600,1800 后退 2300:2700转圈 3050:3200 右平移 3650:3900 前左 4000:4180 前右 4500:4700 右转 4800:5000坐转
    # 播放 
    vizPlay.robotPlay(data)
    print("!!!!!!!!!!!1")
 
