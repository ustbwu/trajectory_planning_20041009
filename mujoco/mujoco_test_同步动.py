import numpy as np
from gym import spaces, logger
from math import sin, cos
import torch
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
import bisect
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import sys
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\ik_solver.py')
from ik_solver import IK_solver

joint_num = 81
#坐标系转换
height_base = 1.702 +0.205
# 部分碰撞参数
main_arm_len = 5.8705
push_pole_len = 2.590
push_pole_radius = 0.25
main_arm_radius = 0.25
joint_radius = 0.25
# safe_distance = 0.01
safe_distance = 0.15
arcade_num = 11
base2car_top = 1.000
base2car_side = 1.000
base2tunnel_side = 2.1
safe_scale = 0.15
arm = load_model_from_path('/home/yhq/桌面/DrillArmSimu/drillarm/xml/arm.xml')
sim = MjSim(arm)
viewer = MjViewer(sim)
viewer.cam.distance = 10
viewer.cam.azimuth = 190
viewer.cam.elevation = -30
viewer.cam.lookat[1] = -4.7
viewer._hide_overlay = True
arcade_point = []
with open("/home/yhq/桌面/DrillArmSimu/drill_data2.csv", 'r') as drill_data:
    data = list(csv.reader(drill_data))
    for i in range(arcade_num):
        arcade_point.append(list(map(float, data[46 + i][1:3])))
def mujoco_target(target):
    target_position = np.zeros(3)
    target_position[0] = target[1] + 1
    target_position[1] = -target[0]
    target_position[2] = target[2]
    return target_position
def mujoco_render(state, start_target, end_target):   #读取这个函数
    step = np.zeros(8)
    step[0] = -state[0]
    step[1] = -state[1] - 90 * np.pi / 180
    step[2] = -state[2] + 2.59
    step[3] = state[3] - 90 * np.pi / 180
    step[4] = state[4] + 90 * np.pi / 180
    step[5] = state[5] - 90* np.pi / 180
    step[6] = state[6]
    step[7] = 2.4665 - state[7]
    sim_state = sim.get_state()

    for i in range(8):
        sim_state.qpos[i] = step[i]
    sim.set_state(sim_state)
    sim.step()
    sim.set_state(sim_state)
    sim.forward()
    viewer_frame = viewer.render()
    sim.model.body_pos[9] = mujoco_target(start_target)
    sim.model.body_pos[11] = mujoco_target(end_target)

    return viewer_frame

"""pid"""

def interpolate_single_joint(start_joints, target_joints, joint_idx, T):
    joint_sequence = torch.zeros(T, start_joints.size(1))
    joint_sequence[:, joint_idx] = torch.linspace(start_joints[0, joint_idx], target_joints[0, joint_idx], T)
    joint_sequence[:, :joint_idx] = start_joints[0, :joint_idx]
    joint_sequence[:, joint_idx+1:] = start_joints[0, joint_idx+1:]
    return joint_sequence
def interpolate_multiple_joints(start_joints, target_joints, joint_idxs, T):
    num_joints = start_joints.size(1)
    joint_sequence = torch.zeros(T, num_joints)
    joint_sequence[0] = start_joints[0]
    for t in range(1, T):
        current_joints = joint_sequence[t - 1].clone()
        for idx in joint_idxs:
            target_inner_joints = target_joints.clone()
            target_inner_joints[0, :idx] = current_joints[:idx]

            current_joints[idx] = torch.lerp(start_joints[0, idx],
                                             target_joints[0, idx],
                                             t / (T - 1))
            for i in range(idx):
                current_joints[i] = torch.lerp(start_joints[0, i],
                                               target_inner_joints[0, i],
                                               t / (T - 1))
        joint_sequence[t] = current_joints
    return joint_sequence
def dh_matrix(alpha,a,d,theta):
    alpha = torch.true_divide(alpha, 180) * np.pi
    cos_theta = torch.cos(theta);sin_theta = torch.sin(theta)
    cos_alpha = torch.cos(alpha);sin_alpha = torch.sin(alpha)
    matrix = torch.zeros(( 4, 4), device='cpu')
    matrix[ 0, 0] = cos_theta
    matrix[0, 1] = -sin_theta
    matrix[0, 2] = 0
    matrix[ 0, 3] = a
    matrix[ 1, 0] = sin_theta * cos_alpha
    matrix[ 1, 1] = cos_theta * cos_alpha
    matrix[1, 2] = -sin_alpha
    matrix[1, 3] = -sin_alpha * d
    matrix[2, 0] = sin_theta * sin_alpha
    matrix[2, 1] = cos_theta * sin_alpha
    matrix[2, 2] = cos_alpha
    matrix[2, 3] = cos_alpha * d
    matrix[ 3, 0] = 0
    matrix[3, 1] = 0
    matrix[ 3, 2] = 0
    matrix[ 3, 3] = 1.0
    return matrix  # 返回旋转矩阵
def FORWARD1(output_denoise):
    joint_num = 8
    hole_depth = torch.tensor([[0.0], [0.0], [1]], device='cpu')  # 后续末端向量相乘
    hole_depth_1 = torch.tensor([[0.0], [0.0], [0.45]], device='cpu')  # 后续末端向量相乘
    joint_alpha = torch.tensor([[0., -90., -90., 90., -90., -90., 90., -90.]], device='cpu')
    joint_a = torch.tensor([[0, 0.16, 0.07, 0, 0.1334, 0, 0.15-0.035, 0.3625]], device='cpu')
    pose_V = torch.zeros((6,))  # 储存位姿差值
    q1 = output_denoise[0]
    q2 = output_denoise[1]
    d3 = output_denoise[2]
    q4 = output_denoise[3]
    q5 = output_denoise[4]
    q6 = output_denoise[5]
    q7 = output_denoise[6]
    d8 = output_denoise[7]
    # 正运动学计算
    joint_d = torch.tensor([[0, 0, d3, 0, -0.1316, 1.0105, 0.52, d8]], device='cpu')
    joint_theta = torch.tensor([[q1, q2, 0, q4, q5, q6, q7, 0]], device='cpu')

    joint_hm = []
    for j in range(joint_num):
        joint_hm.append(dh_matrix(joint_alpha[:, j], joint_a[:, j], joint_d[:, j], joint_theta[:, j]))
    # -----------连乘计算----------------------
    for j in range(joint_num - 1):
        joint_hm[j + 1] = torch.matmul(joint_hm[j], joint_hm[j + 1])
    end_poser = joint_hm[7]  # 最后将第八个变换矩阵给了末端，为4X4矩阵
    matrix_end = end_poser[:3, :3]  # 从 BB 中提取出前三行三列的子矩阵，并赋值给变量 B_3
    end_poser1 = torch.matmul(matrix_end, hole_depth)  # 生成孔内的投影
    end_poser2 = torch.matmul(matrix_end, hole_depth_1)  # 生成孔内的投影
    # 得到位姿向量
    pose_V[0] = end_poser[0, 3]
    pose_V[1] = end_poser[1, 3]
    pose_V[2] = end_poser[2, 3] + 1.702+0.205
    pose_V[3] = end_poser1[ 0, 0] + end_poser[ 0, 3]  # 将 L 的第一列第一行元素加上 BB 的第一列第四个元素，并将结果赋值给 BBB 的第四列。
    pose_V[4] = end_poser1[ 1, 0] + end_poser[ 1, 3]
    pose_V[5] = end_poser1[2, 0] + end_poser[ 2, 3] + 1.702+0.205

    return pose_V
def arcade_data( ):
    with open("/home/yhq/桌面/DrillArmSimu/drill_data2.csv", 'r') as drill_data:
        data = list(csv.reader(drill_data))
        arcade = []
        for i in range(11):
            arcade.append(list(map(float, data[46 + i][1:3])))  #提取CSV文件中第46+i行的第2和第3列数据（下标从0开始）
        return arcade
def collision(joint_tf,push_pole_len):   #检测杆8与关节1、2、3的碰撞
    joint_num = 8
    height_base = 1.702
    main_arm_len = 5.8705
    # push_pole_len = 2.590  # 杆长度
    push_pole_radius = 0.2  #杆1包络半径
    main_arm_radius_y = 0.2   # 杆2包络半径
    main_arm_radius_z = 0.15     # 杆2包络半径
    joint_radius = 0.2
    safe_distance = 0.01    #安全距离
    base2car_top = 1.000
    base2car_side = 1.000
    base2tunnel_side = 2.25
    safe_scale = 0.05  # 隧道碰撞安全量
    # push_pole_len = test3['joint_3'][2]   #这个控制状态来自哪里   得到test3之后大臂的数据
    xyz = np.array([dh[0:3, 3] for dh in joint_tf])  #从dh矩阵中提取出前3行和第四列的元素   8个
    col_lrwall = False
    col_ground = False
    col_body = False
    col_finalbody = False
    col_arch = False
    col_self = False  #都是布尔类型的标志，用于表示不同类型的碰撞情况。
    col_leg = False
    #表示出杆3、8的旋转矩阵
    matrix_3 = joint_tf[2][0:3, 0:3]  #索引2处的元素中提取的3行3列的子矩阵
    matrix_8 = joint_tf[7][0:3, 0:3]
    # 表示出杆3的dh坐标y,z方向单位向量
    y_3 = np.dot(matrix_3, np.transpose([0, 1, 0]))   #方向投影
    z_3 = np.dot(matrix_3, np.transpose([1, 0, 0]))  #关节坐标系中 x 方向的投影  ???
    # 杆3、8各自端点坐标
    line = np.zeros((4, 3))  #形状为(4, 3)的全零数组，表示四个点的坐标，每个点有三个坐标值（x、y、z）
    line[0] = xyz[1]   #得到各个端点的三维坐标，第二、四、七、八个关节
    line[1] = xyz[3]
    line[2] = xyz[6]
    line[3] = xyz[7]   #第四行   为第八个关节坐标
    # 单位向量
    main_arm_vector = np.dot(matrix_8, np.transpose([0, 0, -1]))  #反向z轴投影，与车身碰撞检测
    final_test_point = main_arm_vector * main_arm_len + line[3]   #self.main_arm_len = -5.8705+现在第八个关节的坐标   类似于孔深的算法
    line[2] = final_test_point   #得到最终的杆件8反向末端坐标
    # 将推进杆分段
    point8 = 0.001 * (final_test_point - line[3])    #还是回到了移动关节的投影，杆8的两段，  得到划分间隔的三维坐标  杆8拆分，它的各个点到杆3的距判定
    # 将8个点的坐标存入, 坐标系转换至隧道坐标系
    point_i = np.empty(18)    #18数组长度
    for i in range(joint_num):
        point_i[2 * i] = xyz[i][1]  #y坐标在基坐标系数组存储在偶数组
        point_i[2 * i + 1] = xyz[i][2] + height_base    #self.height_base = 1.702
    # 加入杆件末端点
    point_i[16] = final_test_point[1]      #引入杆件8的y坐标
    point_i[17] = final_test_point[2] + height_base     #z坐标
    # 左右壁面、地面碰撞检测
    for i in range(joint_num + 1):
        if abs(point_i[2 * i]) >= base2tunnel_side - joint_radius:  #两臂2.1-0.25=1.85m   2.05
            col_lrwall = True
            # print('col_lrwall',col_lrwall)
        if point_i[2 * i + 1] < 0.2-0.15:  #地面
            col_ground = True
    # 隧道拱顶碰撞检测,
    arcade_point = arcade_data()   #导入拱形桥数据
    # 检测对象：钻臂8个端点
    for m in range(joint_num + 1):
        # 遍历拱廊上11个点
        arcade_num = 11
        for i in range(arcade_num - 1):
            # 遍历拱廊相邻两点间等分点
            for j in range(4):
                point_plane = [j * 0.25 * (arcade_point[i + 1][0] - arcade_point[i][0]) + arcade_point[i][0],
                               j * 0.25 * (arcade_point[i + 1][1] - arcade_point[i][1]) + arcade_point[i][1]]  #精度0.25划分，新的数据就两列y、z
                if ((point_plane[0] - point_i[2 * m]) * point_plane[0] <= 0) and point_plane[1] <= point_i[
                    m * 2 + 1] + safe_scale:   #self.safe_scale = 0.15  # 隧道碰撞安全量  〖(O_iy-O〗_y)∙O_iy≤0且O_z≥O_iz来判断是否在隧道范围外，以简化计算量
                    col_arch = True   #
                    print('col_arch', col_arch)
    # 车身碰撞检测
    for i in range(joint_num):
        if xyz[i][0] < 0 and xyz[i][2] < base2car_top and abs(xyz[i][1]) < base2car_side:
            col_body = True
        # print('col_arch', col_arch)  #输出了
    if final_test_point[0] < 0.86 + 0.05 and abs(final_test_point[1]) < 1.5 and final_test_point[2] + height_base < 0.75:
        col_leg = True
    if final_test_point[0] < 0 and abs(final_test_point[1]) < 1 and final_test_point[2] + height_base < base2car_top:
        col_body = True
    for i in range(3):    #
        if max(line[0][i], line[1][i]) < min(final_test_point[i], line[3][i]) or min(line[0][i], line[1][i]) > max(
                final_test_point[i], line[3][i]):
            col_self = False
    for i in range(1000):
        point_detection = i * point8 + line[3] - line[0]  #第八个关节-第二个关节坐标+1000个沿着8关节的点
        distance_x = np.dot(point_detection, line[1] - line[0])   #二、四坐标距离*各个点，得到x方向距离
        distance_y = np.dot(point_detection, y_3)    #y方向的投影    点乘
        distance_z = np.dot(point_detection, z_3)  #z方向的投影
        safe_x = push_pole_len ** 2 + safe_distance   #259cm平方+1cm
        safe_y = push_pole_radius + main_arm_radius_y + safe_distance   #0.25+0.2+0.01=0.46  0.41
        safe_z = push_pole_radius + main_arm_radius_z + safe_distance  # 0.2+0.2+0.01=0.41    0.36
        if not (safe_x > distance_x > -safe_distance and   #  杆8的    -1到67082
                safe_y > distance_y > -safe_y and     #-51cm  到51cm     0.35
                safe_z > distance_z > -safe_z):    #0.25
            col_self = False
        else:
            col_self = True
            break
    # print('col_self',col_self)
    col_data = [col_lrwall,col_ground,col_body,col_leg,col_arch, col_self]
    col = col_lrwall or col_ground or col_body  or col_arch or col_self or col_leg
    return col,col_data

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = torch.zeros(8)
        self.prev_error = torch.zeros(8)

    def control(self, pose_current, pose_target):
        error = pose_target - pose_current

        error = error.detach().numpy().squeeze(0)
        error = torch.tensor(error)
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        control_output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control_output

def main(current_joints,current_pose , target_joints,target_pose):

    # PID控制器参数
    kp = np.array([1.])
    ki = np.array([0.00])
    kd = np.array([0.0])
    """控制顺序已经订好、避撞在轨迹生成"""
    pid_controller = PIDController(kp, ki, kd)
    """控制顺序、避撞"""

    T = 2 # 每个关节插值步数
    num_joints = current_joints.size(1)
    joint_order = [5, 2, (3, 1), (4, 0), 6, 7]
    min_error_threshold = 2
    collision_free = False
    best_trajectory = None
    best_target_joints = None
    joint_actual_trajectory = []
    predicted_trajectory = []
    ideal_trajectory = []
    while not collision_free:
        target_joints_total = IK_solver(target_pose)  # 生成3组解

        for i in range(len(target_joints_total)):
            target_joints = target_joints_total[i].flatten().tolist()
            joint_actual_trajectory = []
            collision_occurred = False

            for joint_idxs in joint_order:
                # 生成当前关节的插值序列(100,8)
                if isinstance(joint_idxs, tuple):
                    # 生成多个关节的插值序列
                    joint_sequence = interpolate_multiple_joints(current_joints, target_joints, joint_idxs, T)
                    # 对多个关节进行同时控制
                    actual_trajectory = torch.zeros(T, num_joints)
                    for t in range(T):
                        # 计算控制输出
                        control_output = pid_controller.control(current_joints, joint_sequence[t].unsqueeze(0))
                        current_joints += control_output
                        actual_trajectory[t] = current_joints
                        # 计算末端位姿
                        end_effector_pos = FORWARD1(current_joints[0])
                        predicted_trajectory.append(end_effector_pos)
                        # 计算理想末端位姿
                        joints_ideal = joint_sequence[t].unsqueeze(0)
                        ideal_pos = FORWARD1(joints_ideal[0])
                        ideal_trajectory.append(ideal_pos)
                    joint_actual_trajectory.append(actual_trajectory)
                    print(f"Joints {', '.join(str(idx + 1) for idx in joint_idxs)} control process completed.")

                else:
                    # 单个关节的控制逻辑保持不变
                    joint_idx = joint_idxs
                    joint_sequence = interpolate_single_joint(current_joints, target_joints, joint_idx, T)

                    actual_trajectory = torch.zeros(T, num_joints)
                    for t in range(T):
                        control_output = pid_controller.control(current_joints, joint_sequence[t].unsqueeze(0))
                        current_joints += control_output
                        actual_trajectory[t] = current_joints
                        end_effector_pos = FORWARD1(current_joints[0])
                        predicted_trajectory.append(end_effector_pos)
                        joints_ideal = joint_sequence[t].unsqueeze(0)
                        ideal_pos = FORWARD1(joints_ideal[0])
                        ideal_trajectory.append(ideal_pos)
                    joint_actual_trajectory.append(actual_trajectory)
                    print(f"Joint {joint_idx + 1} control process completed.")

            print("Final Joints:", current_joints)
            # 碰撞检测
            for num in range(len(joint_actual_trajectory)):
                collision_joint = joint_actual_trajectory[num][1]
                collision_joint_reshaped = torch.tensor(collision_joint.reshape(1, 8))
                pose_V, joint_hm, matrix_end, d3 = FORWARD1(collision_joint_reshaped.cuda())
                matrix_total = []
                for tensor in joint_hm:
                    matrix = tensor.detach().cpu().numpy()
                    matrix_total.append(np.squeeze(matrix))

                done = collision(matrix_total, d3)
                print(f"Collision check for target set {i + 1}, trajectory {num + 1}: {done}")
                if done:
                    print(f"Collision detected for trajectory {num + 1} in target set {i + 1}. Generating new targets...")
                    collision_occurred = True
                    break  # 退出当前循环，重新生成目标

                # 如果没有碰撞，进行误差检测
                if not collision_occurred:
                    print(f"Checking if total error < min_error_threshold for target set {i + 1}")
                    collision_joint_final = joint_actual_trajectory[-1][1]
                    collision_joint_final_reshaped = torch.tensor(collision_joint_final.reshape(1, 8))
                    pose_V_end, joint_hm, matrix_end, d3 = FORWARD1(collision_joint_final_reshaped.cuda())
                    # pose_V_end = pose_V_end.to(device)
                    moni_output = 100 * pose_V_end.detach().cpu().numpy()  # 模型输出的位姿向量cm
                    real_input = target_pose # 实际真实的位姿
                    dist0_0 = ((moni_output[:, 0] - real_input[:, 0]) ** 2) + (
                                (moni_output[:, 1] - real_input[:, 1]) ** 2) + (
                                      (moni_output[:, 2] - real_input[:, 2]) ** 2)
                    dist0_1 = np.sqrt(dist0_0)  # 起始点误差
                    dist1_0 = ((moni_output[:, 3] - real_input[:, 3]) ** 2) + (
                                (moni_output[:, 4] - real_input[:, 4]) ** 2) + (
                                      (moni_output[:, 5] - real_input[:, 5]) ** 2)
                    dist1_1 = np.sqrt(dist1_0)  # 末端点误差
                    total_error = dist0_1 + dist1_1  # 计算总误差
                    print(f"Total error for target set {i + 1}: {total_error}")

                    if total_error < min_error_threshold:
                        print("Total error is below the threshold. Returning best trajectory and target joints.")
                        best_trajectory = joint_actual_trajectory
                        best_target_joints = target_joints
                        collision_free = True  # 符合要求，跳出while循环
                        break
                    else:
                        print(f"Total error for target set {i + 1} exceeds threshold. Generating new targets...")
                        collision_free = False  # 错误超出阈值，继续寻找新解
    if best_trajectory is None or best_target_joints is None:
        print("No valid solution found, returning None.")

    print("Collision-free and precision-satisfactory solution found.")

    return predicted_trajectory, ideal_trajectory,joint_actual_trajectory,  current_pose  , target_pose
T = 2
all_states = []
if __name__ == "__main__":
        current_joints = torch.from_numpy([]).float().unsqueeze(0)
        target_joints = torch.from_numpy([] ).float().unsqueeze(0)
        target_pose_INITIAL = FORWARD1(current_joints)
        target_pose_end =FORWARD1(target_joints)

        predicted_trajectory, ideal_trajectory,joint_actual_trajectory, current_pose, target_pose = main(current_joints,target_pose_INITIAL, target_joints, target_pose_end)

        images = []  # 存储所有的图像帧
        for joint_idx in range(len(joint_actual_trajectory)):
            joint_states = joint_actual_trajectory[joint_idx]
            for t in range(T):

                state = joint_states[t]  # 获取实际的关节状态
                state = state.numpy() # 转换为 NumPy 数组并压缩维度

                start_target = current_pose
                end_target = target_pose
                start_target = start_target.numpy().squeeze(0)
                end_target = end_target.numpy().squeeze(0)
                print('state=',state)

                viewer_frame = mujoco_render(state, start_target, end_target)
                images.append(viewer_frame)
                time.sleep(0.1)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制预测的轨迹线
        # 遍历预测轨迹列表中的每个张量
        for trajectory in predicted_trajectory:
            # 将张量转换为 NumPy 数组
            trajectory_np = trajectory.numpy()
            # 绘制当前轨迹
            ax.plot(trajectory_np[0], trajectory_np[1], trajectory_np[ 2], marker='o', markersize=5,
                    label='Fact Trajectory')
        for i in range(len(ideal_trajectory) - 1):
            current_ideal = ideal_trajectory[i]
            next_ideal = ideal_trajectory[i + 1]
            ax.plot([current_ideal[0], next_ideal[0]],
                    [current_ideal[1], next_ideal[1]],
                    [current_ideal[2], next_ideal[2]],
                    marker='o', markersize=2, c='orange', label='Ideal Trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 调整图像布局，避免切割标题和标签
        # ax.legend()
        plt.tight_layout()
        # 将张量转换为 NumPy 数组
        # 保存图像为高分辨率PNG文件
        plt.savefig('1_avoid.png', dpi=600)
        plt.show()