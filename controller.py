import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\nagavation.py')
from nagavation import compute_transfer_matrix
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\ik_solver.py')
from ik_solver import IK_solver
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\collision_detect.py')
from collision_detect import collision
from DDIM import FORWARD
device = torch.device("cuda")
# 单关节插值
def interpolate_single_joint(start_joints, target_joints, joint_idx, T):
    joint_sequence = np.zeros((T, len(start_joints)))
    joint_sequence[:, joint_idx] = np.linspace(start_joints[joint_idx], target_joints[joint_idx], T)
    joint_sequence[:, :joint_idx] = start_joints[:joint_idx]
    joint_sequence[:, joint_idx + 1:] = start_joints[joint_idx + 1:]
    return joint_sequence

# 双关节插值
def interpolate_multiple_joints(start_joints, target_joints, joint_idxs, T):
    num_joints = len(start_joints)
    joint_sequence = np.zeros((T, num_joints))
    joint_sequence[0] = start_joints

    for t in range(1, T):
        current_joints = joint_sequence[t - 1].copy()
        for idx in joint_idxs:
            target_inner_joints = target_joints.copy()
            target_inner_joints[:idx] = current_joints[:idx]

            current_joints[idx] = np.linspace(start_joints[idx], target_joints[idx], T)[t]
            for i in range(idx):
                current_joints[i] = np.linspace(start_joints[i], target_inner_joints[i], T)[t]

        joint_sequence[t] = current_joints
    return joint_sequence

# 最简单的PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(8)
        self.prev_error = np.zeros(8)

    def control(self, pose_current, pose_target):
        error = pose_target - pose_current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        control_output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control_output

def main(current_joints, a):
    kp = np.array([1.])
    ki = np.array([0.00])
    kd = np.array([0.0])
    pid_controller = PIDController(kp, ki, kd)
    T = 2
    num_joints = len(current_joints)
    joint_order = [5, 2, (3, 1), (4, 0), 6, 7]  # 控制顺序
    min_error_threshold = 2
    collision_free = False

    # 初始化 best_trajectory 和 best_target_joints
    best_trajectory = None
    best_target_joints = None

    while not collision_free:
        target_joints_total = IK_solver(a)  # 生成3组解

        for i in range(len(target_joints_total)):
            target_joints = target_joints_total[i].flatten().tolist()
            joint_actual_trajectory = []
            collision_occurred = False

            # 生成实际的关节轨迹
            for joint_idxs in joint_order:
                if isinstance(joint_idxs, tuple):
                    joint_sequence = interpolate_multiple_joints(current_joints, target_joints, joint_idxs, T)
                    actual_trajectory = np.zeros((T, num_joints))
                    for t in range(T):
                        control_output = pid_controller.control(current_joints, joint_sequence[t])
                        current_joints += control_output
                        actual_trajectory[t] = current_joints
                    joint_actual_trajectory.append(actual_trajectory)
                    print(f"Joints {', '.join(str(idx + 1) for idx in joint_idxs)} control process completed.")
                else:
                    joint_idx = joint_idxs
                    joint_sequence = interpolate_single_joint(current_joints, target_joints, joint_idx, T)
                    actual_trajectory = np.zeros((T, num_joints))
                    for t in range(T):
                        control_output = pid_controller.control(current_joints, joint_sequence[t])
                        current_joints += control_output
                        actual_trajectory[t] = current_joints
                    joint_actual_trajectory.append(actual_trajectory)
                    print(f"Joint {joint_idx + 1} control process completed.")

            # 碰撞检测
            for num in range(len(joint_actual_trajectory)):
                collision_joint = joint_actual_trajectory[num][1]
                collision_joint_reshaped = torch.tensor(collision_joint.reshape(1, 8))
                pose_V, joint_hm, matrix_end, d3 = FORWARD(collision_joint_reshaped.cuda(), batchsize=1)
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
                pose_V_end, joint_hm, matrix_end, d3 = FORWARD(collision_joint_final_reshaped.cuda(), batchsize=1)
                pose_V_end = pose_V_end.to(device)
                moni_output = 100 * pose_V_end.detach().cpu().numpy()  # 模型输出的位姿向量cm
                real_input = a  # 实际真实的位姿
                dist0_0 = ((moni_output[:, 0] - real_input[:, 0]) ** 2) + ((moni_output[:, 1] - real_input[:, 1]) ** 2) + (
                        (moni_output[:, 2] - real_input[:, 2]) ** 2)
                dist0_1 = np.sqrt(dist0_0)  # 起始点误差
                dist1_0 = ((moni_output[:, 3] - real_input[:, 3]) ** 2) + ((moni_output[:, 4] - real_input[:, 4]) ** 2) + (
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
    return best_trajectory, best_target_joints



if __name__ == "__main__":
    """示例"""
    joint_initial = np.array([-8.660458, -79.42416, 3.440263 * 1000, 77.7766, -98.93389, 61.22102,
                              4.9999924, 3.130865 * 1000], dtype=np.double)
    # joint_initial = np.array([0, -90, 2590, 90, -90.0, 90, 0, 2466.5], dtype=np.double)
    joint_initial[2] = joint_initial[2] / 1000
    joint_initial[7] = joint_initial[7] / 1000
    """#调用车身基坐标系相对于巷道基坐标系（0,0，0）的转移矩阵M"""
    bias = 0.
    data = np.array([[bias, 1.5, 0.0, bias, 1.5, -1]], dtype=np.float32)
    M = compute_transfer_matrix(data, joint_initial)
    # M_inv = np.linalg.inv(M)
    M_inv = np.linalg.inv(M)
    current_joints = [-8.660458, -79.42416, 3.440263 , 77.7766, -98.93389, 61.22102,
                      4.9999924, 3.130865 ]

    current_joints[0] = current_joints[0] /180*np.pi
    current_joints[1] = current_joints[1] /180*np.pi
    current_joints[3] = current_joints[3] /180*np.pi
    current_joints[4] = current_joints[4] /180*np.pi
    current_joints[5] = current_joints[5] /180*np.pi
    current_joints[6] = current_joints[6] /180*np.pi

    """传感器读数"""
    a = np.array([[0.744, 2.5, 0, 0.744, 2.5, 1]], dtype=np.float32)  # 单位m
    a[:, -1] = a[:, -1] * -1
    a = a * 100
    a[0, :3] = np.dot(M_inv[:3, :3], a[0, :3])
    a[0, 3:6] = np.dot(M_inv[:3, :3], a[0, 3:6])
    a[0, 0] = a[0, 0] + 780
    a[0, 3] = a[0, 3] + 780
    print(a)

    # 调用主函数处理
    joint_actual_trajectory ,target_joints = main(current_joints, a)
    print("joint_actual_trajectory:",joint_actual_trajectory)
    print("target_joints_rads:",target_joints)
    target_joints[0] = target_joints[0] *180/np.pi
    target_joints[1] = target_joints[1] *180/np.pi
    target_joints[3] = target_joints[3] *180/np.pi
    target_joints[4] = target_joints[4] *180/np.pi
    target_joints[5] = target_joints[5] *180/np.pi
    target_joints[6] = target_joints[6] *180/np.pi
    print("target_joints_degrees:",target_joints)
    aaa = 123



