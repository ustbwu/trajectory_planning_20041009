import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import datetime
import time
import os
import ctypes
import torch.nn.functional as F
os.environ["OMP_NUM_THREADS"] = str(6)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
device = torch.device("cpu")
def dh_matrix(alpha,a,d,theta):
    alpha = torch.true_divide(alpha, 180) * np.pi
    cos_theta = torch.cos(theta);sin_theta = torch.sin(theta)
    cos_alpha = torch.cos(alpha);sin_alpha = torch.sin(alpha)
    matrix = torch.zeros((4, 4))
    matrix[0, 0] = cos_theta
    matrix[0, 1] = -sin_theta
    matrix[ 0, 2] = 0
    matrix[ 0, 3] = a
    matrix[1, 0] = sin_theta * cos_alpha
    matrix[1, 1] = cos_theta * cos_alpha
    matrix[1, 2] = -sin_alpha
    matrix[ 1, 3] = -sin_alpha * d
    matrix[2, 0] = sin_theta * sin_alpha
    matrix[ 2, 1] = cos_theta * sin_alpha
    matrix[ 2, 2] = cos_alpha
    matrix[ 2, 3] = cos_alpha * d
    matrix[ 3, 0] = 0
    matrix[ 3, 1] = 0
    matrix[ 3, 2] = 0
    matrix[3, 3] = 1.0
    return matrix  # 返回旋转矩阵
def FORWARD(output_denoise):
    joint_num=8
    hole_depth = torch.tensor([0.0 ,0.0 ,1])
    joint_alpha = torch.tensor([0., -90., -90., 90., -90., -90., 90., -90.])
    joint_a = torch.tensor([0, 0.16, 0.07, 0, 0.1334, 0, 0.15-0.035, 0.3625])
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
    joint_d = torch.stack([torch.tensor(0), torch.tensor(0), d3, torch.tensor(0), torch.tensor(-0.1316), torch.tensor(1.0105), torch.tensor(0.52), d8])
    joint_theta = torch.stack(  [q1, q2, torch.tensor(0), q4, q5, q6, q7,torch.tensor(0)])
    joint_hm = []
    for j in range(joint_num):
        joint_hm.append(dh_matrix(joint_alpha[j], joint_a[ j], joint_d[j], joint_theta[ j]))
    # -----------连乘计算----------------------
    for j in range(joint_num - 1):
        joint_hm[j + 1] = torch.matmul(joint_hm[j], joint_hm[j + 1])
    end_poser = joint_hm[7]  # 最后将第八个变换矩阵给了末端，为4X4矩阵
    matrix_end = end_poser[:3, :3]  # 从 BB 中提取出前三行三列的子矩阵，并赋值给变量 B_3
    end_poser1 = torch.matmul(matrix_end, hole_depth)  # 生成孔内的投影
    # 得到位姿向量
    pose_V[0] = end_poser[ 0, 3]  # 将L 的第一列四个元素
    pose_V[1] = end_poser[ 1, 3]  # 将L 的第二列四个元素
    pose_V[2] = end_poser[2, 3] + 1.702+0.205  #
    pose_V[3] = end_poser1[  0] + end_poser[ 0, 3]  # 将 L 的第一列第一行元素加上 BB 的第一列第四个元素，并将结果赋值给 BBB 的第四列。
    pose_V[4] = end_poser1[ 1] + end_poser[ 1, 3]
    pose_V[5] = end_poser1[ 2] + end_poser[ 2, 3] + 1.702+0.205
    return pose_V , joint_hm, matrix_end  ,d3

def improved_annealed_langevin_dynamics(s0, action, n_steps):
    device = action.device
    v = torch.zeros_like(action).to(device)
    beta2 = 0.95
    epsilon = 1e-8  # Smaller epsilon for numerical stability
    # Adaptive learning rate parameters
    initial_lr = 1e-3
    min_lr = 1e-5
    # Noise schedule parameters
    initial_noise_scale = 1e-4
    final_noise_scale = 1e-6
    best_action = action.clone()
    best_loss = float('inf')
    joint_high = torch.tensor([35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                               -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]).to(device)
    joint_low = torch.tensor([-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                              -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]).to(device)
    # 定义约束参数
    height = 1.702+0.205
    d_ground = 0.05+0.205
    d_wall = 1.95
    safety_distance_y = 0.43 + 0.10
    safety_distance_z = 0.36 + 0.10

    # 拱顶约束的多项式系数
    y_data = np.array([-2.1, -2.003, -1.605, -1.107, -0.565, 0, 0.565, 1.107, 1.605, 2.003, 2.1])
    z_data = np.array([2.4, 2.975, 3.454, 3.726, 3.893, 3.95, 3.893, 3.726, 3.454, 2.975, 2.4])
    coefficients = np.polyfit(y_data, z_data, 10)

    def compute_constraints(action):
        pose_V, joint_hm, matrix, d3 = FORWARD(action)

        # 提取需要的数据
        matrix_3 = joint_hm[2][:3, :3]
        y_zhou = torch.tensor([0.0, 1.0, 0.0])
        z_zhou = torch.tensor([1.0, 0.0, 0.0])
        zf_zhou = torch.tensor([0.0, 0.0, -1.0])
        y_3 = torch.matmul(matrix_3, y_zhou)
        z_3 = torch.matmul(matrix_3, z_zhou)
        z_8_reverse = torch.matmul(matrix, zf_zhou)

        dh = [joint_hm[i] for i in range(8)]
        dh_end = [dh1[0:3, 3] for dh1 in dh]
        dh_end_point = [dh2 for dh2 in dh_end]

        # 拱顶约束
        y_point = dh_end_point[7][1].cpu()
        z_point = (dh_end_point[7][2] + height).cpu()
        mask = ((abs(y_point) <= 2.1) & (z_point > 2.4))
        selected_y = y_point[mask]
        selected_z = z_point[mask]
        if selected_y.numel() > 0:
            z_fit = np.polyval(coefficients, selected_y.detach().numpy().flatten())
            constrain_arch_z = torch.relu(selected_z - torch.tensor(z_fit) ).mean()
        else:
            constrain_arch_z = torch.tensor(0.0).to(device)

        # 地面约束
        constrain_ground_z = sum(torch.relu(d_ground - dh[  2] - height).mean() for dh in dh_end_point)

        # 墙面约束
        loss_wall_mask = torch.zeros( 1, 1, dtype=torch.bool, device=device)
        for dh in dh_end_point:
            loss_wall_mask |= (abs(dh[ 1]) > d_wall) & (dh[ 2] + height < 2.4)
        constrain_wall_y = sum(
            torch.relu(abs(dh[1]) - d_wall).mean() for dh in dh_end_point) * loss_wall_mask.float().mean()

        # 自身碰撞约束
        start_point8 = z_8_reverse * 5.8705 + dh_end_point[7]
        point8 = (start_point8 - dh_end_point[7]) * 0.001
        weights = [999, 950, 920, 900, 880, 850, 820, 800, 750, 700, 650]
        distances_y = []
        distances_z = []
        for weight in weights:
            point_direction = weight * point8 + dh_end_point[7] - dh_end_point[1]
            distances_y.append(torch.sum(point_direction * y_3))
            distances_z.append(torch.sum(point_direction* z_3))

        distances_y = torch.stack(distances_y)  # 形状将是 (11,)
        distances_z = torch.stack(distances_z)  # 形状将是 (11,)
        loss_self_mask = (safety_distance_y > abs(distances_y)) & (safety_distance_z > abs(distances_z))
        constrain_distances_y = F.softplus(safety_distance_y - torch.abs(distances_y))
        constrain_distances_z = F.softplus(safety_distance_z - torch.abs(distances_z))
        constrain_self = ((constrain_distances_z + constrain_distances_y) * loss_self_mask.float()).mean()

        return constrain_arch_z + constrain_ground_z + constrain_wall_y + constrain_self

    for ite in range(n_steps):
        # Compute score (gradient)
        action.requires_grad_(True)
        pose_V, _, _, _ = FORWARD(action)
        pose_V = pose_V.to(device)
        loss = F.mse_loss(100 * pose_V, s0) + compute_constraints(action)
        grad = torch.autograd.grad(loss.mean(), action, create_graph=True)[0]
        with torch.no_grad():
            # Update momentum term
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            v_hat = v / (1 - beta2 ** (ite + 1))
            # Adaptive learning rate
            lr = initial_lr * min(ite / (n_steps * 0.1), (1 - ite / n_steps) ** 0.5) + min_lr
            # Gradient clipping
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(grad, max_grad_norm)
            # Noise schedule
            noise_scale = initial_noise_scale * (1 - ite / n_steps) ** 2 + final_noise_scale
            # Update action
            step_size = lr / (torch.sqrt(v_hat) + epsilon)
            proposed_action = action - step_size * grad + torch.rand_like(action) * noise_scale
            # Enforce joint limits
            within_limits = torch.logical_and(proposed_action <= joint_high, proposed_action >= joint_low)
            action = torch.where(within_limits, proposed_action, action)
            # Check for improvement
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_action = action.clone()
            # Early stopping
            if current_loss < 1e-4:
                print(f"Converged at iteration {ite}")
                break
            # Periodic resetting to best known solution (to escape local minima)
            if ite % 25 == 0 and ite > 0:
                action = best_action.clone() + torch.rand_like(action) * noise_scale
                # action = best_action.clone()
        action = action.detach()  # Detach for next iteration
        if ite % 10 == 0:
            print(f"Iteration {ite}, Loss: {current_loss:.6f}, LR: {lr:.6f}, Noise: {noise_scale:.6f}")
    return best_action

def IK_solver(x):
    x = torch.tensor(x)

    output_denoise_list = []
    start_point = [];end_point = []
    joint_high = torch.tensor([35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                               -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]).to(device)
    joint_low = torch.tensor([-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                              -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]).to(device)
    posev_list = []
    n_step  = 1
    start_time = time.time()
    for j in range(1,n_step+1):
        noise = torch.rand(8, device=device)
        noise = noise * (joint_high - joint_low) + joint_low
        input_data = x[0].to(device)
        samples = improved_annealed_langevin_dynamics(input_data, noise, n_steps=2500)
        pose_V, joint_tf, matrix_end, d3 = FORWARD(samples)
        pose_V = pose_V.to(device)
        samples = samples.detach().cpu().numpy()
        moni_optput = 100 * pose_V.detach().cpu().numpy()  # 模型输出的位姿向量cm
        real_input = input_data.detach().cpu().numpy()  # 实际真实的位姿
        dist0_0 = ((moni_optput[0] - real_input[0]) ** 2) + ((moni_optput[1] - real_input[1]) ** 2) + (
                (moni_optput[2] - real_input[2]) ** 2)
        dist0_1 = np.sqrt(dist0_0)  # 起始点
        dist1_0 = ((moni_optput[3] - real_input[3]) ** 2) + ((moni_optput[4] - real_input[4]) ** 2) + (
                (moni_optput[5] - real_input[5]) ** 2)
        dist1_1 = np.sqrt(dist1_0)  # 末端点
        start_point.append(dist0_1)
        end_point.append(dist1_1)
        output_denoise_list.append(samples)
        posev_list.append(pose_V.detach().cpu().numpy())
        end_time = time.time()
        output_denoise_list.append(samples)
        posev_list.append(pose_V.detach().cpu().numpy())
        end_time = time.time()
        print('第{}个样本的运行时间为{}s'.format(j, end_time - start_time))
    return output_denoise_list



