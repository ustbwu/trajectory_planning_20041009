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
device = torch.device("cuda")
from DDIM import FORWARD
# def FORWARD(joint_value):
#     dll = ctypes.CDLL("C:\\Users\\DELL\\Desktop\\算法_mm\\8.10self_constrain\\forward.dll")
#     # Define the function signature
#     dll.calculateTransformMatrix.argtypes = [
#         np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
#         np.ctypeslib.ndpointer(dtype=np.double, ndim=2, shape=(4, 4)),
#         np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(3,)),
#         np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(3,)),
#     ]
#     dll.calculateTransformMatrix.restype = None
#     # Create output arrays
#     tf_end = np.zeros((4, 4), dtype=np.double)
#     pos_z = np.zeros(3, dtype=np.double)
#     pos_z_end = np.zeros(3, dtype=np.double)
#     # Call the function
#     dll.calculateTransformMatrix(joint_value, tf_end, pos_z, pos_z_end)
#     return pos_z,pos_z_end

def improved_annealed_langevin_dynamics(s0, action, n_steps):
    device = action.device
    v = torch.zeros_like(action).to(device)
    beta2 = 0.95
    epsilon = 1e-8  # Smaller epsilon for numerical stability
    # Adaptive learning rate parameters
    initial_lr = 1e-3
    min_lr = 1e-5
    # Noise schedule parameters
    initial_noise_scale = 1e-3
    final_noise_scale = 1e-5
    best_action = action.clone()
    best_loss = float('inf')
    for ite in range(n_steps):
        # Compute score (gradient)
        action.requires_grad_(True)
        pose_V , _,_,_ = FORWARD(action, batchsize=action.size(0))

        pose_V = pose_V.to(device)
        loss = F.mse_loss(100 * pose_V, s0)
        grad = torch.autograd.grad(loss.mean(), action, create_graph=True)[0]
        joint_high = torch.tensor([[35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                                    -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]]).to(device)
        joint_low = torch.tensor([[-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                                   -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]]).to(device)
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
            proposed_action = action - step_size * grad + torch.randn_like(action) * noise_scale
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
            if ite % 50 == 0 and ite > 0:
                action = best_action.clone() + torch.randn_like(action) * noise_scale * 0.1
        action = action.detach()  # Detach for next iteration
        if ite % 10 == 0:
            print(f"Iteration {ite}, Loss: {current_loss:.6f}, LR: {lr:.6f}, Noise: {noise_scale:.6f}")
    return best_action

def IK_solver(x):
    x = torch.tensor(x)
    joint_high = torch.tensor([[35 * np.pi / 180, -60 * np.pi / 180, 394 / 100, 155 * np.pi / 180,
                                    -55 * np.pi / 180, 180 * np.pi / 180, 5 * np.pi / 180, 371 / 100]]).to(device)
    joint_low = torch.tensor([[-35 * np.pi / 180, -155 * np.pi / 180, 259 / 100, 60 * np.pi / 180,
                                   -125 * np.pi / 180, -180 * np.pi / 180, -90 * np.pi / 180, 250 / 100]]).to(device)
    output_denoise_list = []
    start_point = [];end_point = []

    posev_list = []
    n_step  = 1
    start_time = time.time()
    for j in range(1,n_step+1):
                batchsize1 = len(x)
                noise = torch.rand(batchsize1, 8, device=device)
                noise = noise * (joint_high - joint_low) + joint_low
                input_data = x[0].to(device)
                samples = improved_annealed_langevin_dynamics(input_data, noise, n_steps=2500)
                pose_V , joint_hm, matrix_end  ,d3 = FORWARD(samples, batchsize=batchsize1)

                pose_V = pose_V.to(device)
                samples = samples.detach().cpu().numpy()
                moni_optput = 100 * pose_V.detach().cpu().numpy() # 模型输出的位姿向量cm
                real_input = x.detach().cpu().numpy()  # 实际真实的位姿
                dist0_0 = ((moni_optput[:, 0] - real_input[:, 0]) ** 2) + ((moni_optput[:, 1] - real_input[:, 1]) ** 2) + (
                        (moni_optput[:, 2] - real_input[:, 2]) ** 2)
                dist0_1 = np.sqrt(dist0_0)  # 起始点
                dist1_0 = ((moni_optput[:, 3] - real_input[:, 3]) ** 2) + ((moni_optput[:, 4] - real_input[:, 4]) ** 2) + (
                        (moni_optput[:, 5] - real_input[:, 5]) ** 2)
                dist1_1 = np.sqrt(dist1_0)  # 末端点
                start_point.append(dist0_1)
                end_point.append(dist1_1)
                output_denoise_list.append(samples)
                posev_list.append(pose_V.detach().cpu().numpy())
                end_time = time.time()
                print('第{}个样本的运行时间为{}s'.format(j, end_time - start_time))
    return output_denoise_list
