import numpy as np
import time
from scipy.linalg import pinv
from math import radians, sin, cos
def euler_angles_from_rotation_matrix(matrix):
    if matrix[2, 0] != 1 and matrix[2, 0] != -1:
        ry1 = -np.arcsin(matrix[2, 0])
        ry2 = np.pi - ry1
        rx1 = np.arctan2(matrix[2, 1]/np.cos(ry1), matrix[2, 2]/np.cos(ry1))
        rx2 = np.arctan2(matrix[2, 1]/np.cos(ry2), matrix[2, 2]/np.cos(ry2))
        rz1 = np.arctan2(matrix[1, 0]/np.cos(ry1), matrix[0, 0]/np.cos(ry1))
        rz2 = np.arctan2(matrix[1, 0]/np.cos(ry2), matrix[0, 0]/np.cos(ry2))
        return [rx1, ry1, rz1], [rx2, ry2, rz2]
    else:
        rz = 0
        if matrix[2, 0] == -1:
            ry = np.pi/2
            rx = rz + np.arctan2(matrix[0, 1], matrix[0, 2])
        else:
            ry = -np.pi/2
            rx = -rz + np.arctan2(-matrix[0, 1], -matrix[0, 2])
        return [rx, ry, rz], None
def dh_matrix(alpha, a, d, theta):
    #
    # theta = theta / 180 * np.pi
    matrix = np.identity(4)
    matrix[0, 0] = cos(theta)
    matrix[0, 1] = -sin(theta)
    matrix[0, 2] = 0
    matrix[0, 3] = a
    matrix[1, 0] = sin(theta) * cos(alpha)
    matrix[1, 1] = cos(theta) * cos(alpha)
    matrix[1, 2] = -sin(alpha)
    matrix[1, 3] = - d * sin(alpha)
    matrix[2, 0] = sin(theta) * sin(alpha)
    matrix[2, 1] = cos(theta) * sin(alpha)
    matrix[2, 2] = cos(alpha)
    matrix[2, 3] = d * cos(alpha)
    matrix[3, 0] = 0
    matrix[3, 1] = 0
    matrix[3, 2] = 0
    matrix[3, 3] = 1
    return matrix
def calculate_transform_matrix(joint_values):
    joint_num = len(joint_values)
    joints_alpha= [0, -90*np.pi/180, 90*np.pi/180, 90*np.pi/180, -90*np.pi/180, 90*np.pi/180, -90*np.pi/180]
    joints_a = [0, 0, 0, 0, 0, 0, 0]
    q1 = joint_values[0]
    q2 = joint_values[1]
    q3 = joint_values[2]
    q4 = joint_values[3]
    q5 = joint_values[4]
    q6 = joint_values[5]
    q7 = joint_values[6]
    joints_d = [0.1695, 0.0, 0.1155, 0.0, 0.12783, 0.0, 0.06598]

    joints_theta = [q1, q2, q3, q4, q5, q6, q7]

    tf = []
    for i in range(joint_num):
        tf.append(dh_matrix(joints_alpha[i], joints_a[i], joints_d[i], joints_theta[i]))
    for j in range(joint_num - 1):
        tf[j + 1] = np.dot(tf[j], tf[j + 1])
    pos_z = [0, 0, 0]

    tf_end = tf[6]
    # 求点坐标
    pos_z[0] = tf_end[0, 3]
    pos_z[1] = tf_end[1, 3]
    pos_z[2] = tf_end[2, 3]
    rotation_matrix = tf_end[:3, :3]
    # Calculate Euler angles
    euler_angles, _ = euler_angles_from_rotation_matrix(rotation_matrix)
    return pos_z , euler_angles


joint_limits = [
    (-160 * np.pi / 180, 160 * np.pi / 180),
    (-80 * np.pi / 180, 80 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180),
    (-100 * np.pi / 180, 80 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180),
    (-110 * np.pi / 180, 110 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180)
]


def calculate_jacobian(joint_values):   #关节限位避免的指标。
    epsilon = 1e-6
    jacobian = np.zeros((6, 7))

    for i in range(7):
        q_plus = joint_values.copy()
        q_plus[i] += epsilon
        q_minus = joint_values.copy()
        q_minus[i] -= epsilon
        pos_plus, ori_plus = calculate_transform_matrix(q_plus)
        pos_minus, ori_minus = calculate_transform_matrix(q_minus)

        jacobian[:3, i] = (np.array(pos_plus) - np.array(pos_minus)) / (2 * epsilon)
        jacobian[3:, i] = (np.array(ori_plus) - np.array(ori_minus)) / (2 * epsilon)

    return jacobian


# def performance_index(q):
#     # 这里使用关节限位避免作为性能指标
#     q_mid = np.mean(joint_limits, axis=1)
#     q_range = np.diff(joint_limits, axis=1).flatten()
#     return -np.sum(((q - q_mid) / q_range) ** 2)


def performance_index_gradient(q):
    q_mid = np.mean(joint_limits, axis=1)
    q_range = np.diff(joint_limits, axis=1).flatten()
    return -2 * (q - q_mid) / (q_range ** 2)


def gradient_projection_ik(target_pos, target_ori, initial_guess, alpha=0.01, max_iterations=1000, tolerance=1e-4):
    q = np.array(initial_guess)

    for iteration in range(max_iterations):
        current_pos, current_ori = calculate_transform_matrix(q)

        error = np.concatenate([
            np.array(target_pos) - np.array(current_pos),
            np.array(target_ori) - np.array(current_ori)
        ])

        if np.linalg.norm(error) < tolerance:
            print(f"Converged after {iteration} iterations")
            return q

        J = calculate_jacobian(q)
        J_pinv = np.linalg.pinv(J)

        # 计算性能指标梯度
        gradient_H = performance_index_gradient(q)

        # 计算步长，包括主任务和次要任务
        step_primary = np.dot(J_pinv, error)
        step_secondary = alpha * np.dot(np.eye(7) - np.dot(J_pinv, J), gradient_H)
        step = step_primary + step_secondary

        # 更新关节角度
        q_new = q + step

        # 投影到关节限制范围内
        q_new = np.clip(q_new, [limit[0] for limit in joint_limits], [limit[1] for limit in joint_limits])

        # 更新 q
        q = q_new

    print("Max iterations reached without convergence")
    return q


# 使用示例
# initial_guess = [0, 0, 0, 0, 0, 0, 0]  # 初始猜测值
initial_guess = [0, -np.pi / 4, 0, np.pi / 2, 0, np.pi / 4, 0]
target_pos = [0.14832457601338145, 0.12390496113374928, 0.16395866041350055]  # 目标位置
target_ori = [-3.1402235926706665, -0.005804820187516991, -2.4448900417466883]  # 目标姿态（欧拉角）

start_time = time.time()
result = gradient_projection_ik(target_pos, target_ori, initial_guess)
end_time = time.time()

print("Solved joint values:", result)
print("Execution time:", end_time - start_time, "seconds")

# 验证结果
final_pos, final_ori = calculate_transform_matrix(result)
print("Final position:", final_pos)
print("Final orientation:", final_ori)
print("Position error:", np.linalg.norm(np.array(target_pos) - np.array(final_pos)))
print("Orientation error:", np.linalg.norm(np.array(target_ori) - np.array(final_ori)))