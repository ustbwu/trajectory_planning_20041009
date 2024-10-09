import numpy as np
import time
from scipy.linalg import pinv
from math import radians, sin, cos
def dh_matrix(alpha, a, d, theta):
    alpha = alpha / 180 * np.pi
    matrix = np.identity(4)
    matrix[0, 0] = cos(theta)
    matrix[0, 1] = -sin(theta)
    matrix[0, 2] = 0
    matrix[0, 3] = a
    matrix[1, 0] = sin(theta) * cos(alpha)
    matrix[1, 1] = cos(theta) * cos(alpha)
    matrix[1, 2] = -sin(alpha)
    matrix[1, 3] = -sin(alpha) * d
    matrix[2, 0] = sin(theta) * sin(alpha)
    matrix[2, 1] = cos(theta) * sin(alpha)
    matrix[2, 2] = cos(alpha)
    matrix[2, 3] = cos(alpha) * d
    matrix[3, 0] = 0
    matrix[3, 1] = 0
    matrix[3, 2] = 0
    matrix[3, 3] = 1
    return matrix
def calculate_transform_matrix(joint_values):
    joint_num = len(joint_values)
    joints_alpha = [0, -90, -90, 90, -90, -90, 90, -90]
    joints_a = [0, 0.16, 0.07, 0, 0.1334, 0.0, 0.15-0.035, 0.3625]
    d3 = joint_values[2]  # assuming d3 is the 3rd value
    d8 = joint_values[7]  # assuming d8 is the 8th value
    q1 = joint_values[0]
    q2 = joint_values[1]
    q4 = joint_values[3]
    q5 = joint_values[4]
    q6 = joint_values[5]
    q7 = joint_values[6]
    joints_d = [0, 0, d3, 0, -0.1316, 1.0105, 0.52, d8]
    joints_theta = [q1, q2, 0, q4, q5, q6, q7, 0]

    tf = []
    for i in range(joint_num):
        tf.append(dh_matrix(joints_alpha[i], joints_a[i], joints_d[i], joints_theta[i]))

    for j in range(joint_num - 1):
        tf[j + 1] = np.dot(tf[j], tf[j + 1])
    pos_z=[0,0,0]  ;pos_z_end=[0,0,0]
    tf_end = tf[7]
    tf_6 = tf[5]

    #求点坐标
    pos_z[0] = tf_end[0, 3]
    pos_z[1] = tf_end[1, 3]
    pos_z[2] = tf_end[2, 3]+1.702+0.205
    hole = [0,0,1]
    pos_end = np.dot(tf_end[:3, :3],hole)
    pos_z_end[0] =  tf_end[0, 3]  + pos_end[0,]
    pos_z_end[1] =  tf_end[1, 3]  + pos_end[1,]
    pos_z_end[2] =  tf_end[2, 3]  + pos_end[2,] +1.702+0.205
    return pos_z ,pos_z_end



joint_limits = [
    (-35 * np.pi / 180, 35 * np.pi / 180),
    (-155 * np.pi / 180, -60 * np.pi / 180),
    (259 / 100, 394/100),
    (60 * np.pi / 180, 155 * np.pi / 180),
    (-125 * np.pi / 180, -55 * np.pi / 180),
    (-180 * np.pi / 180, 180 * np.pi / 180),
    (-90 * np.pi / 180, 5 * np.pi / 180),
    (250 / 100, 371 / 100)
]


def calculate_jacobian(joint_values):
    epsilon = 1e-6
    jacobian = np.zeros((6, 8))

    for i in range(8):
        q_plus = joint_values.copy()
        q_plus[i] += epsilon
        q_minus = joint_values.copy()
        q_minus[i] -= epsilon

        pos_plus, ori_plus = calculate_transform_matrix(q_plus)
        pos_minus, ori_minus = calculate_transform_matrix(q_minus)

        jacobian[:3, i] = (np.array(pos_plus) - np.array(pos_minus)) / (2 * epsilon)
        jacobian[3:, i] = (np.array(ori_plus) - np.array(ori_minus)) / (2 * epsilon)

    return jacobian


def gradient_projection_ik(target_pos, target_ori, initial_guess, max_iterations=10000, tolerance=1e-4):
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

        # Compute pseudoinverse of Jacobian
        J_pinv = np.linalg.pinv(J)

        # Compute step
        step = np.dot(J_pinv, error)

        # Update joint values
        q_new = q + step

        # Project onto joint limits
        for i in range(8):
            q_new[i] = np.clip(q_new[i], joint_limits[i][0], joint_limits[i][1])

        # Update q
        q = q_new

    print("Max iterations reached without convergence")
    return q


# 使用示例
initial_guess = [0, -90* np.pi / 180, 2.59, 90* np.pi / 180, -90* np.pi / 180,90* np.pi / 180, 0,2.5]  # 初始猜测值
# initial_guess = [0, -np.pi / 4, 0, np.pi / 2, 0, np.pi / 4, 0]
target_pos = [7, -0.2, 1.4]  # 目标位置
target_ori = [8, -0.2, 1.4]  # 目标姿态（欧拉角）

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