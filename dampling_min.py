import numpy as np
from scipy.linalg import pinv
from math import radians, sin, cos
import time
from mujoco.drill_boom.tidutouying import start_time, end_time


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


def apply_joint_limits(thetas):
    return np.array([np.clip(theta, low, high) for theta, (low, high) in zip(thetas, joint_limits)])


def jacobian(theta):
    """
    Compute the Jacobian matrix for a 7-DOF robot arm.
    """
    J = np.zeros((6, 7))
    delta = 1e-6

    for i in range(7):
        theta_plus = theta.copy()
        theta_plus[i] += delta
        pos_plus, angle_plus = calculate_transform_matrix(theta_plus)

        theta_minus = theta.copy()
        theta_minus[i] -= delta
        pos_minus, angle_minus = calculate_transform_matrix(theta_minus)

        J[:3, i] = (np.array(pos_plus) - np.array(pos_minus)) / (2 * delta)
        J[3:, i] = (np.array(angle_plus) - np.array(angle_minus)) / (2 * delta)

    return J


def damped_least_squares_ik(target_pose, initial_guess, max_iterations=1000, tolerance=1e-4, damping=0.1):
    """
    Solve inverse kinematics using the Damped Least Squares method with joint limits.
    """
    thetas = apply_joint_limits(initial_guess)

    for _ in range(max_iterations):
        current_pose, current_angles = calculate_transform_matrix(thetas)

        error = np.zeros(6)
        error[:3] = target_pose[:3] - current_pose
        error[3:] = target_pose[3:] - current_angles

        if np.linalg.norm(error) < tolerance:
            break

        J = jacobian(thetas)  #计算当前关节角度下的雅可比矩阵。
        JT = J.T
        delta_theta = JT @ pinv(J @ JT + damping ** 2 * np.eye(6)) @ error  #使用阻尼最小二乘法更新关节角度，公式为：
        thetas = apply_joint_limits(thetas + delta_theta)

    return thetas

# Example usage
if __name__ == "__main__":
    # Set a target pose [x, y, z, rx, ry, rz]
    target_pose = np.array([-0.022441229,	-0.018804293,	0.47633795,	-0.020022569	,0.097867918	,-3.064458542
])
    start_time = time.time()
    # Initial guess for joint angles
    initial_guess = np.array([0, -np.pi / 4 , 0, np.pi / 2 , 0, np.pi / 4, 0])
    # initial_guess = np.array([0, 0, 0, 0, 0,0, 0])

    # Solve IK
    solution = damped_least_squares_ik(target_pose, initial_guess)

    print("IK solution (joint angles in radians):")
    print(solution)

    # Verify the solution
    final_pose, final_angles = calculate_transform_matrix(solution)
    print("\nResulting end-effector pose:")
    print(final_pose)
    print("\nResulting end-effector angles:")
    print(final_angles)

    print("\nTarget pose:")
    print(target_pose)
    end_time  =time.time()
    time_delt = end_time-start_time
    print("time_delt:",time_delt)
    print("\nPose error:")
    print(np.linalg.norm(final_pose - target_pose[:3]))
    print(np.linalg.norm(final_angles - target_pose[3:]))

    print("\nJoint limits check:")
    for i, (theta, (low, high)) in enumerate(zip(solution, joint_limits)):
        print(f"Joint {i + 1}: {theta:.4f} rad, Limit: [{low:.4f}, {high:.4f}] rad")




