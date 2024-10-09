import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from math import cos, sin

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
    joints_alpha = [0, -90*np.pi/180, 90*np.pi/180, 90*np.pi/180, -90*np.pi/180, 90*np.pi/180, -90*np.pi/180]
    joints_a = [0, 0, 0, 0, 0, 0, 0]
    joints_d = [0.1695, 0.0, 0.1155, 0.0, 0.12783, 0.0, 0.06598]
    joints_theta = joint_values

    tf = []
    for i in range(joint_num):
        tf.append(dh_matrix(joints_alpha[i], joints_a[i], joints_d[i], joints_theta[i]))
    for j in range(joint_num - 1):
        tf[j + 1] = np.dot(tf[j], tf[j + 1])

    tf_end = tf[6]
    pos_z = tf_end[:3, 3]
    rotation_matrix = tf_end[:3, :3]

    euler_angles, _ = euler_angles_from_rotation_matrix(rotation_matrix)

    return tf_end, pos_z, euler_angles

# 创建7自由度机械臂链
my_chain = Chain(name='arm', links=[
    OriginLink(),
    URDFLink(
        name="joint1",
        origin_translation=[0, 0, 0.1695],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
    ),
    URDFLink(
        name="joint2",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, -np.pi/2, 0],
        rotation=[0, 1, 0],
    ),
    URDFLink(
        name="joint3",
        origin_translation=[0, 0.1155, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 1, 0],
    ),
    URDFLink(
        name="joint4",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, np.pi/2, 0],
        rotation=[0, 1, 0],
    ),
    URDFLink(
        name="joint5",
        origin_translation=[0, 0.12783, 0],
        origin_orientation=[0, -np.pi/2, 0],
        rotation=[0, 0, 1],
    ),
    URDFLink(
        name="joint6",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, np.pi/2, 0],
        rotation=[0, 1, 0],
    ),
    URDFLink(
        name="joint7",
        origin_translation=[0, 0.06598, 0],
        origin_orientation=[0, -np.pi/2, 0],
        rotation=[0, 0, 1],
    )
])

# 设置目标位置和方向
target_position = [0.006187395,	-0.000816779	,0.478506591
]
target_orientation = [	0.01986855,	-0.092714762,	2.572186617
]

try:
    # 进行逆运动学求解，仅使用目标位置
    ik_solution = my_chain.inverse_kinematics(target_position)

    # 使用自定义函数计算实际达到的位置和方向
    _, actual_position, actual_orientation = calculate_transform_matrix(ik_solution[1:])

    # 计算位置误差和方向误差
    position_error = np.linalg.norm(np.array(target_position) - actual_position)
    orientation_error = np.linalg.norm(np.array(target_orientation) - actual_orientation)

    print(f"IK Solution (joint angles in radians):\n{ik_solution[1:]}")
    print(f"\nTarget Position: {target_position}")
    print(f"Actual Position: {actual_position}")
    print(f"Position Error: {position_error:.6f}")
    print(f"\nTarget Orientation: {target_orientation}")
    print(f"Actual Orientation: {actual_orientation}")
    print(f"Orientation Error: {orientation_error:.6f}")

except ValueError as e:
    print(f"Error occurred during inverse kinematics calculation: {e}")
    print("Please check your target position and make sure it is within the robot's workspace.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")