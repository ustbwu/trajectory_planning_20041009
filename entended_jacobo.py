import numpy as np
from scipy.linalg import pinv


def euler_angles_from_rotation_matrix(matrix):
    if matrix[2, 0] != 1 and matrix[2, 0] != -1:
        ry1 = -np.arcsin(matrix[2, 0])
        ry2 = np.pi - ry1
        rx1 = np.arctan2(matrix[2, 1] / np.cos(ry1), matrix[2, 2] / np.cos(ry1))
        rx2 = np.arctan2(matrix[2, 1] / np.cos(ry2), matrix[2, 2] / np.cos(ry2))
        rz1 = np.arctan2(matrix[1, 0] / np.cos(ry1), matrix[0, 0] / np.cos(ry1))
        rz2 = np.arctan2(matrix[1, 0] / np.cos(ry2), matrix[0, 0] / np.cos(ry2))
        return [rx1, ry1, rz1], [rx2, ry2, rz2]
    else:
        rz = 0
        if matrix[2, 0] == -1:
            ry = np.pi / 2
            rx = rz + np.arctan2(matrix[0, 1], matrix[0, 2])
        else:
            ry = -np.pi / 2
            rx = -rz + np.arctan2(-matrix[0, 1], -matrix[0, 2])
        return [rx, ry, rz], None


def dh_matrix(alpha, a, d, theta):
    matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])
    return matrix


def calculate_transform_matrix(joint_values):
    joint_num = len(joint_values)
    joints_alpha = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2]
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
    return pos_z, euler_angles


def calculate_jacobian(joint_values):
    delta = 1e-6
    jacobian = np.zeros((6, 7))

    for i in range(7):
        q_plus = joint_values.copy()
        q_plus[i] += delta
        q_minus = joint_values.copy()
        q_minus[i] -= delta

        pos_plus, euler_plus = calculate_transform_matrix(q_plus)
        pos_minus, euler_minus = calculate_transform_matrix(q_minus)

        jacobian[:3, i] = (pos_plus - pos_minus) / (2 * delta)
        jacobian[3:, i] = (np.array(euler_plus) - np.array(euler_minus)) / (2 * delta)

    return jacobian


def inverse_kinematics_augmented_jacobian(target_pos, target_euler, initial_guess, joint_limits, max_iterations=1000,
                                          tolerance=1e-6):
    current_joints = np.array(initial_guess)

    for iteration in range(max_iterations):
        current_pos, current_euler = calculate_transform_matrix(current_joints)

        error = np.concatenate([
            target_pos - current_pos,
            np.array(target_euler) - np.array(current_euler)
        ])

        if np.linalg.norm(error) < tolerance:
            return current_joints

        J = calculate_jacobian(current_joints)

        # Augment Jacobian with joint limit constraints
        J_aug = np.zeros((6 + 7, 7))
        J_aug[:6, :] = J
        error_aug = np.zeros(6 + 7)
        error_aug[:6] = error

        for i in range(7):
            if current_joints[i] < joint_limits[i][0]:
                J_aug[6 + i, i] = -1
                error_aug[6 + i] = joint_limits[i][0] - current_joints[i]
            elif current_joints[i] > joint_limits[i][1]:
                J_aug[6 + i, i] = 1
                error_aug[6 + i] = joint_limits[i][1] - current_joints[i]

        delta_q = np.dot(pinv(J_aug), error_aug)

        new_joints = current_joints + delta_q

        if np.allclose(new_joints, current_joints, atol=tolerance):
            return new_joints

        current_joints = new_joints

    print("Warning: Max iterations reached without convergence")
    return current_joints


# Example usage
joint_limits = [
    (-160 * np.pi / 180, 160 * np.pi / 180),
    (-80 * np.pi / 180, 80 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180),
    (-100 * np.pi / 180, 80 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180),
    (-110 * np.pi / 180, 110 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180)
]

# Set a target pose
target_position = np.array([0.14832457601338145, 0.12390496113374928, 0.16395866041350055])
target_euler = np.array([-3.1402235926706665, -0.005804820187516991, -2.4448900417466883])  # Example Euler angles

# Set an initial guess
initial_guess = np.zeros(7)


# Solve inverse kinematics
solution = inverse_kinematics_augmented_jacobian(target_position, target_euler, initial_guess, joint_limits)

print("Inverse Kinematics Solution:")
print(solution)

# Verify the solution
final_pos, final_euler = calculate_transform_matrix(solution)
print("\nFinal Position:")
print(final_pos)
print("\nFinal Euler Angles:")
print(final_euler)
print("Position error:", np.linalg.norm(np.array(target_position) - np.array(final_pos)))
print("Orientation error:", np.linalg.norm(np.array(target_euler) - np.array(final_euler)))