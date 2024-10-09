import numpy as np
from math import cos, sin
import PyKDL

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

def create_new_robot_chain():
    chain = PyKDL.Chain()

    # DH parameters
    alpha = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2]
    a = [0, 0, 0, 0, 0, 0, 0]
    d = [0.1695, 0.0, 0.1155, 0.0, 0.12783, 0.0, 0.06598]
    theta = [0, 0, 0, 0, 0, 0, 0]  # Initial joint angles

    for i in range(7):
        chain.addSegment(PyKDL.Segment(
            PyKDL.Joint(PyKDL.Joint.RotZ),
            PyKDL.Frame(
                PyKDL.Rotation.RotZ(theta[i]) * PyKDL.Rotation.RotX(alpha[i]),
                PyKDL.Vector(a[i], -d[i] * sin(alpha[i]), d[i] * cos(alpha[i]))
            )
        ))

    return chain


def compute_forward_kinematics(chain, joint_angles):
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    q = PyKDL.JntArray(7)
    for i in range(7):
        q[i] = joint_angles[i]

    end_effector_frame = PyKDL.Frame()
    fk_flag = fk.JntToCart(q, end_effector_frame)

    position = end_effector_frame.p
    rotation = end_effector_frame.M

    return np.array([position.x(), position.y(), position.z()]), rotation


def compute_inverse_kinematics(chain, target_pose):
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    ik_v = PyKDL.ChainIkSolverVel_pinv(chain)
    ik = PyKDL.ChainIkSolverPos_NR(chain, fk, ik_v)

    target_frame = PyKDL.Frame(
        PyKDL.Rotation.RPY(target_pose[3], target_pose[4], target_pose[5]),
        PyKDL.Vector(target_pose[0], target_pose[1], target_pose[2])
    )

    initial_guess = PyKDL.JntArray(chain.getNrOfJoints())
    result = PyKDL.JntArray(chain.getNrOfJoints())

    # Solve IK
    ret = ik.CartToJnt(initial_guess, target_frame, result)

    if ret >= 0:
        # Convert KDL JntArray to Python list
        joint_angles = [result[i] for i in range(chain.getNrOfJoints())]
        return joint_angles
    else:
        print("IK solution not found")
        return None


def apply_joint_limits(joint_angles, joint_limits):
    return [max(min(angle, upper), lower) for angle, (lower, upper) in zip(joint_angles, joint_limits)]


if __name__ == "__main__":
    # Create robot chain
    chain = create_new_robot_chain()

    # Set target pose [x, y, z, roll, pitch, yaw]
    target_pose = [0.5, 0.3, 0.4, 0.1, 0.0, 0.05]

    # Compute inverse kinematics
    joint_angles = compute_inverse_kinematics(chain, target_pose)

    if joint_angles:
        print("Computed joint angles:", joint_angles)

        # Apply joint limits
        joint_limits = [
            (-160 * np.pi / 180, 160 * np.pi / 180),
            (-80 * np.pi / 180, 80 * np.pi / 180),
            (-165 * np.pi / 180, 165 * np.pi / 180),
            (-100 * np.pi / 180, 80 * np.pi / 180),
            (-165 * np.pi / 180, 165 * np.pi / 180),
            (-110 * np.pi / 180, 110 * np.pi / 180),
            (-165 * np.pi / 180, 165 * np.pi / 180)
        ]
        limited_joint_angles = apply_joint_limits(joint_angles, joint_limits)
        print("Joint angles after applying limits:", limited_joint_angles)

        # Verify result with forward kinematics
        position, rotation = compute_forward_kinematics(chain, limited_joint_angles)
        print("Resulting end-effector position:", position)
        euler_angles, _ = euler_angles_from_rotation_matrix(rotation)
        print("Resulting end-effector orientation (Euler angles):", euler_angles)

        print("Pose error:", np.linalg.norm(np.array(target_pose[:3]) - position))
        print("Orientation error:", np.linalg.norm(np.array(target_pose[3:]) - euler_angles))
    else:
        print("Failed to find an IK solution")