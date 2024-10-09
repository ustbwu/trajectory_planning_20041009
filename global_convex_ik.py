import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 机械臂参数
n_joints = 7  # 机械臂的关节数量
n_discretize = 20  # 每个关节角度离散化为20个值

# 关节范围（弧度）
joint_limits = [
    (-160 * np.pi / 180, 160 * np.pi / 180),
    (-80 * np.pi / 180, 80 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180),
    (-100 * np.pi / 180, 80 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180),
    (-110 * np.pi / 180, 110 * np.pi / 180),
    (-165 * np.pi / 180, 165 * np.pi / 180)
]


def create_discretized_angles():
    return [np.linspace(lower, upper, n_discretize) for lower, upper in joint_limits]

# Piecewise linear approximation function (unchanged)
def create_piecewise_linear_approximation(joint_limit, n_segments):
    lower, upper = joint_limit
    breakpoints = np.linspace(lower, upper, n_segments + 1)
    return breakpoints

# Optimization model creation with updated objective function and params
def create_optimization_model(target, discretized_angles):
    model = gp.Model("Global_IK_Solver")
    # Binary variables
    z = {}
    for i in range(n_joints):
        for j in range(n_discretize):
            z[i, j] = model.addVar(vtype=GRB.BINARY, name=f'z_{i}_{j}')

    # Continuous variables for joint angles
    theta = model.addVars(n_joints, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")

    # Each joint must select one discrete angle
    for i in range(n_joints):
        model.addConstr(gp.quicksum(z[i, j] for j in range(n_discretize)) == 1)

    # Joint angle relationship with binary variables
    for i in range(n_joints):
        model.addConstr(theta[i] == gp.quicksum(z[i, j] * discretized_angles[i][j] for j in range(n_discretize)))

    # End effector position variables
    end_effector = model.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="end_effector")

    # Piecewise linear approximation for the transform
    breakpoints = [create_piecewise_linear_approximation(joint_limit, n_discretize - 1) for joint_limit in joint_limits]
    for i in range(n_joints):
        pwl_y = [calculate_transform_matrix([0] * i + [bp] + [0] * (n_joints - i - 1)) for bp in breakpoints[i]]
        for j in range(3):  # x, y, z
            model.addGenConstrPWL(theta[i], end_effector[j], breakpoints[i], [y[j] for y in pwl_y])

    # Joint limits constraints
    for i in range(n_joints):
        model.addConstr(theta[i] >= joint_limits[i][0])
        model.addConstr(theta[i] <= joint_limits[i][1])

    # Objective function: minimize squared L2 error
    error = model.addVars(3, lb=0, name="error")
    for i in range(3):
        model.addConstr(error[i] >= end_effector[i] - target[i])
        model.addConstr(error[i] >= target[i] - end_effector[i])

    # Minimizing squared error
    model.setObjective(gp.quicksum(error[i] * error[i] for i in range(3)), GRB.MINIMIZE)

    # Gurobi parameters
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = 60  # Set a 60-second time limit

    return model, theta

# Solver for global inverse kinematics
def solve_global_ik(target):
    discretized_angles = create_discretized_angles()
    model, theta_vars = create_optimization_model(target, discretized_angles)

    # Solve the model
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = [theta_vars[i].X for i in range(n_joints)]
        return np.array(solution)
    else:
        print("No solution found")
        return None

def visualize(theta, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制机械臂
    positions = calculate_transform_matrix(theta)
    ax.plot([0, positions[0]], [0, positions[1]], [0, positions[2]], 'bo-')

    # 绘制目标点
    ax.plot([target[0]], [target[1]], [target[2]], 'r*', markersize=15)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('机械臂全局逆运动学')
    plt.show()


def main():
    target = np.array([0.006187, -0.00082, 0.478507])  # 目标位置
    print("目标位置:", target)

    solution = solve_global_ik(target)

    if solution is not None:
        print("求解的关节角度:", solution)
        final_position = calculate_transform_matrix(solution)
        print("最终末端执行器位置:", final_position)
        print("误差:", np.linalg.norm(np.array(final_position) - target))
        visualize(solution, target)
    else:
        print("未找到解")


if __name__ == "__main__":
    main()