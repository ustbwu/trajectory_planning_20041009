import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from math import sin, cos
import time


active_links_mask = [False] + [True] * 7

joints_alpha = [0, -90 * np.pi / 180, 90 * np.pi / 180, 90 * np.pi / 180, -90 * np.pi / 180, 90 * np.pi / 180,
                -90 * np.pi / 180]
joints_a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
joints_d = [0.1695, 0.0, 0.1155, 0.0, 0.12783, 0.0, 0.06598]

def dh_matrix(alpha, a, d, theta):
    matrix = np.array([[cos(theta), -sin(theta), 0, a],
                       [sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -d * sin(alpha)],
                       [sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), d * cos(alpha)],
                       [0, 0, 0, 1]])
    return matrix

def joint_dh(joints):
    joints_theta = [joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], joints[6]]
    joint_dh = []
    for i in range(7):
        joint_dh.append(dh_matrix(joints_alpha[i], joints_a[i], joints_d[i], joints_theta[i]))
    for i in range(7 - 1):
        joint_dh[i + 1] = np.dot(joint_dh[i], joint_dh[i + 1])
    return joint_dh

my_chain = Chain(name='my_chain', links=[
    OriginLink(),
    URDFLink(
      name="joint_1",
      origin_translation=[0, 0, 0.1695],
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 1],
    ),
    URDFLink(
      name="joint_2",
      origin_translation=[0, 0, 0.1695],
      origin_orientation=[-1.57, 0, 0],
      rotation=[0, 0, 1],
    ),
    URDFLink(
      name="joint_3",
      origin_translation=[0, 0, 0.285],
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 1],
    ),
    URDFLink(
      name="joint_4",
      origin_translation=[0, 0, 0.285],
      origin_orientation=[1.57, 0, 0],
      rotation=[0, 0, 1],
    ),
    URDFLink(
      name="joint_5",
      origin_translation=[0, 0, 0.41283],
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 1],
    ),
    URDFLink(
      name="joint_6",
      origin_translation=[0, 0, 0.41283],
      origin_orientation=[1.57, 0, 0],
      rotation=[0, 0, 1],
    ),
    URDFLink(
      name="joint_7",
      origin_translation=[0, 0, 0.47881],
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 1],
    )
],
    active_links_mask = active_links_mask)
target_position = [-0.002009894,	0.002737329,	0.478251064]
# target_position = [0., 0., 0.]
t1 = time.process_time_ns()
list_input = my_chain.inverse_kinematics(target_position)[1:]  # 从第二个元素开始到列表末尾
t2 = time.process_time_ns()
# 将这些元素转换为字符串
list_output = [element for element in list_input]
print("The angles of each joints are : ", list_output)

dh = joint_dh(list_output)
print('time_test(ms)', (t2 - t1)/1000000)
print('target', target_position)

print('final', dh[6][:3, 3])
pos_error = np.linalg.norm(dh[6][:3, 3]-target_position)
print('pos_error(cm)', pos_error*100)