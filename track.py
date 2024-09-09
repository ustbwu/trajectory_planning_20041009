import ctypes
import numpy as np
import onnxruntime

#调用车身基坐标系相对于巷道基坐标系（0,0，0）的转移矩阵M
import sys
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\nagavation.py')
from nagavation import compute_transfer_matrix
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\ik_solver.py')
from ik_solver import IK_solver
sys.path.append('D:\\taiche\\mpc\\trajectory_planning_20240905\\collision_detect.py')
from collision_detect import collision
"""示例"""
joint_initial = np.array([ -8.660458,  -79.42416  ,   3.440263*1000 ,  77.7766   , -98.93389  , 61.22102,
    4.9999924  , 3.130865* 1000], dtype=np.double)
# joint_initial = np.array([0, -90, 2590, 90, -90.0, 90, 0, 2466.5], dtype=np.double)
joint_initial[2] = joint_initial[2]/1000
joint_initial[7] = joint_initial[7]/1000
"""#调用车身基坐标系相对于巷道基坐标系（0,0，0）的转移矩阵M"""
bias = 0.
data = np.array([[bias, 1.5, 0.0,bias, 1.5 ,-1]], dtype=np.float32)
M  = compute_transfer_matrix(data,joint_initial)
a = np.dot(M[:3,:3],[0,0,1])
#M_inv = np.linalg.inv(M)
M_inv = np.linalg.inv(M)

# 加载 DLL文件  生成轨迹控制量
dll_length = ctypes.cdll.LoadLibrary("C:\\Users\\DELL\Desktop\\算法_mm\\new_v2_705\\track\\track.dll")
dll = ctypes.cdll.LoadLibrary("C:\\Users\\DELL\\Desktop\\算法_mm\\new_v2_705\\track\\track_length.dll")
# Example usage  弧度制、m[ 8.1070611  -0.79259148  2.07882368]
# current_joints = [ 0.29540455 ,-1.69349253 , 3.09224749  ,1.59738731 ,-1.3882612  , 0.47943187,  0.08726644  ,2.50093508]
current_joints =[ -8.660458,  -79.42416  ,   3.440263*1000 ,  77.7766   , -98.93389  , 61.22102,
    4.9999924  , 3.130865* 1000]

"""传感器读数"""
a = np.array([[0.744,2.5,0,0.744,2.5,1]], dtype=np.float32)   #单位m
a[:,-1] = a[:,-1]*-1
a = a*100
a[0,:3] = np.dot( M_inv[:3,:3],a[0,:3])
a[0,3:6] = np.dot(M_inv[:3,:3],a[0,3:6])
a[0, 0] = a[0, 0] + 780
a[0, 3] = a[0, 3] + 780

ort_inputs = {"input": a}
"""  八个结果为单位弧度和m"""
target_joints = IK_solver(a)
target_joints_array = target_joints[0]
target_joints_array[0,0] = target_joints_array[0,0]*180/np.pi
target_joints_array[0,1] = target_joints_array[0,1]*180/np.pi
target_joints_array[0,2] = target_joints_array[0,2]*1000
target_joints_array[0,3] = target_joints_array[0,3]*180/np.pi
target_joints_array[0,4] = target_joints_array[0,4]*180/np.pi
target_joints_array[0,5] = target_joints_array[0,5]*180/np.pi
target_joints_array[0,6] = target_joints_array[0,6]*180/np.pi
target_joints_array[0,7] = target_joints_array[0,7]*1000
print(target_joints_array[0,7])
target_joints_array[0,7] = target_joints_array[0,7]
target_joints = target_joints_array.flatten().tolist()
print('target_joints',target_joints)
"""生成控制量部分"""
dll_length.SetJoints((ctypes.c_double * len(current_joints))(*current_joints),
              (ctypes.c_double * len(target_joints))(*target_joints),
              len(current_joints))
dll_length.GetTrajectory_length.argtypes = []
dll_length.GetTrajectory_length.restype = ctypes.c_int
fact_trajectory_length = dll_length.GetTrajectory_length()
# 输出轨迹长度和类型
print("需要的轨迹长度:", fact_trajectory_length)
print("轨迹长度的数据类型:", type(fact_trajectory_length))

dll.SetJoints.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
dll.SetJoints.restype = None
dll.GetTrajectory.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
dll.GetTrajectory.restype = ctypes.POINTER(ctypes.c_double)
# 设置当前和目标关节信息
dll.SetJoints((ctypes.c_double * len(current_joints))(*current_joints),
              (ctypes.c_double * len(target_joints))(*target_joints),
              len(current_joints))
# 生成轨迹
num_trajectories = ctypes.c_double(0)
trajectory_length = ctypes.c_double(0)
num_joints = ctypes.c_int(0)
trajectory_ptr = dll.GetTrajectory(ctypes.byref(num_trajectories),
                                    ctypes.byref(trajectory_length),
                                    ctypes.byref(num_joints))
trajectory_length = int(trajectory_length.value)
num_joints = num_joints.value
# 将指针转换为NumPy数组
trajectory_array = np.ctypeslib.as_array(trajectory_ptr, shape=(int(num_trajectories.value), int(trajectory_length * num_joints)))
trajectory_array = np.reshape(trajectory_array, (int(num_trajectories.value), int(trajectory_length), num_joints))

print("需要控制的关节个数:", num_joints)
print("长度:", trajectory_length)
print("生成的关节轨迹信息:",trajectory_array)