import ctypes
import numpy as np
def compute_transfer_matrix(data,joint_value):
    data_start = data[0, :3]
    # 固定点相对于巷道基坐标系的转移矩阵A
    STABLE = data_start
    ground = np.array([[0, 0, 0]], dtype=np.float32)
    # 计算从地面到固定点的平移向量
    translation_vector = STABLE - ground
    # 构造转换矩阵A
    A = np.eye(4)  # 单位矩阵
    A[:3, 3] = translation_vector.flatten()  # 将平移向量赋值给矩阵的最后一列
    print('固定点（0.0 , 1.5 , 0.0）相对于巷道基坐标系（0 , 0 , 0）的转移矩阵 A:\n', A)

    # 钻臂末端坐标相对于固定点的转移矩阵B
    """回转角度对应的转移矩阵"""
    theta_6 = joint_value[5]* np.pi/180
    theta_z =  -theta_6     # 绕z轴顺时针旋转theta
    # 绕Z轴旋转矩阵
    B_1 = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                    [np.sin(theta_z), np.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    theta_y = np.pi   # 绕y轴逆时针旋转90°
    # 绕y轴旋转矩阵
    B_2 = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                    [0, 0, 0, 1]])
    # theta_x = np.pi   # 绕x轴逆时针旋转90°
    # # 绕x轴旋转矩阵
    # B_3 = np.array([[1, 0, 0, 0],
    #                 [0, np.cos(theta_x), -np.sin(theta_x), 0],
    #                 [0, np.sin(theta_x), np.cos(theta_x), 0],
    #                 [0, 0, 0, 1]])
    # B_trans = np.dot(B_3,np.dot(B_1, B_2))
    # B_trans  = np.dot(B_2,B_3)
    # B_trans  = np.dot(B_1,B_3)
    B_trans = np.dot(B_1,B_2)
    print('钻臂末端点坐标相对于固定点的转移矩阵B:\n', B_trans)

    dll = ctypes.CDLL("C:\\Users\\DELL\\Desktop\\算法_mm\\8.10self_constrain\\forward.dll")
    # Define the function signature
    dll.calculateTransformMatrix.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, shape=(4, 4)),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(3,)),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(3,)),
    ]
    dll.calculateTransformMatrix.restype = None
    # Create output arrays
    tf_end = np.zeros((4, 4), dtype=np.double)
    pos_z = np.zeros(3, dtype=np.double)
    pos_z_end = np.zeros(3, dtype=np.double)
    # Call the function
    dll.calculateTransformMatrix(joint_value, tf_end, pos_z, pos_z_end)

    """末端点相对车身的转移矩阵"""
    C = tf_end
    """车身/末端点转移矩阵"""
    C_inv = np.linalg.inv(C)


    M = np.dot(np.dot(B_trans, A), C_inv)
    print('末端点相对车身基坐标系:\n',C )
    print('车身基坐标系相对于巷道基坐标系:\n', M)
    return M  

if __name__ == "__main__":
    """示例使用   断面的固定点坐标"""

    data = np.array([[0.0, 1.5, 0.0, 0.0, 1.5 ,-1]], dtype=np.float32)
    """读到的关节值"""
    joint_initial = np.array([-0.1, -91.1, 3204, 88.61, -90.0, 26.18, 0, 2880], dtype=np.double)
    joint_initial[2] = joint_initial[2]/1000
    joint_initial[7] = joint_initial[7]/1000
    M   = compute_transfer_matrix(data,joint_initial)
    # M_inv = np.linalg.inv(M)
    # print(M_inv)
