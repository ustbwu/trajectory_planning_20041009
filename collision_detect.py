import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import time
import csv
def arcade_data( ):
    with open("D:\\taiche\\data_craete\\new_NN\\drill_data2.csv", 'r') as drill_data:
        data = list(csv.reader(drill_data))
        arcade = []
        for i in range(11):
            arcade.append(list(map(float, data[46 + i][1:3])))  #提取CSV文件中第46+i行的第2和第3列数据（下标从0开始）
        return arcade

def collision(joint_tf,push_pole_len):   #检测杆8与关节1、2、3的碰撞
    joint_num = 8
    height_base = 1.702 + 0.205
    main_arm_len = 5.8705
    # push_pole_len = 2.590  # 杆长度
    push_pole_radius = 0.2  #杆1包络半径
    main_arm_radius_y = 0.2   # 杆2包络半径
    main_arm_radius_z = 0.15     # 杆2包络半径
    joint_radius = 0.2
    safe_distance = 0.01    #安全距离
    base2car_top = 1.000
    base2car_side = 1.000
    base2tunnel_side = 2.25
    safe_scale = 0.05  # 隧道碰撞安全量
    # push_pole_len = test3['joint_3'][2]   #这个控制状态来自哪里   得到test3之后大臂的数据
    xyz = np.array([dh[0:3, 3] for dh in joint_tf])  #从dh矩阵中提取出前3行和第四列的元素   8个
    col_lrwall = False
    col_ground = False
    col_body = False
    col_finalbody = False
    col_arch = False
    col_self = False  #都是布尔类型的标志，用于表示不同类型的碰撞情况。
    col_leg = False
    #表示出杆3、8的旋转矩阵
    matrix_3 = joint_tf[2][0:3, 0:3]  #索引2处的元素中提取的3行3列的子矩阵
    matrix_8 = joint_tf[7][0:3, 0:3]
    # 表示出杆3的dh坐标y,z方向单位向量
    y_3 = np.dot(matrix_3, np.transpose([0, 1, 0]))   #方向投影
    z_3 = np.dot(matrix_3, np.transpose([1, 0, 0]))  #关节坐标系中 x 方向的投影  ???
    # 杆3、8各自端点坐标
    line = np.zeros((4, 3))  #形状为(4, 3)的全零数组，表示四个点的坐标，每个点有三个坐标值（x、y、z）
    line[0] = xyz[1]   #得到各个端点的三维坐标，第二、四、七、八个关节
    line[1] = xyz[3]
    line[2] = xyz[6]
    line[3] = xyz[7]   #第四行   为第八个关节坐标
    # 单位向量
    main_arm_vector = np.dot(matrix_8, np.transpose([0, 0, -1]))  #反向z轴投影，与车身碰撞检测
    final_test_point = main_arm_vector * main_arm_len + line[3]   #self.main_arm_len = -5.8705+现在第八个关节的坐标   类似于孔深的算法
    line[2] = final_test_point   #得到最终的杆件8反向末端坐标
    # 将推进杆分段
    point8 = 0.001 * (final_test_point - line[3])    #还是回到了移动关节的投影，杆8的两段，  得到划分间隔的三维坐标  杆8拆分，它的各个点到杆3的距判定
    # 将8个点的坐标存入, 坐标系转换至隧道坐标系
    point_i = np.empty(18)    #18数组长度
    for i in range(joint_num):
        point_i[2 * i] = xyz[i][1]  #y坐标在基坐标系数组存储在偶数组
        point_i[2 * i + 1] = xyz[i][2] + height_base    #self.height_base = 1.702
    # 加入杆件末端点
    point_i[16] = final_test_point[1]      #引入杆件8的y坐标
    point_i[17] = final_test_point[2] + height_base     #z坐标
    # 左右壁面、地面碰撞检测
    for i in range(joint_num + 1):
        if abs(point_i[2 * i]) >= base2tunnel_side - joint_radius:  #两臂   2.25-0.2=2.05m
            col_lrwall = True
            print('col_lrwall',col_lrwall)
        if point_i[2 * i + 1] < 0.2-0.15:  #地面
            col_ground = True
    # 隧道拱顶碰撞检测,
    arcade_point = arcade_data()   #导入拱形桥数据
    # 检测对象：钻臂8个端点
    for m in range(joint_num + 1):
        # 遍历拱廊上11个点
        arcade_num = 11
        for i in range(arcade_num - 1):
            # 遍历拱廊相邻两点间等分点
            for j in range(4):
                point_plane = [j * 0.25 * (arcade_point[i + 1][0] - arcade_point[i][0]) + arcade_point[i][0],
                               j * 0.25 * (arcade_point[i + 1][1] - arcade_point[i][1]) + arcade_point[i][1]]  #精度0.25划分，新的数据就两列y、z
                if ((point_plane[0] - point_i[2 * m]) * point_plane[0] <= 0) and point_plane[1] <= point_i[
                    m * 2 + 1] + safe_scale:   #self.safe_scale = 0.15  # 隧道碰撞安全量  〖(O_iy-O〗_y)∙O_iy≤0且O_z≥O_iz来判断是否在隧道范围外，以简化计算量
                    col_arch = True   #
                    print('col_arch', col_arch)
    # 车身碰撞检测
    for i in range(joint_num):
        if xyz[i][0] < 0 and xyz[i][2] < base2car_top and abs(xyz[i][1]) < base2car_side:
            col_body = True
 #   print(final_test_point)  #输出了
    if final_test_point[0] < 0.86 + 0.05 and abs(final_test_point[1]) < 1.5 and final_test_point[2] + height_base < 0.75:
        col_leg = True
    if final_test_point[0] < 0 and abs(final_test_point[1]) < 1 and final_test_point[2] + height_base < base2car_top:
        col_body = True
    for i in range(3):    #
        if max(line[0][i], line[1][i]) < min(final_test_point[i], line[3][i]) or min(line[0][i], line[1][i]) > max(
                final_test_point[i], line[3][i]):
            col_self = False
    for i in range(1000):
        point_detection = i * point8 + line[3] - line[0]  #第八个关节-第二个关节坐标+1000个沿着8关节的点
        distance_x = np.dot(point_detection, line[1] - line[0])   #二、四坐标距离*各个点，得到x方向距离
        distance_y = np.dot(point_detection, y_3)    #y方向的投影    点乘
        distance_z = np.dot(point_detection, z_3)  #z方向的投影
        safe_x = push_pole_len ** 2 + safe_distance   #259cm平方+1cm
        safe_y = push_pole_radius + main_arm_radius_y + safe_distance   #0.25+0.2+0.01=0.46  0.41
        safe_z = push_pole_radius + main_arm_radius_z + safe_distance  # 0.2+0.2+0.01=0.41    0.36
        if not (safe_x > distance_x > -safe_distance and   #  杆8的    -1到67082
                safe_y > distance_y > -safe_y and     #-51cm  到51cm     0.35
                safe_z > distance_z > -safe_z):    #0.25
            col_self = False
        else:
            col_self = True
            break
    print('col_self',col_self)
    col_data = [col_lrwall,col_ground,col_body,col_leg,col_arch, col_self]
    col = col_lrwall or col_ground or col_body  or col_arch or col_self or col_leg
    return col