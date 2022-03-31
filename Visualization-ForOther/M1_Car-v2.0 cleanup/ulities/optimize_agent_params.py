#!/usr/bin/env python3

"""
para_optimizer.py 用于检测action前后pos的拟合程度，以指导对agent模型参数进行修改

    原理：
        按步长乘以刷新次数得到Joint每次转动的偏移角（单位弧度），公式=sin(i * 0.01*4 * np.pi) ，其所生成的正弦曲线与舌的行走姿态相似。
            其中：dt为mujoco系统sim的常用变量，一般默认值为0.01。
                  num_steps为刷新帧数，即do_simulation(a, self.frame_skip)中的frame_skip，默认值为4。
                  dt * n_frames = 0.01 * 4
        这里，采取了control-based的方式，使得蛇的行进 action='波形主姿态+随机偏移'，波形主姿态占60%、随机偏移占40%。
        希望以此方式减少valueless的action，又能给强化学习提供所需的随机样本。

    注意：在正式训练agent前，强烈建议通过模型参数优化程序来检查和调整xml参数，确认蛇的基本走形是否正确后再开始训练。


    *para_optimizer.py*功能说明：
    ----------
    功能：
    观测转换前后caculate数值和real数值之间的偏差，来指导参数调整，使二者尽可能拟合。

    基本逻辑：
    1. 随机生成一个转角，并在此基础上形成计算得出下一步(含干扰)转动角度。转动角= 随机目标角与90%当前pos差 + 角速度*0.05弧度/秒
    2. 然后记录转动前后角度变化，前后二者之差形成PDF分析结果
    3. 分析pdf来调整参数，优化后再进行测试。

    结论：
    程序是用来分析分析指令发出（sim.data.ctrl[:]）多长后Joint能执行到指定的位置，采用近距逼近的方式实现。
    发现与当前角度姿态、角速度、分步逼近次数三个参数有关：every step takes 0.002s, the sensor samples every 0.1s. So one control step happens in 0.1s.
    所以，基于xml模型分析结果pdf，来调整PP、VEL、num_steps三个参数，以较好拟合caculate-real两条曲线。
    如：本项目模型适合采用num_steps=10，PP=1.0，VEL=0.05，可以较好拟合caculate和sim两条曲线。
"""
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import os
from pprint import pprint
import time
import numpy as np
import matplotlib.pyplot as plt

import csv

# time_list = []
#
# with open('nowtime.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         time_list.append(row)
#
# time_list_array = np.array(time_list[0:100])
#
#
#
# measured_angle_list = []
#
# with open('now.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         measured_angle_list.append(float(row[0]) - 90)
#
# calculated_angle_list = []
#
# with open('real.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         calculated_angle_list.append(float(row[0]) - 90)
#
# read_index = np.arange(0, 700)

import argparse

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--render', help='render', type=bool, default=False)
# args = parser.parse_args()

def plot_result(target_position_list, sim_position_list):

    step_index = np.arange(0, len(sim_position_list))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(step_index*10/len(sim_position_list), target_position_list, '-', alpha=0.2)
    # ax.plot(step_index*0.1, calculated_angle_list[0:len(step_index)], '--', alpha=0.2)
    # ax.plot(step_index*0.1, measured_angle_list[0:len(step_index)])
    ax.plot(step_index*10/len(sim_position_list), sim_position_list, '.', alpha=0.5)
    plt.xlim(3, 4)
    plt.grid()
    ax.legend(["calculate", "sim_target", "measured", "observe (mujoco)"], loc="upper left")
    plt.xlabel("time (s)")

    fig.tight_layout()
    plt.savefig('test_snake.pdf', bbox_inches='tight')
    plt.show()

    print(target_position_list)
    print(sim_position_list)




def analyse_Swimmer_parameters(PP=1.0, VEL=0.05, n_frames=4):

    model = load_model_from_path("swimmer.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    joints = ['rot2', 'rot3']
    joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))

    target_position_list = []
    sim_position_list = []
    flag = -1

    start_time = time.time()
    for i in range(500):
        target_pos_angle = 60 * np.sin(i * 0.01 * 4 * np.pi) + 0     # 生成一个随机转动角度（正负30度正则化），单位：degree
        target_position = target_pos_angle / 180 * np.pi             # 单位：radian

        pos_before_action = sim.data.qpos[joints_idx]
        wave_a = [1, -1]
        wave_a = np.array(wave_a) * target_position

        action = np.random.randint(-10, 10, 2)/180 * np.pi
        # TODO！！按照target_position的值正则化
        act_max = np.max(abs(action))
        if 0 < act_max:
            action = action/act_max * target_position

        action = np.array([0]*2)
        # 这里，action='波形主姿态+随机偏移'，波形主姿态占80%、随机偏移占20%。
        target_action = wave_a * 1.0 + action * 0.3          # a为网络传入的action，按1.5弧度正则化后最大约为90度
        diff_pos = target_action - pos_before_action

        # TODO!! pgos赋target，一步step就可以在内存执行到位；而ctrl赋diff+step次数，还不确定执行效果？？？
        sim.data.qpos[joints_idx] = target_action
        sim.data.ctrl[:] = diff_pos
        # sim.data.ctrl[:] = target_action
        # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL

        for j in range(n_frames):
            sim.step()
        if True:
            viewer.render()
            pass

        # # time.sleep(0.01)

        target_position = target_action[1]
        target_position_list.append(target_position*180/np.pi)  # 单位为degree
        real_position = sim.data.qpos[joints_idx][1]            # 读取转动后的Pos
        sim_position_list.append(real_position*180/np.pi)       # 单位为degree

        # sim.data.qpos[joints_idx] = [0] * 6
        # for j in range(n_frames):
        #     sim.step()
        # viewer.render()

    plot_result(target_position_list, sim_position_list)



def analyse_M2Simple_parameters(PP=1.0, VEL=0.05, n_frames=4):
    # TODO！！every step takes 0.002s, the sensor samples every 0.1s. So one control step happens in 0.1s.
    diff_pos_integrated = 0
    # PP = 1
    # VEL = 0.02
    # D = 2

    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # file = os.path.join(ROOT_DIR, 'test_M1_parameters.xml')
    # # env = MujocoEnv(file, frame_skip=n_frames)
    # # env = MujocoEnv()
    # # env = gym.make()

    # model = load_model_from_path("./M2-Snake-XML/M2-Snake-Head2Tail.xml")
    model = load_model_from_path("M1_simple_7Modules.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    joints = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5', 'Joint_6']
    joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))

    target_position_list = []
    sim_position_list = []
    flag = -1

    start_time = time.time()
    for i in range(1000):
        # sim系统dt默认值为0.01/s，所以此处
        target_pos_angle = 60 * np.sin(i * 0.01 * 4 * np.pi) + 0     # 生成一个随机转动角度（正负30度正则化），单位：degree
        target_position = target_pos_angle / 180 * np.pi             # 单位：radian

        pos_before_action = sim.data.qpos[joints_idx]

        wave_a = [0, 1, 0, -1, 0, 1]
        wave_a = [0, -1, 0, 1, 0, -1]
        # flag *= -1
        wave_a = np.array(wave_a) * flag * target_position

        action = np.random.randint(-10, 10, 6)/180 * np.pi
        # TODO！！按照target_position的值正则化
        act_max = np.max(abs(action))
        if 0 < act_max:
            action = action/act_max * target_position

        # action = np.array([0]*6)
        # 这里，action='波形主姿态+随机偏移'，波形主姿态占80%、随机偏移占20%。
        target_action = wave_a * 1.0 + action * 0.5          # a为网络传入的action，按1.5弧度正则化后最大约为90度
        diff_pos = target_action - pos_before_action

        diff_pos[0] = 0
        diff_pos[2] = 0
        diff_pos[4] = 0

        target_action[0] = 0
        target_action[2] = 0
        target_action[4] = 0

        # TODO!!  以下两种方式蛇运动的幅度有很大不同，.qpos[]方式运动很快(赋目标target位姿)，.ctrl[:]方式运动平稳(.qpos[]方式运动很快(赋目标target位姿)，.ctrl[:]方式运动平稳(赋diff角度差)
        # TODO!!  --- ctrl[:]方式赋diff角度差 还是 赋diff角度差 ???！！！
        sim.data.qpos[joints_idx] = target_action   # 方式1：.qpos[]方式
        sim.data.ctrl[:] = diff_pos
        for j in range(n_frames):
            sim.step()

        # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL
        # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL

        # print("Control Value: ", diff_pos * 350 + diff_pos_integrated * 2.5)
        if True:
            viewer.render()
            pass



        target_position = target_action[1]
        target_position_list.append(target_position*180/np.pi)  # 单位为degree
        real_position = sim.data.qpos[joints_idx][1]            # 读取转动后的Pos
        sim_position_list.append(real_position*180/np.pi)       # 单位为degree

        # sim.data.qpos[joints_idx] = [0] * 6
        # for j in range(n_frames):
        #     sim.step()
        # viewer.render()


        if os.getenv('TESTING') is not None:
            break


    end_time = time.time()
    # print("Elapse Time", end_time - start_time)     # 历时
    # print("Steps", len(sim_position_list))

    plot_result(target_position_list, sim_position_list)


## Todo！！ 下面两个函数要与real环境类函数保持一致
class M1Snake():
    def __init__(self, sim):
        self.sim = sim
        self.data = sim.data
        self.joints_idx = None
        self.i_step = 0

    ## 使蛇的行进 action='波形主姿态+随机偏移'，波形主姿态占60%、随机偏移占40%。函数将noisy_ac作为控制ac并返回叠加了noisy_ac的wave
    def generate_wave_action(self, noisy_ac, n_frames=4):

        # Todo!! 以下为采用control-based的分步step方式：
        #  计算一个Joint每次转动的偏移角（单位弧度），公式=sin(i * 0.01*4 * np.pi) ，其所生成的正弦曲线与舌的行走姿态相似。
        pos_before_action = self.data.qpos[self.joints_idx]
        wave_a = [0, 1, 0, -1, 0, 1]
        # wave_a = [1, 0, -1, 0, 1, 0]
        target_angle = 50 * np.sin(self.i_step * 0.01 * n_frames * np.pi)      # 生成一个随机转动角度（正负60度正则化），单位：degree
        target_position = target_angle / 180 * np.pi    # 单位：radian
        wave_a = np.array(wave_a) * target_position

        # TODO！！按照target_position的值正则化
        act_max = np.max(abs(noisy_ac))
        if 0 < act_max:
            ac = noisy_ac/act_max * target_position
        else:
            ac = noisy_ac

        # a = np.array([0]*6)
        # 这里，action='波形主姿态+随机偏移'，波形主姿态占80%、随机偏移占20%。
        target_action = wave_a * 1.0 + ac * 0.9          # a为网络传入的action，按1.5弧度正则化后最大约为90度
        diff_pos = target_action - pos_before_action
        # target_action[0] = 0
        # target_action[2] = 0
        # target_action[4] = 0
        #
        # diff_pos[0] = 0
        # diff_pos[2] = 0
        # diff_pos[4] = 0

        # target_action[1] = 0
        # target_action[3] = 0
        # target_action[5] = 0
        #
        # diff_pos[1] = 0
        # diff_pos[3] = 0
        # diff_pos[5] = 0

        return target_action, diff_pos

    def do_wave_action(self, noisy_ac, n_frames=4):
        target_action, diff_pos = self.generate_wave_action(noisy_ac, n_frames=n_frames)

        # self.data.qpos[self.joints_idx] = target_action     # 方式1：.qpos[]方式
        a = diff_pos        # Todo! 用target_action还是diff_pos，fatal最终选择了target_action——————需要进一步确认？？？？？
        a = target_action
        self.data.ctrl[:] = a   # 稳定运动应采用.ctrl[:]方式
        for i in range(n_frames):
            self.sim.step()
        return

def analyse_M2Real_parameters(PP=1.0, VEL=0.05, n_frames=4):

    # TODO！！every step takes 0.002s, the sensor samples every 0.1s. So one control step happens in 0.1s.

    model = load_model_from_path("../envs/assets/M2-Snake-XML/M2-Snake-Head2Tail.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    joints = ['Joint_0_fromHead', 'Joint_1_fromHead', 'Joint_2_fromHead',
              'Joint_3_fromHead', 'Joint_4_fromHead', 'Joint_5_fromHead']
    joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))
    target_position_list = []
    sim_position_list = []

    test_env = M1Snake(sim)
    test_env.joints_idx = joints_idx



    # env_id = "M1-fixed-goal-v1"
    # real_env = gym.make(env_id)
    # sim = real_env.sim

    start_time = time.time()

    for i in range(0, 1000):
        # a = np.array([0] * 6)   # Todo!! xml模型参数应能保证在a在全为0（之后波形没有干扰）的情况下，保证蛇能稳定直线行进
        a = np.random.randint(-10, 10, 6)/180 * np.pi

        ## 测试1：real环境下，do_simulation()方式
        # real_env.i_step = i
        # target_action, diff_pos = real_env.generate_wave_action(noisy_ac=a, n_frames=4)
        # real_env.do_simulation(target_action, n_frames=4)
        # real_env.render()

        n_frames = 4
        ## 测试2：测试环境下，自定义step()方式
        test_env.i_step = i
        target_action, diff_pos = test_env.generate_wave_action(noisy_ac=a, n_frames=n_frames)
        test_env.do_wave_action(target_action, n_frames=n_frames)

        if True:
            viewer.render()
            pass

        # target_position = target_action[1]
        # target_position_list.append(target_position*180/np.pi)  # 单位为degree
        # real_position = sim.data.qpos[joints_idx][1]            # 读取转动后的Pos
        # sim_position_list.append(real_position*180/np.pi)       # 单位为degree
        target_position = target_action[0]
        target_position_list.append(target_position*180/np.pi)  # 单位为degree
        real_position = sim.data.qpos[joints_idx][0]            # 读取转动后的Pos
        sim_position_list.append(real_position*180/np.pi)       # 单位为degree

    end_time = time.time()
    # print("Elapse Time", end_time - start_time)     # 历时
    # print("Steps", len(sim_position_list))

    plot_result(target_position_list, sim_position_list)




def analyse_Ant_parameters(PP=1.0, VEL=0.05, n_frames=4):

    # TODO！！every step takes 0.002s, the sensor samples every 0.1s. So one control step happens in 0.1s.

    model = load_model_from_path("./ant.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    joints = ['hip_4', 'ankle_4', 'hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3']
    joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))

    target_position_list = []
    sim_position_list = []
    flag = -1

    start_time = time.time()
    for i in range(1, 201):
        pos_before_action = sim.data.qpos[joints_idx]
        target_action = np.random.randint(-10, 10, 8)/180 * np.pi
        # # TODO！！按照target_position的值正则化
        # act_max = np.max(abs(action))
        # if 0 < act_max:
        #     action = action/act_max * target_position

        # a = np.array([0]*6)
        # 这里，action='波形主姿态+随机偏移'，波形主姿态占80%、随机偏移占20%。
        # target_action = wave_a * 1.0 + action * 0.5          # a为网络传入的action，按1.5弧度正则化后最大约为90度
        diff_pos = target_action - pos_before_action

        print(f"target_action = {target_action} ")
        print(f"diff_pos={diff_pos} ")

        # 以下两种方式蛇运动的幅度有很大不同，.qpos[]方式运动很快，.ctrl[:]方式运动平稳
        # sim.data.qpos[joints_idx] = diff_pos
        sim.data.ctrl[:] = diff_pos
        # sim.data.ctrl[:] = diff_pos
        # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL
        # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL

        # print("Control Value: ", diff_pos * 350 + diff_pos_integrated * 2.5)
        # for j in range(n_frames):
        for j in range(4):
            sim.step()
        if True:
            viewer.render()
            pass
        sim.data.qpos[joints_idx] = np.array([0]*8)
        sim.step()

        target_position = target_action[1]
        target_position_list.append(target_position*180/np.pi)  # 单位为degree
        real_position = sim.data.qpos[joints_idx][1]            # 读取转动后的Pos
        sim_position_list.append(real_position*180/np.pi)       # 单位为degree


    plot_result(target_position_list, sim_position_list)



def analyse_Mujoco_delay(PP=1.0, VEL=0.05, n_frames=4):
    model = load_model_from_path("swimmer.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    joints = ['rot2', 'rot3']
    joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))

    target_position_list = []
    sim_position_list = []
    flag = -1

    # TODO！！对mujoco模拟器执行结果分析的过程：
    # ======================
    #  1. 读取‘back’要素的初始坐标xyz0、原始Pos0
    x, y, z = sim.data.get_body_xpos('back')[:3]
    print("real_position 0=", sim.data.qpos[joints_idx], f"xy={x:.2f}, {y:.2f}")

    # 2. 采用qpos[]方式直接赋Pos1，之后马上读取Pos和xyz1，发现Pos=Pos1、xyz1=xyz0
    sim.data.qpos[joints_idx] = (1.5, 1.5)
    x, y, z = sim.data.get_body_xpos('back')[:3]
    print("real_position 1=", sim.data.qpos[joints_idx], f"xy={x:.2f}, {y:.2f}")

    # 3. 只要qpos[]方式直接赋posX，之后无论是否执行step再读取的pos=posX，但xyz与step(一次以上效果一样)高度相关
    for j in range(5):
        sim.step()
    x, y, z = sim.data.get_body_xpos('back')[:3]
    print("real_position 2=", sim.data.qpos[joints_idx], f"xy={x:.2f}, {y:.2f}")

    # 4. 直接qpos[]方式赋Pos3、.ctrl[:]赋-pos3，执行step前读取Pos3和xyz3，发现Pos=Pos3、xyz未变（=xyz2）。
    #   但只要执行一步step再读取Pos4和xyz4，发现Pos=Pos3=Pos4、xyz更新为与Pos3对应的位置————.qpos[]只需要一步step即可生效
    #   ctrl[]的执行效果与step()的次数有很大关系，需要针对xml的模型针对性测试后找到合适次数————.ctrl[:]对step()次数高度相关
    sim.data.qpos[joints_idx] = (-1.5, -1.5)
    sim.step()
    sim.data.ctrl[:] = (0.5, 0.5)   # ctrl[:]对step()次数高度相关

    x, y, z = sim.data.get_body_xpos('back')[:3]
    print("real_position 3=", sim.data.qpos[joints_idx], f"xy={x:.2f}, {y:.2f}")
    for j in range(5):
        sim.step()
    viewer.render()
    x, y, z = sim.data.get_body_xpos('back')[:3]
    print("real_position 4=", sim.data.qpos[joints_idx], f"xy={x:.2f}, {y:.2f}")

    if True:
        viewer.render()
        pass
    # time.sleep(0.01)

    # target_position = target_action[1]
    # target_position_list.append(target_position*180/np.pi)  # 单位为degree
    # real_position = sim.data.qpos[joints_idx][1]            # 读取转动后的Pos
    # sim_position_list.append(real_position*180/np.pi)       # 单位为degree

    # sim.data.qpos[joints_idx] = [0] * 6
    # for j in range(n_frames):
    #     sim.step()
    # viewer.render()

    plot_result(target_position_list, sim_position_list)


def analyse_Joint_parameters(PP=1.0, VEL=0.03, n_frames=4):


    model = load_model_from_path("test_snake.xml")
    # model = load_model_from_path("test_M1_parameters.xml")

    sim = MjSim(model)

    joints = ['Joint_1']
    joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))
    qpos = sim.data.qpos
    joints_pos = qpos[joints_idx]

    viewer = MjViewer(sim)

    sim_state = sim.get_state()

    target_position_list = []
    sim_position_list = []

    i = 0

    start_time = time.time()

    # i的步进为0.01，相当于将180度分为100份，被测试的Joint每次转动4个单位（约3.75度）
    # pos_detla_angle = 60 * np.sin(0.01 * 4 * np.pi)
    # pos_detla = pos_detla_angle / 180 * np.pi
    # while ((time.time() - start_time) <= 10):

    for i in range(1000):
        target_pos_angle = 60 * np.sin(i * 0.01 * 4 * np.pi) + 0     # 生成一个随机转动角度（正负30度正则化），单位：degree
        target_position = target_pos_angle / 180 * np.pi             # 单位：radian
        target_position_list.append(target_pos_angle)                # 单位：degree

        # TODO！！every step takes 0.002s, the sensor samples every 0.1s. So one control step happens in 0.1s.
        diff_pos_integrated = 0
        PP = 1
        VEL = 0.05
        D = 2
        n_frames = 3

        # pos_before_action = sim.data.qpos[joints_idx]       # Joint此时的角速度
        # diff_pos = target_position - pos_before_action      # 角度差（不含干扰项）
        # move_detla = (target_position - pos_before_action) / n_frames     # 角度差（不含干扰项）

        for j in range(1):
            pos_before_action = sim.data.qpos[joints_idx]       # Joint此时的角速度
            vel_before_action = sim.data.qvel[joints_idx]       # Joint此时的角速度
            diff_pos = target_position - pos_before_action      # 角度差（不含干扰项）
            # diff_pos_integrated += diff_pos

            # 执行转动：含干扰的转动角= 随机目标角与90%当前pos差 + 角速度*0.05弧度/秒。此处暂估角速度为0.05弧度/秒
            # sim.data.ctrl[:] = diff_pos * 5.0 + vel_before_action * 0.03 + diff_pos_integrated * 0.1 * 0
            # sim.data.ctrl[:] = diff_pos * 1.0 + vel_before_action * 0.05 + diff_pos_integrated * 0.1 * 0
            # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL + diff_pos_integrated * 0.1 * D

            sim.data.ctrl[:] = diff_pos
            # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL
            # sim.data.ctrl[:] = diff_pos * PP + vel_before_action * VEL * j

            # print("Control Value: ", diff_pos * 350 + diff_pos_integrated * 2.5)
            for j in range(n_frames):
                sim.step()

            # time.sleep(0.002)
            # j += 1

        # time.sleep(0.01)

        real_position = sim.data.qpos[joints_idx]               # 读取转动后的Pos
        sim_position_list.append(real_position*180/np.pi)       # 单位为degree

        print(f"target_position={target_position*180/np.pi} , real_position = {real_position*180/np.pi}")
        print(f"diff={real_position*180/np.pi-pos_before_action*180/np.pi}")

        # if args.render:
        if True:
            viewer.render()
            pass

        if os.getenv('TESTING') is not None:
            break

        i += 1

    end_time = time.time()

    print("Elapse Time", end_time - start_time)     # 历时
    print("Steps", len(sim_position_list))

    plot_result(target_position_list, sim_position_list)

if __name__ == '__main__':
    # analyse_Mujoco_delay()

    # analyse_Swimmer_parameters()
    # analyse_M2Simple_parameters()
    analyse_M2Real_parameters()
    # analyse_Joint_parameters()

    # analyse_Ant_parameters()