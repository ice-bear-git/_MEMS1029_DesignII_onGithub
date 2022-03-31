
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import time
import pprint

class Params:
    def __init__(self):
        self.N = 4          # number of state variables
        self.M = 2          # number of control variables
        self.T = 10         # Prediction Horizon
        self.DT = 0.2       # discretization step
        self.path_tick = 0.05
        self.L = 0.3            # vehicle wheelbase
        self.MAX_SPEED = 1.5    # m/s
        self.MAX_ACC = 1.0      # m/ss
        self.MAX_D_ACC = 1.0    # m/sss
        self.MAX_STEER = np.radians(30)     # rad
        self.MAX_D_STEER = np.radians(30)   # rad/s

P = Params()

REWARD_TYPES = ('dense', 'sparse')      # dense：连续解； sparse：稀疏的
class M1_CarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 use_direct=False,
                 reward_type='dense',
                 terminate_at_goal=True,
                 goal_radius=0.5,
                 goal_distance=5.0,
                 goal_angle_range=(0, 2*np.pi),
                 velocity_reward_weight=0.5,
                 ctrl_cost_coeff=1e-2,
                 contact_cost_coeff=1e-3,
                 fixed_target_position=[5, -3, 0.5],     # 目前，先与模型保持一致？？？set_ball()还不能移动，后续解决！！！
                 ):
        assert reward_type in REWARD_TYPES
        super().__init__()

        self.USE_DIRECT = use_direct
        self._reward_type = reward_type
        self.terminate_at_goal = terminate_at_goal

        self.goal_radius = goal_radius
        self.goal_distance = goal_distance
        self.goal_angle_range = goal_angle_range

        self.velocity_reward_weight = velocity_reward_weight
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.contact_cost_coeff = contact_cost_coeff

        # Todo! 个性化超量
        self.reward_threshold = 300     # 一轮episode中reward总和的过关阈值
        self.dt = 1/240. * 2            # 时延，时延=每帧simulate时长 * 2帧，即1./120.

        self.i_step = 0
        self.reward_sum = 0
        self.max_reward_sum = 0
        self.fixed_target_position = fixed_target_position
        if fixed_target_position is not None:
            self.target_position = fixed_target_position
            x, y, _ = self.target_position
            self.goal_distance = np.linalg.norm(np.array([x, y]) - np.array([0.0, 0.0]))

        # 连接物理引擎
        if self.USE_DIRECT:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        # p.connect(p.SHARED_MEMORY)
        p.resetDebugVisualizerCamera(cameraDistance=5.5, cameraYaw=0,
                                     cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
        # self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3))           # 3个action输出，值域为-1到1
        # self.observation_space = spaces.Box(np.array([-1]*4), np.array([1]*4))      # 4个状态值。效果也很好？？？

        # action_space：maxForce，steeringAngle=0, acceleration。
        self.action_space = spaces.Box(np.array([0.5, 1.0, -0.8]),
                                       np.array([5.0, 3.0, 0.8]))               # 3个action输出，值域为-1到1
        self.observation_space = spaces.Box(np.array([-500.0]*10),
                                            np.array([500.0]*10))               # 10个状态值，值域为-1到1
        # self.observation_space = spaces.Box(np.array([-500.0]*4),
        #                                     np.array([500.0]*4))                # 4个状态值。效果也很好？？？

    def reset(self):
        self.i_step = 0
        self.reward_sum = 0
        self.max_reward_sum = 0

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        p.setGravity(0, 0, -10)
        p.setTimeStep(1./120.)      # 设置时延参数
        p.setTimeStep(self.dt)      # 设置时延参数

        # 实时仿真
        useRealTimeSim = 1
        p.setRealTimeSimulation(useRealTimeSim)

        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.0,
                                     cameraYaw=-90,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[-0.1, -0.0, 0.65])

        # 载入地面模型，useMaximalCoordinates加大坐标刻度可以加快加载
        p.loadURDF("plane.urdf", useMaximalCoordinates=False)

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.objectUid = p.loadURDF(os.path.join(ROOT_DIR, "URDFs/target_obj.urdf"),
                                    basePosition=self.target_position)


        # Todo！ 加载小车————joint_index对应：2 左后动力、3 右后动力、5 左前动力、7 右前动力；4 左前转向、6 右前转向
        self.URid = p.loadURDF("racecar/racecar.urdf", basePosition=[0, .0, .0])
        self.inactive_wheels = [5, 7]       # 设为随动轮
        self.power_wheels_idx = [2, 3]      # 设为动力轮
        self.steerings_idx = [4, 6]         # 转向轮（两个joint的转角相同）
        self.joints_idx = [2, 4, 6]         # 与观测值有关的joints

        for wheel in self.inactive_wheels:
            p.setJointMotorControl2(self.URid, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setRealTimeSimulation(1)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

        return self.get_obs()

    def get_obs(self):
        """
        """
        rob_pos, rob_orn = p.getBasePositionAndOrientation(self.URid)
        line_vel, ang_vel = p.getBaseVelocity(self.URid)
        pos_list, vel_list = self.get_joint_PosAndVel()
        observation = np.concatenate([
            [rob_pos[0], rob_pos[1]],
            [np.sqrt(line_vel[0]**2 + line_vel[1]**2)],
            [p.getEulerFromQuaternion(rob_orn)[2]],
            pos_list,
            vel_list,
        ]).reshape(-1)
        return observation

    def get_joint_PosAndVel(self):
        joints_pos = p.getJointStates(self.URid, self.joints_idx)
        pos_list = [joint[0] for joint in joints_pos]
        vel_list = [joint[1] for joint in joints_pos]
        return pos_list, vel_list

    def set_ctrl(self, *, currVel, maxForce=10, steeringAngle=0, acceleration=0):
        robotId = self.URid

        gearRatio = 1./21
        # p.setGravity(0, 0, -10)

        targetVelocity = currVel + acceleration*P.DT
        for wheel in self.power_wheels_idx:
            p.setJointMotorControl2(robotId,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    # targetVelocity=targetVelocity/gearRatio,
                                    targetVelocity=targetVelocity,
                                    force=maxForce)

        for steer in self.steerings_idx:
            p.setJointMotorControl2(robotId,
                                    steer,
                                    p.POSITION_CONTROL,
                                    targetPosition=steeringAngle)
    def set_target(self):
        if self.fixed_target_position is None:
            startPos = self.target_position
            startOrientation = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(self.objectUid, startPos, startOrientation)
            p.stepSimulation()

    def go_forward(self, action=[8, 6, -0.5]):
        """
        按照给定的vel, force, steering_angle 行进
        """
        curr_vel, max_force, steering_angle = action[0], action[1], action[2]
        self.set_ctrl(currVel=curr_vel,  maxForce=max_force, steeringAngle=steering_angle, acceleration=0)
        # p.stepSimulation()
        # time.sleep(1./240.)

    def get_body2target_distance(self):
        curr_posi, curr_orn = p.getBasePositionAndOrientation(self.URid)
        x, y, z = curr_posi
        t_x, t_y, _ = self.target_position
        dist = np.linalg.norm(np.array([x, y]) - np.array([t_x, t_y]))
        return dist

    def step(self, action):
        self.i_step += 1

        self.set_target()

        dist_before = self.get_body2target_distance()
        self.set_ctrl(currVel=action[0], maxForce=action[1], steeringAngle=action[2], acceleration=0)
        p.stepSimulation()
        dist_after = self.get_body2target_distance()

        # Todo!! forward_reward原则：实际有效速度 * velocity_reward_weight
        distance_delta = dist_before - dist_after
        velocity_to_target = distance_delta / self.dt   # in m/s wrong with moving target
        forward_reward = velocity_to_target * self.velocity_reward_weight
        ctrl_cost = self.ctrl_cost_coeff
        contact_cost = self.contact_cost_coeff

        # if dist_before > dist_after:
        #     forward_reward = self.velocity_reward_weight

        # TODO！个性化reward如下：
        orient_reward_weight = 0.01     # (optional)鼓励向目标运动
        orient_reward = 0               # (optional)运动方向是否有利于到达目标位置
        goal_reward = 0                 # (optional)距离目标位置远近程度的奖励
        min_vaild_reward = -10.0        # 一轮episode的最低累计reward，低于此值则终止本轮训练

        # TODO！！just for testing!!
        too_far = dist_after > self.goal_distance*1.25
        if too_far:
            forward_reward = -orient_reward_weight * 2

        survive_reward = 0
        reward = (forward_reward + survive_reward - ctrl_cost - contact_cost) \
                 + (goal_reward + orient_reward)

        self.reward_sum += reward
        if self.max_reward_sum < self.reward_sum:
            self.max_reward_sum = self.reward_sum

        goal_reached = dist_after <= self.goal_radius    # 接近达到目标
        done = (self.terminate_at_goal and goal_reached) or too_far
        if (self.reward_sum < min_vaild_reward) :
            done = True
            print(f"!!! self.reward_sum < {min_vaild_reward}")
        if self.reward_sum > self.reward_threshold:
            done = True
            print(f"!!! self.reward_sum>reward_threshold")

        print(f" 第{self.i_step}次：done={done}, "
              f"reward_sum={self.reward_sum:0.2f}, max_reward_sum={self.max_reward_sum:0.2f}")
        print(f" dist_after={dist_after:0.3f}, reward={reward:0.2f},  forward_reward={forward_reward:0.3f},"
              f" ctrl_cost={-ctrl_cost:0.3f}, contact_cost={-contact_cost:0.3f},"
              f" goal_reward={goal_reward:0.3f}, orient_reward={orient_reward:0.3f}")

        ob = self.get_obs()
        return ob, reward, done, dict(reward=reward,
                                      reward_forward=forward_reward,
                                      reward_ctrl=-ctrl_cost,
                                      reward_contact=contact_cost,
                                      )

    def render(self, mode='human'):
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
        #                                                   distance=.7, pitch=-70, roll=0,
        #                                                   upAxisIndex=2)
        # proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(960)/720, nearVal=0.1, farVal=100.0)
        # (_, _, px, _, _) = p.getCameraImage(width=960, height=720,
        #                                     viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        #                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (720, 960, 4))
        # rgb_array = rgb_array[:, :, :3]
        # return rgb_array
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

