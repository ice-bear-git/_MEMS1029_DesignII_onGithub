from datetime import datetime

import torch

from policies.PPO import PPO



import gym
from gym import spaces
import os
import pybullet as p
import pybullet_data
import numpy as np


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

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass





## Training with custom PPO
def train_PPO(env_id, env):
    print("============================================================================================")

    ### initialize environment hyperparameters ###
    has_continuous_action_space = True  # continuous action space; else discrete

    max_training_timesteps = int(3e4)   # break training loop if timeteps > max_training_timesteps
    max_ep_len = 128                    # max timesteps in one episode

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e3)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e3)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 8      # update policy every n timesteps
    K_epochs = 80           # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # env_id = "M1-Car-v1"
    # env = gym.make(env_id)
    # env = M1_CarEnv(use_direct=False)

    state_dim = env.observation_space.shape[0]
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "log_dir"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/PPO_logs/' + env_id + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_id + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_id + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_id folder

    directory = "log_dir/PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_id + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_id, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim,
                    lr_actor, lr_critic,
                    gamma, K_epochs, eps_clip,
                    has_continuous_action_space,
                    action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode, timestep, reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")



#################################### Testing ###################################
def test_PPO(env_id, env):
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_id = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_id = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_id = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    has_continuous_action_space = True
    max_ep_len = 128            # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    # env_id = "M1-Car-v1"
    # env = gym.make(env_id)
    # env = M1_CarEnv(use_direct=False)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "log_dir/PPO_preTrained" + '/' + env_id + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_id, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    env_id = "M1-Car-v1"
    env = M1_CarEnv(use_direct=False)
    # print("training environment name : " + env_id)

    # config = {
    #         "planner": "CEM"
    # }

    train_PPO(env_id, env)
    # test_PPO(env_id, env)
    
    
    
    
