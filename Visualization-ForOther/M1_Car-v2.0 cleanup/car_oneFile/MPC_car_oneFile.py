"""
    Title: The Simulated MPC training scheme without real MuJoCol/PyBullet env
    Author: Ziang Cao
    Data: Mar 24

    Will cause 22.88 s for training on [1X30X10] loops over [5 X 1000 X 6] Action_K_ROLLOUTs
    Needs to pass real env inside of __init__ function
    so call real MuJoCol/PyBullet env.step() inside the corresponding functions.
"""

import torch
import numpy as np
import time


import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger

import gym
from envs.pybullet_M1_car import M1_CarEnv

# 创建新/继承gym环境，用于MPC测试
class TestEnv(M1_CarEnv):
    def __init__(self):
        super().__init__()

    def get_state(self):
        return self.get_obs()


class MPC_ONE_FILE():

    def __init__(self):
        # We didn't implement Batch_Size here ---> no parallel computation here.
        # self.env = AntEnv()

        self.env = TestEnv()
        self.s0 = self.env.reset()
        # self.fn = self.env.fn

        # action_mode表示action的生成模式："sample"对应env.action_space.sample()，"pi"对应pi.act()，"input"表示直接使用输入的action
        self.action_mode = "sample"
        self.action_mode = "input"

        # Should declare reward function here.
        self.M = 1   # Trail Number --- Layer 1
        self.L = 20  # Environment Step --- Layer 2
        self.T = 10  # CEM Iteration --- Layer 3

        self.K = 100
        self.K_top = 30

        self.action_scale = 150  # +- 150 degree
        self.action_scale = 1.0  # +- 1.0 radian

        self.H = 5  # as self.H self.rollout_steps
        self.a_size = 3  # as self.a_size  self.a_size
        self.a_size = 10  # as self.a_size  self.a_size

        # Corresponding to real action applied after the training of MPC.
        self.reset_State = 0

        # Will be reused during CEM training
        self.sim_State = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return

    def get_CurRealState(self):
        # return self.reset_State
        if self.s0 is None:
            self.s0 = self.env.reset()
            s = self.s0
        else:
            s = self.env.get_state()
        return s

    def rollouts(self, action_K_ROLLOUTs, s0):
        """
        Given [rollout_steps * K * num_joints] ---> [H * K * a_size]
        """
        # prepare the return list: Rollouts_rewards [K]
        Rollouts_rewards_list = []
        # for each sample in K
        for i in range(0, self.K):
            # Make [action_1_ROLLOUT] be [H * a_size]
            action_1_ROLLOUT = action_K_ROLLOUTs[:, i, :].view(self.H, self.a_size)
            # print("For Sample index: {}, action_1_ROLLOUT: \n{} ".format(i, action_1_ROLLOUT))
            rollout_reward, _ = self.sim_steps(action_1_ROLLOUT, s0)
            # append to return list
            Rollouts_rewards_list.append(rollout_reward.item())
        return torch.tensor(Rollouts_rewards_list)  # Convert list to tensor

    # @staticmethod
    # def sim(s, a=None):
    #     # simulate the step in mujocol!
    #     return s + 1.1 * torch.randn(1, device="cpu"), False
    def sim(self, s, a=None):
        # simulate the step in mujocol!

        # Todo! 训练时用env.action_space.sample()？？
        if self.action_mode == "pi":
            # ac = self.fn.act(stochastic=False, ob=s)[0]
            pass
        elif self.action_mode == "sample":
            ac = self.env.action_space.sample()
        else:
            ac = a.tolist()    # tensor转为列表

        ob, reward, done, _ = self.env.step(ac)
        return ob, reward, done, ac


    @staticmethod
    def calc_obs(self, s):
        return 0

    # @staticmethod
    # def calc_reward(s, a=None):
    #     return s * 100
    @staticmethod
    def calc_reward(r, a=None):
        return r * 100

    def sim_steps(self, actions_1_rollout, s0):
        """
        Be called from self.rollout()
        Handle state reset, one rollout simulate and return one reward.
        actions: [rollout_steps * num_joints] action for a certain K
        """
        # print("Require State s and action a to simulate within rollout.")
        if actions_1_rollout is None:
            actions_1_rollout = torch.zeros(self.H, self.a_size, device=self.device)

        # reInit sim_State to reset_State
        self.sim_State = s0
        # self.sim_State = torch.zeros(self.H, self.a_size, device=self.device)
        # idx = [i for i in range(self.H)]
        # self.sim_State.index_fill_(dim=0, index=idx, value=np.array(s0))
        # self.sim_State = torch.from_numpy(np.array([s0]*self.H))

        # self.sim_State =np.array([s0]*self.H)
        # prepare the reward r
        r = 0
        # continuously simulating based on [rollout_steps X num_joints] action
        for i in range(0, self.H):
            self.sim_State, reward, isStuck, ac = self.sim(self.sim_State, actions_1_rollout[i])
            if isStuck:
                r -= 100
            r += reward  # Can consider actions as well.

            if i == 0:
                a0 = ac

        # r += self.calc_reward(self.sim_State)  # Can consider actions as well.

        # return o, r, True         # can implement obs as well

        # Todo！ 因为action可能是由pi函数/sample()生成的，所以统一为将actions_1_rollout[0]更新为a0。列表可以被更新！！
        actions_1_rollout[0] = torch.tensor(a0)
        return r, False

    # def real_step(self, a=None):
    #     # print("Purpose of real_step(): Record real transition by current best action: ", a)
    #     if a is None:
    #         a = torch.zeros(1, self.a_size, device=self.device)
    #     # print("1. Apply action to real agent/env")
    #     self.sim(self.reset_State, a)
    #     # print("2. Update self.reset_State here")
    #     self.reset_State += 1
    def real_step(self, a=None):
        # print("Purpose of real_step(): Record real transition by current best action: ", a)
        if a is None:
            a = torch.zeros(1, self.a_size, device=self.device)
        # print("1. Apply action to real agent/env")
        self.sim(self.reset_State, a)
        # print("2. Update self.reset_State here")
        self.reset_State += 1


    def MPC_arch(self, return_plan=True):
        """
        The Main function: MPC_arch
        Implements the architecture of CEM training.
        requires the return of sequences of final plan
        """
        for trail in range(0, self.M):
            """ [Layer 1] Env Declare 
                Declare the reward function and target position here. --- Only 1 for now.
                Prepare the list of plan_sequences
            """
            plan_sequences = []     # Prepare the list of plan_sequences

            for step in range(0, self.L):
                """ [Layer2] Env step
                    Initialize the Whole Plan of [rollout_steps * K * num_joints] shape
                    a_size: number of dimensions for the space the agent is operating in
                    Then, get current real State
                """
                # Initialize by self.action_scale * N(0, I)
                A_MEANs = torch.zeros(self.H, 1, self.a_size, device=self.device)
                A_STDs = self.action_scale * torch.ones(self.H, 1, self.a_size, device=self.device)
                # print(A_MEANs.shape, A_STDs.shape)  # Size: [H * 1 * a_size]

                # 2-1 Creates: Sequences [rollout_steps * K * num_joints] --> [H * K * a_size]
                Action_K_ROLLOUTs = A_MEANs + A_STDs * torch.randn(self.H, self.K, self.a_size,
                                                                   dtype=torch.float, device=self.device)

                print("Action_K_ROLLOUTs[0,0]=", Action_K_ROLLOUTs[0,0])
                # Todo! 训练时用env.action_space.sample()？？！！！
                # a = self.env.action_space.sample()

                # When need Backtracking on actions: srcTensor = srcTensor.clone().detach().requires_grad_(True)
                # print("The Original Action_K_ROLLOUTs: ")
                # print(Action_K_ROLLOUTs.shape)
                # print(Action_K_ROLLOUTs)

                # 2-2 Get current real State
                s0 = self.get_CurRealState()
                # print("current real State: ", s0)

                for CEM_Iteration in range(0, self.T + 1):
                    """ [Layer3] CEM_Iteration 
                    Notes: last iteration will resample actions, but won't helps for finding the first rank action
                    Hence, we use T+1 in range()
                    3-1. [Rollouts] ---> Based on s0 + a0  ---> get s1 + a1 --> ... ---> s_end
                                    ---> Return the total_r_list [Rollouts_rewards], 
                                    ---> where each r is only based on each s_end
                    3-2. [ReSample] ---> cal new mean and std for top K rollouts 
                                    ---> make K-K_top resample 
                                    ---> replace the bottom K-K_top rollout actions
                    """
                    # Gain the Rollouts_rewards of [1 * K * 1] --> [1 * K] size
                    Rollouts_rewards = self.rollouts(Action_K_ROLLOUTs, s0).view(1, self.K)

                    # As indices are in same order for [Rollouts_rewards] and [Action_K_ROLLOUTs]
                    _, topk_indices = \
                        Rollouts_rewards.reshape(1, self.K).topk(self.K_top, dim=1, largest=True, sorted=True)
                    # Fetch [best_actions]: the K_top numbers of Action_K_ROLLOUTs
                    best_actions = Action_K_ROLLOUTs[:, topk_indices.view(-1), :].reshape(self.H,
                                                                                          self.K_top, self.a_size)
                    # Calculate the new [A_MEANs] & [A_STDs] based on [best_actions]
                    A_MEANs = best_actions.mean(dim=1, keepdim=True)
                    A_STDs = best_actions.std(dim=1, unbiased=False, keepdim=True)
                    # print(A_MEANs.shape, A_STDs.shape)  # Should be [H X 1 X a_size]

                    # Figure out the num of resample_actions
                    K_resample = self.K - self.K_top
                    # Make K_resample resample_actions
                    resample_actions = (A_MEANs +
                                        A_STDs * torch.randn(self.H, K_resample, self.a_size, dtype=torch.float,
                                                             device=self.device)).view(self.H, 1 * K_resample,
                                                                                       self.a_size)
                    # Find indices for the bottom (K-K_top) rank actions
                    _, botnk_indices = Rollouts_rewards.reshape(1, self.K).topk(K_resample, dim=1,
                                                                                largest=False, sorted=False)
                    # Put bottom (K-K_top) num of resample_actions back to [Action_K_ROLLOUTs]
                    Action_K_ROLLOUTs.data[:, botnk_indices.view(-1)] = resample_actions.data

                    # End of Layer 3

                """ Back to Layer 2 """
                # 2-3. Execute first action from the highest rollout sequences
                # print("Only Execute the first action from the Best rollout")
                best_plan = best_actions[0, 0, :]
                # print(best_plan.shape)
                # print(best_plan)
                # print(type(best_plan))
                plan_sequences.append(best_plan.tolist())
                # 2-4. Record real transition in D
                self.real_step(best_plan)

        if return_plan:
            return plan_sequences
        else:
            return None


def demo_without_mpc():
    env = gym.make("Ant-v2")

    # env_id = 'Ant-v2'
    # env = make_mujoco_env(env_id, seed=None)
    #
    # pi = policy_fn('pi', env.observation_space, env.action_space)
    # model_path = os.path.join(logger.get_dir(), 'PPO1-gym-Ant-v2')
    # U.load_state(model_path)

    while True:
        ob = env.reset()
        sum_reward = 0
        for i_step in range(1000):
            action = env.action_space.sample()
            # action = pi.act(stochastic=False, ob=ob)[0]
            ob, reward, done, dict = env.step(action)
            sum_reward += reward
            print(f"第{i_step}步： sum_reward={sum_reward:0.3f}, reward={reward:0.3f}, dict={dict}")     # 默认为times_episode=1000步
            env.render()
            if done:
                # ob = env.reset()
                break

def demo_with_mpc():
    start_time = time.time()  # StartTime

    mpc = MPC_ONE_FILE()
    whole_plans = mpc.MPC_arch()
    print("The whole Sequences Plans: ")
    print("The total lengths of the real applied actions: ", len(whole_plans))
    print(whole_plans)
    whole_plans = np.array(whole_plans)
    np.save("MyMPC_plans.npy", whole_plans)

    print("Total Execution time: {} s".format(round(time.time() - start_time, 2)))  # EndTime

    while True:
        ob = mpc.env.reset()
        sum_reward = 0
        for i_step in range(len(whole_plans)):
            # action = mpc.env.action_space.sample()
            action = whole_plans[i_step]
            ob, reward, done, dict = mpc.env.step(action)
            sum_reward += reward
            print(f"第{i_step}步： sum_reward={sum_reward:0.3f}, reward={reward:0.3f}, dict={dict}")     # 默认为times_episode=1000步
            mpc.env.render()
            if done:
                # ob = env.reset()
                break

if __name__ == "__main__":
    # start_time = time.time()  # StartTime
    #
    # my_MPC = MPC_ONE_FILE()
    # whole_plans = my_MPC.MPC_arch()
    # print("The whole Sequences Plans: ")
    # print("The total lengths of the real applied actions: ", len(whole_plans))
    # print(whole_plans)
    # whole_plans = np.array(whole_plans)
    # np.save("MyMPC_plans.npy", whole_plans)
    #
    # print("Total Execution time: {} s".format(round(time.time() - start_time, 2)))  # EndTime

    # demo_without_mpc()
    demo_with_mpc()
