import numpy as np
import torch
from torch import jit
from torch import nn, optim
import cvxpy as opt

from envs.pybullet_M1_car import M1_CarEnv
import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from gym.envs.mujoco.ant_v3 import AntEnv
from baselines.ppo1 import mlp_policy

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

# 创建新/继承gym环境，用于MPC测试
class MpcEnv(M1_CarEnv):
    def __init__(self, env_id='M1-Car-v1', model_id='PPO1-M1-Car-v1'):
        super().__init__()

        # env_id = 'Ant-v2'
        # env_id = 'M1-Car-v1'
        # model_path = os.path.join(logger.get_dir(), 'PPO1-gym-Ant-v2')
        # model_path = os.path.join(logger.get_dir(), 'PPO1-M1-Car-v1')
        model_path = os.path.join(logger.get_dir(), model_id)

        # construct the model object, load pre-trained model and render
        pi = policy_fn('pi', self.observation_space, self.action_space)
        U.load_state(model_path)
        self.fn = pi

    # def get_state(self):
    #     # return self._get_obs()
    #     return super.get_obs()

    # def step(self, action):
    #     # return self._get_obs()
    #     return super.step(action)

class MPC():
    def __init__(self, planning_horizon, time_horizon, total_samples, top_samples,
                 env, device, grad_clip=True):
        super().__init__()

        self.H = planning_horizon       # horizon指探索范围，即通过几步推导。如多少组S-A-R-S，时间序列的state-action对
        self.T = time_horizon           # horizon指探索范围，即通过几步推导。如多少组S-A-R-S，时间序列的state-action对
        self.K = total_samples          # total K，每步推导有K组并行模拟同样的SARS
        self.top_K = top_samples        # The Top K，执行完成n步探索后，计算每组模拟结果，并从中选出reward最高的top_K组
        # self.B = batch_size             # 每次MPC推导使用batch_size组K，默认为1。即每次用一批次K和top_k

        self.device = device            # comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.opt_iters = opt_iters      # opt_iters = CEM iteration t   ———— 是指连续优化多少步吗？？
        self.grad_clip = grad_clip      # True

        # self.set_env(env)
        self.env = env
        self.s_size = self.env.observation_space.shape[0]    # state_dim
        self.a_size = self.env.action_space.shape[0]         # action_dim
        # self.state_cost = Q
        # self.action_cost = R


    # def set_env(self, env):
    #     self.env = env
    #
    #     self.s_size = env.observation_space.shape[0]    # state_dim
    #     self.a_size = env.action_space.shape[0]         # action_dim
    #     # self.state_cost = Q
    #     # self.action_cost = R

    def rollout(self, actions_K, obs_0, return_traj=False):
        """
        :param obs_0: t0时刻的状态，从env.get_obs()返回值，即所有试预测的初始状态。
        :param actions_K: K个预预测样本的action序列串
        :return: torch（各步的state、reward/最后的sum_reward）
        """
        # Uncoditional action sequence rollout
        # actions: shape: TxBxA (time, batch, action)
        # assert actions.dim() == 3
        # assert actions.size(1) == self.B, "{}, {}".format(actions.size(1), self.B)
        # assert actions.size(2) == self.a_size
        # T = actions.size(0)     # T 表示time_horizon步数
        ss = []                 # 存放state的列表，即观测值
        rs = []                 # 存放reward的列表
        b = 0
        # Todo! requires_grad=False?? or True?? 我认为False为不修改原有网络
        returns_as = torch.zeros(self.K, self.T, self.a_size, device=self.device)    # 存放K个试预测的action序列串

        # T为向前预测步长。计算连续T步后，得到各模拟sequence的过程（含各步的state、reward和最后的sum_returns）
        for i in range(self.K):
            one_actions = actions_K[b, i]      # 从样本中逐条取出actions序列串（含T步）
            ob = obs_0
            sum_reward = 0
            done = False
            for j in range(self.T):         # 每个样本执行T步试预测
                a = self.env.fn.act(stochastic=False, ob=ob)[0]
                ob, reward, done, _ = self.env.step(a)
                returns_as[i, j, :] = torch.Tensor(a)
                sum_reward += reward
                if done:
                    break
            ss.append(ob)
            rs.append([sum_reward])

        # if return_traj:
        #     return rs, ss
        # else:
        #     return rs
        return returns_as, ss, rs


    # optimize_linearized_model(self, A, B, C, initial_state, target, time_horizon=10, Q=None, R=None, verbose=False):
    def optimize_action_choose(self,
                               initial_state=None,
                               batch_size=1,
                               return_plan=False,
                               return_plan_each_iter=False):
        """
        Note：系统为本次MPC生成的全部随机样本，总数=horizon步数 * （Batch大小 * 每批的样本数K） * action空间

        """
        assert len(initial_state) == self.s_size

        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size      # 每次MPC推导使用batch_size组K，默认为1。即每次用一批次K和top_k

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = torch.zeros(self.H, B, 1, self.a_size, device=self.device)
        a_std = torch.ones(self.H, B, 1, self.a_size, device=self.device)

        # Sample actions (T x (B*K) x A)
        # actions：系统为本次MPC生成的全部随机样本，总数=horizon步数 * （Batch大小 * 每批的样本数K） * action空间
        ini_states = torch.zeros(self.H, B, self.K, self.s_size, device=self.device)
        actions_K = (a_mu + a_std * torch.randn(1, B, self.K, self.a_size,
                                                device=self.device)).view(self.H, B * self.K, self.a_size)
        # actions = torch.tensor(actions_K, requires_grad=True)
        actions_K = torch.tensor(actions_K, requires_grad=False).detach()    # 控制step()不改变现有网络结构和参数
        # detach()返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，
        # 得到的这个Variable永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad

        # # optimizer = optim.SGD([actions], lr=0.1, momentum=0)
        # optimizer = optim.RMSprop([actions], lr=0.1)
        # plan_each_iter = []

        # Create variables
        x = opt.Variable((self.s_size, self.H + 1), name='states')
        u = opt.Variable((self.a_size, self.H), name='actions')

        # # Returns (B*K)
        # returns = self.rollout(self.H, actions_K, return_traj=False)
        # sum_returns = returns.sum()
        # # (-sum_returns).backward()        # Todo! 不需要修改网络需要做反向传播？？
        obs_0 = self.env.get_obs()
        actions, ss, rs = self.rollout(actions_K, obs_0)    # actions=tenser(K, time_horizon, s_size)
        states = torch.Tensor(ss)       # tenser(K, s_size)
        returns = torch.Tensor(rs)      # tenser(K, 1)

        # 从模拟规划中选择top k的sequence，并将排序第一的作为best_plan
        _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
        # topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
        # best_actions = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)
        # best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
        best_actions = actions[topk.view(-1), :].reshape(self.top_K, self.T, self.a_size)
        best_states = states[topk.view(-1), :].reshape(self.top_K, 1, self.s_size)
        best_returns = returns[topk.view(-1), :].reshape(self.top_K, 1)

        best_plan = best_actions[0]
        plan_obs = best_states[0]
        plan_sumReward = best_returns[0]

        # for t in range(time_horizon):
        #
        #     _cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + \
        #             opt.quad_form(u[:, t], R)
        #
        #     _constraints = [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C,
        #                     u[0, t] >= -P.MAX_ACC, u[0, t] <= P.MAX_ACC,
        #                     u[1, t] >= -P.MAX_STEER, u[1, t] <= P.MAX_STEER]
        #     #opt.norm(target[:, t + 1] - x[:, t + 1], 1) <= 0.1]

        # actions = actions_K.detach()
        # Re-fit belief to the K best action sequences
        # _, topk = actions.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
        # print("actions.element_size():" , actions.element_size())
        # print(topk)
        #
        # best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size)

        if return_plan_each_iter:
            return best_actions, best_states, best_returns
        if return_plan:
            return best_plan, plan_obs, plan_sumReward
        else:
            return best_actions





def demo_run_with_mpc():

    pass

def demo_run_without_mpc():

    pass


if __name__ == "__main__":
    # B = 2
    B = 1
    K = 10
    top_K = 5
    Test_env = MpcEnv()
    planner = MPC(planning_horizon=1, time_horizon=5, total_samples=10, top_samples=5,
                  env=Test_env, device=torch.device('cpu'))
    planner.env.enable_render = True
    s0 = planner.env.reset()


    # action = planner.forward(B)
    # action = action.cpu().numpy()


    # demo_run_with_mpc()
    # demo_run_without_mpc()

    while True:
        for i_step in range(1000):
            ob = s0
            # action = pi.act(stochastic=False, ob=ob)[0]
            # action = env.action_space.sample()

            acs, ob, sumReward = planner.optimize_action_choose(initial_state=ob, return_plan=True, return_plan_each_iter=False)
            for j in range(planner.T):
                s0, reward, done, _ = planner.env.step(acs[j])

                if done:
                    s0 = planner.env.reset()
                    break