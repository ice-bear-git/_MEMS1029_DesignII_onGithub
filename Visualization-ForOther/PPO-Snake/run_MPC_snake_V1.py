import torch
from envs.pybullet_MPC_snake import MPC_Basic_Snake
from policies.MPC import MPC
import time
import pybullet as p
import numpy as np


class MPC_AGENT(MPC):
    def __init__(self, env, action_scale=1, a_size=6, H=5, L=128, T=10, K=1000, K_top=300):
        super().__init__(action_scale, a_size, H, L, T, K, K_top)
        self.env = env

    def save_cur_real_state(self, stepIndex):
        if stepIndex == 0:
            self.env.save_state(isFirstSave=True)    # this is required!
        else:
            self.env.save_state()    # The default isFirstSave is False
        pass

    def real_step(self, picked_action=None):
        if picked_action is None:
            picked_action = torch.zeros(1, self.a_size, device=self.device).tolist()

        # restore_state to eliminate the effect of the last rollout simulation among K Rollouts
        self.env.restore_state()
        # Then, use step to really perform the picked_action
        # leave the save_state to self.save_cur_real_state() that called in the second layer of run_MPC() agent
        reward, _, done = self.env.step(picked_action)
        # Whether finished or not
        if done:
            print(" Already reach the goal! Finish all of the training. -- Break From the MPC second loop.")
            return True, reward  # Earlier Finish
        else:
            return False, reward  # Earlier Finish

    # def Sim_1_Rollout(self, Action_1_ROLLOUT, sum_allReward=True):
    #     """
    #     Must consider action scaling are different!!!!!
    #     :param Action_1_ROLLOUT:
    #     :return: unsuccess
    #     """
    #     p.setRealTimeSimulation(0)

    #     if Action_1_ROLLOUT is None:
    #         Action_1_ROLLOUT = torch.zeros(self.H, self.a_size, device=self.device)

    #     self.env.restore_state()

    #     if sum_allReward:
    #         reward = 0
    #         for i in range(0, self.H):
    #             action = (Action_1_ROLLOUT[i]).tolist()
    #             each_reward, abandon, done = self.env.step(action)
    #             reward += each_reward
    #             if abandon or done:
    #                 return reward, abandon, done
    #             if i == (self.H-1):
    #                 return reward, abandon, done
    #     else:
    #         for i in range(0, self.H):
    #             action = (Action_1_ROLLOUT[i]).tolist()
    #             reward, abandon, done = self.env.step(action)
    #             if abandon or done:
    #                 return reward, abandon, done
    #             if i == (self.H-1):
    #                 return reward, abandon, done

    def Sim_1_Rollout(self, Action_1_ROLLOUT):
        """
        Must consider action scaling are different!!!!!
        :param Action_1_ROLLOUT:
        :return: unsuccess
        """
        p.setRealTimeSimulation(0)

        if Action_1_ROLLOUT is None:
            Action_1_ROLLOUT = torch.zeros(self.H, self.a_size, device=self.device)

        self.env.restore_state()
        reward = 0
        for i in range(0, self.H):
            action = (Action_1_ROLLOUT[i]).tolist()
            each_reward, abandon, done = self.env.step(action)
            reward += ((self.H-i+5)/self.H) * each_reward
            if abandon or done:
                return reward, abandon, done
            if i == (self.H-1):
                return reward, abandon, done


class Snake_MPC_V1(MPC_Basic_Snake):
    def __init__(self, target_position=None, action_Scaling_list=None):
        if target_position is None:
            target_position = [1, 0, 0.5]
        if action_Scaling_list is None:
            action_Scaling_list = [1, 1, 1, 1, 1, 1]
        super().__init__(target_position=target_position, action_Scaling_list=action_Scaling_list)


# def train_MPC(env_id=None, env=None):
#     """
#     :param env_id:
#     :param env:
#     :return:
#     """
#
#     # TODO!!!!! -------------训练----------------
#     p.connect(p.DIRECT)
#     # p.connect(p.GUI)
#
#     # create a PyBullet env
#     env = RaceCar_MPC_V1(target_position=[1, 0, 0.5])
#     print("re-setting environment")
#     env.reset()
#
#     # initialize a MPC agent
#     MPC_agent = MPC_AGENT(env, action_scale=1, a_size=3, H=10, L=128, T=4, K=600, K_top=120)
#     # p.setRealTimeSimulation(0)    # 视频录制 --- log
#     start_time = time.time()  # StartTime
#     plan_sequences = MPC_agent.run_MPC()
#     Execution_Time = time.time() - start_time
#     print("Total Execution time: {} s".format(round(Execution_Time, 2)))  # EndTime
#
#     np.save("MyMPC_Apr7.npy", plan_sequences)
#     np.save("Execution_Time_Apr7.npy", np.array([Execution_Time]))
#

def train_MPC_OnColab():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--Target', help='Target Position', nargs='+', type=float, default=[1, 1, 0.5])
    parser.add_argument('--action_Scaling_list',
                        help='action_Scaling_list', nargs='+', type=float, default=[1, 1, 1, 1, 1, 1])
    parser.add_argument('--Name', help='Trail name', type=str, default="MPC_APR7")
    parser.add_argument('--H', help='MPC Looking ahead how many steps', type=int, default=5)
    parser.add_argument('--L', help='MPC second loop --- num of real actions', type=int, default=256)
    parser.add_argument('--a_size', help='a_size', type=int, default=6)

    args = parser.parse_args()

    # p.connect(p.GUI)
    p.connect(p.DIRECT)
    # create a PyBullet env
    print("target_position: ", args.Target)
    env = Snake_MPC_V1(target_position=args.Target, action_Scaling_list=args.action_Scaling_list)
    print("re-setting environment")
    env.reset()
    # initialize a MPC agent
    MPC_agent = MPC_AGENT(env, action_scale=1, a_size=args.a_size, H=args.H, L=args.L, T=5, K=10000, K_top=1000)
    start_time = time.time()  # StartTime
    plan_sequences = MPC_agent.run_MPC()
    Execution_Time = time.time() - start_time
    print("Total Execution time: {} s".format(round(Execution_Time, 2)))  # EndTime

    Plan_File_Name = args.Name + "_plan.npy"
    np.save(Plan_File_Name, plan_sequences)
    TIME_File_Name = args.Name + "_time.npy"
    np.save(TIME_File_Name, np.array([Execution_Time]))
    return


if __name__ == "__main__":
    # train_MPC()
    train_MPC_OnColab()