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


class MPC_ONE_FILE():

    def __init__(self):
        # We didn't implement Batch_Size here ---> no parallel computation here.

        # Should declare reward function here.
        self.M = 1  # Trail Number --- Layer 1
        self.L = 30  # Environment Step --- Layer 2
        self.T = 10  # CEM Iteration --- Layer 3

        self.K = 1000
        self.K_top = 300

        self.action_scale = 150  # +- 150 degree

        self.H = 5  # as self.H self.rollout_steps
        self.a_size = 6  # as self.a_size  self.a_size

        # Corresponding to real action applied after the training of MPC.
        self.reset_State = 0

        # Will be reused during CEM training
        self.sim_State = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return

    def get_CurRealState(self):
        return self.reset_State

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

    @staticmethod
    def sim(s, a=None):
        # simulate the step in mujocol!
        return s + 1.1 * torch.randn(1, device="cpu"), False

    @staticmethod
    def calc_obs(self, s):
        return 0

    @staticmethod
    def calc_reward(s, a=None):
        return s * 100

    def sim_steps(self, actions_1_rollout, s0):
        """
        Be called from self.rollout()
        Handle state reset, one rollout simulate and return one reward.
        actions: [rollout_steps * num_joints] action for a certain K
        """
        # print("Require State s and action a to simulate within rollout.")
        # [H * a_size]
        if actions_1_rollout is None:
            actions_1_rollout = torch.zeros(self.H, self.a_size, device=self.device)

        # reInit sim_State to reset_State
        self.sim_State = s0
        # prepare the reward r
        r = 0
        # continuously simulating based on [H X a_size] action
        for i in range(0, self.H):
            self.sim_State, isStuck = self.sim(self.sim_State, actions_1_rollout[i])
            if isStuck:
                r += -100
        r += self.calc_reward(self.sim_State)  # Can consider actions as well.

        # return o, r, True     # can implement obs as well
        return r, False

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
                # Initialize by [self.action_scale * N(0, I)]
                A_MEANs = torch.zeros(self.H, 1, self.a_size, device=self.device)
                A_STDs = self.action_scale * torch.ones(self.H, 1, self.a_size, device=self.device)
                # print(A_MEANs.shape, A_STDs.shape)  # Size: [H * 1 * a_size]

                # 2-1 Creates: Sequences [rollout_steps * K * num_joints] --> [H * K * a_size]
                Action_K_ROLLOUTs = A_MEANs + A_STDs * torch.randn(self.H, self.K, self.a_size,
                                                                   dtype=torch.float, device=self.device)
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
                    Rollouts_rewards = (self.rollouts(Action_K_ROLLOUTs, s0)).view(1, self.K)

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


if __name__ == "__main__":
    start_time = time.time()  # StartTime

    my_MPC = MPC_ONE_FILE()
    whole_plans = my_MPC.MPC_arch()
    print("The whole Sequences Plans: ")
    print("The total lengths of the real applied actions: ", len(whole_plans))
    print(whole_plans)
    whole_plans = np.array(whole_plans)
    np.save("MyMPC_plans.npy", whole_plans)

    print("Total Execution time: {} s".format(round(time.time() - start_time, 2)))  # EndTime
