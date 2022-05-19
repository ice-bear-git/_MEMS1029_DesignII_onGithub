import os

import gym
import pybullet as p
import pybullet_data
import numpy as np
import math

# from gym import spaces
import torch


class MPC_Basic_Snake(gym.Env):
    """
    Will require the implementation of function to reward inside of run_MPC_car_v2.py file.
    """

    def __init__(self,
                 fix_target_position=True,
                 target_position=None,
                 action_Scaling_list=None):
        """
        Currently fixed all parameters here.
        """
        super().__init__()  # Initialize gym.Env

        # When to stop
        self.terminate_at_goal = True

        # Relates to target
        self.fix_target_position = fix_target_position
        if target_position is None:
            self.target_position = [1, -2, 0.5]
            print("it should be updates")
        else:
            self.target_position = target_position

        # Relates to Goal Geometry
        self.goal_radius = 0.15
        self.near_to_goal = 0.2

        # Relates to Action Scaling
        if action_Scaling_list is None:
            action_Scaling_list = [1, 1, 1, 1, 1, 1]
        self.action_Scaling_list = action_Scaling_list
        # self.action_Scaling_list = [10, 10, 10, 10, 10, 10]

        # Relates to Rewards
        self.reward_threshold = 1e5  # The upper boundary for rewards in one episode.

        self.velocity_reward_weight = 0.3
        self.distance_reward_weight = 0.7
        self.orient_diff_reward_weight = 0.6
        self.orient_change_reward_weight = 0.4

        # Relates to saved state
        self.dist_At_savedState = 0
        self.angle_diff_before = 0
        self.head_pos_before = [0, 0]
        self.body_pos_before = [0, 0]

        # Relates to visualization
        self.delay = (1 / 240.) * 2  # self.dt
        self.DT = 0.2

        # Relates to PyBullet model
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        useRealTimeSim = 0
        p.setRealTimeSimulation(useRealTimeSim)
        # 2. load plane
        self.planeUid = p.loadURDF("plane.urdf", useMaximalCoordinates=True)

        # 3. Setup Target!
        # self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # self.targetUid = p.loadURDF(os.path.join(self.ROOT_DIR, "URDFs/target_obj.urdf"),
        #                             basePosition=self.target_position)
        halfEx = [0.1, 0.1, 0.1]  # length width height
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=halfEx,
            rgbaColor=[1, 0, 0, 0.9]
        )
        collison_box_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=halfEx
        )
        objectUid = p.createMultiBody(
            baseMass=100,
            baseCollisionShapeIndex=collison_box_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.target_position
        )
        self.targetUid = objectUid

        # 5. load the Snake
        """加载小车————joint_index对应：2 左后动力、3 右后动力、5 左前动力、7 右前动力；4 左前转向、6 右前转向"""
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        # self.targetUid = p.loadURDF(os.path.join(self.ROOT_DIR, "URDFs/target_obj.urdf"),
        #                             basePosition=self.target_position)

        # self.URid = p.loadURDF("racecar/racecar.urdf", basePosition=[0, .0, .0])
        self.URid = p.loadURDF(os.path.join(self.ROOT_DIR, "URDF_Snake/6-Module-Mar9-V1.urdf"),
                               useFixedBase=0,
                               flags=(p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE))
        self.num_motor = 6
        self.startPosi = [0, 0, 0]
        self.startPose = [0, 0, 0]

        self.print_agent_info()

        # Relates to MPC -- resetting for different rollouts
        self.savedStateID = None

        # TODO!!!!! 此处赋动作空间，以便于MPC算法调用（环境和算法分开），三个值的mean分别为3.0、2.0、0.0
        # action_space：maxForce，steeringAngle=0, acceleration。
        # self.action_space = spaces.Box(np.array([6.0, 5.0, -0.8]),
        #                                np.array([10.0, 15.0, 0.8]))               # 3个action输出，值域为-1到1
        # self.observation_space = spaces.Box(np.array([-500.0]*10),
        #                                     np.array([500.0]*10))               # 10个状态值，值域为-1到1

    # TODO!!!!! 非必要功能放到了外面。且lities目录中提供了两种可视化测试agent的URDF/mujoco的工具。
    def print_agent_info(self):
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.URid)
        for i in range(numJoints):
            info = p.getJointInfo(self.URid, i)
            jointID = info[0]
            jointName = info[1].decode('utf-8')
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            print(jointID, " ", jointName, " ", jointType, " ", jointLowerLimit, " ",
                  jointUpperLimit, " ", jointMaxForce, " ", jointMaxVelocity)
        """
        0   Joint_0_fromHead   REVOLUTE   -1.74   1.74   0.0   0.0
        1   Joint_1_fromHead   REVOLUTE   -1.74   1.74   0.0   0.0
        2   Joint_2_fromHead   REVOLUTE   -1.74   1.74   0.0   0.0
        3   Joint_3_fromHead   REVOLUTE   -1.74   1.74   0.0   0.0
        4   Joint_4_fromHead   REVOLUTE   -1.74   1.74   0.0   0.0
        5   Joint_5_fromHead   REVOLUTE   -1.74   1.74   0.0   0.0
        """
        return

    def simulate_save_state_for_Replay(self):
        self.dist_At_savedState = self.get_body2target_distance()
        self.angle_diff_before = self.get_orientation_angel_diffs(body_pos_before=self.body_pos_before,
                                                                  body_pos_after=self.get_body_position())
        self.body_pos_before = self.get_body_position()

    def save_state(self, isFirstSave=False):
        # In-memory snapshot of state
        if not isFirstSave:
            p.removeState(self.savedStateID)
        self.savedStateID = p.saveState()
        # On-disk snapshot of state -- [unUsed]
        # p.saveBullet(os.path.join(self.ROOT_DIR, "bulletBuffer/preRealState.bullet"))
        self.dist_At_savedState = self.get_body2target_distance()
        self.angle_diff_before = self.get_orientation_angel_diffs(body_pos_before=self.body_pos_before,
                                                                  body_pos_after=self.get_body_position())
        self.body_pos_before = self.get_body_position()

    def restore_state(self):
        # Restore from In-memory snapshot of state
        p.restoreState(self.savedStateID)
        # Restore form On-disk snapshot of state -- [unUsed]
        # p.resortState(fileName=os.path.join(self.ROOT_DIR, "bulletBuffer/preRealState.bullet"))

    def reset(self):
        """
        Must be Implemented by the interface gym.Env
        :return:
        """
        # Initializing
        # p.resetSimulation()

        p.setGravity(0, 0, -9.8)
        # TODO: 这一行的含义还不确定是否有用。我先注释掉了
        # p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setTimeStep(self.delay)  # 设置时延参数
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # setup camera
        p.resetDebugVisualizerCamera(cameraDistance=4.0,
                                     cameraYaw=-90,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[-0.1, -0.0, 0.65])
        # For video recording --- Might be required for storing the trained results
        useRealTimeSim = 0
        p.setRealTimeSimulation(useRealTimeSim)

        # Setup the Snake
        p.resetBasePositionAndOrientation(self.URid, self.startPosi, p.getQuaternionFromEuler(self.startPose))

        """ A convenient way to change the dynamics parameters  
        --- <p.setJointMotorControlArray> NOT WORKING!!!
        https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.d6og8ua34um1
        Also see example below:
        https://github.com/bulletphysics/bullet3/blob/e306b274f1885f32b7e9d65062aa942b398805c2/examples/pybullet/tensorflow/humanoid_running.py#L196
        """
        for i in range(self.num_motor):
            # Useful!
            # p.changeDynamics(S.URid, linkIndex=i, lateralFriction=0.0001, spinningFriction=0.005)
            p.changeDynamics(self.URid, linkIndex=i, lateralFriction=0.02, spinningFriction=0.02)
            p.setJointMotorControl2(bodyUniqueId=self.URid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=5,
                                    maxVelocity=5)

        # Setup target
        target_startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.targetUid, self.target_position, target_startOrientation)

        # step stepSimulation --- must be done at the end of reset function
        for i in range(100):
            p.stepSimulation()  # Make sure the target will move to the desired position
        # end the for loop

        pass

    def render(self, mode='human'):
        """
        Must be Implemented by the interface gym.Env
        :param mode:
        :return:
        """
        pass

    def input_action(self, *, action):
        """
        用此函数控制运动，简洁
        *: 要求调用function的时候，必须要写变量名=输入值
        """
        for i in range(self.num_motor):
            p.setJointMotorControl2(bodyUniqueId=self.URid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=action[i] * self.action_Scaling_list[i])

        # p.stepSimulation()
        # reset a special joint's position（like a worm）
        # tail_joint = self.num_motor
        # r = random.uniform(-0.002, 0.002)
        # p.resetJointState(self.URid, jointIndex=tail_joint, targetValue=r)
        p.stepSimulation()

    def step(self, action=None):
        """
        Must be Implemented by the interface gym.Env
        1. apply action by p.stepSimulation()
        2. calculate the reward after the action
        3. return reward --- we didn't need obs for the non-predictive training Net.
        :param action: currVel=action[0], maxForce=action[1], steeringAngle=action[2]
        :return: reward, abandon, done
        """

        # print("action=", action)

        if action is None:
            action = [8, 6, -0.5]  # 按照给定的 [vel, force, steering_angle]

        """ Initialize the return values """
        reward = 0
        abandon = False
        done = False

        """ Record the distance before and after applying the action. """
        # dist_before = self.get_head2target_distance()  # Before applying the action
        dist_before = self.dist_At_savedState  # Before applying the action
        body_pos_before = self.get_body_position()
        angle_diff_before = self.angle_diff_before
        head_distance_before = self.get_head2target_distance()

        self.input_action(action=action)  # Apply the action and scaling it inside.

        head_distance_after = self.get_head2target_distance()
        dist_after = self.get_head2target_distance()  # After applying the action
        body_pos_after = self.get_body_position()
        # head_pos_before, head_pos_after)
        angle_diff_after = self.get_orientation_angel_diffs(body_pos_before, body_pos_after)
        # print("angle_diff_after: {}".format(angle_diff_after))

        """ Calculate the key indicators for performances from the records """
        # Distance and velocity
        distance_delta = dist_before - dist_after
        velocity_to_target = distance_delta / (self.DT * 4)  # in m/s toward target; 6 should be H in MPC
        head_distance_delta = head_distance_before - head_distance_after

        # Orientation and angular_velocity
        angle_diff = angle_diff_after
        angular_velocity = angle_diff_before - angle_diff_after
        # print("angular_velocity {} = angle_diff_before {} "
        #       "- angle_diff_after {} ".format(angular_velocity, angle_diff_before, angle_diff_after))

        """ initialize all reward aspects here """
        forward_reward, distance_reward, survive_reward, \
            ctrl_cost, contact_cost, goal_reward, orient_reward = 0, 0, 0, 0, 0, 0, 0

        # [1-1 Head]
        # [1-2 Body] forward_reward -- Velocity
        # forward_reward += 100 * velocity_to_target * self.velocity_reward_weight
        # forward_reward += 1 * velocity_to_target * self.velocity_reward_weight
        forward_reward += 1e2 * velocity_to_target * self.velocity_reward_weight

        # motors_accumulativeLength
        survive_reward += 5e6 * self.get_motors_relative_position()

        # print("Velocity reward: {}".format(forward_reward))
        if dist_after >= dist_before:
            # Very care about the distance!!!
            # distance_reward += -1000 * abs(distance_delta) * self.distance_reward_weight
            distance_reward += -1000 * self.distance_reward_weight
            distance_reward += 2 * head_distance_delta * self.distance_reward_weight
            reward += -20
        else:
            distance_reward += 500 * abs(distance_delta) * self.distance_reward_weight

            distance_reward += 100 * head_distance_delta * self.distance_reward_weight

        # print("Distance reward: {}".format(distance_reward))

        # [2] goal_reward
        if dist_after <= self.near_to_goal:
            goal_reward += 5

        # [3] orient_reward
        if angle_diff < 20:
            coeff_diff = -0.2
            coeff_change = 2  # angular_velocity itself bring the positive/negative sign!
        elif angle_diff < 50:
            coeff_diff = -1
            coeff_change = 2  # angular_velocity itself bring the positive/negative sign!
        else:
            coeff_diff = -2
            coeff_change = 3  # angular_velocity itself bring the positive/negative sign!

        orient_diff_reward = coeff_diff * angle_diff
        # print("angle_diff: {}, orient_diff_reward: {}".format(angle_diff, orient_diff_reward))

        # angular_velocity itself bring the positive/negative sign!
        # orient_change_reward = coeff_change * angular_velocity
        orient_change_reward = 0 * angular_velocity
        # print("angular_velocity: {}, orient_change_reward: {}".format(angular_velocity, orient_change_reward))
        orient_reward += orient_change_reward * self.orient_change_reward_weight
        orient_reward += orient_diff_reward * self.orient_diff_reward_weight

        """ Get Overall reward """
        reward += 5e2 * (distance_reward + forward_reward) + ((orient_reward + goal_reward)*3.14)/180
        reward += (survive_reward - ctrl_cost - contact_cost)

        """ get_body_orientation_diff """
        body_orientation_diff = self.body_orientation_diff()
        if body_orientation_diff < 90:
            body_angle_diff = -0.2
        elif body_orientation_diff < 140:
            body_angle_diff = -1
        else:
            body_angle_diff = -5

        reward += body_angle_diff * body_orientation_diff

        """ Whether we reach the goal or not """
        goal_reached = dist_after <= self.goal_radius  # 接近达到目标
        if self.terminate_at_goal and goal_reached:
            done = True
        """ Whether we abandon this trail or not """
        if dist_after >= (dist_before * 1.001):
            # TODO：如果是snake，joint的卡死就会导致abandon=True。这里是小车，所以只有too_far会引发abandon
            abandon = True

        return reward, abandon, done

    def get_body_position(self):
        x_position = []
        y_position = []
        for i in range(self.num_motor):
            curr_posi = p.getLinkState(self.URid, linkIndex=i)[0]
            x_position.append(curr_posi[0])
            y_position.append(curr_posi[1])
        # torch.tensor
        body_x = np.mean(x_position)
        body_y = np.mean(y_position)
        return body_x, body_y

    def get_motors_relative_position(self):
        target_x, target_y = self.get_target_position()
        motors_distance = []

        for i in range(self.num_motor):
            curr_posi = p.getLinkState(self.URid, linkIndex=i)[0]
            motors_distance.append(np.linalg.norm(np.array([target_x,target_y])-np.array([curr_posi[0],curr_posi[1]])))

        motors_accumulativeLength = np.sum(np.diff(motors_distance))
        return motors_accumulativeLength

    def get_body2target_distance(self):
        x_position = []
        y_position = []
        for i in range(self.num_motor):
            curr_posi = p.getLinkState(self.URid, linkIndex=i)[0]
            x_position.append(curr_posi[0])
            y_position.append(curr_posi[1])
        body_x = np.mean(x_position)
        body_y = np.mean(y_position)

        target_x, target_y = self.get_target_position()
        return np.linalg.norm(np.array([target_x, target_y]) - np.array([body_x, body_y]))

    def get_head2target_distance(self):
        head_x, head_y = self.get_head_position()
        target_x, target_y = self.get_target_position()
        return np.linalg.norm(np.array([target_x, target_y]) - np.array([head_x, head_y]))

    def get_head_position(self):
        curr_posi, _ = p.getBasePositionAndOrientation(self.URid)
        # print("Body: x: {}, y: {}".format(curr_posi[0], curr_posi[1]))
        return curr_posi[0], curr_posi[1]

    def print_body_position(self):
        for i in range(self.num_motor):
            curr_posi = p.getLinkState(self.URid, linkIndex=i)[0]
            curr_pose = p.getLinkState(self.URid, linkIndex=i)[1]
            curr_local_posi = p.getLinkState(self.URid, linkIndex=i)[2]
            curr_local_pose = p.getLinkState(self.URid, linkIndex=i)[5]
            print("Link: index: {}, position: {} pose: {}".format(i, curr_posi,
                                                                  math.degrees(p.getEulerFromQuaternion(curr_pose)[2])))
            print("\tcurr_local_posi: {}, curr_local_pose: {}".format(i, curr_local_posi,
                                                                      math.degrees(
                                                                          p.getEulerFromQuaternion(curr_local_pose)[
                                                                              2])))
        pass

    def get_Body_angles_mean(self):
        orients = []
        for i in range(self.num_motor):
            curr_pose = p.getLinkState(self.URid, linkIndex=i)[1]
            orients.append(math.degrees(p.getEulerFromQuaternion(curr_pose)[2]))
        # torch.tensor
        mean = torch.mean(torch.tensor(orients)).item()
        # print("mean: {}".format(mean))
        return mean

    def body_orientation_diff(self):
        return abs(self.get_Body_angles_mean() - self.get_head_degree_around_z())

    def get_target_position(self):
        target_pos = self.target_position
        return target_pos[0], target_pos[1]

    def get_target_degree_In_body_view(self):
        """
        向左为正，向右为负
        """
        body_x, body_y = self.get_body_position()
        target_x, target_y = self.get_target_position()

        adjacent = target_x - body_x
        opposite = target_y - body_y

        target_In_head_view = math.degrees(math.atan2(opposite, adjacent))
        # print("target_In_head_view: {}".format(target_In_head_view))

        return target_In_head_view

    # def get_orientation_angel_diffs(self):
    #     head_degree = self.get_head_degree_around_z()
    #     target_degree = self.get_target_degree_In_body_view()
    #     # velocity__degree = self.get_velocity_degree_In_body_view()
    #     orientation_angel_diffs = abs(head_degree - target_degree)
    #
    #     # print("head_angel_around_Z: {}".format(head_angel_around_Z))
    #     # print("orientation_angel_diffs: {}".format(orientation_angel_diffs))
    #
    #     return orientation_angel_diffs

    def get_head_degree_around_z(self):
        """
        NOT BEING USED
        向左为正，向右为负
        """
        curr_posi, curr_orn = p.getBasePositionAndOrientation(self.URid)
        """
        getEulerFromQuaternion returns a list of 3 floating point values, a vec3.
        The rotation order is first roll around X,
        then pitch around Y and finally yaw around Z,
        as in the ROS URDF rpy convention.
        """
        head_angel_around_Z = math.degrees(p.getEulerFromQuaternion(quaternion=curr_orn)[2])  # 向右为负
        # print("head_angel_around_Z: {}".format(head_angel_around_Z))
        # self.get_body_positions_mean()
        return head_angel_around_Z

    @staticmethod
    def get_velocity_degree_In_body_view(body_pos_before, body_pos_after):
        """
       向左为正，向右为负
       """
        body_x_before, body_y_before = body_pos_before
        body_x_after, body_y_after = body_pos_after

        adjacent = body_x_after - body_x_before
        opposite = body_y_after - body_y_before

        velocity_In_head_view = math.degrees(math.atan2(opposite, adjacent))
        return velocity_In_head_view

    def get_orientation_angel_diffs(self, body_pos_before, body_pos_after):
        target_degree = self.get_target_degree_In_body_view()
        velocity_degree = self.get_velocity_degree_In_body_view(body_pos_before=body_pos_before,
                                                                body_pos_after=body_pos_after)
        orientation_angel_diffs = abs(target_degree - velocity_degree)
        return orientation_angel_diffs
