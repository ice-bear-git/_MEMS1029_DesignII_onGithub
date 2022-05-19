import math
import os

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from sklearn import preprocessing


class PPO_Basic_Snake(gym.Env):

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

        # Relates to saved state
        # self.head_pos_before = [0, 0]
        # self.body_pos_before = [0, 0]

        # Relates to visualization
        self.delay = (1 / 240.) * 2  # self.dt

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
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.startPosi = [0.2, 0, 0.3]
        self.startPose = [0, 0, 0]

        self.URid = p.loadURDF(os.path.join(self.ROOT_DIR, "URDF_Snake/6-Module-Mar9-V1.urdf"),
                               useFixedBase=0,
                               basePosition=self.startPosi,
                               # globalScaling=3.0,
                               globalScaling=2.0,
                               # basePosition=[0, .0, .0],      # TODO: 不知道该不该启用
                               flags=(p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE))
        self.num_motor = 6

        self.print_agent_info()

        # Relates to MPC -- resetting for different rollouts
        self.savedStateID = None

        # TODO：这个应该是PPO里必须包含的--Action_space && Observations_space
        # self.action_space = spaces.MultiDiscrete(nvec=[15, 15, 15, 15, 15, 15])
        # self.action_space = spaces.Discrete(90)
        # self.action_space = spaces.MultiDiscrete(nvec=[60,60,60,60,60,60])
        # self.action_space = spaces.MultiDiscrete(nvec= np.array([-3.0]*9), np.array([-3.0]*9))
        self.action_space = spaces.Box(low=np.array([-90]*self.num_motor),
                                       high=np.array([90]*self.num_motor), dtype=np.float64)

        # self.observation_space = spaces.Box(np.array([-200.0]*11),
        #                                     np.array([200.0]*11))               # 10个状态值，值域为-1到1
        # self.observation_space = spaces.Box(low=np.array([-3.0]*11),
        #                                     high=np.array([3.0]*11), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-3.0]*6),
                                            high=np.array([3.0]*6), dtype=np.float64)       # 10个状态值，值域为-1到1
        # TODO!!!!! 非必要功能放到了外面。且ulities目录中提供了两种可视化测试agent的URDF/mujoco的工具。

    """ Reset """
    def reset(self, **kwargs):
        """
        Must be Implemented by the interface gym.Env
        :return:
        """
        # Initializing
        # p.resetSimulation()
        # p.setGravity(0, 0, -9.8)
        p.setGravity(0, 0, -98)
        # TODO: 这一行的含义还不确定是否有用。我先注释掉了
        # p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setTimeStep(self.delay)  # 设置时延参数
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # setup camera
        p.resetDebugVisualizerCamera(cameraDistance=1.0,
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
            # p.changeDynamics(self.URid, linkIndex=i, lateralFriction=0.0001, spinningFriction=0.005)
            # p.changeDynamics(self.URid, linkIndex=i, lateralFriction=0.02, spinningFriction=0.02)
            p.changeDynamics(self.URid, linkIndex=i, lateralFriction=0.2, spinningFriction=0.1)
            p.setJointMotorControl2(bodyUniqueId=self.URid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0)

        # Setup target
        # target_startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        # p.resetBasePositionAndOrientation(self.targetUid, self.target_position, target_startOrientation)

        # step stepSimulation --- must be done at the end of reset function
        for i in range(100):
            p.stepSimulation()  # Make sure the target will move to the desired position
        # end the for loop

        return self.get_obs()

    """ Convert degrees action to radians command """
    def input_action(self, *, action):
        """
        用此函数控制运动，简洁
        *: 要求调用function的时候，必须要写变量名=输入值
        """

        # print(action)

        for i in range(self.num_motor):
            # index = self.num_motor-i-1
            index = i
            # For "self.action_space = spaces.MultiDiscrete(nvec=[5, 5, 5, 5, 5, 5])"
            # degree = (action[index]-7)*30 * self.action_Scaling_list[index]
            # degree = (action[index]-0)*70 * self.action_Scaling_list[index]
            # degree = (action[index]-0)*180/math.pi * self.action_Scaling_list[index]
            degree = action[index] * self.action_Scaling_list[index]
            # print(degree)
            p.setJointMotorControl2(bodyUniqueId=self.URid,
                                    jointIndex=index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=math.radians(degree))
        p.stepSimulation()

    """ Get Observations """
    def get_obs(self):
        """
        Decide what should be included inside of the observation.
        Head Position;
        Shape--relative angles and distance;
        Head2Target Orientation;
        """
        # proprioceptive_observation = super().get_obs()
        # (-1.1913180463086002e-05, -9.211808486818358e-05, 5.9127429286265354e-05)
        # (0.0014092154753881235, 0.0007372734890095624, 0.0020955224062138554)
        # line_vel, ang_vel = p.getBaseVelocity(self.URid)

        # head_x, head_y = self.get_head_position()
        # self.get_head_position()
        # self.get_head_degree_around_z()
        # self.print_body_position()
        # print(head_x)
        # head_x = preprocessing.normalize(np.array(head_x).reshape(-1, 1))
        # print(head_x)s
        joints_angle_list, joints_pos_list = self.get_joint_PosAndVel()
        # print(joints_angle_list)
        # joints_angle_list = preprocessing.normalize(joints_angle_list.reshape(-1, 1))
        # print(joints_angle_list)

        # self.get_body2target_distance()
        # (self.get_body2target_distance() - self.get_head2target_distance()) * 1e3
        # self.get_motors_accumulativeLength() * 1e3
        # abs(self.get_body_orientation_diff())
        # abs(self.get_toward_target_orientation_diff())

        observation = np.concatenate([
            # [head_x, head_y],                 # 2
            # [self.get_body2target_distance()],    # 1
            # [(self.get_body2target_distance() - self.get_head2target_distance()) * 1e3],      # 1
            # [self.get_motors_accumulativeLength() * 1e3],       # 1
            # [self.get_body_orientation_diff()],            # 1
            # [self.get_toward_target_orientation_diff()],   # 1
            # [self.get_body_orientation_diff()],             # 1
            joints_angle_list,                          # 6
            # joints_pos_list,                            # 6
        ]).reshape(-1)
        # Define axis used to normalize the data along.
        # print(observation)

        # output = output.clamp(-5, 5).sigmoid()*1

        # If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
        normalized_arr = preprocessing.normalize(observation.reshape(1, -1), axis=1)*1e3    # preprocessing for NN
        # input!!!
        # normalized_arr = preprocessing.normalize(observation.reshape(1, -1))    # preprocessing for NN input!!!

        print("normalized_arr", normalized_arr)
        return normalized_arr

    def get_joint_PosAndVel(self):
        joints_states = p.getJointStates(self.URid, range(self.num_motor))
        angle_pos_list = [math.degrees(jointInfo[0]) for jointInfo in joints_states]
        angle_vel_list = [jointInfo[1] for jointInfo in joints_states]
        return angle_pos_list, angle_vel_list

    def render(self, mode='human'):
        """
        Must be Implemented by the interface gym.Env
        :param mode:
        :return:
        """
        pass

    """ [Private] Position """
    def get_target_position(self):
        target_pos = self.target_position
        return target_pos[0], target_pos[1]

    def get_head_position(self):
        head_position_frame_world = p.getLinkState(self.URid, linkIndex=0)[4]
        head_x, head_y = head_position_frame_world[0], head_position_frame_world[1]
        print("head_posi: x: {}, y: {}".format(head_x, head_y))
        return head_x, head_y

    def get_tail_position(self):
        tail_position_frame_world = p.getLinkState(self.URid, linkIndex=self.num_motor-1)[4]
        tail_x, tail_y = tail_position_frame_world[0], tail_position_frame_world[1]
        print("head_posi: x: {}, y: {}".format(tail_x, tail_y))
        return tail_x, tail_y

    def get_bodyMean_position(self):
        x_position = []
        y_position = []
        for i in range(self.num_motor):
            position_frame_world = p.getLinkState(self.URid, linkIndex=i)[4]
            x_position.append(position_frame_world[0])
            y_position.append(position_frame_world[1])

        body_x = np.mean(x_position)
        body_y = np.mean(y_position)

        print("body_x: {}, body_y: {}".format(body_x, body_y))
        return body_x, body_y

    """ [Public] Distance + Velocity """
    def cal_tail2head_length(self):
        head_x, head_y = self.get_head_position()
        tail_x, tail_y = self.get_tail_position()
        return np.linalg.norm(np.array([tail_x, tail_y]) - np.array([head_x, head_y]))

    def cal_head2target_distance(self):
        head_x, head_y = self.get_head_position()
        target_x, target_y = self.get_target_position()
        return np.linalg.norm(np.array([target_x, target_y]) - np.array([head_x, head_y]))

    # def cal_bodyMean_velocity(self):

    """ [Private] YAW degrees """
    def get_head_yaw_degree(self):
        world_rotation_frame = p.getLinkState(self.URid, linkIndex=0)[5]
        head_yaw = math.degrees(p.getEulerFromQuaternion(world_rotation_frame)[2])
        return head_yaw

    def get_bodyMean_yaw_degree(self):
        yaw_degrees = []
        for i in range(self.num_motor):
            world_rotation_frame = p.getLinkState(self.URid, linkIndex=i)[5]
            yaw_degrees.append(math.degrees(p.getEulerFromQuaternion(world_rotation_frame)[2]))

        bodyMean_yaw_degree = np.mean(yaw_degrees)
        print("bodyMean_yaw_degree: {}".format(bodyMean_yaw_degree))
        return bodyMean_yaw_degree

    def get_head2target_yaw_degree(self):
        """ Front-Left is positive"""
        head_x, head_y = self.get_head_position()
        target_x, target_y = self.get_target_position()

        adjacent = target_x - head_x
        opposite = target_y - head_y

        head2target_yaw_degree = math.degrees(math.atan2(opposite, adjacent))
        return head2target_yaw_degree

    def get_bodyMean2target_yaw_degree(self):
        """ Front-Left is positive"""
        body_x, body_y = self.get_bodyMean_position()
        target_x, target_y = self.get_target_position()

        adjacent = target_x - body_x
        opposite = target_y - body_y

        bodyMean2target_yaw_degree = math.degrees(math.atan2(opposite, adjacent))
        return bodyMean2target_yaw_degree

    def get_bodyMean_velocity_degree(self):
        x_velocity = []
        y_velocity = []
        for i in range(self.num_motor):
            position_frame_world = p.getLinkState(self.URid, linkIndex=i, computeLinkVelocity=1)[6]
            x_velocity.append(position_frame_world[0])
            y_velocity.append(position_frame_world[1])

        body_Vx = np.mean(x_velocity)
        body_Vy = np.mean(y_velocity)

        print("body_Vx: {}, body_Vy: {}".format(body_Vx, body_Vy))

        adjacent = body_Vx
        opposite = body_Vy

        bodyMean_velocity_degree = math.degrees(math.atan2(opposite, adjacent))
        return bodyMean_velocity_degree

    """ [Public] Orientation angles """
    def cal_body2Target_velocityOrientDiff(self):
        """
        MOST IMPORTANT
        When it's positive, the body should turn left!
        """

        bodyMean2target_yaw_degree = self.get_bodyMean2target_yaw_degree()
        bodyMean_velocity_degree = self.get_bodyMean_yaw_degree()
        bodyVelocity_orientDiff = bodyMean2target_yaw_degree - bodyMean_velocity_degree
        print("bodyMean2target_yaw_degree: {}, bodyMean_velocity_degree: {}, bodyVelocity_orientDiff: {}".format(
            bodyMean2target_yaw_degree, bodyMean_velocity_degree, bodyVelocity_orientDiff))
        return bodyVelocity_orientDiff

    def cal_headBody2Target_towardOrientDiff(self):
        """ When it's positive, it should rotate the head to the Front-Left """
        head2target_yaw_degree = self.get_head2target_yaw_degree()
        bodyMean2target_yaw_degree = self.get_bodyMean2target_yaw_degree()
        targetToward_orientDiff = head2target_yaw_degree - bodyMean2target_yaw_degree
        return targetToward_orientDiff

    def cal_bodyShape_orientDiff(self):
        """ When it's positive, the head is more front-left than the average links """
        head_yaw_degree = self.get_head_yaw_degree()
        bodyMean_yaw_degree = self.get_bodyMean_yaw_degree()
        bodyShape_orientDiff = head_yaw_degree - bodyMean_yaw_degree

        return bodyShape_orientDiff

    """ Printer Agent """
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

    def print_body_position(self):
        for i in range(self.num_motor):
            print("The {} link!".format(i))
            linkStates = p.getLinkState(self.URid, linkIndex=i)
            position_linkcom_world = linkStates[0]
            world_rotation_linkcom = linkStates[1]
            position_linkcom_frame = linkStates[2]
            frame_rotation_linkcom = linkStates[3]
            position_frame_world = linkStates[4]
            world_rotation_frame = linkStates[5]
            # worldLinkLinearVelocity = p.getLinkState(self.URid, linkIndex=i, computeLinkVelocity=1)[6]
            """
            position_linkcom_world, world_rotation_linkcom,
            position_linkcom_frame, frame_rotation_linkcom, -- want!!!
            position_frame_world, world_rotation_frame,
            linearVelocity_linkcom_world, angularVelocity_linkcom_world
            """
            print("0. position_linkcom_world: ", position_linkcom_world)
            """ Convert quaternion [x,y,z,w] to Euler [roll, pitch, yaw] as in URDF/SDF convention """
            print("1. world_rotation_linkcom: ", math.degrees(p.getEulerFromQuaternion(world_rotation_linkcom)[2]))

            print("2. position_linkcom_frame: ", position_linkcom_frame)
            print("3. frame_rotation_linkcom: ", math.degrees(p.getEulerFromQuaternion(frame_rotation_linkcom)[2]))

            print("4. position_frame_world: ", position_frame_world)
            print("5. world_rotation_frame: ", math.degrees(p.getEulerFromQuaternion(world_rotation_frame)[2]))

            # print("worldLinkLinearVelocity: ", worldLinkLinearVelocity)

            # curr_posi = p.getLinkState(self.URid, linkIndex=i)[0]
            # curr_pose = p.getLinkState(self.URid, linkIndex=i)[1]
            # curr_local_posi = p.getLinkState(self.URid, linkIndex=i)[2]
            # curr_local_pose = p.getLinkState(self.URid, linkIndex=i)[5]
            # print("Link: index: {}, position: {} pose: {}".format(i, curr_posi,
            #                                                       math.degrees(p.getEulerFromQuaternion(curr_pose)[2])))
            # print("\tcurr_local_posi: {}, curr_local_pose: {}".format(i, curr_local_posi,
            #                                                           math.degrees(
            #                                                               p.getEulerFromQuaternion(curr_local_pose)[
            #                                                                   2])))
        pass

    """ Save state """
    def save_state(self, isFirstSave=False):
        # In-memory snapshot of state
        if not isFirstSave:
            p.removeState(self.savedStateID)
        self.savedStateID = p.saveState()

    """ restore from saved state ID """
    def restore_state(self):
        # Restore from In-memory snapshot of state
        p.restoreState(self.savedStateID)
        # Restore form On-disk snapshot of state -- [unUsed]
        # p.resortState(fileName=os.path.join(self.ROOT_DIR, "bulletBuffer/preRealState.bullet"))

    """ Real step and reward """
    def step(self, action=None):
        """
        Must be Implemented by the interface gym.Env
        1. apply action by p.stepSimulation() --- input_action() method
        2. calculate the reward after the action
        3. return obs, reward, done, info
        """

        """ Initialize the return values """
        reward = 0
        done = False
        abandon = False

        """ Apply the action """
        self.input_action(action=action)  # Apply the action and scaling it inside.

        """ initialize all reward aspects here """
        forward_reward, distance_reward, survive_reward, shape_reward, \
            ctrl_cost, contact_cost, goal_reward, orient_reward = 0, 0, 0, 0, 0, 0, 0, 0

        """ Calculate the key indicators for performances from the records """
        # [1-1 Head]
        # [1-2 Body] forward_reward -- Velocity

        # [2] goal_reward
        """ distance_reward """
        tail2head_length = self.cal_tail2head_length()
        print("tail2head_length: ", tail2head_length)
        head2target_distance = self.cal_head2target_distance()
        print("head2target_distance: ", head2target_distance)

        """ get_body_orientation_diff """
        targetToward_orientDiff = self.cal_headBody2Target_towardOrientDiff()
        print("targetToward_orientDiff: ", targetToward_orientDiff)
        bodyShape_orientDiff = self.cal_bodyShape_orientDiff()
        print("bodyShape_orientDiff: ", bodyShape_orientDiff)
        bodyVelocity_orientDiff = self.cal_body2Target_velocityOrientDiff()
        print("bodyVelocity_orientDiff: ", bodyVelocity_orientDiff)

        """ Whether we reach the goal or not """

        """ Whether we abandon this trail or not """

        """ Get Overall reward """
        reward += distance_reward + shape_reward + 1e-2 * orient_reward
        # reward += 5e2 * (distance_reward + 0) + (orient_reward + goal_reward)
        reward += (survive_reward - ctrl_cost - contact_cost) + goal_reward
        # print("reward", reward)
        next_observation = self.get_obs()
        # return next_observation, reward, done, info
        return next_observation, reward, done, abandon


