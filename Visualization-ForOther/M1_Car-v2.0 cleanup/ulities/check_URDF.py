import os
import pybullet as p
import pybullet_data
import time
import gym

import numpy as np
from gym import error, spaces, utils
from collections import namedtuple
from attr_dict import AttrDict
import pprint

from envs.M1_car import M1_CarEnv

# global physicsCilent

gym.logger.set_level(40)
class UREnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=True):
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(os.getcwd())
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.URid = -1

        # 加载地面
        # p.loadURDF("plane.urdf")
        # self.URid = p.loadURDF("URDF/snake.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        # 加载小测
        self.URid = p.loadURDF("racecar/racecar.urdf")

        # conInfo = p.getConnectionInfo(physicsCilent)
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

        # 调整相机视角，初始对准目标的position
        object_pos = [0, 0 , 0]
        # object_pos = [random.uniform(0.5, 0.8), random.uniform(-0.2, 0.2), 0.05]
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=3, cameraPitch=-40, cameraTargetPosition=object_pos)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)   # 打开模拟窗口
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)         # 关闭左侧控件

        p.setRealTimeSimulation(1)


    def close(self):
        p.disconnect()

    def setJoint_by_controlBar(self):
        control_joint_name = self.ctrlJointNames
        joints = self.joints

        #p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)  # 允许机械臂慢慢渲染
        parameter = []
        for i in range(len(control_joint_name)):
            parameter.append(p.readUserDebugParameter(i))
            jointName = control_joint_name[i]
            joint = joints[i]
            parameter_sim = parameter[i]
            p.setJointMotorControl2(self.URid, joint.id, p.POSITION_CONTROL,
                                    targetPosition=parameter_sim,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
        p.stepSimulation()

    def viewControlBar(self):
        '''
        在模拟窗口右上角显示各关节的控件bar
        :return:
        '''
        # 登记各个节点的信息
        jointTypeList = ["REVOLUTE", "CONTINUOUS", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.URid)
        jointInfo = namedtuple("jointInfo", ["id", "name", "type",
                                             "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])
        joints_list = []
        for i in range(numJoints):
            info = p.getJointInfo(self.URid, i)
            jointID = info[0]
            jointName = info[1].decode('utf-8')
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType,
                                   jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
            # joints[singleInfo.name] = singleInfo
            joints_list.append(singleInfo)

        pprint.pprint(joints_list)

        # 确定控制bar中的关节的list
        # position_control_joint_name = [jointName for jointName in joints]
        position_control_joint_name = [jointInfo.name for jointInfo in joints_list]
        # position_control_joint_name.remove("panda_hand_joint")    # 删除不需要显示的关节

        position_control_group = []
        for singleInfo in joints_list:
            if singleInfo.name in position_control_joint_name:
                # # 设置控制bar的默认值为中间值，第4、5关节的默认值为指定数值
                # if jointName == "panda_joint4":
                #     defautVal = singleInfo.upperLimit   # 第4连接点设置为最大值，AttrDict采用A.attrName的取值方式
                # elif jointName == "panda_joint6":
                #     defautVal = singleInfo.lowerLimit   # 第6连接点设置为最小值，AttrDict采用A.attrName的取值方式
                # else:
                #     defautVal = (singleInfo.lowerLimit + singleInfo.upperLimit)/2

                if singleInfo.lowerLimit<=0 and singleInfo.upperLimit>=0:
                    defautVal = 0
                else:
                    defautVal = (singleInfo.lowerLimit+singleInfo.upperLimit)/2
                position_control_group.append(
                    p.addUserDebugParameter(singleInfo.name, singleInfo.lowerLimit, singleInfo.upperLimit, defautVal))

        p.stepSimulation()

        print(f"共有{len(position_control_group)}个joints：{position_control_joint_name}")
        self.ctrlJointNames = position_control_joint_name
        self.joints = joints_list


def create_snakeEnv():
    U = UREnv(gui=True)
    U.reset()

    for i in range(U.stepNum):
        print("step=", i+1)
        U.step()
        time.sleep(1./240.)

        # if i < U.stepNum-1:
        #     U.position = np.array(p.getLinkState(U.pandaUid, U.link_index)[4], dtype=float)
        #     print("\t最后target{}:".format(U.link_index), U.robotEndPos)
        #     print("\t最后real{}:".format(U.link_index), U.position)
        #     print("\t最后diff:", U.robotEndPos-U.position)
        # else:
        #     U.position = np.array(p.getLinkState(U.pandaUid, U.link_index)[4], dtype=float)
        #     print("\t最后target{}:".format(U.link_index), U.robotEndPos)
        #     print("\t最后real{}:".format(U.link_index), U.position)
        #     print("\t最后diff:", U.robotEndPos-U.position)

    time.sleep(5)
    U.close()


def create_env_shape():
    # 连接引擎

    # 添加资源路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(numSolverIterations=10)

    # 载入地面模型，useMaximalCoordinates加大坐标刻度可以加快加载
    p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

    p.setRealTimeSimulation(1)

    # 创建过程中不渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # 不展示GUI的套件
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # 禁用 tinyrenderer
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    # 创建视觉模型和碰撞箱模型时共用的两个参数
    shift = [0, -0.02, 0]
    scale = [1, 1, 1]

    # 创建视觉形状和碰撞箱形状
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName="duck.obj",
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.4, 0.4, 0],
        visualFramePosition=shift,
        meshScale=scale
    )
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName="duck_vhacd.obj",
        collisionFramePosition=shift,
        meshScale=scale
    )

    # # # 使用创建的视觉形状和碰撞箱形状使用createMultiBody将两者结合在一起
    # p.createMultiBody(
    #     baseMass=5,
    #     baseCollisionShapeIndex=collision_shape_id,
    #     baseVisualShapeIndex=visual_shape_id,
    #     basePosition=[0, 0, 0],
    #     # baseOrientation=[0, math.pi/2, 0, 0.8],
    #     # baseOrientation=[0, 0, 0, 2.5],
    #     useMaximalCoordinates=True
    # )

    # 使用创建的视觉形状和碰撞箱形状使用createMultiBody将两者结合在一起
    for i in range(3):
        p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 1.5 * i, 1],
            useMaximalCoordinates=True
        )

    halfEx = [5, 0.5, 2]    # 长宽高
    # 创建一面墙
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=halfEx
    )

    collison_box_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=halfEx
    )
    #
    wall_id = p.createMultiBody(
        baseMass=10,
        baseCollisionShapeIndex=collison_box_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[2, 6, 2]
    )

    # 创建结束，重新开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    while (1):
        p.stepSimulation()
        time.sleep(1./240.)






def check_params():
    # # 连接物理引擎
    # p.connect(p.GUI)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # p.resetSimulation()
    #
    # print(pybullet_data.getDataPath())
    # # reset重力
    # p.setGravity(0, 0, -10)
    #
    # # 实时仿真
    # useRealTimeSim = 1
    #
    # # 加载地面
    # p.loadURDF("plane.urdf")
    # # p.loadSDF("stadium.sdf")
    #
    # # p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # # print(os.path.join(pybullet_data.getDataPath(), "ramdom_urdfs/000/000.urdf"))
    # p.loadURDF(os.path.join(pybullet_data.getDataPath(), "random_urdfs/001/001.urdf"))
    # # p.loadURDF(os.path.join(pybullet_data.getDataPath(), "a1/a1.urdf"))
    #
    # # 加载小测
    # car = p.loadURDF("racecar/racecar.urdf")

    env = M1_CarEnv()
    ob = env.reset()
    car = env.URid

    inactive_wheels = [3, 5, 7]
    wheels = [2]

    for wheel in inactive_wheels:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # 转向轮
    steering = [4, 6]

    # 自定义参数滑块，分别为速度，转向，驱动力
    targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -10, 10, 0)
    maxForceSlider = p.addUserDebugParameter("maxForce", -10, 10, 0)
    steeringSlider = p.addUserDebugParameter("steering", -1.5, 1.5, 0)


    p.setRealTimeSimulation(1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    # 开始仿真
    while 1:
        # 读取速度，转向角度，驱动力参数
        maxForce = p.readUserDebugParameter(maxForceSlider)
        targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = p.readUserDebugParameter(steeringSlider)
        # targetVelocity, maxForce, steeringAngle = [10, 10, -0.5]

        # 根据上面读取到的值对关机进行设置
        for wheel in wheels:
            p.setJointMotorControl2(car,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=targetVelocity,
                                    force=maxForce)

        for steer in steering:
            p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)





def test_pybullet():
    U = UREnv(gui=True)
    U.reset()

    textColor = [1, 1, 0]
    # 先设置一个空内容
    debug_text_id = p.addUserDebugText(
        text="",
        textPosition=[0, 0, 2],
        textColorRGB=textColor,
        textSize=2.5
    )

    print(os.getcwd() + "/log/keyboard2.mp4")
    stepNum = 100
    logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.getcwd() + "/log/keyboard2.mp4")
    while (stepNum):
        stepNum = stepNum-1
        U.step()
        print("stepNum=", stepNum)

        # 按下“W”切换线框视角，按下“G”打开或关闭GUI组件。我们还可以自定义键盘事件和鼠标事件:
        key_dict = p.getKeyboardEvents()
        if len(key_dict):
            print("key_dict:", key_dict)
            if p.B3G_UP_ARROW in key_dict and p.B3G_LEFT_ARROW in key_dict:     # 左前

                debug_text_id = p.addUserDebugText(
                    text="up + left",
                    textPosition=[0, 0, 2],
                    textColorRGB=textColor,
                    textSize=2.5,
                    replaceItemUniqueId=debug_text_id
                )

            elif p.B3G_UP_ARROW in key_dict and p.B3G_RIGHT_ARROW in key_dict:  # 右前

                debug_text_id = p.addUserDebugText(
                    text="up + right",
                    textPosition=[0, 0, 2],
                    textColorRGB=textColor,
                    textSize=2.5,
                    replaceItemUniqueId=debug_text_id
                )

            elif p.B3G_UP_ARROW in key_dict:        # 向前

                debug_text_id = p.addUserDebugText(
                    text="down",
                    textPosition=[0, 0, 2],
                    textColorRGB=textColor,
                    textSize=2.5,
                    replaceItemUniqueId=debug_text_id
                )


            elif p.B3G_LEFT_ARROW in key_dict:        # 原地左转
                # p.setJointMotorControlArray(
                #     bodyUniqueId=robot_id,
                #     jointIndices=[2, 3, 6, 7],
                #     controlMode=p.VELOCITY_CONTROL,
                #     targetVelocities=[maxV / t, maxV / t, -maxV / t, -maxV / t],
                #     forces=[maxF / t, maxF / t, maxF / t, maxF / t]
                # )
                debug_text_id = p.addUserDebugText(
                    text="left",
                    textPosition=[0, 0, 2],
                    textColorRGB=textColor,
                    textSize=2.5,
                    replaceItemUniqueId=debug_text_id
                )

            elif p.B3G_RIGHT_ARROW in key_dict:        # 原地右转

                debug_text_id = p.addUserDebugText(
                    text="right",
                    textPosition=[0, 0, 2],
                    textColorRGB=textColor,
                    textSize=2.5,
                    replaceItemUniqueId=debug_text_id
                )

        else:           # 没有按键，则停下

            debug_text_id = p.addUserDebugText(
                text="",
                textPosition=[0, 0, 2],
                textColorRGB=textColor,
                textSize=2.5,
                replaceItemUniqueId=debug_text_id
            )
    p.stopStateLogging(logging_id)
    U.close()




if __name__ == '__main__':
    ''' -- CHECK1: joint params --'''
    # check_params()
    # exit(0)


    ''' -- CHECK2: create new shape --'''
    # U = UREnv(gui=True)

    # p.setGravity(0, 0, -10)
    # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=3, cameraPitch=-40, cameraTargetPosition=[0,0,0])
    # create_env_shape()
    # exit()

    # 加载地面
    # p.loadURDF("plane.urdf", useFixedBase=True)

    # S = Snake(model=1, gui=True)
    # S.reset()
    # p.setGravity(0, 0, -10)
    # p.setGravity(0, 0, -8)
    #
    # p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=3, cameraPitch=-40, cameraTargetPosition=[0,0,0])
    # numModular, stepInverval, actionType, Poslist = read_sim_file("./control_files/LinearProgressionx4")

    # robotStepPos = list(U.startPos_array + U.step_array)
    # jointPoses = p.calculateInverseKinematics(U.URid, endEffectorLinkIndex=U.link_index,
    #                                           targetPosition=robotStepPos,
    #                                           # targetOrientation=self.robotEndPosture,
    #                                           maxNumIterations=U.maxIter)
    # p.setJointMotorControlArray(U.URid, range(len(jointPoses)), p.POSITION_CONTROL, targetPositions=jointPoses)

    # for jointPoses in Poslist:         # 移动时间
    #     p.setJointMotorControlArray(S.URid, range(len(jointPoses)), p.POSITION_CONTROL, targetPositions=jointPoses)
    #
    #     p.stepSimulation()      # 这个是一定要写的，每次移动都要调用这个方法
    #     time.sleep(1./240.)
    #     time.sleep(0.1)
    #
    # exit(0)


    ''' -- DEMO 3 --'''
    U = UREnv(gui=True)

    U.viewControlBar()
    while True:
        key_dict = p.getKeyboardEvents()
        if len(key_dict):
            print("exit !")
            exit(0)
        else:
            time.sleep(0.01)