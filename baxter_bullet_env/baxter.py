"""
This code provide a simple control interface to control baxter robot left arm
osc control -- [dx, dy, dz, da]
gripper control -- control the state of the gripper
"""

import operator
import os, inspect
import pybullet
import pybullet as p
import numpy as np
import copy
import math
import pybullet_data

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)
pb_path = pybullet_data.getDataPath()
baxter_path = "./models/baxter_description/urdf/baxter.urdf"


class BaxterLeft:
    """
    control left hand of baxter robot
    """

    def __init__(self, baxter_id, timeStep=0.01, os_min=[-1, -1, -1], os_max=[1, 1, 1]):

        # define the range of the work space
        self.os_min = os_min
        self.os_max = os_max
        self.timeStep = timeStep

        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerForce = 10

        self.finger_a_id = 49
        self.finger_b_id = 51

        self.endEffector_id = 48
        self.camera_axis_id = 44

        self.finger_tips_a_id = 50
        self.finger_tips_b_id = 52

        self.gripper_open = 0.020833
        self.gripper_close = 0

        # define the end effector initial position, gripper state
        self.ini_pos = [0.5, 0.2, 0, 0]

        # loading baxter model and plane
        self.baxter_path = baxter_path
        self.baxterId = baxter_id
        self.endEffector_id = 48
        self.camera_axis_id = 44

        # create control dict for baxter
        self.numJoints = p.getNumJoints(self.baxterId)
        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.baxterId, i)
            # print(jointInfo)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

        # define lower limits, upper limits, joint range, rest pose
        self.ll, self.ul, self.jr = self.getJointRanges(self.baxterId)

        # set the rest pose as the home position
        self.rp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   -0.8462368022380607, -1.630866219617225,
                   -0.08415513820451972, 2.0843043904431457,
                   -0.002042621644835222, 1.1254427955750927,
                   -0.1461959296684458, 0.0, 0.0)

        self.reset(self.ini_pos)

    def getJointRanges(self, bodyId, includeFixed=False):
        """
        lowerLimits : [ float ] * numDofs
        upperLimits : [ float ] * numDofs
        jointRanges : [ float ] * numDofs
        """
        lowerLimits, upperLimits, jointRanges = [], [], []
        numJoints = p.getNumJoints(bodyId)

        for i in range(numJoints):
            jointInfo = p.getJointInfo(bodyId, i)

            if includeFixed or jointInfo[3] > -1:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                lowerLimits.append(ll)
                upperLimits.append(ul)
                jointRanges.append(jr)
        return [lowerLimits, upperLimits, jointRanges]

    def reset(self, pose_gripper):
        """
        reset the robot state with desired end_effector position and angel, gripper state
        :param pose_angle:
        :return:
        """
        p.resetBasePositionAndOrientation(self.baxterId, [0.000, 0.000000, 0.00000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])

        # ik  and reset position
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        pose = pose_gripper[0:3]
        jointPoses = p.calculateInverseKinematics(self.baxterId,
                                                  self.endEffector_id,
                                                  pose, orn,
                                                  lowerLimits=self.ll,
                                                  upperLimits=self.ul,
                                                  jointRanges=self.jr,
                                                  restPoses=self.rp,
                                                  solver=0,
                                                  maxNumIterations=100,
                                                  residualThreshold=.005)

        for i in range(len(self.motorIndices)):
            p.resetJointState(self.baxterId, self.motorIndices[i], jointPoses[i])

        # set gripper state --  close state
        p.resetJointState(self.baxterId, self.finger_a_id, self.gripper_close)
        p.resetJointState(self.baxterId, self.finger_b_id, self.gripper_close)

        # move away the right arm
        right_s0 = 12
        p.resetJointState(self.baxterId, right_s0, -np.pi/2)

        right_s1 = 13
        p.resetJointState(self.baxterId, right_s1, np.pi/2)

    # operation space control
    def osc(self, motorCommands):
        """
        :param motorCommands: change of x, y, z
        :return:
        """
        d_position = motorCommands
        orientation = p.getQuaternionFromEuler([0, -math.pi, 0])

        # get current state
        state = p.getLinkState(self.baxterId, self.endEffector_id)
        end_pose = np.array(state[4])

        # new pos
        new_end_pose = end_pose + d_position
        new_end_pose = np.clip(new_end_pose, self.os_min, self.os_max)

        # ik
        jointPoses = p.calculateInverseKinematics(self.baxterId, self.endEffector_id,
                                                  new_end_pose, orientation,
                                                  lowerLimits=self.ll, upperLimits=self.ul,
                                                  jointRanges=self.jr, restPoses=self.rp, solver=0,
                                                  maxNumIterations=100,
                                                  residualThreshold=.005)

        p.setJointMotorControlArray(bodyUniqueId=self.baxterId, jointIndices=self.motorIndices[-9:],
                                    controlMode=p.POSITION_CONTROL, targetPositions=jointPoses[-9:],
                                    targetVelocities=[0]*9, forces=[self.maxForce]*9)

    def gripper_control(self, gripper_state):
        """
        the code to control gripper
        :param gripper_state: [-1, 1]
        0 to 0.020833
        """
        # the state of the gripper
        gripper_state = (gripper_state + 1) * 0.5
        gripper_state_a = np.clip(gripper_state * self.gripper_open, self.gripper_close, self.gripper_open)

        gripper_state_b = np.clip(-gripper_state * self.gripper_open, -self.gripper_open, self.gripper_close)

        p.setJointMotorControl2(bodyUniqueId=self.baxterId,
                                jointIndex=self.finger_a_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_state_a,
                                force=self.fingerForce)
        p.setJointMotorControl2(bodyUniqueId=self.baxterId,
                                jointIndex=self.finger_b_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_state_b,
                                force=self.fingerForce)

    def gripper_close_open(self, gripper_command):
        """
        use command to close or open the gripper

        """
        if gripper_command == "close":
            target_position = self.gripper_close
        elif gripper_command == "open":
            target_position = self.gripper_open
        else:
            target_position = p.getJointState(self.baxterId, self.finger_a_id)[0]

        p.setJointMotorControl2(bodyUniqueId=self.baxterId,
                                jointIndex=self.finger_a_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position,
                                force=self.fingerForce)
        p.setJointMotorControl2(bodyUniqueId=self.baxterId,
                                jointIndex=self.finger_b_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=-target_position,
                                force=self.fingerForce)


if __name__ == "__main__":
    p.connect(pybullet.GUI)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pb_path)
    p.resetDebugVisualizerCamera(2., 135, 0., [0.52, 0.2, np.pi / 4.])

    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
    baxter_id = p.loadURDF("./models/baxter_description/urdf/baxter.urdf", useFixedBase=True)
    baxter = BaxterLeft(baxter_id)
    baxter.osc([0, 0, 0.1])
    baxter.gripper_control(0.0)
    baxter.gripper_control(0.020833)


