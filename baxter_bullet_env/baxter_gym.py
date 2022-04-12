import os, inspect
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from baxter import BaxterLeft
import random
import pybullet_data
import operator
import pygame
import sys
from scipy.spatial.transform import Rotation

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

pb_path = pybullet_data.getDataPath()
baxter_path = os.getcwd() + "/baxter_bullet_env/models/baxter_description/urdf/paddle_gripper.urdf"
table_path = os.getcwd() + "/baxter_bullet_env/models/objects/table/table.urdf"
block_path = os.getcwd() + "/baxter_bullet_env/models/objects/block/model.urdf"
wall_path = os.getcwd() + "/baxter_bullet_env/models/objects/block/black_wall.urdf"

RENDER_HEIGHT = 512
RENDER_WIDTH = 512


class BaxterGymEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=1,
                 isEnableSelfCollision=True,  max_episode_steps=200,
                 renders=True, pygame_renders=True,
                 camera_view=True, reward_type="dense"):
        """
        :param urdfRoot:
        :param actionRepeat:
        :param isEnableSelfCollision:
        :param renders:
        :param camera_view:
        :param reward_type: dense or sparse
        """
        self._timeStep = 1. / 240.
        self.control_time = 1/20
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._camera_view = camera_view
        self._reward_types = reward_type

        self._observation = []
        self.observation_space = {}
        self.action_dim = 4
        self.v_end = 0.05   # the velocity of end-effector
        self.v_gripper = 1  # need at least 1 steps to close

        self._envStepCounter = 0
        self.max_episode_steps = max_episode_steps
        self._renders = renders
        self._pygame_render = pygame_renders
        self._width = RENDER_HEIGHT
        self._height = RENDER_HEIGHT

        # whether
        self.terminated = 0
        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, 0.2, np.pi / 4.])
        else:
            p.connect(p.DIRECT)
        self.viewer = None

        self.seed()

        # get the observation
        obs = self.reset()

        # get the observation dim, action dim
        if self._camera_view:
            self.observation_space['physic'] = spaces.Box(low=-1, high=1, shape=obs[0].shape, dtype=float)
            self.observation_space["image"] = spaces.Box(low=0, high=255,
                                                         shape=obs[1].shape, dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=obs.shape, dtype=float)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=float)

    def action_spec(self):
        low = -1 * np.ones(self.action_dim)
        high = 1 * np.ones(self.action_dim)
        return low, high

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.terminated = 0
        self._envStepCounter = 0
        self._grasped_time = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        # loading the models of robot table, objects
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -0.9])
        p.loadURDF(wall_path, [-1, -1, -0.9], useFixedBase=True)

        self.baxterId = p.loadURDF(baxter_path, useFixedBase=True)
        self.table_id = p.loadURDF(table_path, [0.5, 0.4, -.9],
                                   [0.0, 0.0, 0, 1])

        # reset the robot and block
        self.robot = BaxterLeft(self.baxterId)

        obj_pose = list(map(operator.add, [0.35, 0.2, -0.1], [random.random() * 0.2, random.random() * 0.2, 0]))

        ang = np.pi * random.random()
        obj_orn = list(p.getQuaternionFromEuler([0, 0, ang]))

        self.blockUid = p.loadURDF(block_path, obj_pose, obj_orn)

        p.changeDynamics(self.blockUid, -1, lateralFriction=1.3, spinningFriction=0.001, rollingFriction=0.0005,
                         restitution=0)
        p.changeDynamics(self.table_id, -1, lateralFriction=1, restitution=0)

        p.changeDynamics(self.baxterId, self.robot.finger_tips_a_id, lateralFriction=1, spinningFriction=0.001,
                         rollingFriction=0.0005, restitution=0)
        p.changeDynamics(self.baxterId, self.robot.finger_tips_b_id, lateralFriction=1, spinningFriction=0.001,
                         rollingFriction=0.0005, restitution=0)


        for _ in range(int(self.control_time/self._timeStep) * 10):
            p.stepSimulation()

        cube_pose, _ = p.getBasePositionAndOrientation(self.blockUid)
        self.cube_init_z = cube_pose[2]

        if self._pygame_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self._width, self._height))

        # return the obs
        return self.getObservation()

    # -------------------------------------------------------------------------------------------------
    def getObservation(self):
        self.obs = []
        if self._camera_view:
            self.obs.append(self.get_physic_state())
            self.obs.append(self.get_camera_state())
            return self.obs
        else:
            self.obs.append(self.get_physic_state())
            return self.obs[0]

    def get_camera_state(self):
        # ---------------------------- get camera obs -----------------
        camera = []
        # --------------- head camera -------------
        camEyePos = [0.5, 1.22, 0.16]
        # look through y axis
        cam_targe_pos = [0.5, 0.1, 0.16]
        up_vector = [0, 0, 1]
        viewMat = p.computeViewMatrix(camEyePos, cam_targe_pos, up_vector)

        fov = 54
        aspect = self._width / self._height
        near = 0.07
        far = 10
        projMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        front_img_arr = p.getCameraImage(width=self._width,
                                         height=self._height,
                                         viewMatrix=viewMat,
                                         projectionMatrix=projMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        front_img = np.reshape(front_img_arr[2], (self._height, self._width, 4))
        camera.append(front_img[:, :, 0:3])

        # --------------- hand camera ---------------
        fov2 = 60
        gripperState = p.getLinkState(self.baxterId, self.robot.camera_axis_id)
        projMatrix = p.computeProjectionMatrixFOV(fov2, aspect, near, far)
        # define hand_camera position
        hand_came_pose = np.array(gripperState[0] - np.array([0, 0, 0.035]))
        hand_targe_pos = np.array(hand_came_pose) - np.array([0, 0, 1])
        r = Rotation.from_quat(gripperState[1]).as_matrix()
        up_vector = r[:, 0]
        viewMat = p.computeViewMatrix(hand_came_pose, hand_targe_pos, up_vector)

        hand_img_arr = p.getCameraImage(width=self._width,
                                        height=self._height,
                                        viewMatrix=viewMat,
                                        projectionMatrix=projMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        hand_img = np.reshape(hand_img_arr[2], (self._height, self._width, 4))
        camera.append(hand_img[:, :, 0:3])
        # stacking the image
        camera = np.concatenate(camera, axis=2)
        return camera

    def get_physic_state(self):
        # get physic state
        """
        The physic state contain:
        1. joint cos ().  3.end_effector pose, orientation.
        4. normalized gripper_state. 5.gripper to cube. 6. cube position.
        7. cube orientation
        """
        motor_id = self.robot.motorIndices[-9:]
        joint_states = p.getJointStates(self.baxterId, motor_id)
        joint_cos_p = [np.cos(x[0]) for x in joint_states[:-2]]
        # joint_cos_v = [np.cos(x[1]) for x in joint_states[:-2]]
        end_effector_p = list(p.getLinkState(self.robot.baxterId, self.robot.endEffector_id, computeLinkVelocity=True)[4])
        end_effector_v = list(p.getLinkState(self.robot.baxterId, self.robot.endEffector_id, computeLinkVelocity=True)[6])

        cube_pose, cube_orn = p.getBasePositionAndOrientation(self.blockUid)
        cube_v, _ = p.getBaseVelocity(self.blockUid)

        # get joint state -- normalized to [0, 1]
        gripper_state = (joint_states[-1][0] / self.robot.gripper_open - 0.5) * 2
        dist = np.linalg.norm(np.array(gripper_state) - np.array(cube_pose))
        physic_state = np.concatenate((end_effector_p, cube_pose, [gripper_state],
                                       [dist]), axis=0)
        return physic_state
    # ---------------------------------------------------------------------------------------------------

    def step(self, action):
        # take the action
        self._envStepCounter += 1
        d_pose = action[0:3] * self.v_end
        self.robot.osc(d_pose)

        d_gripper = action[3] * self.v_gripper

        self.robot.gripper_control(d_gripper)

        # update simulation
        for _ in range(int(self.control_time/self._timeStep)):
            p.stepSimulation()

        # update obs
        obs = self.getObservation()
        # update termination
        done = self._envStepCounter >= self.max_episode_steps  # or self._success()
        # update reward
        reward = self._reward()
        # update info
        info = {}
        if self._success():
            info["success"] = "True"
        else:
            info["success"] = "False"
        return obs, reward, done, info

    def _success(self):
        cube_pose, _ = p.getBasePositionAndOrientation(self.blockUid)
        return cube_pose[2] - self.cube_init_z > 0.05

    def _reward(self):
        reward = 0

        # lift reward
        if self._success():
            reward = 2.25

        if self._reward_types == "dense":

            # reaching reward
            cube_pose, _ = p.getBasePositionAndOrientation(self.blockUid)
            gripper_pose = list(p.getLinkState(self.robot.baxterId, self.robot.endEffector_id)[4])
            dist = np.linalg.norm(np.array(gripper_pose) - np.array(cube_pose))
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasped reward
            if self._grasped():
                reward += 0.25

            return reward / 2.25

    def _grasped(self):
        """
        check the whether block is grasped
        :return:
        """
        left_contact = p.getContactPoints(self.baxterId, self.blockUid, self.robot.finger_tips_a_id)
        right_contact = p.getContactPoints(self.baxterId, self.blockUid, self.robot.finger_tips_b_id)
        return left_contact != () and right_contact != ()

    def render(self, mode='human', close=False):
        """
        Create a pygame render show front view
        :param mode:
        :param close:
        :return:
        """
        if self._pygame_render:
            if mode != "rgb_array":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

            image = np.array(self.obs[1][:, :, 3:]).astype("uint8")
            surf = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
            # surf = pygame.surfarray.make_surface(image)
            self.screen.blit(surf, (0, 0))
            pygame.display.update()
        else:
            return

    def __del__(self):
        p.disconnect()

if __name__ == "__main__":
    env = BaxterGymEnv()
    # random action
    # Get action limits
    low, high = env.action_spec()

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        print(i)
        print(reward)
        env.render()



