import sys
import time
import math
import cv2 as cv
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

""" TODO
  
INTERESTING LINKS

(Code example with nnabla)
 - https://github.com/sony/nnabla-examples/tree/master/reinforcement_learning/dqn

(Blog nnabla)
 - https://blog.nnabla.org/examples/deep-reinforcement-algorithm-dqn-deep-q-learning/


tc = TimeControl(0.05)
while True:
    self.read_joystick()
    self.sim.setJointTargetVelocity(self.left_wheel, self.vel_left)
    self.sim.setJointTargetVelocity(self.right_wheel, self.vel_right)
    # img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
    # img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    # img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    #print(self.velocity[0], self.velocity[1])
    # cv2.imshow('', img)
    # cv2.waitKey(1)
    self.client.step()
    tc.wait()

"""

class EnvKinova():
    def __init__(self):
        """ Starts an API client, oads the scene, takes all the important objects
            from it and starts the simulation. """
        print('Program started')
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.action_space = [0, 0, 0, 0, 0]
        self.possible_values = [-1, 0, 1]
        self.period = 20
        self.current_time = 0

        # Scene
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        # TODO: Relative path
        self.sim.loadScene("/home/robocomp/robocomp/components/RoboticaAvanzada/rl_env/etc/kinova_rl.ttt")
        print('Scene loaded')

        # Env's center
        self.rs_zero = self.sim.getObjectHandle('gen3')
        
        # Agent
        self.agent = {
            "tip":    self.sim.getObjectHandle('tip'),
            "goal":   self.sim.getObjectHandle('goal'), 
            "target": self.sim.getObjectHandle('target'),
            "wrist":  self.sim.getObjectHandle('Actuator6'),
            "camera": self.sim.getObjectHandle("camera_arm"),
        }
        self.arm_init_pose = self.sim.getObjectPose(self.agent["tip"], self.rs_zero)
        
        # Block
        self.block = self.sim.getObjectHandle('block')
        self.block_init_pose = self.sim.getObjectPose(self.block, self.rs_zero)

        # Test iter index, only for testing purposes, will be deleted later
        self.testITER = 0

        # self.client.setStepping(True)
        self.sim.startSimulation()

    def close(self):
        """ Stops the simulation """
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        # self.sim.closeScene() # NO FUNCIONA CORRECTAMENTE: ¿no soportada?
        print('Program ended')

    def reset(self):
        """ Moves the arm and the block to their initial positions, also resets the time """
        self.move_arm(self.arm_init_pose)
        self.sim.setObjectPose(self.block, self.rs_zero, self.block_init_pose)
        self.current_time = 0
        pass

    def action_space_sample(self):
        """ TODO """
        pass

    def step(self, action):
        """ Executes the chosen action, evaluates the post-action state and stores info """
        pos, wrist, grip = self.__interpretate_action(action)
        self.__move_arm(pos)
        self.__move_wrist(wrist)
        self.__move_grip(grip)
        # self.client.step()
        observation = self.__observate()
        # reward = self.__calculate_reward()
        # done = self.__check_if_done()
        # info = self.__register_info()
        self.current_time += 1
        # return observation, reward, done, info
        pass

    def __move_arm(self, tg_pos):
        """ Moves the arm to a target position """
        self.sim.setObjectPose(self.agent["goal"], self.rs_zero, tg_pos)
        dist = sys.float_info.max
        while dist > 0.1:
            pose = self.sim.getObjectPose(self.agent["tip"], self.rs_zero)
            dist = LA.norm(np.array(pose[:3]) - np.array(tg_pos[:3]))

    def __move_wrist(self, action):
        """ Rotates the arm's wrist """
        rot = self.sim.getObjectOrientation(self.agent["tip"], self.rs_zero)
        posJoint = self.sim.getJointPosition(self.agent['wrist'])

        rot[1] += action / 10
        posJoint += action / 10

        self.sim.setObjectOrientation(self.agent['tip'], self.rs_zero, rot)
        self.sim.setJointPosition(self.agent['wrist'], posJoint)

    def __move_grip(self, action):
        """ Executes the grip's subaction """
        if action == 1:
            self.sim.callScriptFunction("open@ROBOTIQ_85", 1)
        elif action == 0:
            self.sim.callScriptFunction("stop@ROBOTIQ_85", 1)
        elif action == -1:
            self.sim.callScriptFunction("close@ROBOTIQ_85", 1)

    def __interpretate_action(self, action):
        """ Translates the chosen action to 3 (arm move, wrist rotation and grip) subactions """
        is_correct = all(list(map(lambda x: x in self.possible_values, action)))
        if is_correct:
            current_pose = self.sim.getObjectPose(self.agent["tip"], self.rs_zero)
            delta_pose = list(map(lambda x: x/1000, action[:3])) + [0, 0, 0, 0]
            new_pose = [a+b for a,b in list(zip(current_pose,delta_pose))]
            # print(delta_pose, new_pose)
            wrist_action, grip_action = action[3], action[4]

            return new_pose, wrist_action, grip_action
        else:
            print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

    def __observate(self):
        """  """
        imgBuffer, resX, resY = self.sim.getVisionSensorCharImage(self.agent["camera"])
        # img = np.frombuffer(imgBuffer, dtype=np.uint8).reshape(resY, resX, 3)
        # print(np.shape(img))
        # img = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)
        # img = cv.rectangle(img, (200, 290), (300, 390), (0, 0, 255), 2)
        # cv.imshow('RBG', img)


        depthBuffer = self.sim.getVisionSensorDepthBuffer(self.agent["camera"])
        print(np.shape(depthBuffer), type(depthBuffer))
        depth = np.array(depthBuffer, dtype=np.float)
        depth.reshape(resY, resX)
        print(depth.shape)
        # depth = cv.flip(cv.cvtColor(depth, cv.COLOR_BGR2RGB), 0)
        depth = cv.rectangle(depth, (200, 290), (300, 390), (0, 0, 255), 2)
        """
        np.frombuffer(all.depth.depth, np.float32).reshape(all.depth.height, all.depth.width)
        """
        cv.imshow('D', depth)

        cv.waitKey(1)
        return True

    def __calculate_reward(self):
        """  """
        # return reward
        pass

    def __check_if_done(self):
        """  """
        # return done
        pass

    def __register_info(self):
        """  """
        # return info
        pass

    def test(self):
        """ Public test method, that allow to use all the private and public methods
            of the class. Only for testing purposes, will be deleted later """
        self.testITER = (self.testITER + 1) % 4
        it = self.testITER -1
        if it == 2:
            it = 0
        # print(it)
        x = it 
        y = it
        z = it
        wrist = it
        grip = it

        self.step([x, y, z, wrist, grip])




