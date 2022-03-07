import sys
import time
import math
import threading
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
            "tip":     self.sim.getObjectHandle('tip'),
            "goal":    self.sim.getObjectHandle('goal'), 
            "target":  self.sim.getObjectHandle('target'),
            "wrist":   self.sim.getObjectHandle('Actuator6'),
            "camera":  self.sim.getObjectHandle("camera_arm"),
            "gripL": self.sim.getObjectHandle("ROBOTIQ_85_attachForceSensorFingerLeft"), 
            "gripR": self.sim.getObjectHandle("ROBOTIQ_85_attachForceSensorFingerRight"), 
            "fingerL":   self.sim.getObjectHandle("ROBOTIQ_85_attachForceSensorTipLeft"), 
            "fingerR":   self.sim.getObjectHandle("ROBOTIQ_85_attachForceSensorTipRight"), 
            "gripper": self.sim.getObjectHandle("ROBOTIQ_85_active1")
        }
        self.arm_init_pose = self.sim.getObjectPose(self.agent["tip"], self.rs_zero)
        
        # Block
        self.block = self.sim.getObjectHandle('block')
        self.block_init_pose = self.sim.getObjectPose(self.block, self.rs_zero)

        # Test iter index, only for testing purposes, will be deleted later
        self.testITER = 0

        self.EXPLORE = 0.05

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

    def step(self, action):
        """ Executes the chosen action, evaluates the post-action state and stores info """
        if self.__interpretate_action(action):
            self.sim.callScriptFunction("do_step@gen3", 1, action)
        else:
            print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        # self.client.step()
        # reward, exit = self.__reward_and_or_exit(observation)
        # reward = self.__calculate_reward()
        # done = self.__check_if_done()
        # info = self.__register_info()
        # return observation, reward, done, info


    def __interpretate_action(self, action):
        """ Translates the chosen action to 3 (arm move, wrist rotation and grip) subactions """
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        """ DOCU """

        return self.sim.callScriptFunction("get_observation@gen3", 1) 

    def __reward_and_or_exit(self, observation):
        reward = 0

        if observation["fl"] > 1 or  observation["fr"] > 1:
            exit = True
            reward = -100
        else:
            pass
            
        return exit, reward

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
        # self.testITER = (self.testITER + 1) % 4
        # it = self.testITER -1
        # if it == 2:
        #     it = 0
        # # print(it)
        # x = it 
        # y = it
        # z = it
        # wrist = it
        # grip = it

        # self.step([x, y, -1, 0, -1])
        self.step([1, 0, 0, 0, 0])

    def action_space_sample(self):
        if np.random.rand() < self.EXPLORE:
            res = self.rand_step()
        else:
            res = self.algo_step()
        
        return res

    def algo_step(self):
        observation = self.__observate()
        jointPos = observation["gripper"]

        # print("right", np.array(observation["gripR"][1]), LA.norm(np.array(observation["gripR"][1])))
        # print("left", np.array(observation["gripL"][1]), LA.norm(np.array(observation["gripL"][1])))

        if observation["depth"][0] > 0.17 and jointPos > -0.02:
            return [0, 0, -1, 0, 0]

        limit = 0.3
        if jointPos < -0.04 or LA.norm(np.array(observation["gripR"][1])) > limit or LA.norm(np.array(observation["gripL"][1])) > limit:
            return [0, 0, 1, 0, 0]
            
        return [0, 0, 0, 0, -1]

    def rand_step(self):
        act = np.random.choice(self.possible_values, size=5).tolist()
        return act

    def end_of_episode(self):
        observation = self.__observate()
        # return observation["fr"] > 1 or observation["fl"] > 1 or 


    