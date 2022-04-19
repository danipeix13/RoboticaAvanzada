import random
from re import A
import sys
import time
import math
import threading
import cv2 as cv
from cv2 import sqrt
import numpy as np
from collections import deque
from numpy import dtype, linalg as LA
import matplotlib.pyplot as plt
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

import gym
from gym import spaces

""" TODO
  
INTERESTING LINKS

(Code example with nnabla)
 - https://github.com/sony/nnabla-examples/tree/master/reinforcement_learning/dqn

(Blog nnabla)
 - https://blog.nnabla.org/examples/deep-reinforcement-algorithm-dqn-deep-q-learning/
"""

class EnvKinova_gym(gym.Env):
    def __init__(self):
        super(EnvKinova_gym, self).__init__()
        """ Starts an API client, oads the scene, takes all the important objects
            from it and starts the simulation. """
        print('Program started')
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # self.action_space = [0, 0, 0, 0, 0]
        self.possible_values = [-1, 0, 1]
        self.max_steps = 200
        self.current_step = 0

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
        self.testIndex = 0
        self.SIZE = 20
        self.EXPLORE = 0.45

        # # time series
        # plt.ion()
        # self.visible = 120
        # self.dopening = deque(np.zeros(self.visible), self.visible)
        # self.ddistance = deque(np.zeros(self.visible), self.visible)
        # self.dforce_left = deque(np.zeros(self.visible), self.visible)
        # self.dforce_right = deque(np.zeros(self.visible), self.visible)
        # self.dforce_left_tip = deque(np.zeros(self.visible), self.visible)
        # self.dforce_right_tip = deque(np.zeros(self.visible), self.visible)
        # self.dx = deque(np.zeros(self.visible), self.visible)
        # self.data_length = np.linspace(0, 121, num=120)

        # # plt
        # self.fig = plt.figure(figsize=(8, 3))
        # self.ah1 = self.fig.add_subplot()
        # plt.margins(x=0.001)
        # self.ah1.set_ylabel("Gripper", fontsize=14)
        # self.opening, = self.ah1.plot(self.dx, self.dopening, color='yellow', label="Closing (x10)", linewidth=1.0)
        # self.distance, = self.ah1.plot(self.dx, self.ddistance, color='orange', label="Distance (x10)", linewidth=1.0)
        # self.force_left, = self.ah1.plot(self.dx, self.dforce_left, color='red', label="L-Force", linewidth=1.0)
        # self.force_right, = self.ah1.plot(self.dx, self.dforce_right, color='magenta', label="R-Force", linewidth=1.0)
        # self.force_left_tip, = self.ah1.plot(self.dx, self.dforce_left_tip, color='blue', label="LT-Force", linewidth=1.0)
        # self.force_right_tip, = self.ah1.plot(self.dx, self.dforce_right_tip, color='green', label="RT-Force", linewidth=1.0)
        # self.ah1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)
        # self.x_data = 0


        # ACTION SPACE
        self.action_space = spaces.MultiDiscrete([3, 3]) # 0:-, 1:nop, 2:+ --> -1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.goal = [0, 0]

        # self.client.setStepping(True)
        self.sim.startSimulation()

    def step(self, action):
        """ Executes the chosen action, evaluates the post-action state and stores info """

        sim_act = [int(action[0]-1), int(action[1]-1), 0, 0, 0]
        if self.__interpretate_action(sim_act):
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()
        exit, reward = self.__reward_and_or_exit(observation)
        self.current_step += 1
        # print (self.current_step, observation["pos"][0][0], action[2], reward)
        
        if exit:
            with open('output.txt', 'a') as file:  # Use file to refer to the file object
                file.write(str(reward) + "\n")
        # print('obs=', observation["pos"][0][:2], 'reward=', reward, 'exit=', exit)
        # print("REWARD: ", reward)
        return np.array([observation["dist_x"], observation["dist_y"]], dtype=np.float32), reward, exit, {}


    def reset(self):
        """ Moves the arm and the block to their initial positions, also resets the time """
        # self.sim.callScriptFunction("reset@gen3", 1)

        print("RESET")
        print(self.current_step)
        self.sim.stopSimulation()
        time.sleep(.1)
        self.sim.startSimulation()
        time.sleep(.5)
        aux_goal = self.sim.callScriptFunction("move_to_random_x@gen3", 1)
        self.goal = aux_goal[:2]
        # print (self.goal)

        self.current_step= 0
        obs = self.__observate()
        ret = np.array(obs["pos"][0][:2], dtype=np.float32)
        # print(ret.shape)

        return ret

    def close(self):
        """ Stops the simulation """
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        # self.sim.closeScene() # NO FUNCIONA CORRECTAMENTE: ¿no soportada?
        print('Program ended')


############################################################################################################

    def __interpretate_action(self, action):
        """ Translates the chosen action to 3 (arm move, wrist rotation and grip) subactions """
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        """ DOCU """
        obs = {"pos": [[0, 0, 0]]}
        
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        
        # self.draw_gripper_series(obs)
        return obs

    def __reward_and_or_exit(self, observation):
        reward = 0
        exit = False

        #No alejarse
        if math.sqrt(observation["dist_x"]**2 + observation["dist_y"]**2) > 0.1:
            return True, -10000

        # Posición en XY
        dist = math.sqrt(observation["dist_x"]**2 + observation["dist_y"]**2)
        reward += (1 - self.__normalize(dist, 0, 0.1)) * 100
        if dist < 0.005:
            reward += 10000

        # Pinza abierta
        reward += (1 - self.__normalize(observation["gripper"], -0.002, -0.01)) * 10

        # No chocarse
        # if LA.norm(np.array(observation["fingerL"][1])) > .006 or LA.norm(np.array(observation["fingerR"][1])) > .006:
        #     reward += -100
        #     exit = True

        # Posición Z
        # reward += (1 - self.__normalize(observation["dist_z"], 0, 0.1)) * 100

        # # Pinza cerrada
        # if len(observation["gripL"]) == 3:
        #     if LA.norm(np.array(observation["gripL"][1])) > .15 or LA.norm(np.array(observation["gripR"][1])) > .15:
        #         reward += 10

        # print("Reward: ", reward, " exit: ", exit, "\t", random.random(), "\t", random.random(), "\t", random.random(), "\t", random.random())

        return exit, reward
        
        '''
        if LA.norm(np.array(observation["fingerL"][1])) > .15:
            return True, -1000
        if LA.norm(np.array(observation["fingerR"][1])) > .15:
            return True, -1000

        # TODO: tip
        
        dist = math.sqrt(observation["dist_x"]**2 + observation["dist_y"]**2 + observation["dist_z"]**2)

        # print("GRIPPER: ", observation["gripper"])

        if dist > 0.1:
            return True, -1000
        elif dist < 0.008:
            return True, 10000
        else:
            exit = self.current_step >= 300
            return exit, -dist*10
        '''

    def __normalize(self, x, min_val, max_val):
        return ((x - min_val) / (max_val + min_val))

    def test(self):
        """ Public test method, that allow to use all the private and public methods
            of the class. Only for testing purposes, will be deleted later """
        pass

    def action_space_sample(self):
        if np.random.rand() < self.EXPLORE:
            res = self.rand_step()
        else:
            res = self.algo_step()
        print (res)
        return res

    def algo_step(self):
        observation = self.__observate()
        jointPos = observation["gripper"]
        print(jointPos)

        # print("right", np.array(observation["gripR"][1]), LA.norm(np.array(observation["gripR"][1])))
        # print("left", np.array(observation["gripL"][1]), LA.norm(np.array(observation["gripL"][1])))

        if observation["depth"][0] > 0.16 and jointPos > -0.01:
            return [0, 0, -1, 0, 0]
        # if np.mean(self.rforce) + np.mean(self.lforce) > 0.5:
        #     return [0, 0, 1, 0, 0]
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

    # def draw_gripper_series(self, gdata):
    #     # print(gdata)
    #     # update data
    #     # self.dopening.extend([gdata["gripper"]])
    #     # self.ddistance.extend(gdata["depth"])
    #     self.dforce_right.extend([LA.norm(gdata["fingerR"][1]) * 10])
    #     self.dforce_left.extend([LA.norm(gdata["fingerL"][1]) * 10])
    #     self.dforce_left_tip.extend([LA.norm(gdata["gripL"][1])*10])
    #     self.dforce_right_tip.extend([LA.norm(gdata["gripR"][1])*10])
    #     self.dx.extend([self.x_data])

    #     # update plot
    #     self.opening.set_ydata(self.dopening)
    #     self.opening.set_xdata(self.dx)
    #     self.distance.set_ydata(self.ddistance)
    #     self.distance.set_xdata(self.dx)
    #     self.force_left.set_ydata(self.dforce_left)
    #     self.force_left.set_xdata(self.dx)
    #     self.force_right.set_ydata(self.dforce_right)
    #     self.force_right.set_xdata(self.dx)
    #     self.force_left_tip.set_ydata(self.dforce_left_tip)
    #     self.force_left_tip.set_xdata(self.dx)
    #     self.force_right_tip.set_ydata(self.dforce_right_tip)
    #     self.force_right_tip.set_xdata(self.dx)

    #     # set axes
    #     self.ah1.set_ylim(0, 10)
    #     self.ah1.set_xlim(self.x_data-self.visible, self.x_data)

    #     # control speed of moving time-series
    #     self.x_data += 1

    #     self.fig.canvas.draw()
    #     # self.fig.canvas.flush_events()
    