import gym, sys, time, math
from gym import spaces
import utilities as U
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient


class EnvKinova_gym(gym.Env):

    #################################
    ## -- GYM INTERFACE METHODS -- ##
    #################################
    def __init__(self):
        super(EnvKinova_gym, self).__init__()
        print('PROGRAM STARTED')
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # VARS
        self.possible_values = [-1, 0, 1]
        self.max_steps = 200
        self.current_step = 0

        # SCENE
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/RoboticaAvanzada/rl_env/etc/kinova_rl.ttt")
        self.sim.startSimulation()
        time.sleep(1)

        # SPACES
        self.action_space = U.set_action_space()
        action = self.action_space.sample()
        observation, _, done, _ = self.step(action)
        assert not done
        self.observation_space = U.set_observation_space(observation)
        self.goal = [0, 0]

    def step(self, action):
        sim_act = [int(action[0]), int(action[1]), 0, 0, 0]
        
        if self.__interpretate_action(sim_act):
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()
        print("OBSERVATION IN INIT", observation)
        exit, reward = self.__reward_and_or_exit(observation)
        self.current_step += 1
        
        if exit:
            with open('output.txt', 'a') as file: 
                file.write(str(reward) + "\n")

        return observation, reward, exit, {}


    def reset(self):
        print("RESET", "STEP:", self.current_step)
        self.sim.stopSimulation()
        self.sim.startSimulation()
        aux_goal = self.sim.callScriptFunction("move_to_random_x@gen3", 1)
        self.goal = aux_goal[:2]

        self.current_step = 0
        obs = self.__observate()
        ret = np.array(obs["pos"][0][:2], dtype=np.float32)
        return ret

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    ####################################
    ## -- PRIVATE AUXILIAR METHODS -- ##
    ####################################

    def __interpretate_action(self, action):
        return all(list(map(lambda x: x in self.possible_values, action)))

    def __observate(self):
        obs = {"pos": [[0, 0, 0]]}
        obs = self.sim.callScriptFunction("get_observation@gen3", 1) 
        return {"distX":obs["dist_x"], "distY":obs["dist_y"]}

    def __reward_and_or_exit(self, observation):
        reward = 0
        exit = False

        #No alejarse
        if math.sqrt(observation["distX"]**2 + observation["distY"]**2) > 0.1:
            return True, -10000

        # Posición en XY
        dist = math.sqrt(observation["distX"]**2 + observation["distY"]**2)
        reward += (1 - self.__normalize(dist, 0, 0.1)) * 100
        if dist < 0.005:
            reward += 10000

        # Pinza abierta
        # reward += (1 - self.__normalize(observation["gripper"], -0.002, -0.01)) * 10

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

    def __normalize(self, x, min_val, max_val):
        return ((x - min_val) / (max_val + min_val))

    ###################################
    ## -- ALGORITHMIC STEP METHOD -- ##
    ###################################
    def algo_step(self):
        observation = self.__observate()
        jointPos = observation["gripper"]

        if observation["depth"][0] > 0.16 and jointPos > -0.01:
            return [0, 0, -1, 0, 0]

        # if np.mean(self.rforce) + np.mean(self.lforce) > 0.5:
        #     return [0, 0, 1, 0, 0]

        limit = 0.3
        if jointPos < -0.04 or LA.norm(np.array(observation["gripR"][1])) > limit or LA.norm(np.array(observation["gripL"][1])) > limit:
            return [0, 0, 1, 0, 0]
        return [0, 0, 0, 0, -1]

    