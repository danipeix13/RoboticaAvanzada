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
        print('Program started')
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
        #print("-------ACTION SPACE", self.action_space)
        action = self.action_space.sample()
        #print("-------ACTION", action)
        observation, _, done, _ = self.step(action)
        assert not done
        self.observation_space, self.n = U.set_observation_space(observation)
        self.goal = [0, 0]

    def step(self, action):
        print("ACTION", action)
        sim_act = [int(action[0]), int(action[1]), 0, 0, 0]
        
        if self.__interpretate_action(sim_act):
            self.sim.callScriptFunction("do_step@gen3", 1, sim_act)
        else:
            #print("INCORRECT ACTION: values not in [-1, 0, 1]")
            return None

        observation = self.__observate()
        #print("OBSERVATION IN STEP", observation)
        exit, reward, arrival, far, dist = self.__reward_and_or_exit(observation)
        self.current_step += 1
        
        if exit:
            with open('output.txt', 'a') as file: 
                file.write(str(reward) + "\n")

        info = {
            "arrival": arrival,
            "far": far,
            "dist": dist,
        }

        return observation, reward, exit, info


    def reset(self):
        #print("RESET", "STEP:", self.current_step)
        self.goal = self.sim.callScriptFunction("reset@gen3", 1) 

        self.current_step = 0
        obs = self.__observate()
        print(obs)
        return obs

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
        exit = False
        reward = 0
        arrival = 0
        far = 0
        dist = math.sqrt(observation["distX"]**2 + observation["distY"]**2)

        if dist > 0.1:
            exit = True
            reward = -10000
            far = 1
        
        else:
            reward += (1 - self.__normalize(dist, 0, 0.1)) * 100
            
            if dist < 0.005:
                exit = True
                reward += 10000
                arrival = 1

        return exit, reward, arrival, far, dist

        '''
        Se choca:       True, -1
        No se choca:    False, 0
        Llega:          True, 1
        '''

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

    