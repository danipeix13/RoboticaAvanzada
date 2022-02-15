import time
import sys
from numpy import linalg as LA
import numpy as np
sys.path.append('/home/alumno/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

constants = {
    "SCENE": "/home/robocomp/robocomp/components/manipulation_kinova_gen3/etc/kinova_env_dani_bloquesApilados.ttt",
    "EPOCHS": 100,
    "TIMESTAMPS_PER_EPOCH": 100,
}

env_data = {
    "n_actions": 5,
    "possible_values": [-1, 0, 1],
    "seconds_per_epoch": 15,
    "n_epochs": 1000,

}


class Env():
    def __init__(self, env_data):
        self.action_space = [0 for _ in range(env_data["n_actions"])]
        self.possible_values = env_data["possible_values"]
        self.period = env_data["seconds_per_epoch"]
        self.epochs = 

        """ESTRUCTURA (tabla gorda!?)"""

        print('Program started')
        self.client = RemoteAPIClient()
        self.sim = client.getObject('sim')

        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.sim.loadScene(constants["SCENE"])
        print('Scene loaded')

        self.sim.startSimulation()

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    def reset(self):
        # Poner el cubo en la posición inicial
        # Mover el brazo a la posición previa al grasping
        # Poner tiempo a cero
        pass

    def action_space_sample(self):
        # Devolver accion random o de la tabla
        pass

    def step(self, action):
        # return observation, reward, done, info
        pass





# MAIN

rl_env = Env()
for i in constants["EPOCHS"]
    observation = rl_env.reset()
    for ts in constants["TIMESTAMPS_PER_EPOCH"]
        print(observation)
        action = rl_env.action_space_sample()
        observation, reward, done, info = rl_env.step(action)
        if done:
            print(f"Episode finished after {ts+1} timestamps")
            break
rl_env.close()