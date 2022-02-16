import sys
import time
import numpy as np
from numpy import linalg as LA
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

class EnvKinova():
    def __init__(self):
        print('Program started')
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # Data and structures
        """ESTRUCTURA (tabla gorda!?)"""

        self.action_space = [0, 0, 0, 0, 0]
        self.possible_values = [-1, 0, 1]
        self.period = 20
        self.current_time = 0

        # Scene
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.sim.loadScene("/home/robocomp/robocomp/components/manipulation_kinova_gen3/etc/kinova_env_dani_bloquesApilados.ttt") # MODIFICAR ENTORNO: un solo cubo
        print('Scene loaded')

        # Agent
        base = self.sim.getObject('/customizableTable/gen3')
        tip = self.sim.getObjectHandle('tip')
        target = self.sim.getObjectHandle('target')
        goal = self.sim.getObjectHandle('goal')
        self.agent = {
            "base": base,
            "tip": tip,
            "goal": goal, 
            "target": target,
            # OBTENER MUÑECA
            # OBTENER PINZA
        }

        # OBTENER BLOQUE

        # client.setStepping(True)
        self.sim.startSimulation()

    def close(self):
        """ Parar simulación (y destruir escena?) """
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        print('Program ended')

    def reset(self):
        """ Cubo a posición inicial, brazo a posición inicial, tiempo a cero """
        self.__reset_arm()
        self.__reset_block()
        self.current_time = 0
        pass

    def action_space_sample(self):
        """ Devolver accion random o de la tabla """
        pass

    def step(self, action):
        """  """
        # return observation, reward, done, info
        pass

    def __reset_arm(self):
        """ Mandar el brazo a la posición inicial (posición previa al grasping) """
        pass

    def __reset_block(self):
        """ Mover el bloque a la posición inicial """
        pass

    def __moveArm(self, tg_pos):
        """  """
        current_pose = self.sim.getObjectPose(self.agent["tip"], self.agent["base"])
        goal_pose = self.sim.getObjectPose(tg_pos, self.agent["base"])
        delta_pose = goal_pose - current_pose
        self.sim.setObjectPose(self.agent["tip"], self.agent["base"], delta_pose)
        dist = sys.float_info.max
        while dist > 0.1:
            print(dist)
            pAux = self.sim.getObjectPose(self.agent["tip"], self.agent["goal"])
            dist = LA.norm(np.array([pAux[0], pAux[1], pAux[2]]))

