import sys
import time
import math
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

        # TODO: Mover escena.ttt a la carpeta de robotica avanzada
        self.sim.loadScene("/home/robocomp/robocomp/components/RoboticaAvanzada/rl_env/etc/kinova_rl.ttt")
        print('Scene loaded')

    
        self.rs_zero = self.sim.getObjectHandle('gen3')
        
        # Agent
        self.agent = {
            "tip":    self.sim.getObjectHandle('tip'),
            "goal":   self.sim.getObjectHandle('goal'), 
            "target": self.sim.getObjectHandle('target'),
            "wrist":  self.sim.getObjectHandle('Actuator6')
            # OBTENER PINZA
        }
        self.arm_init_pose = self.sim.getObjectPose(self.agent["tip"], self.rs_zero)
        self.testITER = 0

        self.block = self.sim.getObjectHandle('block')
        self.block_init_pose = self.sim.getObjectPose(self.block, self.rs_zero)

        # self.client.setStepping(True)
        self.sim.startSimulation()

    def close(self):
        """ Parar simulación (y destruir escena?) """
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        # self.sim.closeScene() # NO FUNCIONA CORRECTAMENTE: ¿no soportada?
        print('Program ended')

    def reset(self):
        """ Cubo a posición inicial, brazo a posición inicial, tiempo a cero """
        self.move_arm(self.arm_init_pose)
        self.sim.setObjectPose(self.block, self.rs_zero, self.block_init_pose)
        self.current_time = 0
        pass

    def action_space_sample(self):
        """ Devolver accion random o de la tabla """
        pass

    def step(self, action):
        """ Ejecuta el paso con la acción elegida y aporta feedback sobre la misma """
        pos, wrist, grip = self.__interpretate_action(action)
        self.__move_arm(pos)
        self.__move_wrist(wrist)
        # self.__move_grip(grip)
        # coppelia step
        # observation = ...
        # reward = self.__calculate_reward()
        # done = self.__check_if_done()
        # info = self.__register_info()
        self.current_time += 1
        # return observation, reward, done, info
        pass

    def __move_arm(self, tg_pos):
        """ Mueve el brazo a una determinada posición """
        self.sim.setObjectPose(self.agent["goal"], self.rs_zero, tg_pos)
        dist = sys.float_info.max
        while dist > 0.1:
            # print(dist)
            pose = self.sim.getObjectPose(self.agent["tip"], self.rs_zero)
            dist = LA.norm(np.array(pose[:3]) - np.array(tg_pos[:3]))

    def __move_wrist(self, action):
        # rot = self.sim.getObjectOrientation(self.agent["wrist"], self.rs_zero)
        # rot[2] += action
        # self.sim.setObjectOrientation(self.agent['wrist'], self.rs_zero, rot)
        jointAngle = self.sim.getJointPosition(self.agent["wrist"])
        targetAngle = (jointAngle + action) 
        while abs(jointAngle - targetAngle) > 0.1 * math.pi / 180:
            vel = self.__computeTargetVelocity(jointAngle, targetAngle)
            self.sim.setJointTargetVelocity(self.agent["wrist"], vel)
            self.sim.setJointMaxForce(self.agent["wrist"], 100)
            # self.client.step()
            jointAngle = self.sim.getJointPosition(self.agent["wrist"])
            print(f"V: {vel}, deltaAngle: {jointAngle}")
    
    def __computeTargetVelocity(self, jointAngle, targetAngle):
        dynStepSize = 0.005
        velUpperLimit = 360 * math.pi / 180
        PID_P = 0.1
        errorValue = targetAngle - jointAngle
        sinAngle = math.sin(errorValue)
        cosAngle = math.cos(errorValue)
        errorValue = math.atan2(sinAngle, cosAngle)
        ctrl = errorValue * PID_P

        # Calculate the velocity needed to reach the position
        # in one dynamic time step:
        velocity = ctrl / dynStepSize
        if velocity > velUpperLimit:
            velocity = velUpperLimit

        if velocity < -velUpperLimit:
            velocity = -velUpperLimit

        return velocity


    def __move_grip(self, action):
        if action == 1:
            self.sim.callScriptFunction("open@ROBOTIQ_85", 1)
        elif action == 0:
            self.sim.callScriptFunction("stop@ROBOTIQ_85", 1) # TODO: Implementar/Corregir
        elif action == -1:
            self.sim.callScriptFunction("close@ROBOTIQ_85", 1)

    def __interpretate_action(self, action):
        """ Pasa de una acción del epacio de acciones a un movimiento """
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

    def __calculate_reward(self):

        # return reward
        pass

    def __check_if_done(self):

        # return done
        pass

    def __register_info(self):

        # return info
        pass

    def test(self):
        self.testITER = (self.testITER + 1) % 4
        it = self.testITER -1
        if it == 2:
            it = 0
        # print(it)
        self.step([0, 0, 0, 1, 0])




