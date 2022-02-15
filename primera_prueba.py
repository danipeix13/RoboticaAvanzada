# Make sure to have the add-on "ZMQ remote API"
# running in CoppeliaSim
#
# All CoppeliaSim commands will run in blocking mode (block
# until a reply from CoppeliaSim is received). For a non-
# blocking example, see simpleTest-nonBlocking.py

import time
import sys
from numpy import linalg as LA
import numpy as np
sys.path.append('/home/alumno/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient



###################################################################
# ACTION'S SPACE
###################################################################

# https://www.coppeliarobotics.com/helpFiles/en/apiFunctions.htm

###################################################################
# CONSTANTS & VARIABLES
###################################################################
constants = {
    "EPOCHS": 3,
    "SCENE": "/home/robocomp/robocomp/components/manipulation_kinova_gen3/etc/kinova_env_dani_bloquesApilados.ttt",
    "TIMESTAMPS_PER_EPOCH": 2,
    "POS": 1,
    "ZERO": 0,
    "NEG": -1,
    "ARM_PATH": "/customizableTable/gen3",
    "CAMERA_PATH": "/customizableTable/Actuator8/\
                    Shoulder_Link_respondable0/Actuator0/\
                    HalfArm1_Link_respondable0/Actuator2/\
                    HalfArm2_Link_respondable/Actuator3/\
                    ForeArm_Link_respondable0/Actuator14/\
                    SphericalWrist1_link_respondable0/Actuator5/\
                    SphericalWrist2_Link_respondable0/Actuator6/\
                    Bracelet_Link_respondable0/Bracelet_link_visual0/camera_arm"
}

###################################################################
# ACTION SPACE
###################################################################
class Action():
    def __init__(self, x, y, z, gripper, wrist):
        self.x = x
        self.y = y
        self.z = z
        self.gripper = gripper
        self.wrist = wrist

###################################################################
# ENV FUNCTIONS
###################################################################
def step(action):
    #COPPELIA STEP
    pass

def reset():
    pass

def next_action():
    pass

def reward():
    pass 



###################################################################
# MAIN
###################################################################

#if __name__ == "main":
print('Program started')

###################################################################
# OPEN ENV
###################################################################
client = RemoteAPIClient()
sim = client.getObject('sim')

defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

sim.loadScene(constants["SCENE"])
print('Scene loaded')

###################################################################
# FUNCTIONS
###################################################################
base_obj = sim.getObject('/customizableTable/gen3')
# camera = sim.getObject(constants["CAMERA_PATH"])
target_obj = sim.getObjectHandle('target')
goal_obj = sim.getObjectHandle('goal')
tip_obj = sim.getObjectHandle('tip')

def moveArm(objToMove, referencedTo):
    pos = sim.getObjectPose(objToMove, referencedTo)
    pos += pose
    sim.setObjectPose(objToMove, referencedTo, pos)
    dist = sys.float_info.max
    while dist > 0.1:
        print(dist)
        pAux = sim.getObjectPose(objToMove, goal_obj)
        dist = LA.norm(np.array([pAux[0], pAux[1], pAux[2]]) - np.array([pose[0], pose[1], pose[2]]))

    # while (t := sim.getSimulationTime()) < 3:
    #     print("CACA")

#client.setStepping(True)
sim.startSimulation()

###################################################################
# LEARNING LOOP
###################################################################
# while (t := sim.getSimulationTime()) < 1:
#     sim.callScriptFunction("close@ROBOTIQ_85", 1)
# while (t := sim.getSimulationTime()) < 2:
#     sim.callScriptFunction("open@ROBOTIQ_85", 1)
moveArm(goal_obj, base_obj, [0, 0, 0])
moveArm(goal_obj, base_obj, [0.1, 0.1, 0.1])


# arm.

''' # Bucle de aprendizaje (OpenAI Gym)
for i in constants["EPOCHS"]
    observation = reset()
    for ts in constants["TIMESTAMPS_PER_EPOCH"]
        print(observation)
        action = next_action()
        observation, reward, done, info = step(action)
        if done:
            print(f"Episode finished after {ts+1} timestamps")
'''
###################################################################
# CLOSE ENV
###################################################################
sim.stopSimulation()
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)
print('Program ended')
