import time
import sys
from numpy import linalg as LA
import numpy as np
sys.path.append('/home/alumno/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient



arm = {
    "base": sim.getObject('/customizableTable/gen3'),
    "tip": sim.getObjectHandle('tip'),
    "target": sim.getObjectHandle('target'),
    "goal": sim.getObjectHandle('goal'),
}

def moveArm(tg_pos):
    current_pose = sim.getObjectPose(arm["tip"], arm["base"])
    goal_pose = sim.getObjectPose(tg_pos, arm["base"])
    delta_pose = goal_pose - current_pose
    sim.setObjectPose(arm["tip"], arm["base"], delta_pose)
    dist = sys.float_info.max
    while dist > 0.1:
        print(dist)
        pAux = sim.getObjectPose(arm["tip"], arm["goal"])
        dist = LA.norm(np.array([pAux[0], pAux[1], pAux[2]]))

    # while (t := sim.getSimulationTime()) < 3:
    #     print("CACA")

#client.setStepping(True)
sim.startSimulation()



## MAIN ##
moveArm(goal_obj)
# moveArm(goal_obj)





sim.stopSimulation()
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)
print('Program ended')
