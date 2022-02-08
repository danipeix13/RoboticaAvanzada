# Make sure to have the add-on "ZMQ remote API"
# running in CoppeliaSim
#
# All CoppeliaSim commands will run in blocking mode (block
# until a reply from CoppeliaSim is received). For a non-
# blocking example, see simpleTest-nonBlocking.py

import time

from zmqRemoteApi import RemoteAPIClient


# print('Program started')

# client = RemoteAPIClient()
# sim = client.getObject('sim')

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
# defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
# sim.setInt32Param(sim.intparam_idle_fps, 0)

# sim.loadScene('/home/robocomp/robocomp/components/manipulation_kinova_gen3/etc/kinova_env_dani_bloquesApilados.ttt')

# # Create a few dummies and set their positions:
# handles = [sim.createDummy(0.01, 12 * [0]) for _ in range(50)]
# for i, h in enumerate(handles):
#     sim.setObjectPosition(h, -1, [0.01 * i, 0.01 * i, 0.01 * i])

# # Run a simulation in asynchronous mode:
# sim.startSimulation()
# while (t := sim.getSimulationTime()) < 3:
#     s = f'Simulation time: {t:.2f} [s] (simulation running asynchronously '\
#         'to client, i.e. non-stepped)'
#     print(s)
#     sim.addLog(sim.verbosity_scriptinfos, s)
# sim.stopSimulation()
# # If you need to make sure we really stopped:
# while sim.getSimulationState() != sim.simulation_stopped:
#     time.sleep(0.1)

# # Run a simulation in stepping mode:
# client.setStepping(True)
# sim.startSimulation()
# while (t := sim.getSimulationTime()) < 3:
#     s = f'Simulation time: {t:.2f} [s] (simulation running synchronously '\
#         'to client, i.e. stepped)'
#     print(s)
#     sim.addLog(sim.verbosity_scriptinfos, s)
#     client.step()  # triggers next simulation step
# sim.stopSimulation()

# # Remove the dummies created earlier:
# for h in handles:
#     sim.removeObject(h)

# Restore the original idle loop frequency:
# sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

# print('Program ended')


# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================


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

#client.setStepping(True)
sim.startSimulation()

###################################################################
# LEARNING LOOP
###################################################################
while (t := sim.getSimulationTime()) < 1:
    sim.callScriptFunction("close@ROBOTIQ_85", 1)
while (t := sim.getSimulationTime()) < 2:
    sim.callScriptFunction("open@ROBOTIQ_85", 1)
while (t := sim.getSimulationTime()) < 3:
    sim.callScriptFunction("close@ROBOTIQ_85", 1)

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
#else: 
#    print(caca)
