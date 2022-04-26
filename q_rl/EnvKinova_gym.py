from collections import OrderedDict
from typing import Optional
from re import A
import sys
import time
import math
import numpy as np
from numpy import linalg as LA, uint8
sys.path.append('/home/robocomp/software/CoppeliaSim_Edu_V4_3_0_Ubuntu20_04/programming/zmqRemoteApi/clients/python')
from zmqRemoteApi import RemoteAPIClient

import gym
from gym import spaces


class EnvKinova_gym(gym.Env):
    def __init__(self):
        super(EnvKinova_gym, self).__init__()
        """ Starts an API client, oads the scene, takes all the important objects
            from it and starts the simulation. """
        print('Program started')
        
        # 1. ENVIRONMENT
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        # 1.1 Scene
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.loadScene("/home/robocomp/robocomp/components/RoboticaAvanzada/rl_env/etc/kinova_rl.ttt")
        print('Scene loaded')

        # 1.2 Env's reference system origin
        self.rs_zero = self.sim.getObjectHandle('gen3')

        # 1.3 Agent
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
         
        # 1.4 Block
        self.block = self.sim.getObjectHandle('block')
        self.block_init_pose = self.sim.getObjectPose(self.block, self.rs_zero)

        # 2. ACTION AND OBSERVATION SPACES 
        self.__set_action_space()
        action = self.action_space.sample()
        observation, _, done, _ = self.step(action)
        assert not done
        self.__set_observation_space(observation)

        # TODO: Parámetros antigupos, revisar si son necearios o no
        self.testITER = 0
        self.testIndex = 0
        self.SIZE = 20
        self.EXPLORE = 0.45
        self.possible_values = [-1, 0, 1]
        self.max_steps = 200
        self.current_step = 0

        # 3 START THE SIMULAITON
        self.sim.startSimulation()

    # __init__() AUX METHODS
    def __set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def __set_observation_space(self, observation):
        self.observation_space = self.__convert_observation_to_space(observation)
        return self.observation_space

    def __convert_observation_to_space(self, observation):
        if isinstance(observation, dict):
            space = spaces.Dict(
                OrderedDict(
                    [
                        (key, self.__convert_observation_to_space(value))
                        for key, value in observation.items()
                    ]
                )
            )
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -float("inf"), dtype=np.float32)
            high = np.full(observation.shape, float("inf"), dtype=np.float32)
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)

        return space

################################################## Mojoco basic env methods ############################################

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.sim.reset()
        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

#######################################################################################################################



################################### MUJUCO INHERITING ENV METHODS #####################################################
    import numpy as np

    from gym import utils
    from gym.envs.mujoco import mujoco_env

    DEFAULT_CAMERA_CONFIG = {
        "trackbodyid": 1,
        "distance": 4.0,
        "lookat": np.array((0.0, 0.0, 2.0)),
        "elevation": -20.0,
    }


    def mass_center(model, sim):
        mass = np.expand_dims(model.body_mass, axis=1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

    class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        def __init__(
            self,
            xml_file="humanoid.xml",
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            contact_cost_weight=5e-7,
            contact_cost_range=(-np.inf, 10.0),
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
        ):
            utils.EzPickle.__init__(**locals())

            self._forward_reward_weight = forward_reward_weight
            self._ctrl_cost_weight = ctrl_cost_weight
            self._contact_cost_weight = contact_cost_weight
            self._contact_cost_range = contact_cost_range
            self._healthy_reward = healthy_reward
            self._terminate_when_unhealthy = terminate_when_unhealthy
            self._healthy_z_range = healthy_z_range

            self._reset_noise_scale = reset_noise_scale

            self._exclude_current_positions_from_observation = (
                exclude_current_positions_from_observation
            )

            mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        @property
        def healthy_reward(self):
            return (
                float(self.is_healthy or self._terminate_when_unhealthy)
                * self._healthy_reward
            )

        def control_cost(self, action):
            control_cost = self._ctrl_cost_weight * np.sum(np.square(self.sim.data.ctrl))
            return control_cost

        @property
        def contact_cost(self):
            contact_forces = self.sim.data.cfrc_ext
            contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
            min_cost, max_cost = self._contact_cost_range
            contact_cost = np.clip(contact_cost, min_cost, max_cost)
            return contact_cost

        @property
        def is_healthy(self):
            min_z, max_z = self._healthy_z_range
            is_healthy = min_z < self.sim.data.qpos[2] < max_z

            return is_healthy

        @property
        def done(self):
            done = (not self.is_healthy) if self._terminate_when_unhealthy else False
            return done

        def _get_obs(self):
            position = self.sim.data.qpos.flat.copy()
            velocity = self.sim.data.qvel.flat.copy()

            com_inertia = self.sim.data.cinert.flat.copy()
            com_velocity = self.sim.data.cvel.flat.copy()

            actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
            external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

            if self._exclude_current_positions_from_observation:
                position = position[2:]

            return np.concatenate(
                (
                    position,
                    velocity,
                    com_inertia,
                    com_velocity,
                    actuator_forces,
                    external_contact_forces,
                )
            )

        def step(self, action):
            xy_position_before = mass_center(self.model, self.sim)
            self.do_simulation(action, self.frame_skip)
            xy_position_after = mass_center(self.model, self.sim)

            xy_velocity = (xy_position_after - xy_position_before) / self.dt
            x_velocity, y_velocity = xy_velocity

            ctrl_cost = self.control_cost(action)
            contact_cost = self.contact_cost

            forward_reward = self._forward_reward_weight * x_velocity
            healthy_reward = self.healthy_reward

            rewards = forward_reward + healthy_reward
            costs = ctrl_cost + contact_cost

            observation = self._get_obs()
            reward = rewards - costs
            done = self.done
            info = {
                "reward_linvel": forward_reward,
                "reward_quadctrl": -ctrl_cost,
                "reward_alive": healthy_reward,
                "reward_impact": -contact_cost,
                "x_position": xy_position_after[0],
                "y_position": xy_position_after[1],
                "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
                "x_velocity": x_velocity,
                "y_velocity": y_velocity,
                "forward_reward": forward_reward,
            }

            return observation, reward, done, info

        def reset_model(self):
            noise_low = -self._reset_noise_scale
            noise_high = self._reset_noise_scale

            qpos = self.init_qpos + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            )
            qvel = self.init_qvel + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nv
            )
            self.set_state(qpos, qvel)

            observation = self._get_obs()
            return observation

        def viewer_setup(self):
            for key, value in DEFAULT_CAMERA_CONFIG.items():
                if isinstance(value, np.ndarray):
                    getattr(self.viewer.cam, key)[:] = value
                else:
                    setattr(self.viewer.cam, key, value)
    }
#######################################################################################################################

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

    