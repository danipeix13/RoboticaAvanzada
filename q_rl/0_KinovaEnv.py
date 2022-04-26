import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

X = 0
Y = 1
Z = 2
G = 3
W = 4

X_SIZE = 100 # mm
Y_SIZE = 100 # mm
Z_SIZE = 100 # mm
G_SIZE = 60  # degrees
W_SIZE = 45  # degrees

class KinovaEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (X_SIZE, Y_SIZE, Z_SIZE, G_SIZE, W_SIZE)

        nS = np.prod(self.shape) # Number of states
        nA = 5 # Number of actions

        # Cube
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][X] = self._calculate_transition_prob(position, [-1, 1], winds)
            P[s][Y] = self._calculate_transition_prob(position, [-1, 1], winds)
            P[s][Z] = self._calculate_transition_prob(position, [-1, 1], winds)
            P[s][G] = self._calculate_transition_prob(position, [-1, 1], winds)
            P[s][W] = self._calculate_transition_prob(position, [-1, 1], winds)

        # Starting position
        isd = np.zeros(nS)

        super(KinovaEnv, self).__init__(nS, nA, P, isd)

    def render(self): # delete if not necessary
        pass