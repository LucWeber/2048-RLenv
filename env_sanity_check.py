from tkinter import Frame, Label, CENTER
import random
import gym
from gym import spaces
import numpy as np
import torch as t

BASE_PATH = '/home/lucas/Desktop/2048-python'

import sys

sys.path.insert(1, BASE_PATH)

import logic
import constants as c

'''
What I did:
- get necessary portions for a gym environment from 
https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
- copy over corresponding parts from python_2048 game from github

TODO: 
- DONE: solve logic from Frame and integrate it with gym_env
- check which aspects of official gym-env are missing and implement them 
- DONE: implement baseline
- implement simple RL-algorithm to learn gym_env
- afterwards, take care of visualization (reintegrate tkinter)
'''

N_DISCRETE_ACTIONS = 4
SIZE_GRID = (c.GRID_LEN, c.GRID_LEN)
MAX_VALUE = 2048


class EnvDummy(gym.Env, Frame):
    """2048 environment that follows gym interface

    Description:
        The well-known 2048 game. Move all tiles in one of four possible directions. If neighbouring tiles in direction
        of movement match, they merge and their value is being summed. Merging tiles increases the score. After every
        action a tile of value 2 is added.
    Source:
        Original game by Gabriele Cirulli.
    Observation:
        Type: Box(4)
        Num     Observation                                     Min           Max
        0       Tile matrix (GRID_SIZE x GRID_SIZE)             0             Inf (or less, depending on GRID_SIZE)
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Move all tiles up
        1     Move all tiles down
        2     Move all tiles left
        3     Move all tiles right
    Reward:
        Score produced by merging tiles. Score is equal to sum of values of all merged tiles.
    Starting State:
        Two value 2 tiles in random position.
    Episode Termination:
        The whole tile matrix is filled and there is no move that merges any tiles

    """

    def __init__(self, mode):
        super(EnvDummy, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects

        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.total_score = 0
        self.mode = mode
        self.n_illegal_actions = 0
        self.matrix = self.reset()
        self.observation_space = spaces.Box(low=0, high=MAX_VALUE, shape=SIZE_GRID, dtype=np.uint8)
        self.history_matrices = []

        # VISUALIZATION
        self.grid_visualization = None
        #self.render()

    def init_grid(self, grid_visualization):
        self.grid_visualization = grid_visualization

    def step(self, action):
        # Execute one time step within the environment
        # obs, reward, done = 1, 1, False -> obs  == self.matrix
        score, done = self._take_action(action)
        self.total_score += score
        reward = self.reward_func(score)

        return self.matrix, reward, done, {}

    def reward_func(self, score):
        return score

    def reset(self):
        # Reset the state of the environment to an initial state
        self.matrix = logic.new_game(c.GRID_LEN)
        self.total_score = 0
        self.n_illegal_actions = 0
        return self.matrix

    def render(self, mode=None, grid_visualization=None):
        '''see puzzle.update_grid_cells()'''

        if mode is None:
            mode = self.mode
        if mode == 'human':
            grid_visualization.update_grid_cells(self.matrix)
        elif mode == 'agent':
            print(self.matrix)

    def _next_observation(self):
        '''
        This is already implemented in logic.add_2 ?
        see puzzle.generate_next()
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

        return obs
        '''
        raise NotImplementedError

    def _take_action(self, action):
        ''' see puzzle.keydown()'''

        self.matrix, action_executed, score = self.action_logic_map(action, self.matrix)
        done = True if self._get_game_state() == 'lose' else False
        if action_executed:
            '''
            if not done:
                self.matrix = logic.add_two(self.matrix)
            # record last move
            self.history_matrices.append(self.matrix)
            #self.render()'''
        else:
            #print('Illegal action! Reward: -1')
            score = -1
            self.n_illegal_actions += 1

        """
        This is for visualization during human play; we don't need this right now!
        if logic.game_state(self.matrix) == 'win':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        if logic.game_state(self.matrix) == 'lose':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        """

        return score, done

    def action_logic_map(self, action, matrix):
        if (action == 0 and matrix[0][0] == 1) or \
            (action == 1 and matrix[0][0] == 2) or \
            (action == 2 and matrix[0][0] == 3) or \
            (action == 3 and matrix[0][0] == 0):
            score = 1
        else:
            score = -1

        matrix = [[action for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        return matrix, True, score

    def _get_game_state(self):
        return logic.game_state(self.matrix)
