import sys
import os
import time
import pickle
import random
import textwrap
import textDisplay

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from game import Agent, Directions, Game
from pacman import GameState, runGames, readCommand, ClassicGameRules
import layout as pac_layout
import ghostAgents
from graphicsDisplay import PacmanGraphics
from training_utils import get_output_path


ACTION_LIST = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]

ENV_WINDOW_SIZE = 2  # Play with this parameter to test different observation windows



# This function is basically the same logic as the one from Programming Assignment 2, all we're doing here is turning each observed
# state into a single integer so that it's easier to create and then lookup optimal actions from our Q-table.
def hashObservation(obs):
    '''
    Hash the observed state into a single integer using base-7 encoding.
    Encoding:
    0 -> wall/out of bounds
    1 -> empty
    2 -> food
    3 -> capsule
    4 -> ghost
    5 -> scared ghost
    6 -> pacman (center cell)
    '''
    window_size = obs.get('window_size', ENV_WINDOW_SIZE)
    window = obs.get('window', {})
    base = 7
    value = 0
    multiplier = 1

    for dx in range(-window_size, window_size + 1):
        for dy in range(-window_size, window_size + 1):
            cell = window.get((dx, dy))
            code = 0
            if cell and cell['in_bounds'] and not cell['wall']:
                code = 1
                if cell['ghosts']:
                    code = 4
                elif cell['scared_ghosts']:
                    code = 5
                elif cell['capsule']:
                    code = 3
                elif cell['food']:
                    code = 2
            if dx == 0 and dy == 0:
                code = 6
            value += code * multiplier
            multiplier *= base
    return value

# Taken from my PA2 assignment, this method is just a helper to help with plotting out the running average of rewards
# over a training cycle.  Helps identify if the agent is actually getting better or just shitty the whole time.
def avg_training_rewards_plot(reward_df, num_episodes, decay_rate):
    reward_df = reward_df.sort_values('episode')

    plt.figure(figsize=(10, 6))
    plt.plot(
        reward_df['episode'],
        reward_df['avg_reward'],
        color='#0072B2',
        linewidth=2.0,
        label='Running Avg. Reward'
    )

    long_title = f"Running Average of Rewards per Episode ({num_episodes:,.0f} episodes, decay rate = {decay_rate})"
    wrapped_title = "\n".join(textwrap.wrap(long_title, width=50))
    plt.title(wrapped_title, fontsize=14, weight="bold", pad=15)
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Running Avg. Reward", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_name = get_output_path(f"avg_reward_plot_{num_episodes}_{decay_rate}_{ENV_WINDOW_SIZE}.png")
    plt.savefig(out_name, dpi=300)
    plt.close()


def training_rewards_overtime(reward_df, num_episodes, decay_rate):
    reward_df = reward_df.sort_values('episode')

    plt.figure(figsize=(10, 6))
    plt.plot(
        reward_df['episode'],
        reward_df['static_rewards'],
        color='#0072B2',
        linewidth=2.0,
        label='Reward per Episode'
    )

    long_title = f"Episode Rewards Over ({num_episodes:,.0f} Episodes, Decay Rate = {decay_rate})"
    wrapped_title = "\n".join(textwrap.wrap(long_title, width=50))
    plt.title(wrapped_title, fontsize=14, weight="bold", pad=15)
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Reward per Episode", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_name = get_output_path(f"rewards_overtime_plot_{num_episodes}_{decay_rate}_{ENV_WINDOW_SIZE}.png")
    plt.savefig(out_name, dpi=300)
    plt.close()

# And here's the big guy, the class that holds our Q-learning agent.
class QLearningAgent(Agent):
    '''
    Q-learning agent that uses partial observability like PA2, with specified window size from above.
    The agent implements the Agent interface from the OG pacman environment we forked, that way it should
    provide some easy access for us to implement advancements to the Q-learning algorithm down the road.
    '''

    def __init__(self, 
                 window_size=ENV_WINDOW_SIZE,
                 gamma=0.9,
                 epsilon=1.0,
                 decay_rate=0.999,
                 min_epsilon=0.05,
                 qfile=None,
                 load_qtable=False):
        Agent.__init__(self)
        self.window_size = window_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.qfile = qfile
        self.needsDeepCopy = False
        
        self.q_table = {}
        self.n_updates = {}
        self.n_actions = len(ACTION_LIST)
        self.training = True  # Default to training mode
        
        # Episode tracking
        self.last_state_hash = None
        self.last_action_idx = None
        self.last_score = 0.0
        self.episode_reward = 0.0
        self.episode_rewards = []

        # Tracking statistics during evaluation:
        self.total_actions = 0
        self.qtable_actions = 0
        self.random_actions = 0
        self.new_states_encountered = 0
        
        # Load Q-table if specified
        if load_qtable and qfile:
            self._load_qtable()

    def _load_qtable(self):
        '''Load Q-table from file.'''
        if not self.qfile or not os.path.exists(get_output_path(self.qfile)):
            return
        try:
            with open(get_output_path(self.qfile), 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'q_table' in data:
                    self.q_table = data['q_table']
                    self.n_updates = data.get('n_updates', {})
                else:
                    self.q_table = data
                    self.n_updates = {k: np.zeros(self.n_actions, dtype=np.float64) 
                                     for k in self.q_table}
            print(f"[QLearningAgent] Loaded Q-table from {self.qfile} ({len(self.q_table)} states)")
        except Exception as e:
            print(f"[QLearningAgent] Failed to load Q-table: {e}")

    def _save_qtable(self):
        '''Save Q-table to file.'''
        if not self.qfile:
            return
    
        try:
            data = {'q_table': self.q_table, 'n_updates': self.n_updates}
            path = get_output_path(self.qfile, agent_type='qlearning')
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[QLearningAgent] Failed to save Q-table: {e}")

    def _ensure_state(self, state_hash):
        '''
        Handy-dandy helper that we use to immediately check if the state we're in has been seen before.
        If not, then initalize a row in the Q-table for this newly observed state.
        '''
        if state_hash not in self.q_table:
            self.new_states_encountered += 1
            self.random_actions += 1
            self.q_table[state_hash] = np.zeros(self.n_actions, dtype=np.float64)
            self.n_updates[state_hash] = np.zeros(self.n_actions, dtype=np.float64)
        else:
            self.qtable_actions += 1

    def registerInitialState(self, state):
        '''Called at the start of each game.'''
        self.last_state_hash = None
        self.last_action_idx = None
        self.last_score = state.getScore()
        self.episode_reward = 0.0

    def observationFunction(self, state):
        '''In the game.py file, the pre-existing environment makes a deepcopy of the game state
           at every step, becoming EXTREMELY costly for training my agent.  This method should
           help with mitigating that extra time cost.'''
        return state

    def getAction(self, state):
        '''
        Method takes in the current state and returns the most optimal action to take (or a random
        legal action if the agent can't make that choice for whatever reason).  The environment setup
        makes this super easy here because we can just define this getAction() method for whatever agent
        we're building and the environment takes care of the boring stuff like ensuring compatability with
        display, etc..
        '''
        # We have current state (awesome), just use the buildObservation method to get that partially observable
        # window around Pacman, then hash it to make our next moves.
        obs = state.buildObservation(window_size=self.window_size)
        state_hash = hashObservation(obs)
        self._ensure_state(state_hash)

        # Update Q-table from previous state-action pair
        if self.last_state_hash is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            reward = current_score - self.last_score
            self.last_score = current_score
            self.episode_reward += reward
            
            # Check if terminal
            terminal = state.isWin() or state.isLose()
            
            # Q-learning update
            self._update_q(self.last_state_hash, self.last_action_idx, 
                          reward, state_hash, terminal)

        # Get legal actions
        legal_actions = state.getLegalActionsIndices(ACTION_LIST)
        if not legal_actions:
            legal_actions = [0]

        if not self.training:
            self.total_actions += 1

        # Want to use argmax for time constraints while we're training, but use softmax during evaluation.
        if self.training:
            if np.random.random() < self.epsilon:
                action_index = np.random.choice(legal_actions)
            else:
                q_values = self.q_table[state_hash][legal_actions]
                best_index = np.argmax(q_values)
                action_index = legal_actions[best_index]
        else:
            try:
                action_index = np.random.choice(legal_actions, p=softmax(self.q_table[state_hash][legal_actions]))
            except Exception as e:
                action_index = np.random.choice(legal_actions)

        # Store for next update
        self.last_state_hash = state_hash
        self.last_action_idx = action_index

        # Return the direction (not index)
        direction = ACTION_LIST[action_index]
        legal = state.getLegalPacmanActions()
        
        # Ensure direction is legal
        if direction not in legal:
            if Directions.STOP in legal and len(legal) > 1:
                legal = [d for d in legal if d != Directions.STOP]
            direction = legal[0] if legal else Directions.STOP
            # Update action_index to match
            self.last_action_idx = ACTION_LIST.index(direction)

        return direction

    def _update_q(self, state_hash, action_index, reward, next_state_hash, terminal):
        """Update Q-value using Q-learning rule."""
        if not self.training:
            return
        
        # Adaptive learning rate
        eta = 1.0 / (1.0 + self.n_updates[state_hash][action_index])
        self.n_updates[state_hash][action_index] += 1
        
        # Q-learning update
        current_q = self.q_table[state_hash][action_index]
        if terminal or next_state_hash is None:
            next_max = 0.0
        else:
            self._ensure_state(next_state_hash)
            next_max = float(np.max(self.q_table[next_state_hash]))
        
        td_target = reward + self.gamma * next_max
        self.q_table[state_hash][action_index] = (1.0 - eta) * current_q + eta * td_target

    def final(self, state):
        '''
        Called at the end of each game.
        Update Q-table with final reward and decay epsilon.
        '''
        # Final Q-update if we have a pending state-action pair
        if self.last_state_hash is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            reward = current_score - self.last_score
            self.episode_reward += reward
            
            # Final update (terminal state)
            self._update_q(self.last_state_hash, self.last_action_idx, 
                          reward, None, True)

        # Store episode reward
        self.episode_rewards.append(self.episode_reward)

        # Decay epsilon
        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        # Saves Q-table periodically for testing purposes, remove in practice, can be costly
        # if self.qfile and len(self.episode_rewards) % 5_000 == 0:
        #     self._save_qtable()

    def setTraining(self, training):
        '''Set whether agent is in training mode.'''
        self.training = training
        if not training:
            self.epsilon = 0.0  # No exploration during evaluation

    def getEvalStats(self):
        '''Method returns our agent stats during evaluation mode.'''
        if self.total_actions == 0:
            return {
                'total_actions': 0,
                'qtable_actions': 0,
                'random_actions': 0,
                'qtable_percentage': 0.0,
                'random_percentage': 0.0,
                'new_states_encountered': self.new_states_encountered
            }
        
        return {
            'total_actions': self.total_actions,
            'qtable_actions': self.qtable_actions,
            'random_actions': self.random_actions,
            'qtable_percentage': self.qtable_actions / self.total_actions,
            'random_percentage': self.random_actions / self.total_actions,
            'new_states_encountered': self.new_states_encountered
        }
# Keeping softmax, helps agent optimally select actions.
def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)
