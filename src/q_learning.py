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

BOLD = '\033[1m'
RESET = '\033[0m'

ACTION_LIST = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]

ENV_WINDOW_SIZE = 2  # Play with this parameter to test different observation windows
# ENV_WINDOW_SIZE = 3  # Let's try 3, this would be a 7x7 observation window, so pretty big... idk i'm curious if he'll do better

OUTPUT_DIR = 'q_learning_data'  # Just ensuring we output every file to one folder to keep things clean and organized

MAX_STEPS = 1000 # Let's use this parameter to control our max_steps for the agent, let's start by decreasing to 500

# Just some functionality for ensuring we output files to the right place
def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR

def get_output_path(filename):
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)

# Here's functionality for recognizing whether we're running our agent in training/evaluation mode, and if we'd like to
# visualize gameplay using the GUI or just let it do its thing under the hood.
train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

# Okay so runGames is a method in the OG pacman.py file that runs a specified number of games.  Below is simply a wrapper (basically
# the exact same function), but it contains functionality for:
# a. Ensuring we don't actually print out the "Pacman died!" stuff every time we run a game.
# b. Displaying or not displaying the GUI depending on the flag we give in the terminal.
def runGamesQuiet(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30, randomRewards=False, max_steps=MAX_STEPS):
    '''
    Wrapper around runGames that suppresses all game output messages.
    '''
    
    rules = ClassicGameRules(timeout, randomRewards=randomRewards, maxSteps=max_steps)
    rules.quiet = True  # This is the parameter to stop the "Pacman died!" stuff
    games = []

    use_gui = isinstance(display, PacmanGraphics)
    
    for i in tqdm(range(numGames), desc="Games", unit="game"):
        # To suppress output from the OG setup, we need to use NullGraphics
        if use_gui:
            gameDisplay = display
        else:
            gameDisplay = textDisplay.NullGraphics()
        game = rules.newGame(layout, pacman, ghosts, gameDisplay, True, catchExceptions)
        game.run()
        games.append(game)  # Always append, unlike original runGames
    
    return games


# This function is basically the same logic as the one from Programming Assignment 2, all we're doing here is turning each observed
# state into a single integer so that it's easier to create and then lookup optimal actions from our Q-table.
def hash_observation(obs):
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
def training_rewards_plot(reward_df, num_episodes, decay_rate):
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
            with open(get_output_path(self.qfile), 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[QLearningAgent] Failed to save Q-table: {e}")

    def _ensure_state(self, state_hash):
        '''
        Handy-dandy helper that we use to immediately check if the state we're in has been seen before.
        If not, then initalize a row in the Q-table for this newly observed state.
        '''
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.n_actions, dtype=np.float64)
            self.n_updates[state_hash] = np.zeros(self.n_actions, dtype=np.float64)

    def registerInitialState(self, state):
        '''Called at the start of each game.'''
        self.last_state_hash = None
        self.last_action_idx = None
        self.last_score = state.getScore()
        self.episode_reward = 0.0

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
        state_hash = hash_observation(obs)
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

        # Epsilon-greedy action selection
        if self.training and random.random() < self.epsilon:
            action_idx = random.choice(legal_actions)
        else:
            # Choose best legal action
            q_values = self.q_table[state_hash]
            best_value = -float('inf')
            action_idx = legal_actions[0]
            for legal_idx in legal_actions:
                if q_values[legal_idx] > best_value:
                    best_value = q_values[legal_idx]
                    action_idx = legal_idx

        # Store for next update
        self.last_state_hash = state_hash
        self.last_action_idx = action_idx

        # Return the direction (not index)
        direction = ACTION_LIST[action_idx]
        legal = state.getLegalPacmanActions()
        
        # Ensure direction is legal
        if direction not in legal:
            if Directions.STOP in legal and len(legal) > 1:
                legal = [d for d in legal if d != Directions.STOP]
            direction = legal[0] if legal else Directions.STOP
            # Update action_idx to match
            self.last_action_idx = ACTION_LIST.index(direction)

        return direction

    def _update_q(self, state_hash, action_idx, reward, next_state_hash, terminal):
        """Update Q-value using Q-learning rule."""
        if not self.training:
            return
        
        # Adaptive learning rate
        eta = 1.0 / (1.0 + self.n_updates[state_hash][action_idx])
        self.n_updates[state_hash][action_idx] += 1
        
        # Q-learning update
        current_q = self.q_table[state_hash][action_idx]
        if terminal or next_state_hash is None:
            next_max = 0.0
        else:
            self._ensure_state(next_state_hash)
            next_max = float(np.max(self.q_table[next_state_hash]))
        
        td_target = reward + self.gamma * next_max
        self.q_table[state_hash][action_idx] = (1.0 - eta) * current_q + eta * td_target

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


def context_from_obs(obs):
    '''Extract context from observation for analysis.'''
    try:
        if obs.get('at_capsule'):
            return 'CAPSULE'
        if obs.get('ghost_in_cell'):
            return 'GHOST'
        if obs.get('scared_ghost_in_cell'):
            return 'SCARED_GHOST'
        if obs.get('at_food'):
            return 'FOOD'
    except Exception:
        pass
    return 'EMPTY'


# Using softmax right now for help at analyzing action selection,
# but might remove this depending on how helpful it actually is.
def softmax(x, temp=1.0):
    '''Softmax function for action selection.'''
    x = np.asarray(x, dtype=np.float64)
    e_x = np.exp((x - np.max(x)) / max(1e-6, temp))
    return e_x / e_x.sum(axis=0)


# Default experiment settings
num_episodes = 25_000
decay_rate = 0.99995
# Let's include the window size because that definitely changes the Q-table.
qfile_name = f'Q_table_{num_episodes}_{decay_rate}_{ENV_WINDOW_SIZE}.pickle'
# qfile_name = f'Q_table_{num_episodes}_{decay_rate}.pickle'

if train_flag:
    # Training mode
    print(f"Training Q-learning agent for {num_episodes} episodes...")
    print(f"Decay rate: {decay_rate}")
    print(f"Q-table will be saved to: {qfile_name}")
    
    # Create agent in training mode
    agent = QLearningAgent(
        window_size=ENV_WINDOW_SIZE,
        gamma=0.9,
        epsilon=1.0,
        decay_rate=decay_rate,
        min_epsilon=0.05,
        qfile=qfile_name
    )
    agent.setTraining(True)
    
    # Use command-line args or defaults
    layout_name = 'mediumClassic'
    num_ghosts = 2
    ghost_type = 'RandomGhost'
    frame_time = 0.0 if not gui_flag else 0.001
    
    # Parse command line for layout/ghosts if provided
    if len(sys.argv) > 1:
        # Simple parsing - can be enhanced
        for arg in sys.argv[1:]:
            if arg.startswith('--layout='):
                layout_name = arg.split('=')[1]
            elif arg.startswith('--ghosts='):
                num_ghosts = int(arg.split('=')[1])
            elif arg.startswith('--ghost-type='):
                ghost_type = arg.split('=')[1]
    
    # Get layout and create ghosts
    layout = pac_layout.getLayout(layout_name)
    ghost_cls = getattr(ghostAgents, ghost_type)
    ghosts = [ghost_cls(i + 1) for i in range(num_ghosts)]
    
    # Setup display
    if gui_flag:
        from graphicsDisplay import PacmanGraphics
        display = PacmanGraphics(frameTime=frame_time)
    else:
        from textDisplay import NullGraphics
        display = NullGraphics()
    
    # Run training games
    time_start = time.perf_counter()
    games = runGamesQuiet(
        layout=layout,
        pacman=agent,
        ghosts=ghosts,
        display=display,
        numGames=num_episodes,
        record=False,
        numTraining=num_episodes,  # All games are training
        catchExceptions=False,
        timeout=30,
        randomRewards=False,
        max_steps=MAX_STEPS
    )
    time_end = time.perf_counter()
    
    print(f'\nTraining Runtime (sec): {time_end - time_start:.2f}')
    
    # Save final Q-table
    agent._save_qtable()
    
    # Generate reward plot
    if len(agent.episode_rewards) > 0:
        running_avg = np.cumsum(agent.episode_rewards) / np.arange(1, len(agent.episode_rewards) + 1)
        reward_df = pd.DataFrame({
            "episode": np.arange(len(agent.episode_rewards)),
            "avg_reward": running_avg
        })
        reward_df.to_csv(get_output_path(f"episode_rewards_{num_episodes}_{decay_rate}_{ENV_WINDOW_SIZE}.csv"), index=False)
        training_rewards_plot(reward_df, num_episodes, decay_rate)
        print(f"Average reward: {np.mean(agent.episode_rewards):.2f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Q-table size: {len(agent.q_table)} states")

else:
    # Evaluation mode
    filename = qfile_name
    input(f"\n{BOLD}Currently loading Q-table from {filename}{RESET}.\n\nPress Enter to confirm, or Ctrl+C to cancel.\n")
    
    # Create agent in evaluation mode (loads Q-table)
    agent = QLearningAgent(
        window_size=ENV_WINDOW_SIZE,
        epsilon=0.0,  # No exploration
        qfile=filename,
        load_qtable=True
    )
    agent.setTraining(False)
    
    print(f'Length of Q-Table: {len(agent.q_table)}')
    
    # Setup for evaluation
    layout_name = 'mediumClassic'
    num_ghosts = 2
    ghost_type = 'RandomGhost'
    frame_time = 0.0 if not gui_flag else 0.001
    num_eval_games = 100
    
    # Parse command line if provided
    for arg in sys.argv[1:]:
        if arg.startswith('--layout='):
            layout_name = arg.split('=')[1]
        elif arg.startswith('--ghosts='):
            num_ghosts = int(arg.split('=')[1])
        elif arg.startswith('--games='):
            num_eval_games = int(arg.split('=')[1])
    
    layout = pac_layout.getLayout(layout_name)
    ghost_cls = getattr(ghostAgents, ghost_type)
    ghosts = [ghost_cls(i + 1) for i in range(num_ghosts)]
    
    if gui_flag:
        from graphicsDisplay import PacmanGraphics
        display = PacmanGraphics(frameTime=frame_time)
    else:
        from textDisplay import NullGraphics
        display = NullGraphics()
    
    # Track statistics
    contexts_order = ["CAPSULE", "GHOST", "SCARED_GHOST", "FOOD", "EMPTY"]
    action_hist_by_ctx = {c: np.zeros(len(ACTION_LIST), dtype=np.int64) for c in contexts_order}
    newly_discovered_states = []
    previsited_states = []
    
    # Run evaluation games
    time_start = time.perf_counter()
    games = runGamesQuiet(
        layout=layout,
        pacman=agent,
        ghosts=ghosts,
        display=display,
        numGames=num_eval_games,
        record=False,
        numTraining=0,  # No training
        catchExceptions=False,
        timeout=30,
        randomRewards=False,
        max_steps=MAX_STEPS
    )
    time_end = time.perf_counter()
    
    # Collect statistics from games
    scores = [g.state.getScore() for g in games]
    wins = [g.state.isWin() for g in games]
    # Try to get episode lengths from move history if available
    try:
        episode_lengths = [len(g.moveHistory) for g in games]
    except AttributeError:
        episode_lengths = []  # Not available
    
    total_evaluation_time = time_end - time_start
    total_eval_time_min = int(total_evaluation_time // 60)
    eval_time_remaining_sec = int(total_evaluation_time % 60)
    
    avg_reward = np.mean(scores) if scores else 0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    win_rate = sum(wins) / len(wins) if wins else 0
    
    print()
    print("Number of Training Episodes:", num_episodes)
    print("Decay Rate:", decay_rate)
    print()
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Win Rate: {win_rate:.2%} ({sum(wins)}/{len(wins)})")
    print(f"Average Episode Length: {avg_length:.2f} moves")
    print()
    print(f"Time taken to run {num_eval_games} evaluations: {total_evaluation_time:.2f} sec")
    print(f"Time taken (minutes): {total_eval_time_min}:{eval_time_remaining_sec:02d}")

