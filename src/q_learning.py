import sys
import time
import pickle
import random
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pacman import GameState, ClassicGameRules
import layout as pac_layout
import ghostAgents
from game import Directions
from util import nearestPoint
from graphicsDisplay import PacmanGraphics

BOLD = '\033[1m'
RESET = '\033[0m'

ACTION_LIST = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]

WINDOW_RADIUS = 2  # 5x5 observation window
LAYOUT_NAME = 'mediumClassic'
GHOST_AGENT = 'RandomGhost'
NUM_GHOSTS = 2
MAX_STEPS = 1000
FRAME_TIME = 0.0

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv


# class ActionSpace:
#     def __init__(self, actions):
#         self.actions = actions
#         self.n = len(actions)

#     def sample(self):
#         return random.randrange(self.n)

#     def to_direction(self, index):
#         index = max(0, min(self.n - 1, int(index)))
#         return self.actions[index]


# class PacmanEnv:
#     def __init__(self, layout_name=LAYOUT_NAME, radius=WINDOW_RADIUS, num_ghosts=NUM_GHOSTS,
#                  ghost_name=GHOST_AGENT, gui=False, frame_time=FRAME_TIME, max_steps=MAX_STEPS):
#         self.layout_name = layout_name
#         self.radius = radius
#         self.num_ghosts = num_ghosts
#         self.ghost_cls = getattr(ghostAgents, ghost_name)
#         self.gui = gui
#         self.frame_time = frame_time
#         self.max_steps = max_steps
#         self.display = None
#         self.rules = ClassicGameRules(timeout=max_steps)
#         self.action_space = ActionSpace([
#             Directions.NORTH,
#             Directions.SOUTH,
#             Directions.EAST,
#             Directions.WEST,
#             Directions.STOP,
#         ])
#         self.state = None
#         self.ghosts = []
#         self.prev_score = 0.0
#         self.steps = 0

#     def reset(self):
#         layout_obj = pac_layout.getLayout(self.layout_name)
#         if layout_obj is None:
#             raise ValueError(f"Layout {self.layout_name} not found.")

#         self.state = GameState()
#         self.state.initialize(layout_obj, self.num_ghosts)
#         self.ghosts = [self.ghost_cls(i + 1) for i in range(self.num_ghosts)]
#         self.prev_score = self.state.getScore()
#         self.steps = 0

#         if self.gui:
#             from graphicsDisplay import PacmanGraphics
#             self.display = PacmanGraphics(frameTime=self.frame_time)
#             self.display.initialize(self.state.data)

#         obs = self._build_observation()
#         return obs, 0.0, False, {}

#     def step(self, action_index):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         direction = self.action_space.to_direction(action_index)
#         legal = self.state.getLegalPacmanActions()
        
#         # Ensure we always use a legal action
#         if direction not in legal:
#             # Remove STOP if there are other options
#             if Directions.STOP in legal and len(legal) > 1:
#                 legal = [a for a in legal if a != Directions.STOP]
#             # Pick first legal action (or STOP if it's the only option)
#             direction = legal[0] if legal else Directions.STOP

#         # Apply Pacman's action
#         self.state = self.state.generateSuccessor(0, direction)

#         if self.gui and self.display:
#             self.display.update(self.state.data)

#         # Only move ghosts if game is not over
#         if not self.state.isWin() and not self.state.isLose():
#             for idx, ghost in enumerate(self.ghosts, start=1):
#                 # Check again in case state became terminal during ghost moves
#                 if self.state.isWin() or self.state.isLose():
#                     break
#                 ghost_action = ghost.getAction(self.state)
#                 self.state = self.state.generateSuccessor(idx, ghost_action)
#                 if self.gui and self.display:
#                     self.display.update(self.state.data)

#         current_score = self.state.getScore()
#         reward = current_score - self.prev_score
#         self.prev_score = current_score
#         self.steps += 1

#         done = self.state.isWin() or self.state.isLose() or self.steps >= self.max_steps
#         obs = self._build_observation()
#         info = {
#             'score': current_score,
#             'win': self.state.isWin(),
#             'lose': self.state.isLose(),
#             'steps': self.steps,
#         }

#         if done and self.display:
#             self.display.finish()

#         return obs, reward, done, info

#     def get_legal_actions(self):
#         """Return list of legal action indices for current state."""
#         if self.state is None:
#             return []
#         legal_dirs = self.state.getLegalPacmanActions()
#         # Remove STOP if there are other options
#         if Directions.STOP in legal_dirs and len(legal_dirs) > 1:
#             legal_dirs = [d for d in legal_dirs if d != Directions.STOP]
#         # Map directions to action indices
#         legal_indices = []
#         for idx, direction in enumerate(self.action_space.actions):
#             if direction in legal_dirs:
#                 legal_indices.append(idx)
#         return legal_indices if legal_indices else [self.action_space.actions.index(Directions.STOP)]

#     def _build_observation(self):
#         pac_pos = self.state.getPacmanPosition()
#         px, py = int(round(pac_pos[0])), int(round(pac_pos[1]))
#         walls = self.state.getWalls()
#         food = self.state.getFood()
#         capsules = set((int(x), int(y)) for x, y in self.state.getCapsules())

#         ghost_map = {}
#         for ghost_state in self.state.getGhostStates():
#             gpos = ghost_state.getPosition()
#             if gpos is None:
#                 continue
#             gx, gy = nearestPoint(gpos)
#             gx, gy = int(gx), int(gy)
#             ghost_map.setdefault((gx, gy), []).append(
#                 ghost_state.scaredTimer > 0
#             )

#         window = {}
#         for dx in range(-self.radius, self.radius + 1):
#             for dy in range(-self.radius, self.radius + 1):
#                 x, y = px + dx, py + dy
#                 in_bounds = 0 <= x < walls.width and 0 <= y < walls.height
#                 cell = {
#                     'in_bounds': in_bounds,
#                     'wall': False,
#                     'food': False,
#                     'capsule': False,
#                     'ghosts': [],
#                     'scared_ghosts': [],
#                 }
#                 if in_bounds:
#                     cell['wall'] = walls[x][y]
#                     cell['food'] = bool(food[x][y])
#                     cell['capsule'] = (x, y) in capsules
#                     for scared in ghost_map.get((x, y), []):
#                         if scared:
#                             cell['scared_ghosts'].append(True)
#                         else:
#                             cell['ghosts'].append(True)
#                 window[(dx, dy)] = cell

#         obs = {
#             'pacman_position': (px, py),
#             'window': window,
#             'at_food': bool(food[px][py]),
#             'at_capsule': (px, py) in capsules,
#             'ghost_in_cell': bool(ghost_map.get((px, py))),
#             'scared_ghost_in_cell': any(ghost_map.get((px, py), [])),
#             'radius': self.radius,
#         }
#         return obs


# env = PacmanEnv(gui=gui_flag)

layout_obj = pac_layout.getLayout(LAYOUT_NAME)
ghost_cls = getattr(ghostAgents, GHOST_AGENT)
ghosts = [ghost_cls(i + 1) for i in range(NUM_GHOSTS)]

# GUI setup (if needed)
display = None
if gui_flag:
    display = PacmanGraphics(frameTime=FRAME_TIME)

def reset_game():
    '''Create a new game state and return initial observation.'''
    state = GameState()
    state.initialize(layout_obj, NUM_GHOSTS)
    
    if display:
        display.initialize(state.data)
        display.update(state.data)
    
    obs = state.buildObservation(radius=WINDOW_RADIUS)
    return state, obs, state.getScore()


def hash(obs):
    '''
    Just like PA2, we can simply hash the observed state into a single integer.  In this case
    we will use a base-7 system to encode the observation window.  The encoding will be as follows:
    0 -> wall/out of bounds
    1 -> empty
    2 -> food
    3 -> capsule
    4 -> ghost
    5 -> scared ghost
    6 -> pacman (center cell)
    '''
    radius = obs.get('radius', WINDOW_RADIUS)
    window = obs.get('window', {})
    base = 7
    value = 0
    multiplier = 1

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
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


# Helper method for plotting the running average of rewards over training episodes:
def training_rewards_plot(reward_df: pd.DataFrame = None, num_episodes: int = None, decay_rate: float = None):
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

    out_name = f"avg_reward_plot_{num_episodes}_{decay_rate}.png"
    plt.savefig(out_name, dpi=300)
    plt.close()


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.999):
    '''
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    '''
    Q_table = {}
    max_steps = MAX_STEPS
    n_actions = len(ACTION_LIST)
    N_updates = {}

    # Keep track of running rewards:
    ep_rewards = np.empty(num_episodes, dtype=np.float64)
    running_sum = 0.0

    time_start = time.perf_counter()

    for episode in tqdm(range(num_episodes)):
        game_state, obs, prev_score = reset_game()
        done = False
        steps = 0
        episode_reward = 0.0

        while not done and steps < max_steps:
            state = hash(obs)

            if state not in Q_table:
                Q_table[state] = np.zeros(n_actions, dtype=np.float64)
                N_updates[state] = np.zeros(n_actions, dtype=np.float64)

            # Get legal actions for current state
            legal_actions = game_state.getLegalActionsIndices(ACTION_LIST)
            if not legal_actions:
                legal_actions = [0]  # Fallback to first action

            if np.random.random() < epsilon:
                action = random.choice(legal_actions)
            else:
                # Only consider legal actions when choosing best
                q_values = Q_table[state]
                best_value = -float('inf')
                action = legal_actions[0]
                for legal_idx in legal_actions:
                    if q_values[legal_idx] > best_value:
                        best_value = q_values[legal_idx]
                        action = legal_idx

            direction = ACTION_LIST[action]
            legal = game_state.getLegalPacmanActions()
            if direction not in legal:
                legal = [d for d in legal if d != Directions.STOP]
            direction = legal[0] if legal else Directions.STOP
            
            next_game_state = game_state.stepWithGhosts(direction, ghosts)

            current_score = next_game_state.getScore()
            reward = current_score - prev_score
            prev_score = current_score
            done = next_game_state.isWin() or next_game_state.isLose() or steps >= max_steps

            next_obs = next_game_state.buildObservation(radius=WINDOW_RADIUS)
            next_state = hash(next_obs)

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(n_actions, dtype=np.float64)
                N_updates[next_state] = np.zeros(n_actions, dtype=np.float64)

            eta = 1.0 / (1.0 + N_updates[state][action])
            N_updates[state][action] += 1

            td_target = reward + (0.0 if done else gamma * float(np.max(Q_table[next_state])))
            Q_table[state][action] = (1.0 - eta) * Q_table[state][action] + eta * td_target

            obs = next_obs
            steps += 1
            episode_reward += reward

            if display:
                display.update(next_game_state.data)

        epsilon *= decay_rate
        running_sum += episode_reward
        ep_rewards[episode] = running_sum / (episode + 1)

    time_end = time.perf_counter()
    print('Training Runtime (sec):', time_end - time_start)
    reward_df = pd.DataFrame({
        "episode": np.arange(num_episodes),
        "avg_reward": ep_rewards
    })
    reward_df.to_csv(f"episode_rewards_{num_episodes}_{decay_rate}.csv", index=False)
    training_rewards_plot(reward_df=reward_df, num_episodes=num_episodes, decay_rate=decay_rate)

    return Q_table


# Default experiment settings
num_episodes = 10_000
decay_rate = 0.9999


if train_flag:
    Q_table = Q_learning(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate)
    with open('Q_table_' + str(num_episodes) + '_' + str(decay_rate) + '.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


def context_from_obs(obs):
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


def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=np.float64)
    e_x = np.exp((x - np.max(x)) / max(1e-6, temp))
    return e_x / e_x.sum(axis=0)


def refresh(obs, reward, done, info, delay=0.1):
    time.sleep(delay)


if not train_flag:
    rewards = []
    filename = 'Q_table_' + str(num_episodes) + '_' + str(decay_rate) + '.pickle'
    input(f"\n{BOLD}Currently loading Q-table from " + filename + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in q_learning.py).")
    with open(filename, 'rb') as handle:
        Q_table = pickle.load(handle)
    print('Length of Q-Table:', len(Q_table))

    time_start = time.perf_counter()
    episode_lengths = []
    newly_discovered_states = []
    previsited_states = []

    contexts_order = ["CAPSULE", "GHOST", "SCARED_GHOST", "FOOD", "EMPTY"]
    action_hist_by_ctx = {c: np.zeros(len(ACTION_LIST), dtype=np.int64) for c in contexts_order}

    for episode in tqdm(range(100)):
        game_state, obs, prev_score = reset_game()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            state = hash(obs)
            legal_actions = game_state.getLegalActionsIndices(ACTION_LIST)
            if not legal_actions:
                legal_actions = [0]
            
            try:
                q_values = Q_table[state]
                # Only consider legal actions
                legal_q = [q_values[i] for i in legal_actions]
                legal_probs = softmax(legal_q)
                action_idx = np.random.choice(len(legal_actions), p=legal_probs)
                action = legal_actions[action_idx]
                previsited_states.append(state)
            except KeyError:
                action = random.choice(legal_actions)
                newly_discovered_states.append(state)

            ctx = context_from_obs(obs)
            if ctx in action_hist_by_ctx:
                action_hist_by_ctx[ctx][action] += 1

            direction = ACTION_LIST[action]
            legal = game_state.getLegalPacmanActions()
            if direction not in legal:
                if Directions.STOP in legal and len(legal) > 1:
                    legal = [d for d in legal if d != Directions.STOP]
                direction = legal[0] if legal else Directions.STOP

            next_game_state = game_state.stepWithGhosts(direction, ghosts)
            current_score = next_game_state.getScore()
            reward = current_score - prev_score
            prev_score = current_score
            done = next_game_state.isWin() or next_game_state.isLose()

            obs = next_game_state.buildObservation(radius=WINDOW_RADIUS)
            game_state = next_game_state
            total_reward += reward
            steps += 1

            if display:
                display.update(game_state.data)
                time.sleep(0.1)

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if display and done:
            display.finish()

    time_end = time.perf_counter()
    total_evaluation_time = time_end - time_start

    total_eval_time_min = (total_evaluation_time // 60)
    eval_time_remaining_sec = (total_evaluation_time % 60)

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)

    new_states = len(newly_discovered_states)
    new_unique_states = len(np.unique(newly_discovered_states))

    total_actions = max(1, sum(episode_lengths))
    qtable_states_percentage = len(previsited_states) / total_actions
    random_actions_percentage = len(newly_discovered_states) / total_actions

    print()
    print("Number of Episodes:", num_episodes)
    print("Decay Rate:", decay_rate)
    print()
    print("Average Reward:", avg_reward)
    print(f"Average Episode Length: {avg_length:.2f} actions")
    print()
    print(f"Total Number of New UNIQUE States Discovered: {new_unique_states:.0f}")
    print(f"Percentage of actions taken from using the Q-Table: {qtable_states_percentage:.2%}")
    print(f"Percentage of Random Actions taken out of all actions: {random_actions_percentage:.2%}")
    print()
    print(f"Time taken to run 10,000 evaluations: {total_evaluation_time:.2f} sec")
    print(f"Time taken to run 10,000 evaluations (minutes): {total_eval_time_min:.0f}:{eval_time_remaining_sec:.0f}")

    n_actions = len(ACTION_LIST)
    mat = np.zeros((len(contexts_order), n_actions), dtype=float)
    for i, ctx in enumerate(contexts_order):
        counts = action_hist_by_ctx[ctx].astype(float)
        s = counts.sum()
        mat[i, :] = counts / s if s > 0 else counts

    action_names = ["NORTH", "SOUTH", "EAST", "WEST", "STOP"]

    plt.figure(figsize=(1.5 + 0.8 * n_actions, 4.5))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = plt.colorbar(im)
    cbar.set_label("Normalized frequency", fontsize=12)

    plt.yticks(range(len(contexts_order)), contexts_order, fontsize=12)
    plt.xticks(range(n_actions), action_names[:n_actions], rotation=45, ha="right", fontsize=11)
    plt.xlabel("Action", fontsize=13)
    plt.ylabel("Context", fontsize=13)
    plt.title(f"Action distribution by context: {num_episodes:,.0f} eps, decay={decay_rate}", fontsize=14, pad=10)
    plt.tight_layout()

    heatmap_path = f"action_heatmap_{num_episodes}_{decay_rate}.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
