import os
from deep_q_learning.cnn import PacmanCNN, build_full_observation_tensor
from game import Agent, Directions
from training_utils import get_output_path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random



ACTION_LIST = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]


class ReplayMemory:
    """Simple replay buffer for storing transitions."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DeepQLearningAgent(Agent):
    """
    Deep Q-Learning agent for Pacman using a CNN over the full observable grid.
    Optimized to only learn at grid intersections where decisions matter.
    """

    # Hyperparameters
    learning_rate = 0.001
    discount_factor = 0.99
    replay_memory_size = 50000
    batch_size = 64
    target_sync_steps = 1000
    
    # Reward shaping
    backtrack_penalty = 4.0          # extra penalty for immediate reverse
    food_shaping_scale = 0.5         # +/- reward for moving closer/farther to food


    # Exploration
    initial_epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.999

    def __init__(self, qfile=None, load_model=False, device=None, debug_rewards = False):
        super().__init__()

        self.qfile = qfile
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ) if device is None else device

        self.n_actions = len(ACTION_LIST)

        # Networks (lazy-initialized)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # Replay memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Training state
        self.epsilon = self.initial_epsilon
        self.training = True

        # Tracking
        self.last_state_tensor = None
        self.last_action_idx = None
        self.last_score = 0.0

        self.episode_reward = 0.0
        self.episode_rewards = []
        self.global_step = 0

        self.loss_fn = nn.MSELoss()
        self._pending_load = load_model

            # --- NEW: reward debug toggle + counter ---
        self.debug_rewards = False
        self.debug_step_count = 0

    def _closest_food_distance(self, state):
        """Return Manhattan distance from Pacman to nearest food, or None if no food."""
        food = state.getFood()
        px, py = state.getPacmanPosition()
        px, py = int(round(px)), int(round(py))
        width, height = food.width, food.height

        best = None
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    d = abs(x - px) + abs(y - py)
                    if best is None or d < best:
                        best = d
        return best

    def _is_reverse(self, curr_idx, prev_idx):
        """Return True if action curr_idx is the exact reverse of prev_idx."""
        if curr_idx is None or prev_idx is None:
            return False
        a = ACTION_LIST[curr_idx]
        b = ACTION_LIST[prev_idx]
        return (
            (a == Directions.NORTH and b == Directions.SOUTH) or
            (a == Directions.SOUTH and b == Directions.NORTH) or
            (a == Directions.EAST  and b == Directions.WEST)  or
            (a == Directions.WEST  and b == Directions.EAST)
        )


    def _init_networks_from_state(self, state):
        """Initialize networks based on layout size."""
        if self.policy_net is not None:
            return

        layout = state.data.layout
        width, height = layout.width, layout.height

        in_channels = 9  # 9-channel input now
        
        self.policy_net = PacmanCNN(width, height, self.n_actions, in_channels=in_channels).to(self.device)
        self.target_net = PacmanCNN(width, height, self.n_actions, in_channels=in_channels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        if self._pending_load and self.qfile:
            self._load_model()
            self._pending_load = False


    def _save_model(self):
        if not self.qfile or self.policy_net is None:
            return
        path = get_output_path(self.qfile, agent_type='dqn')
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def _load_model(self):
        path = get_output_path(self.qfile, agent_type='dqn')
        if not os.path.exists(path):
            print(f"[DQN] No model file at {path}, starting from scratch.")
            return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.min_epsilon)
            print(f"[DQN] Loaded model from {self.qfile}")
        except Exception as e:
            print(f"[DQN] Failed to load model: {e}")

    def setTraining(self, training: bool):
        self.training = training
        if not training:
            self.epsilon = 0.0

    def registerInitialState(self, state):
        """Called at the start of each game."""
        self._init_networks_from_state(state)

        self.last_state_tensor = None
        self.last_action_idx = None
        self.prev_action_idx = None
        self.last_score = state.getScore()
        self.last_food_dist = None
        self.episode_reward = 0.0

    def store_transition(self, s, a, s_next, r, done):
        """Store one transition in replay memory."""
        self.memory.append((s, a, s_next, r, done))

    def optimize_policy(self):
        """Sample a mini-batch and perform one gradient step."""
        if not self.training:
            return
        if self.policy_net is None or self.target_net is None:
            return
        if len(self.memory) < self.batch_size:
            return
        if not hasattr(self, '_gpu_check_done'):
            print(f"[DQN] optimize_policy running on device: {self.device}")
            print(f"[DQN] policy_net on: {next(self.policy_net.parameters()).device}")
            self._gpu_check_done = True

        mini_batch = self.memory.sample(self.batch_size)

        current_q_list = []
        target_q_list = []

        for state_tensor, action_idx, next_state_tensor, reward, done in mini_batch:
            state_batch = state_tensor.unsqueeze(0).to(self.device)

            if done or next_state_tensor is None:
                target_val = torch.tensor(
                    [reward], dtype=torch.float32, device=self.device
                )
            else:
                with torch.no_grad():
                    next_batch = next_state_tensor.unsqueeze(0).to(self.device)
                    q_next = self.target_net(next_batch).squeeze(0)
                    max_next_q = q_next.max()
                    target_val = torch.tensor(
                        [reward + self.discount_factor * max_next_q.item()],
                        dtype=torch.float32,
                        device=self.device,
                    )

            current_q = self.policy_net(state_batch).squeeze(0)
            target_q = current_q.detach().clone()
            target_q[action_idx] = target_val

            current_q_list.append(current_q)
            target_q_list.append(target_q)

        current_q_tensor = torch.stack(current_q_list)
        target_q_tensor = torch.stack(target_q_list)

        loss = self.loss_fn(current_q_tensor, target_q_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.target_sync_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def getAction(self, state):
        """
        Select and return an action.
        
        We:
          - build a (9, H, W) tensor directly from GameState
          - apply reward shaping based on score diff, distance to food, and backtracking
          - store transitions in replay
          - optimize the policy_net
          - choose epsilon-greedy action
        """
        if self.policy_net is None:
            self._init_networks_from_state(state)

        # Build observation tensor directly from GameState
        state_tensor = build_full_observation_tensor(state).float()

        # Compute current distance to nearest food (for shaping)
        curr_food_dist = self._closest_food_distance(state)

        # --- Learning from the previous transition ---
        if self.training and self.last_state_tensor is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            game_reward = current_score - self.last_score

            # Track components for debug
            base_reward = game_reward
            time_penalty_applied = False
            positive_scale_applied = False
            food_shaping_delta = 0.0
            backtrack_delta = 0.0

            # Base shaping from score difference
            shaped_reward = game_reward

            if game_reward > 0:
                shaped_reward = game_reward * 1.5  

            if self.last_food_dist is not None and curr_food_dist is not None:
                if curr_food_dist < self.last_food_dist:
                    shaped_reward += self.food_shaping_scale
                elif curr_food_dist > self.last_food_dist:
                    shaped_reward -= self.food_shaping_scale

            # Backtrack penalty
            if self._is_reverse(self.last_action_idx, self.prev_action_idx):
                shaped_reward -= self.backtrack_penalty

            # --- DEBUG PRINTS ---
            if self.debug_rewards and self.debug_step_count < 2000:
                print(
                    "[DQN][reward] "
                    f"Δscore={base_reward:.1f} | "
                    f"time_penalty={time_penalty_applied} | "
                    f"scaled_positive={positive_scale_applied} | "
                    f"food_delta={food_shaping_delta:+.2f} | "
                    f"backtrack_delta={backtrack_delta:+.2f} | "
                    f"final_shaped={shaped_reward:+.2f}"
                )
                self.debug_step_count += 1

            self.last_score = current_score
            self.episode_reward += shaped_reward

            done = state.isWin() or state.isLose()
            next_tensor = None if done else state_tensor

            self.store_transition(
                self.last_state_tensor,
                self.last_action_idx,
                next_tensor,
                shaped_reward,
                done,
            )
            self.optimize_policy()

        # --- Action selection (epsilon-greedy) ---
        legal_indices = state.getLegalActionsIndices(ACTION_LIST)
        if not legal_indices:
            legal_indices = [0]

        if self.training and random.random() < self.epsilon:
            action_idx = random.choice(legal_indices)
        else:
            with torch.no_grad():
                q_vals = self.policy_net(
                    state_tensor.unsqueeze(0).to(self.device)
                ).squeeze(0)

            q_vals = q_vals.cpu().tolist()
            best_idx = max(legal_indices, key=lambda i: q_vals[i])
            action_idx = int(best_idx)

        # --- Update tracking for next step ---
        self.prev_action_idx = self.last_action_idx
        self.last_state_tensor = state_tensor
        self.last_action_idx = action_idx
        self.last_food_dist = curr_food_dist

        direction = ACTION_LIST[action_idx]
        return direction


    def final(self, state):
        """Called at the end of each game."""
        # Training: final transition and optimization
        if self.training and self.last_state_tensor is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            reward = current_score - self.last_score
            
            # Reward shaping
            if reward == -1:
                reward = -2
            
            self.episode_reward += reward

            self.store_transition(self.last_state_tensor, self.last_action_idx, None, reward, True)
            self.optimize_policy()

        # Track episode reward
        self.episode_rewards.append(self.episode_reward)

        # Training: decay epsilon
        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)