import os
from deep_q_learning.cnn import PacmanCNN, build_full_observation_tensor
from game import Agent, Directions, Actions
from training_utils import get_output_path
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


ACTION_LIST = [
    Directions.NORTH,
    Directions.SOUTH,
    Directions.EAST,
    Directions.WEST,
]


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
    """

    # Hyperparameters
    learning_rate = 0.001
    discount_factor = 0.99
    replay_memory_size = 50000
    batch_size = 64
    target_sync_steps = 1000
    
    # Reward shaping (slightly stronger ghost avoidance)
    backtrack_penalty = 1.0       # small penalty for reversing
    food_shaping_scale = 2.0      # small incentive for moving toward food
    ghost_safe_scale = 3.0        # reward for moving farther from a dangerous ghost
    ghost_danger_scale = 3.0      # penalty for moving closer to a dangerous ghost
    ghost_safety_radius = 5
    ghost_danger_radius = 3

    # Exploration
    initial_epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.9997

    def __init__(
        self,
        qfile=None,
        load_model=False,
        device=None,
        debug_rewards=False,   # kept for compatibility, but no longer used
        load_path=None,
        use_reward_shaping=True,
    ):
        super().__init__()
        print("Load model: ", load_model)
        self.qfile = qfile          # where to SAVE
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
        self.prev_action_idx = None
        self.last_score = 0.0

        self.episode_reward = 0.0
        self.episode_rewards = []
        self.global_step = 0

        self.loss_fn = nn.SmoothL1Loss()

        # Model loading info
        self._pending_load = load_model
        self.load_path = load_path or qfile   # where to LOAD from

        self.last_food_dist = None
        self.last_ghost_dist = None

        # reward shaping toggle
        self.use_reward_shaping = use_reward_shaping

        # previous tile (no longer used for forced movement, but harmless to keep)
        self.prev_tile = None

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

        in_channels = 6  # 6-channel input

        self.policy_net = PacmanCNN(width, height, self.n_actions, in_channels=in_channels).to(self.device)
        self.target_net = PacmanCNN(width, height, self.n_actions, in_channels=in_channels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        if self._pending_load:
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
        # Prefer explicit load_path; fall back to qfile
        filename = self.load_path
        if filename is None:
            print("[DQN] No load path specified, starting from scratch.")
            return

        path = get_output_path(filename, agent_type='dqn')
        print(path)
        if not os.path.exists(path):
            print(f"[DQN] No model file at {path}, starting from scratch.")
            return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.min_epsilon)
            print(f"[DQN] Loaded model from {filename}")
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
        self.last_ghost_dist = None
        self.episode_reward = 0.0

        self.prev_tile = None

    def store_transition(self, s, a, s_next, r, done, legal_next_indices):
        """
        Store one transition in replay memory.

        legal_next_indices:
          - list of int indices for legal actions in next state
          - or None / [] if terminal or not applicable
        """
        self.memory.append((s, a, s_next, r, done, legal_next_indices))

    def optimize_policy(self):
        """Sample a mini-batch and perform one gradient step."""
        if not self.training:
            return
        if self.policy_net is None or self.target_net is None:
            return
        if len(self.memory) < self.batch_size:
            return

        mini_batch = self.memory.sample(self.batch_size)

        current_q_list = []
        target_q_list = []

        for (
            state_tensor,
            action_idx,
            next_state_tensor,
            reward,
            done,
            legal_next_indices,
        ) in mini_batch:
            state_batch = state_tensor.unsqueeze(0).to(self.device)

            # ----- Build target value -----
            if done or next_state_tensor is None or not legal_next_indices:
                # Terminal state or no valid next actions: no bootstrap term
                target_val = torch.tensor(
                    [reward], dtype=torch.float32, device=self.device
                )
            else:
                with torch.no_grad():
                    next_batch = next_state_tensor.unsqueeze(0).to(self.device)
                    q_next = self.target_net(next_batch).squeeze(0)  # [n_actions]

                    # Mask to only legal next actions
                    q_next_valid = q_next[legal_next_indices]
                    max_next_q = q_next_valid.max()

                    target_val = torch.tensor(
                        [reward + self.discount_factor * max_next_q.item()],
                        dtype=torch.float32,
                        device=self.device,
                    )

            # ----- Current Q and full target vector -----
            current_q = self.policy_net(state_batch).squeeze(0)  # [n_actions]
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

    def _closest_ghost_distance(self, state):
        # non-scared ghosts only
        ghost_positions = []
        for ghost_state in state.getGhostStates():
            if ghost_state.scaredTimer > 0:
                continue
            pos = ghost_state.getPosition()
            if pos is None:
                continue
            gx, gy = int(round(pos[0])), int(round(pos[1]))
            ghost_positions.append((gx, gy))

        if not ghost_positions:
            return None

        px, py = state.getPacmanPosition()
        px, py = int(round(px)), int(round(py))

        return min(abs(gx - px) + abs(gy - py) for gx, gy in ghost_positions)

    def _apply_reward_shaping(
        self,
        base_reward: float,
        last_food_dist,
        curr_food_dist,
        last_ghost_dist,
        curr_ghost_dist,
        last_action_idx,
        prev_action_idx,
    ) -> float:
        """
        Apply reward shaping on top of the base environment reward.
        Returns the final shaped reward.
        """
        if not self.use_reward_shaping:
            return base_reward

        shaped_reward = base_reward

        # --- Distance-to-food shaping ---
        if last_food_dist is not None and curr_food_dist is not None:
            if curr_food_dist < last_food_dist:
                shaped_reward += self.food_shaping_scale   # +2
            elif curr_food_dist > last_food_dist:
                shaped_reward -= self.food_shaping_scale / 2   # -1

        # --- Backtrack penalty (direction-based) ---
        apply_backtrack = True
        if curr_ghost_dist is not None and curr_ghost_dist <= self.ghost_danger_radius:
            # Don't punish reversal when a dangerous ghost is close – might be escaping
            apply_backtrack = False

        if apply_backtrack and self._is_reverse(last_action_idx, prev_action_idx):
            shaped_reward -= self.backtrack_penalty

        # --- Ghost-distance shaping (dangerous ghosts only) ---
        if last_ghost_dist is not None and curr_ghost_dist is not None:
            if (last_ghost_dist <= self.ghost_safety_radius or
                curr_ghost_dist   <= self.ghost_safety_radius):

                if curr_ghost_dist > last_ghost_dist:
                    shaped_reward += self.ghost_safe_scale
                elif curr_ghost_dist < last_ghost_dist:
                    shaped_reward -= self.ghost_danger_scale

        return shaped_reward

    def getAction(self, state):
        """
        Select and return an action.
        """

        if self.policy_net is None:
            self._init_networks_from_state(state)

        # Build observation tensor directly from GameState
        state_tensor = build_full_observation_tensor(state).float()

        # Current distances used for shaping
        curr_food_dist = self._closest_food_distance(state)
        curr_ghost_dist = self._closest_ghost_distance(state)

        # --- Learning from the previous transition ---
        if self.training and self.last_state_tensor is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            game_reward = current_score - self.last_score

            base_reward = game_reward

            shaped_reward = self._apply_reward_shaping(
                base_reward=base_reward,
                last_food_dist=self.last_food_dist,
                curr_food_dist=curr_food_dist,
                last_ghost_dist=self.last_ghost_dist,
                curr_ghost_dist=curr_ghost_dist,
                last_action_idx=self.last_action_idx,
                prev_action_idx=self.prev_action_idx,
            )

            self.last_score = current_score
            self.episode_reward += shaped_reward

            done = state.isWin() or state.isLose()
            next_tensor = None if done else state_tensor

            if done:
                legal_next_indices = None
            else:
                legal_next_indices = state.getLegalActionsIndices(ACTION_LIST)
                if not legal_next_indices:
                    legal_next_indices = []

            self.store_transition(
                self.last_state_tensor,
                self.last_action_idx,
                next_tensor,
                shaped_reward,
                done,
                legal_next_indices,
            )
            self.optimize_policy()

        # --- Action selection ---

        legal_indices = state.getLegalActionsIndices(ACTION_LIST)
        if not legal_indices:
            legal_indices = [0]

        if self.training and random.random() < self.epsilon:
            # Explore among legal moves
            action_idx = random.choice(legal_indices)
        else:
            # Exploit: choose best Q among legal moves
            with torch.no_grad():
                q_vec = self.policy_net(
                    state_tensor.unsqueeze(0).to(self.device)
                ).squeeze(0)

            q_vals = q_vec.cpu().tolist()
            best_idx = max(legal_indices, key=lambda i: q_vals[i])
            action_idx = int(best_idx)

        # --- Update tracking for next step ---
        curr_pos = state.getPacmanPosition()
        cx, cy = int(round(curr_pos[0])), int(round(curr_pos[1]))

        self.prev_action_idx = self.last_action_idx
        self.last_state_tensor = state_tensor
        self.last_action_idx = action_idx
        self.last_food_dist = curr_food_dist
        self.last_ghost_dist = curr_ghost_dist
        self.prev_tile = (cx, cy)

        direction = ACTION_LIST[action_idx]
        return direction

    def final(self, state):
        """Called at the end of each game."""
        if self.training and self.last_state_tensor is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            reward = current_score - self.last_score
            
            self.episode_reward += reward
            # terminal transition: no next state / no legal next indices
            self.store_transition(
                self.last_state_tensor,
                self.last_action_idx,
                None,
                reward,
                True,
                None,
            )
            self.optimize_policy()

        self.episode_rewards.append(self.episode_reward)

        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
