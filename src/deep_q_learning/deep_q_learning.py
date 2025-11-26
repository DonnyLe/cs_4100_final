import os
from deep_q_learning.cnn import PacmanCNN, encode_full_observation_cnn
from game import Agent, Directions
from training_utils import get_output_path  # ADD THIS IMPORT
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
    """

    # Hyperparameters (you can tweak)
    learning_rate = 1e-3
    discount_factor = 0.99
    replay_memory_size = 50_000
    batch_size = 64
    target_sync_steps = 1_000

    # Exploration
    initial_epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.999

    def __init__(self, qfile=None, load_model=False, device=None):
        super().__init__()

        self.qfile = qfile
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ) if device is None else device

        self.n_actions = len(ACTION_LIST)

        # Networks will be lazy-initialized when we see the first state
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # Replay memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Epsilon / training flags
        self.epsilon = self.initial_epsilon
        self.training = True

        # For tracking across steps
        self.last_state_tensor = None
        self.last_action_idx = None
        self.last_score = 0.0

        self.episode_reward = 0.0
        self.episode_rewards = []
        self.global_step = 0

        self.loss_fn = nn.MSELoss()

        self._pending_load = load_model

    def _init_networks_from_state(self, state):
        """Initialize networks based on layout size."""
        if self.policy_net is not None:
            return

        layout = state.data.layout
        width, height = layout.width, layout.height

        self.policy_net = PacmanCNN(width, height, self.n_actions).to(self.device)
        self.target_net = PacmanCNN(width, height, self.n_actions).to(self.device)
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
        self.last_score = state.getScore()
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
        """Select and return an action."""
        if self.policy_net is None:
            self._init_networks_from_state(state)

        obs = state.buildFullObservation()
        state_tensor = encode_full_observation_cnn(obs).float()

        if self.last_state_tensor is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            reward = current_score - self.last_score
            self.last_score = current_score
            self.episode_reward += reward

            done = state.isWin() or state.isLose()
            next_tensor = None if done else state_tensor
            self.store_transition(
                self.last_state_tensor,
                self.last_action_idx,
                next_tensor,
                reward,
                done,
            )
            self.optimize_policy()

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

        self.last_state_tensor = state_tensor
        self.last_action_idx = action_idx

        direction = ACTION_LIST[action_idx]
        return direction

    def final(self, state):
        """Called at the end of each game."""
        if self.last_state_tensor is not None and self.last_action_idx is not None:
            current_score = state.getScore()
            reward = current_score - self.last_score
            self.episode_reward += reward

            self.store_transition(self.last_state_tensor, self.last_action_idx, None, reward, True)
            self.optimize_policy()

        self.episode_rewards.append(self.episode_reward)

        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)