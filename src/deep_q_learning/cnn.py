# cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from game import Actions
from util import nearestPoint

MAX_PELLET_VALUE = 30.0  # For normalization


def build_full_observation_tensor(state):
    """
    Build a (C, H, W) tensor directly from a GameState.

    9 Channels:
      0: Walls (constant)
      1: Food values (0 if no food, normalized 0-1 if food present)
      2: Capsules
      3: Ghosts
      4: Scared ghosts
      5: Pacman position (one-hot)
      6: Food-distance map (1.0 near food, 0.0 far; 0 if no food)
      7: Pacman direction x-component (at Pacman cell only, else 0)
      8: Pacman direction y-component (at Pacman cell only, else 0)
    """
    # Basic grids
    walls = state.getWalls()
    food = state.getFood()  # integer grid: 0 = empty, >0 = pellet reward
    capsules = set((int(x), int(y)) for x, y in state.getCapsules())

    width, height = walls.width, walls.height

    # Pacman position
    px_f, py_f = state.getPacmanPosition()
    px, py = int(round(px_f)), int(round(py_f))

    # Pacman direction as a vector (dx, dy) in [-1, 1]
    pac_state = state.getPacmanState()
    direction = pac_state.configuration.direction
    dx, dy = Actions.directionToVector(direction, 1.0)

    # Ghost positions & scared flags
    ghost_map = {}
    for ghost_state in state.getGhostStates():
        gpos = ghost_state.getPosition()
        if gpos is None:
            continue
        gx, gy = nearestPoint(gpos)
        gx, gy = int(gx), int(gy)
        ghost_map.setdefault((gx, gy), []).append(ghost_state.scaredTimer > 0)

    n_channels = 9
    arr = np.zeros((n_channels, height, width), dtype=np.float32)

    food_positions = []

    # First pass: walls, food value, capsules, ghosts, scared ghosts
    for x in range(width):
        for y in range(height):
            # Channel 0: Walls
            if walls[x][y]:
                arr[0, y, x] = 1.0

            # Channel 1: Food value (normalized)
            val = food[x][y]
            if val > 0:
                norm = min(float(val), MAX_PELLET_VALUE) / MAX_PELLET_VALUE
                arr[1, y, x] = norm
                food_positions.append((x, y))

            # Channel 2: Capsules
            if (x, y) in capsules:
                arr[2, y, x] = 1.0

            # Channels 3 & 4: Ghosts / scared ghosts
            for scared in ghost_map.get((x, y), []):
                if scared:
                    arr[4, y, x] = 1.0
                else:
                    arr[3, y, x] = 1.0

    # Channel 6: Food-distance map (1 near food, 0 far)
    if food_positions:
        max_dist = float(width + height)
        for x in range(width):
            for y in range(height):
                d = min(abs(x - fx) + abs(y - fy) for (fx, fy) in food_positions)
                norm = 1.0 - (d / max_dist)
                if norm < 0.0:
                    norm = 0.0
                arr[6, y, x] = norm
    # else: leave channel 6 as zeros (no food left)

    # Channel 5: Pacman position; 7 & 8: Pacman direction at Pacman cell
    if 0 <= px < width and 0 <= py < height:
        arr[5, py, px] = 1.0
        arr[7, py, px] = float(dx)
        arr[8, py, px] = float(dy)

    return torch.from_numpy(arr)  # (9, H, W)


class PacmanCNN(nn.Module):
    def __init__(self, width, height, n_actions, in_channels=9):
        """
        CNN for Pacman Q-learning.
        
        Args:
            width, height: Layout dimensions
            n_actions: Number of possible actions
            in_channels: Number of input channels (default: 9)
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Compute FC input size after two poolings
        h_out = height // 2 // 2  # after two pools
        w_out = width // 2 // 2   # after two pools
        fc_in_dim = 64 * h_out * w_out

        # Fully-connected layers
        self.fc1 = nn.Linear(fc_in_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, n_actions) of Q-values
        """
        x = F.relu(self.conv1(x))  # (B, 32, H, W)
        x = self.pool(x)           # (B, 32, H/2, W/2)

        x = F.relu(self.conv2(x))  # (B, 64, H/2, W/2)
        x = self.pool(x)           # (B, 64, H/4, W/4)

        x = torch.flatten(x, 1)    # (B, 64 * H/4 * W/4)
        x = F.relu(self.fc1(x))    # (B, 128)
        x = self.fc2(x)            # (B, n_actions)
        return x
