# cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import nearestPoint


def build_full_observation_tensor(state):
    """
    Build a (C, H, W) tensor directly from a GameState.

    6 Channels:
      0: Walls
      1: Food (1.0 where food present)
      2: Capsules
      3: Pacman position
      4: Ghosts (non-scared)
      5: Scared ghosts
    """
    walls = state.getWalls()
    food = state.getFood()
    capsules = set((int(x), int(y)) for x, y in state.getCapsules())

    width, height = walls.width, walls.height

    # Pacman position
    px_f, py_f = state.getPacmanPosition()
    px, py = int(round(px_f)), int(round(py_f))

    # Ghosts
    ghost_normal = set()
    ghost_scared = set()
    for ghost_state in state.getGhostStates():
        pos = ghost_state.getPosition()
        if pos is None:
            continue
        gx, gy = nearestPoint(pos)
        gx, gy = int(gx), int(gy)
        if ghost_state.scaredTimer > 0:
            ghost_scared.add((gx, gy))
        else:
            ghost_normal.add((gx, gy))

    n_channels = 6
    arr = np.zeros((n_channels, height, width), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            # Channel 0: Walls
            if walls[x][y]:
                arr[0, y, x] = 1.0

            # Channel 1: Food
            if food[x][y]:
                arr[1, y, x] = 1.0

            # Channel 2: Capsules
            if (x, y) in capsules:
                arr[2, y, x] = 1.0

            # Channel 4: Non-scared ghosts
            if (x, y) in ghost_normal:
                arr[4, y, x] = 1.0

            # Channel 5: Scared ghosts
            if (x, y) in ghost_scared:
                arr[5, y, x] = 1.0

    # Channel 3: Pacman position
    if 0 <= px < width and 0 <= py < height:
        arr[3, py, px] = 1.0

    return torch.from_numpy(arr)  # (6, H, W)


class PacmanCNN(nn.Module):
    def __init__(self, width, height, n_actions, in_channels=6):
        """
        CNN for Pacman Q-learning.

        Args:
            width, height: Layout dimensions
            n_actions: Number of possible actions
            in_channels: Number of input channels (default: 6)
        """
        super().__init__()

        # Smaller convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,        # was 64
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,         # match conv1 out_channels
            out_channels=64,        # was 128
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Compute FC input size after two poolings
        h_out = height // 4  # after two 2x2 pools
        w_out = width // 4
        fc_in_dim = 64 * h_out * w_out

        # Smaller fully-connected layers
        self.fc1 = nn.Linear(fc_in_dim, 128)  # was 256
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
