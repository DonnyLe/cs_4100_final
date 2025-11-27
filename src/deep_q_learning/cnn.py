# cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MAX_PELLET_VALUE = 30.0  # For normalization


def encode_full_observation_cnn(obs):
    """
    Encode a fully observable Pacman state into a (C, H, W) tensor.
    
    6 Channels:
      0: Walls (constant)
      1: Food values (0 if no food, normalized 0-1 if food present)
         This replaces both "food presence" and "food value" channels!
      2: Capsules
      3: Ghosts
      4: Scared ghosts
      5: Pacman position
    
    obs comes from GameState.buildFullObservation(), with:
      - 'grid': {(x, y) -> cell dict}
      - 'width', 'height'
      - 'pacman_position': (px, py)
    """
    grid_dict = obs['grid']
    width = obs['width']
    height = obs['height']
    px, py = obs['pacman_position']

    n_channels = 6
    arr = np.zeros((n_channels, height, width), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            cell = grid_dict[(x, y)]

            # Channel 0: Walls
            if cell['wall']:
                arr[0, y, x] = 1.0

           
            val = cell.get('food_value', 0)
            if val > 0:
                # Normalize to 0-1 range
                norm = min(float(val), MAX_PELLET_VALUE) / MAX_PELLET_VALUE
                arr[1, y, x] = norm

            # Channel 2: Capsules
            if cell['capsule']:
                arr[2, y, x] = 1.0

            # Channel 3: Regular ghosts
            if cell['ghosts']:
                arr[3, y, x] = 1.0

            # Channel 4: Scared ghosts
            if cell['scared_ghosts']:
                arr[4, y, x] = 1.0

    # Channel 5: Pacman position
    if 0 <= px < width and 0 <= py < height:
        arr[5, py, px] = 1.0

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