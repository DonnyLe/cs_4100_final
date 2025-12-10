# cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import nearestPoint


def build_full_observation_tensor(state):
    """
    Build a tensor from the game state for the CNNs.

    6 Channels:
    0: walls
    1: food/pellets (
    2: capsules
    3: Pacman position
    4: dangerous ghost (non-scared)
    5: scared ghosts
    """
    walls = state.getWalls()
    food = state.getFood()
    capsules = set((int(x), int(y)) for x, y in state.getCapsules())

    width, height = walls.width, walls.height

    # Pacman position
    px_f, py_f = state.getPacmanPosition()
    px, py = int(round(px_f)), int(round(py_f))

    # ghosts
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
            # channel 0: walls
            if walls[x][y]:
                arr[0, y, x] = 1.0

            # channel 1: pellets/food
            if food[x][y]:
                arr[1, y, x] = 1.0

            # channel 2: capsules
            if (x, y) in capsules:
                arr[2, y, x] = 1.0

            # channel 4: dangerous ghosts
            if (x, y) in ghost_normal:
                arr[4, y, x] = 1.0

            # channel 5: scared ghosts
            if (x, y) in ghost_scared:
                arr[5, y, x] = 1.0

    # channel 3: Pacman position
    if 0 <= px < width and 0 <= py < height:
        arr[3, py, px] = 1.0

    return torch.from_numpy(arr)


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
        # after two 2x2 pools
        h_out = height // 4  
        w_out = width // 4
        fc_in_dim = 64 * h_out * w_out

        # dense layers
        self.fc1 = nn.Linear(fc_in_dim, 128)  
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        x: (Batch_size, C, H, W)
        returns: (Batch_size, n_actions)
        """
        x = F.relu(self.conv1(x))  
        x = self.pool(x)           

        x = F.relu(self.conv2(x)) 
        x = self.pool(x)         

        x = torch.flatten(x, 1)   
        x = F.relu(self.fc1(x))   
        x = self.fc2(x)          
        return x
