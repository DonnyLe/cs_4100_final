# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, nearestPoint
from game import Directions
import random, util
import numpy as np
import math
from util import Stack
import os
import pickle

from game import Agent

class Node:

    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


class LocalQAgent(Agent):
    """
    Tabular Q-learning agent that observes a square window centered on Pacman.
    The window is encoded into a compact hash which is used as the state id.
    
    Agent arguments (set via -a key=value):
        radius        (int)   : half-width of observation window (default 2 -> 5x5)
        gamma         (float) : discount factor
        epsilon       (float) : initial exploration rate
        decay_rate  (float) : multiplicative decay applied after each game
        minEpsilon    (float) : lower bound on epsilon during training
        numTraining   (int)   : number of games to train (matches -x)
        qfile         (str)   : path to persist the learned Q-table
    """

    CELL_WALL = 0
    CELL_EMPTY = 1
    CELL_FOOD = 2
    CELL_CAPSULE = 3
    CELL_GHOST = 4
    CELL_SCARED_GHOST = 5
    CELL_PACMAN = 6

    def __init__(self,
                 radius=2,
                 gamma=0.9,
                 epsilon=1.0,
                 decay_rate=0.9999,
                 minEpsilon=0.05,
                 numTraining=0,
                 qfile='local_q_table.pkl'):
        Agent.__init__(self)
        self.radius = int(radius)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initialEpsilon = float(epsilon)
        self.decay_rate = float(decay_rate)
        self.minEpsilon = float(minEpsilon)
        self.qfile = qfile
        self.actionList = [Directions.NORTH, Directions.SOUTH,
                           Directions.EAST, Directions.WEST, Directions.STOP]
        self.actionIndex = {action: idx for idx, action in enumerate(self.actionList)}
        self.q_table = {}
        self.update_counts = {}
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0.0
        self.loadedFromDisk = False
        self.trainingEpisodes = int(numTraining)
        self.remainingTraining = max(0, int(numTraining))
        self.training = self.remainingTraining > 0

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------

    def registerInitialState(self, state):
        if not self.loadedFromDisk:
            self._loadQTable()
            self.loadedFromDisk = True
        self.lastState = None
        self.lastAction = None
        self.lastScore = state.getScore()
        self.training = self.remainingTraining > 0

    def getAction(self, state):
        currentKey = self._encodeState(state)
        self._ensureState(currentKey)

        if self.lastState is not None:
            reward = state.getScore() - self.lastScore
            terminal = state.isWin() or state.isLose()
            self._updateQ(self.lastState, self.lastAction, reward, currentKey, terminal)

        legal = state.getLegalPacmanActions()
        action = self._selectAction(currentKey, legal)

        self.lastState = currentKey
        self.lastAction = action
        self.lastScore = state.getScore()

        return action

    def final(self, state):
        if self.lastState is not None:
            reward = state.getScore() - self.lastScore
            self._updateQ(self.lastState, self.lastAction, reward, None, True)
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0.0
        if self.training:
            self.remainingTraining -= 1
            if self.remainingTraining <= 0:
                self.training = False
                self.epsilon = 0.0
            else:
                self.epsilon = max(self.minEpsilon, self.epsilon * self.decay_rate)
        if self.trainingEpisodes > 0:
            self._saveQTable()

    # ------------------------------------------------------------------
    # Q-learning internals
    # ------------------------------------------------------------------

    def _selectAction(self, stateKey, legalActions):
        legal = [a for a in legalActions if a in self.actionIndex]
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)
        if not legal:
            return Directions.STOP
        explore = self.training and random.random() < self.epsilon
        if explore:
            return random.choice(legal)
        q_values = self.q_table[stateKey]
        bestAction = None
        bestValue = -float('inf')
        for action in legal:
            idx = self.actionIndex[action]
            value = q_values[idx]
            if value > bestValue or bestAction is None:
                bestValue = value
                bestAction = action
        return bestAction

    def _updateQ(self, stateKey, action, reward, nextStateKey, terminal):
        if not self.training or stateKey is None or action is None:
            return
        idx = self.actionIndex[action]
        self._ensureState(stateKey)
        currentQ = self.q_table[stateKey][idx]
        eta = 1.0 / (1.0 + self.update_counts[stateKey][idx])
        self.update_counts[stateKey][idx] += 1
        nextMax = 0.0
        if not terminal and nextStateKey is not None:
            self._ensureState(nextStateKey)
            nextMax = np.max(self.q_table[nextStateKey])
        target = reward + self.gamma * nextMax
        self.q_table[stateKey][idx] = (1 - eta) * currentQ + eta * target

    def _ensureState(self, stateKey):
        if stateKey not in self.q_table:
            self.q_table[stateKey] = np.zeros(len(self.actionList), dtype=np.float32)
            self.update_counts[stateKey] = np.zeros(len(self.actionList), dtype=np.float32)

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encodeState(self, state):
        pos = state.getPacmanPosition()
        px, py = int(round(pos[0])), int(round(pos[1]))
        walls = state.getWalls()
        food = state.getFood()
        capsules = set((int(x), int(y)) for x, y in state.getCapsules())
        ghostMap = {}
        for ghostState in state.getGhostStates():
            ghostPos = ghostState.getPosition()
            if ghostPos is None:
                continue
            gx, gy = nearestPoint(ghostPos)
            gx, gy = int(gx), int(gy)
            code = self.CELL_SCARED_GHOST if ghostState.scaredTimer > 0 else self.CELL_GHOST
            ghostMap[(gx, gy)] = code

        codes = []
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                x, y = px + dx, py + dy
                code = self.CELL_WALL
                if 0 <= x < walls.width and 0 <= y < walls.height and not walls[x][y]:
                    code = self.CELL_EMPTY
                    if (x, y) in ghostMap:
                        code = ghostMap[(x, y)]
                    elif (x, y) in capsules:
                        code = self.CELL_CAPSULE
                    elif food[x][y]:
                        code = self.CELL_FOOD
                if dx == 0 and dy == 0:
                    code = self.CELL_PACMAN
                codes.append(code)
        return tuple(codes)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _loadQTable(self):
        if not self.qfile or not os.path.exists(self.qfile):
            return
        try:
            with open(self.qfile, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'q_table' in data and 'update_counts' in data:
                    self.q_table = data['q_table']
                    self.update_counts = data['update_counts']
                else:
                    self.q_table = data
                    self.update_counts = {k: np.zeros(len(self.actionList), dtype=np.float32) for k in self.q_table}
            print(f"[LocalQAgent] Loaded Q-table from {self.qfile} ({len(self.q_table)} states)")
        except Exception as exc:
            print(f"[LocalQAgent] Failed to load Q-table: {exc}")

    def _saveQTable(self):
        if not self.qfile:
            return
        data = {'q_table': self.q_table, 'update_counts': self.update_counts}
        try:
            with open(self.qfile, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            print(f"[LocalQAgent] Failed to save Q-table: {exc}")

    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    pass

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
