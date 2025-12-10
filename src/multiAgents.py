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

import sys
from util import manhattanDistance, nearestPoint
from game import Directions, Agent
import random, util
import numpy as np
from util import Stack
import os
import pickle
import time
import pacman as pac
import layout as pac_layout
import ghostAgents
from training_utils import runGamesQuiet
from textDisplay import NullGraphics

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
    CELL__GHOST = 5
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
    Minimax agent implementation where Pacman is considered the maximizing player and the Ghosts are
    sequentially looped through as the minimizing player. As we discussed in class, the MAX player
    goes first and the game continues in an alternating fashion until a terminal state is achieved (Pacman
    wins or loses the game).
    """
    def basicUtility(self, gameState):
        """
        Basic utility function that just returns the game score (the same one displayed in the Pacman GUI).
        """
        return gameState.getScore()
    
    def improvedUtility(self, gameState):
        """
        Improved utility function for minimax: adds rewards for capturing pellets, penalizes danger (ghosts),
        rewards being close to and eating food.
        """

        # terminal states - game is already won/lost
        if gameState.isWin():
            return float("inf")
        if gameState.isLose():
            return -float("inf")

        score = gameState.getScore()
        pacman = gameState.getPacmanPosition()
        ghosts = gameState.getGhostStates()
        ghostPositions = gameState.getGhostPositions()
        food = gameState.getFood().asList()
        capsules = gameState.getCapsules()
        epsilon = 1e-6 # used to prevent division by 0 but still allow for high reward/penalty when distances are 0

        # penalize getting closer to ghosts
        for i, ghost in enumerate(ghosts):
            gPosition = ghostPositions[i]
            gDistance = util.manhattanDistance(pacman, gPosition)
            if ghost.scaredTimer > 0 and gDistance > 0:
                # if ghost is scared, Pacman should be rewarded for getting closer to it
                score += 300.0 / gDistance

            else :
                # varying levels of penalties depending on proximity
                if gDistance <= 1:
                    score -= 500
                elif gDistance <= 2:
                    score -= 50
                else:
                    score -= 1.0 / gDistance
            
        # reward for getting closer to food
        if food:
            closestFoodDist = min(util.manhattanDistance(pacman, f) for f in food)
            score += 10.0 / (closestFoodDist + epsilon)

            # prioritizes moving closer to a large amount of food so that Pacman doesn't get stuck on one side of the map with one pellet
            avgFoodDist = sum(util.manhattanDistance(pacman, f) for f in food) / len(food)
            score += 5.0 / (avgFoodDist + epsilon)
            if pacman in food:
                score += 500

        # reward for getting closer to power capsules
        if capsules:
            closestCap = min(util.manhattanDistance(pacman, c) for c in capsules)
            score += 20.0 / (closestCap + epsilon)
            
            # prioritize if ghosts are close
            for gPos, gState in zip(ghostPositions, ghosts):
                gDistance = util.manhattanDistance(pacman, gPos)
                if gState.scaredTimer == 0 and gDistance <= 3:
                    score += 30.0 / (closestCap + epsilon)

        # penalize not eating food to help prevent stalling / oscillation (somewhat like a time penalty)
        # - came from initial observations of Pacman getting stuck at walls/doing back-and-forth motions
        score += 500 / (len(food) + epsilon)

        return float(score)
    
    def _minimax_decision(self, gameState, depth=2):
        """
        Returns the best action for Pacman (agentIndex = 0) using the minimax algorithm.
        """

        legal = [a for a in gameState.getLegalActions(0) if a != Directions.STOP]
        bestAction = legal[0]
        bestValue = -float("inf")

        if not legal:
            return Directions.STOP

        for action in legal:
            successor = gameState.generateSuccessor(0, action)
            value = self._minimax(successor, depth, agentIndex=1)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
    
    def _minimax(self, state, depth, agentIndex):
        """
        Recursively computes the minimax value for the given state using the utility function.
        """

        # terminal state - just return the utility (objective function) value
        if depth == 0 or state.isWin() or state.isLose():
            return self.improvedUtility(state)

        numAgents = state.getNumAgents()

        legal = state.getLegalActions(agentIndex)
        if not legal:
            return self.improvedUtility(state)

        # maximizing player - Pacman (agentIndex = 0)
        if agentIndex == 0:
            maxValue = -float("inf")

            for action in state.getLegalActions(0):
                succ = state.generateSuccessor(0, action)
                maxValue = max(maxValue, self._minimax(succ, depth, agentIndex=1))
            
            return maxValue

        # minimizing player - Ghost (agentIndex >= 1)
        else:
            minValue = float("inf")
            nextAgent = agentIndex + 1
            nextDepth = depth

            # once we reach the last ghost, we cycle back to Pacman agent and decrease depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth -= 1
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                minValue = min(minValue, self._minimax(succ, nextDepth, nextAgent))

            return minValue
        
    def getAction(self, gameState):
        """
        Returns the minimax action (using a depth of 2 for the game tree) from the current gameState for Pacman
        """
        if hasattr(self, "lastPos") and gameState.getPacmanPosition() == self.lastPos:
            self.stuckCounter = getattr(self, "stuckCounter", 0) + 1
        else:
            self.stuckCounter = 0

        self.lastPos = gameState.getPacmanPosition()
        return self._minimax_decision(gameState, depth=2)

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

def parse_args():
    """
    Helper to read arguments for evaluating the Minimax and AlphaBeta agents with a specified depth.
    """
    # default params
    agent_type = "Minimax"
    depth = 2

    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == "--agent" and i + 1 < len(args):
            agent_type = args[i+1]
        elif args[i] == "--depth" and i + 1 < len(args):
            depth = int(args[i+1])

    return agent_type, depth


"""
The remaining code provides the evaluation mode for the Minimax and AlphaBeta agents.
In order to run the evaluation games, you can pass in two arguments:
- agent_type - 'Minimax' or 'AlphaBeta' (defuault: 'Minimax')
- depth - any integer value (default: 2 - should be kept small for time/space complexity)

Here are sample configurations:
- python multiAgents.py
-- Will run 1000 medium-classic games with 2 ghosts, using the Minimax agent with depth 2
- python multiAgents.py --agent AlphaBeta --depth 3
-- Will run 1000 medium-classic games with 2 ghosts, using the AlphaBeta agent with depth 3
"""
if __name__ == "__main__":
    agent_type, depth = parse_args()

    print(f"Evaluation using - Agent: {agent_type}, Depth: {depth}")

    # allows us to compare our Minimax and AlphaBeta agents by running the same evaluation config/params
    if agent_type == "Minimax":
        agent = MinimaxAgent(depth=depth)
    elif agent_type == "AlphaBeta":
        agent = AlphaBetaAgent(depth=depth)
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'")
    
    num_eval_games = 1
    layout_name = 'mediumClassic'
    num_ghosts = 2
    ghost_type = 'RandomGhost'
    frame_time = 0

    layout = pac_layout.getLayout(layout_name)
    ghost_cls = getattr(ghostAgents, ghost_type)
    ghosts = [ghost_cls(i + 1) for i in range(num_ghosts)]
    display = NullGraphics()

    # since these agents have no training, we just run several games to calculate the performance using the average over all games
    print(f"Running {num_eval_games} evaluation games\n")

    time_start = time.perf_counter()
    games = runGamesQuiet(
        layout=layout,
        pacman=agent,
        ghosts=ghosts,
        display=display,
        numGames=num_eval_games,
        record=False,
        numTraining=0,
        catchExceptions=False,
        timeout=30,
        randomRewards=True,
        max_steps=1000
    )
    time_end = time.perf_counter()

    # output score, win rate, and time stats over all games
    scores = [g.state.getScore() for g in games]
    wins = [g.state.isWin() for g in games]
    try:
        lengths = [len(g.moveHistory) for g in games]
    except:
        lengths = []
    avg_score = np.mean(scores)
    win_rate = sum(wins) / num_eval_games
    avg_length = np.mean(lengths) if lengths else 0
    total_time = time_end - time_start
    time_per_game = total_time / num_eval_games

    print(f"{agent} Agent Evaluation Stats")
    print(f"Games played: {num_eval_games}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average game length: {avg_length:.1f} moves")
    print(f"Total runtime: {total_time:.2f} sec")
    print(f"Time per game: {time_per_game:.2f} sec")