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

    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def goalTest(self, gs, pos, flag):
        # Looking for food
        if(flag == 0):
            if(gs.hasFood(pos[0], pos[1])):
                return True
            return False
        # Looking for ghost
        if(flag == 1):
            gpos = gs.getGhostPositions()
            for gp in gpos:
                if(gp == pos):
                    return True
            return False
        

    def DLS(self, currentNode, stack, explored, layer, limit, found, flag):
        explored.append(currentNode)
        if(self.goalTest(currentNode.parent.state, currentNode.state.getPacmanPosition(), flag)):
            stack.push(currentNode)
            return stack, explored, True
        if(layer == limit):
            return stack, explored, False
        stack.push(currentNode)
        actions = currentNode.state.getLegalActions()
        for a in actions:
            newState = currentNode.state.generatePacmanSuccessor(a)
            newNode = Node(newState, currentNode, a, 1)
            if newNode in explored:
                continue
            stack, explored, found = self.DLS(newNode, stack, explored, layer+1, limit, found, flag)
            if(found):
                return stack, explored, True
        stack.pop()
        return stack, explored, False
    
    def IDS(self, sgs, limit, flag):
        found = False
        current_limit = 0
        while(not found and current_limit <= limit):
            current_limit = current_limit + 1
            startNode = Node(sgs, None, None, 0)
            startNode.parent = startNode
            stack = Stack()
            explored = []
            stack, explored, found = self.DLS(startNode, stack, explored, 1, current_limit, False, flag)

        actions = []
        while(not stack.isEmpty()):
            node = stack.pop()
            actions.append(node.action)

        if not actions:
            return actions, found
        
        actions.reverse()
        actions.pop(0)  # Removes start node from actions

        return actions, found


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X
        in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        weights = np.loadtxt("weights.csv", delimiter=",")
        new_weights = np.array(weights)

        # Choose one of the best actions
        scores = []
        for action in legalMoves:
            [s,new_weights] = self.evaluationFunction(gameState, action, weights, new_weights)
            scores.append(s)
        
        bestScore = max(scores)
        allIndices = [index for index in range(len(scores))]
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #chosenIndex = random.choice(allIndices) # Pick randomly among the best


        "Add more of your code here if you want to"

        np.savetxt("weights.csv", new_weights, delimiter=",", fmt='%4.8f')

        return legalMoves[chosenIndex]

    def CalcGhostPos(self, cgs, actions):
        for a in actions:
            cgs = cgs.generatePacmanSuccessor(a)
        return cgs.getPacmanPosition()

    # Find all active and scared ghosts and then turn them into binary features
    def findAllGhosts(self, cgs):
        f1 = 0  # Active ghost one step away (Binary)
        f2 = 0  # Active ghost two steps away (Binary)
        f3 = 0  # Scared ghost one step away (Binary)
        f4 = 0  # Scared ghost two steps away (Binary)
        actions, found = self.IDS(cgs, 3, 1)
        if not found:
            return f1, f2, f3, f4
        ghosts = cgs.getGhostStates()
        ghostPos = self.CalcGhostPos(cgs, actions)
        foundGhostPosition = False
        for g in ghosts:
            if(ghostPos == g.configuration.pos):
                ghost = g
                foundGhostPosition = True
                break
        
        if not foundGhostPosition:
            return f1, f2, f3, f4

        if(ghost.scaredTimer > 0):  # If ghost is scared
            if(len(actions) <= 1):
                f3 = 1
            if(len(actions) == 2):
                f4 = 1
        if(ghost.scaredTimer == 0): # If ghost is active
            if(len(actions) <= 1):
                f1 = 1
            if(len(actions) == 2):
                f2 = 1

        return f1, f2, f3, f4


    # Active ghost one step away (Binary)
    def getFeatureOne(self, cgs):
        actions, found = self.IDS(cgs, 2, 1)
        if(found):
            if(len(actions) <= 1):
                return 1
        else:
            return 0

    # Active ghost two steps away (Binary)
    def getFeatureTwo(self, cgs):
        actions, found = self.IDS(cgs, 3, 1)
        if(found):
            if(len(actions) == 2):
                return 1
        else:
            return 0

    # Scared ghost one step away (Binary)
    def getFeatureThree(self, cgs):
        ghosts = cgs.getGhostStates()
        if not ghosts:
            return 0
        g = ghosts[0]
        if(g.scaredTimer > 0):
            actions, found = self.IDS(cgs, 2, 1)
            if(found):
                if(len(actions) <= 1):
                    return 1
            else:
                return 0
        return 0

    # Scared ghost two steps away (Binary)
    def getFeatureFour(self, cgs):
        ghosts = cgs.getGhostStates()
        if not ghosts:
            return 0
        g = ghosts[0]
        if(g.scaredTimer > 0):
            actions, found = self.IDS(cgs, 3, 1)
            if(found):
                if(len(actions) == 2):
                    return 1
            else:
                return 0
        return 0
        
    # Eating Food (Binary)
    def getFeatureFive(self, cgs, sgs):
        if(self.goalTest(cgs, sgs.getPacmanPosition(), 0)):
            return 1
        return 0

    # Distance to closest food
    def getFeatureSix(self, cgs):
        #actions, found = self.IDS(cgs, 3, 0)
        #if(found):
        #    return 1/len(actions)
        
        food = cgs.getFood()
        pacPos = cgs.getPacmanPosition()
        dist = []
        x_size = food.width
        y_size = food.height
        for x in range(0, x_size):
            for y in range(0, y_size):
                if(food[x][y] == True):
                    dist.append(manhattanDistance(pacPos, (x,y)))
        if not dist:
            return 0
        closestFood = min(dist)
        return 1/closestFood

    # Get instantaneous reward 
    def getReward(self, cgs, sgs):
        pacPos = sgs.getPacmanPosition()
        gpos = cgs.getGhostPositions()
        ghosts = cgs.getGhostStates()
        g = ghosts[0]
        for pos in gpos:
            if(pacPos == pos and g.scaredTimer == 0):
                return -250
            if(pacPos == pos and g.scaredTimer > 1):
                return 100
        if(cgs.hasFood(pacPos[0], pacPos[1])):
            if(cgs.getNumFood() <= 1):
                return 250
            return 1
        return -1
    

    def evaluationFunction(self, currentGameState, action, weights, new_weights):
       
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Get features for current state
        f1, f2, f3, f4 = self.findAllGhosts(successorGameState)
        f5 = self.getFeatureFive(currentGameState, successorGameState)
        f6 = self.getFeatureSix(successorGameState)
        features = np.array([f1, f2, f3, f4, f5, f6])
        Q_s_a = np.dot(weights, np.transpose(features))

        # Generate Q(s', a')
        Qs = []
        legalMoves = successorGameState.getLegalActions()
        for a in legalMoves:
            ngs = successorGameState.generatePacmanSuccessor(a)
            f1_next, f2_next, f3_next, f4_next = self.findAllGhosts(ngs) 
            f5_next = self.getFeatureFive(successorGameState, ngs)
            f6_next = self.getFeatureSix(ngs)
            features_next = np.array([f1_next, f2_next, f3_next, f4_next, f5_next, f6_next])
            Q_next = np.dot(weights, np.transpose(features_next))
            Qs.append(Q_next)
        if not Qs:
            Q_next = 0
        else:
            Q_next = max(Qs)

        r = self.getReward(currentGameState, successorGameState)

        alpha = 0.00001
        gamma = 0.9
        diff = (r + gamma*Q_next) - Q_s_a
        
        for w in range(0,6):
            new_weights[w] = new_weights[w] + alpha*diff*features[w]

        return [Q_s_a, new_weights]    # Q state calculated with old weights

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
    Minimax agent implementation where Pacman is considered the maximizing player and the Ghosts are
    sequentially looped through as the minimizing player(s). As we discussed in class, the Max player
    goes first and the game continues in an alternating fashion until a terminal state is achieved (Pacman
    wins or loses the game).
    """
    def utility(self, gameState):
        """
        Initial utility function for minimax: adds rewards for eating pellets, penalizes danger (ghosts),
        reward being close to food.
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

        # penalize getting closer to ghosts
        for i, ghost in enumerate(ghosts):
            gPosition = ghostPositions[i]
            gDistance = util.manhattanDistance(pacman, gPosition)
            if ghost.scaredTimer > 0:
                # if ghost is scared, Pacman should be rewarded for getting closer to it
                if gDistance > 0:
                    score += 200.0 / gDistance

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
            score += 40.0 / (closestFoodDist + 1e-6)

        # reward for getting closer to pellets
        if capsules:
            closestCap = min(util.manhattanDistance(pacman, c) for c in capsules)
            score += 20.0 / (closestCap + 1e-6)

        # penalize stalling / oscillation
        # - came from initial observations of Pacman getting stuck at walls/doing back-and-forth motions
        score -= len(food) * 0.1  # pressures Pacman to finish the level
        score -= closestFoodDist * 0.2  # if Pacman gets stuck, should prioritize movind toward the food more

        return float(score)
    
    def _minimax_decision(self, gameState, depth=2):
        """
        Returns the best action for Pacman (agentIndex=0) using minimax.
        """

        legal = [a for a in gameState.getLegalActions(0) if a != Directions.STOP]
        bestAction = None
        bestValue = -float("inf")

        for action in legal:
            successor = gameState.generateSuccessor(0, action)
            value = self._minimax(successor, depth, agentIndex=1)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
    
    def _minimax(self, state, depth, agentIndex):
        """
        Recursively computes minimax value for the given state using adverserial search.
        """

        # terminal state - just return the utility (objective function) value
        if depth == 0 or state.isWin() or state.isLose():
            return self.utility(state)

        numAgents = state.getNumAgents()

        # maximizing player - Pacman (agentIndex = 0)
        if agentIndex == 0:
            value = -float("inf")

            for action in state.getLegalActions(0):
                succ = state.generateSuccessor(0, action)
                value = max(value,
                            self._minimax(succ, depth, agentIndex=1))

            return value

        # minimizing player - Ghost (agentIndex >= 1)
        else:
            value = float("inf")

            nextAgent = agentIndex + 1
            nextDepth = depth

            # once we reach the last ghost, cycle back to Pacman agent and decrease depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth -= 1

            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)

                value = min(value,
                            self._minimax(succ, nextDepth, nextAgent))

            return value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        if hasattr(self, "lastPos") and gameState.getPacmanPosition() == self.lastPos:
            self.stuckCounter = getattr(self, "stuckCounter", 0) + 1
        else:
            self.stuckCounter = 0

        self.lastPos = gameState.getPacmanPosition()
        
        return self._minimax_decision(gameState, depth=2)
    
    # Notes:
    # check Pacman speed should be 0.5 so it can't outurn the ghost?
    # increasing penalty if no new pellet is eaten which resets once a pellet is eaten

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