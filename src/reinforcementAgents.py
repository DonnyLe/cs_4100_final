# reinforcementAgents.py

from game import Agent, Directions
from pacman import GameState
import random
import util
import pickle
import os
import math
import time

class SARSAAgent(Agent):
    
    def __init__(self, index = 0, epsilon = 0.1, alpha = 0.005, gamma = 0.9, numTraining = 100):
        self.index = index
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        
        self.weights = util.Counter()
        self.episodesSoFar = 0
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        epsilon_int = int(self.epsilon * 100)
        alpha_int = int(self.alpha * 1000)
        self.weightsFile = f'sarsa_e{epsilon_int}_a{alpha_int}_n{self.numTraining}_{timestamp}.pkl'
        
        if self.numTraining > 0:
            print(f"Training session: {self.weightsFile}")
        
    def saveWeights(self):
        try:
            with open(self.weightsFile, 'wb') as f:
                pickle.dump(self.weights, f)
        except:
            if self.episodesSoFar >= self.numTraining:
                print("failed saving weights")
    
    def getFeatures(self, state, action):
        features = util.Counter()
        
        try:
            successor = state.generatePacmanSuccessor(action)
        except:
            successor = None
            
        if successor is None:
            features['bias'] = 1.0
            features['invalid-action'] = -1.0
            return features
        
        features['bias'] = 1.0
        
        pacmanPos = successor.getPacmanPosition()
        foodList = successor.getFood().asList()
        ghostStates = successor.getGhostStates()
        capsules = successor.getCapsules()
        
        scoreDiff = successor.getScore() - state.getScore()
        features['score-delta'] = scoreDiff / 100.0
        
        if len(foodList) > 0:
            minFoodDist = min([util.manhattanDistance(pacmanPos, food) for food in foodList])
            features['nearest-food'] = 1.0 / (minFoodDist + 1.0)
        
        features['food-remaining'] = len(foodList) / 100.0
        
        activeGhostDistances = []
        scaredGhostDistances = []
        
        for ghost in ghostStates:
            ghostPos = ghost.getPosition()
            dist = util.manhattanDistance(pacmanPos, ghostPos)
            
            if ghost.scaredTimer > 0:
                scaredGhostDistances.append(dist)
            else:
                activeGhostDistances.append(dist)
        
        if len(activeGhostDistances) > 0:
            minActiveGhostDist = min(activeGhostDistances)
            if minActiveGhostDist <= 1:
                features['imminent-danger'] = 1.0
            elif minActiveGhostDist <= 3:
                features['danger'] = 1.0 / minActiveGhostDist
            else:
                features['ghost-distance'] = 1.0 / (minActiveGhostDist + 1.0)
        
        if len(scaredGhostDistances) > 0:
            minScaredDist = min(scaredGhostDistances)
            features['scared-ghost-nearby'] = 1.0 / (minScaredDist + 1.0)
            if minScaredDist <= 1:
                features['eat-ghost-opportunity'] = 1.0
        
        if len(capsules) > 0 and len(activeGhostDistances) > 0:
            minCapsuleDist = min([util.manhattanDistance(pacmanPos, cap) for cap in capsules])
            features['nearest-capsule'] = 1.0 / (minCapsuleDist + 1.0)
        
        if action == Directions.STOP:
            features['stopped'] = 1.0
        
        if scoreDiff > 0:
            features['ate-food'] = 1.0

        return features
    
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        qValue = sum(self.weights[f] * features[f] for f in features)
        
        if math.isnan(qValue) or math.isinf(qValue):
            return 0.0
        return qValue
    
    def getPolicy(self, state):
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) == 0:
            return Directions.STOP
        
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)
        
        qValues = [(action, self.getQValue(state, action)) for action in legalActions]
        
        if len(qValues) == 0:
            return Directions.STOP
        
        maxQ = max(q for _, q in qValues)
        
        bestActions = [action for action, q in qValues if q == maxQ]
        
        if len(bestActions) == 0:
            return random.choice(legalActions)
        
        return random.choice(bestActions)
    
    def getAction(self, state):
        legalActions = state.getLegalPacmanActions()
        
        if len(legalActions) == 0:
            return Directions.STOP
        
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)
        
        if self.episodesSoFar < self.numTraining:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        else:
            action = self.getPolicy(state)
        
        if self.lastState is not None:
            reward = state.getScore() - self.lastState.getScore()
            self.update(self.lastState, self.lastAction, reward, state, action)
        
        self.lastState = state
        self.lastAction = action
        
        return action
    
    def update(self, state, action, reward, nextState, nextAction):
        if nextState.isWin() or nextState.isLose():
            target = reward
        else:
            target = reward + self.gamma * self.getQValue(nextState, nextAction)
        
        currentQ = self.getQValue(state, action)
        difference = target - currentQ
        
        difference = max(-10.0, min(10.0, difference))
        
        features = self.getFeatures(state, action)
        for feature in features:
            update = self.alpha * difference * features[feature]
            self.weights[feature] += update
            
            self.weights[feature] = max(-100.0, min(100.0, self.weights[feature]))
    
    def final(self, state):
        if self.lastState is not None:
            reward = state.getScore() - self.lastState.getScore()
            features = self.getFeatures(self.lastState, self.lastAction)
            currentQ = sum(self.weights[f] * features[f] for f in features)
            difference = reward - currentQ
            
            difference = max(-10.0, min(10.0, difference))
            
            for feature in features:
                update = self.alpha * difference * features[feature]
                self.weights[feature] += update
                self.weights[feature] = max(-100.0, min(100.0, self.weights[feature]))
        
        self.episodesSoFar += 1
        
        if state.isWin():
            result = "WIN"
        elif state.data.steps >= state.data.maxSteps if state.data.maxSteps else False:
            result = "TIMEOUT"
        else:
            result = "LOSS"
        
        if self.episodesSoFar >= self.numTraining:
            print(f"E:{self.episodesSoFar}|S:{state.getScore():.0f}|{result}")
        
        if self.episodesSoFar % 10 == 0 and self.episodesSoFar <= self.numTraining:
            self.saveWeights()
            if self.episodesSoFar == self.numTraining:
                print(f"Training: E:{self.episodesSoFar}|S:{state.getScore():.0f}|{result}")
        
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0
        
        if self.episodesSoFar == self.numTraining:
            self.saveWeights()
            print(f"\nSaved to: {self.weightsFile}")
            print("Final weights:")
            for key, value in sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"  {key}: {value:.4f}")
    
    def registerInitialState(self, state):
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

