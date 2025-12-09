# reinforcementAgents.py

from game import Agent, Directions
from pacman import GameState
import random
import util
import pickle
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
# creates the output directory for sarsa if one doesn't exist, so i can keep all training artifacts in one place
def ensure_sarsa_output_dir():
    if not os.path.exists('sarsa_data'):
        os.makedirs('sarsa_data')
    return 'sarsa_data'
# returns the output path for the sarsa data files, like plots or weight files
def get_sarsa_output_path(filename):
    ensure_sarsa_output_dir()
    return os.path.join('sarsa_data', filename)

# using linear function approximation instead of using q tables cause of the huge state space
class SARSAAgent(Agent):
    def __init__(self, index=0, epsilon=1.0, alpha=0.005, gamma=0.9, numTraining=100,
                 epsilonDecay=0.9999, minEpsilon=0.05, weightsFile=None):
        self.index = index
        self.epsilon = float(epsilon)
        self.initialEpsilon = float(epsilon)
        self.epsilonDecay = float(epsilonDecay)
        self.minEpsilon = float(minEpsilon)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        
        # maintaining a weight vector, assigning each weight to a feature
        self.weights = util.Counter()
        self.episodesSoFar = 0
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0.0
        self.episodeReward = 0.0
        self.allEpisodeRewards = []
        
        if weightsFile:
            self.weightsFile = weightsFile
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.weightsFile = f'sarsa_n{self.numTraining}_e{self.epsilonDecay}_{timestamp}.pkl'
        
        if self.numTraining > 0:
            print(f"training session: {self.weightsFile}")
        
        # loading weights for evaluating the pacman performances w demo runs
        self.loadWeights()
    def loadWeights(self):
        filepath = get_sarsa_output_path(self.weightsFile)
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'weights' in data:
                self.weights = data['weights']
                self.epsilon = data.get('epsilon', self.minEpsilon)
                print(f"loaded weights from {self.weightsFile}")
            else:
                self.weights = data
                print(f"loaded raw weights from {self.weightsFile}")
        except Exception as e:
            print(f"failed loading weights: {e}")
    
    # saving learned weghts
    def saveWeights(self):
        try:
            filepath = get_sarsa_output_path(self.weightsFile)
            data = {
                'weights': self.weights,
                'episode_rewards': self.allEpisodeRewards,
                'epsilon': self.epsilon,
                'episodes_trained': self.episodesSoFar
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            if self.episodesSoFar >= self.numTraining:
                print(f"Failed saving weights: {e}")
    
    # generating plots for runing average reward, showing agents improvement (hopefully)
    def plotAverageRewards(self):
        if len(self.allEpisodeRewards) == 0:
            print("no rewards to plot")
            return
        
        rewards = np.array(self.allEpisodeRewards)
        running_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(running_avg, color='#0072B2', linewidth=1.5, label='Running Avg. Reward')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Running Average Reward', fontsize=12)
        plt.title(f'SARSA: Running Average Reward ({len(rewards)} episodes)', fontsize=13, weight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_filename = f'sarsa_reward_plot_{len(rewards)}_{self.epsilonDecay}.png'
        plt.savefig(get_sarsa_output_path(plot_filename), dpi=300)
        plt.close()
        print(f"Saved reward plot to: {get_sarsa_output_path(plot_filename)}")
        
        csv_filename = f'sarsa_episode_rewards_{len(rewards)}_{self.epsilonDecay}.csv'
        csv_path = get_sarsa_output_path(csv_filename)
        with open(csv_path, 'w') as f:
            f.write('episode,reward,avg_reward\n')
            for i, (r, avg) in enumerate(zip(rewards, running_avg)):
                f.write(f'{i},{r},{avg}\n')
        print(f"Saved episode rewards CSV to: {csv_path}")
    
    # transforming raw game states into feature reppresentations 
    # using current state and action being evaled
    def getFeatures(self, state, action):
        features = util.Counter()
        
        try:
            successor = state.generatePacmanSuccessor(action)
        except:
            successor = None
        
        # invalid actions
        if successor is None:
            features['bias'] = 1.0
            features['invalid-action'] = -1.0
            return features
        
        
        # baseline q value
        # for linear regression
        features['bias'] = 1.0
        
        pacmanPos = successor.getPacmanPosition()
        foodList = successor.getFood().asList()
        ghostStates = successor.getGhostStates()
        capsules = successor.getCapsules()
        
        #normalizing the score change by diving 100 to keep values small
        scoreDiff = successor.getScore() - state.getScore()
        features['score-delta'] = scoreDiff / 100.0
        
        # food features
        if len(foodList) > 0:
            # im inversing the distance to nearest food pellet, so closer food gives higher values
            minFoodDist = min([util.manhattanDistance(pacmanPos, food) for food in foodList])
            features['nearest-food'] = 1.0 / (minFoodDist + 1.0)
        
        features['food-remaining'] = len(foodList) / 100.0
        
        # ghost features
        activeGhostDistances = []
        scaredGhostDistances = []
        
        for ghost in ghostStates:
            ghostPos = ghost.getPosition()
            dist = util.manhattanDistance(pacmanPos, ghostPos)
            
            if ghost.scaredTimer > 0:
                scaredGhostDistances.append(dist)
            else:
                activeGhostDistances.append(dist)
        
        # this is for handling active ghosts
        if len(activeGhostDistances) > 0:
            minActiveGhostDist = min(activeGhostDistances)
            if minActiveGhostDist <= 1:
                # ghost is next to the pacman, high priroirty cause next step is improtant for surivival
                features['imminent-danger'] = 1.0
            elif minActiveGhostDist <= 3:
                # not as high priority, cause ghost is close
                # distance is inversed so closer ghosts give higher values
                features['danger'] = 1.0 / minActiveGhostDist
            else:
                # far ghosts
                features['ghost-distance'] = 1.0 / (minActiveGhostDist + 1.0)
        
        # for scared ghosts
        if len(scaredGhostDistances) > 0:
            minScaredDist = min(scaredGhostDistances)
            # encourage closing distance to scared ghosts 
            features['scared-ghost-nearby'] = 1.0 / (minScaredDist + 1.0)
            if minScaredDist <= 1:
                features['eat-ghost-opportunity'] = 1.0
        

        # capsule features
        if len(capsules) > 0 and len(activeGhostDistances) > 0:
            minCapsuleDist = min([util.manhattanDistance(pacmanPos, cap) for cap in capsules])
            features['nearest-capsule'] = 1.0 / (minCapsuleDist + 1.0)
        
        # no stopping, penalized
        if action == Directions.STOP:
            features['stopped'] = 1.0
        
        if scoreDiff > 0:
            features['ate-food'] = 1.0

        return features
    
    # this is for calculating q values using linear function approximation
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        # sum of weight * feature for all of them
        qValue = sum(self.weights[f] * features[f] for f in features)
        
        if math.isnan(qValue) or math.isinf(qValue):
            return 0.0
        return qValue
    
    # getting best policy based on current q values
    # so it should be choosing the action w the highest q value
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
        
        # random select in order to handle ties between actions w same q values
        return random.choice(bestActions)
    
    # selecting actions using epsilon greedy policy
    # explore vs exploit
    # probability epsilon ? choose random (explore), otherwise choose best action (exploit)
    def getAction(self, state):
        legalActions = state.getLegalPacmanActions()
        
        if len(legalActions) == 0:
            return Directions.STOP
        
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)
        
        if self.episodesSoFar < self.numTraining:
            if util.flipCoin(self.epsilon):
                # random actions
                action = random.choice(legalActions)
            else:
                # exloiting by using best q value actions
                action = self.getPolicy(state)
        else:
            action = self.getPolicy(state)
        
        # sarsa update from prev transition
        if self.lastState is not None:
            currentScore = state.getScore()
            reward = currentScore - self.lastScore
            self.lastScore = currentScore
            self.episodeReward += reward
            self.update(self.lastState, self.lastAction, reward, state, action)
        
        self.lastState = state
        self.lastAction = action
        
        return action
    
    # this func performing sarsa weight update
    def update(self, state, action, reward, nextState, nextAction):
        if nextState.isWin() or nextState.isLose():
            target = reward
        else:
            # using q value of the action the agent ACTUALLY took, not max q value action
            target = reward + self.gamma * self.getQValue(nextState, nextAction)
        
        
        currentQ = self.getQValue(state, action)
        difference = target - currentQ
        
        # clipping td error to implement gradient clipping 
        # this is for preventing big gradient updates that could cause instability
        difference = max(-10.0, min(10.0, difference))
        
        # updating the weights
        # positive td error means underestimate so increase weight, negative means otherwise
        features = self.getFeatures(state, action)
        for feature in features:
            update = self.alpha * difference * features[feature]
            self.weights[feature] += update
            
            # clipping weights to range of -100 100
            self.weights[feature] = max(-100.0, min(100.0, self.weights[feature]))
    
    def final(self, state):
        if self.lastState is not None:
            currentScore = state.getScore()
            reward = currentScore - self.lastScore
            self.episodeReward += reward
            
            # terminal state, target === reward cause q terminal = 0
            features = self.getFeatures(self.lastState, self.lastAction)
            currentQ = sum(self.weights[f] * features[f] for f in features)
            difference = reward - currentQ
            
            difference = max(-10.0, min(10.0, difference))
            
            for feature in features:
                update = self.alpha * difference * features[feature]
                self.weights[feature] += update
                self.weights[feature] = max(-100.0, min(100.0, self.weights[feature]))
        
        # storing episode reward for vis
        self.allEpisodeRewards.append(self.episodeReward)
        
        self.episodesSoFar += 1
        
        # decay epsilon during training, but min is 0.05
        if self.episodesSoFar < self.numTraining:
            self.epsilon = max(self.minEpsilon, self.epsilon * self.epsilonDecay)
        
        if state.isWin():
            result = "WIN"
        elif state.data.maxSteps and state.data.steps >= state.data.maxSteps:
            result = "TIMEOUT"
        else:
            result = "LOSS"
        
        # log progress every 100 episodes
        if self.episodesSoFar <= self.numTraining and self.episodesSoFar % 100 == 0:
            avgReward = np.mean(self.allEpisodeRewards[-100:]) if len(self.allEpisodeRewards) >= 100 else np.mean(self.allEpisodeRewards)
            print(f"e:{self.episodesSoFar}|avgR:{avgReward:.1f}|eps:{self.epsilon:.4f}|{result}")
        
        if self.episodesSoFar > self.numTraining:
            print(f"Eval E:{self.episodesSoFar - self.numTraining}|S:{state.getScore():.0f}|{result}")
        
        # this is for saving periodically during training
        if self.episodesSoFar % 500 == 0 and self.episodesSoFar <= self.numTraining:
            self.saveWeights()
        
        self.lastState = None
        self.lastAction = None
        self.episodeReward = 0.0
        
        # just for saving final weights and drawing plots
        if self.episodesSoFar == self.numTraining:
            self.saveWeights()
            self.plotAverageRewards()
            
            print(f"\n{'='*50}")
            print("Training Complete!")
            print(f"{'='*50}")
            print(f"saved weights to: {get_sarsa_output_path(self.weightsFile)}")
            print(f"total episodes: {self.numTraining}")
            print(f"final epsilon: {self.epsilon:.4f}")
            print(f"avg reward: {np.mean(self.allEpisodeRewards[-100:]):.2f}")
            print(f"\nLearned weights ({len(self.weights)} features):")
            for key, value in sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)[:15]:
                print(f"  {key}: {value:.4f}")
    
    def registerInitialState(self, state):
        self.lastState = None
        self.lastAction = None
        self.lastScore = state.getScore()
        self.episodeReward = 0.0

