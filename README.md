# Pac-Man: Through an Artificial Intelligence Lens
### CS 4100: Team 14 Final Project

This repository was forked from @tadowney 's GitHub repo linked [here](https://github.com/tadowney/ai_pacman).  The work seen here was collaboratively developed by John Kraunelis, Riya Roy, Donny Le, Kenneth Fan, and Swar Kewalia - Team 14 of Northeastern University's CS 4100: Artificial Intelligence course.  This course was taught by Professor Rajagopal Venkatesaramani in the Khoury College of Computer Sciences.

## Introduction

The goal of this project was to implement multiple AI agents into the classic *Pac-Man* game, then compare and contrast the results of each agent to come to general conclusions about best practices for this environment.  We tackled this problem through the lens of two different types of agents, and got more granular from there.  Below you will find information on each model we implemented, and how you can run these agents on your own to visualize the work we've done here.

<p align='center'>
  Example of Q-Learning Pac-Man Agent (Training Episodes = 200,000; Decay Rate = 0.99997)
</p>

![Pac-Man Demo](/images/pacman_winning.gif)

**Also, please note that some preferred package versions are displayed in the `requirements.txt` file.**

### Reinforcement Learning Agents:

#### 1. SARSA Agent

The first reinforcement learning algorithm was the SARSA (State-Action-Reward-State-Action) algorithm. SARSA is an op-policy temporal difference learning method that learns Q-values based on the actual actions chosen by the agent, including random exploratory actions. The crucial difference between SARSA and Q-learning is that SARSA learns the value of the policy being followed rather than the optimal policy. This makes SARSA strategies comparatively more conservative in dangerous environments. Please visit [this](https://rajagopalvenkat.com/teaching/resources/AI/ch6.html#rl) link for information on the theoretical foundations of SARSA.

This particular implementation uses **linear function approximation** rather than a Q-table, because Pac-Man demands a large state space. So instead of storing Q-values in a table, the agent learns feature weights and compute Q-values as a weighted sum of game features, such as distance to the nearest food pellet, and power capsule. 

**File Structure**
- The SARSA agent implementation lives within the `/src/sarsa/reinforcementAgents.py` file. All training data, including learned weights, reward plots, and performance metrics, are automatically saved to the `/src/sarsa/sarsa_data` directory upon completion.

**Agent Setup Instructions**
- The SARSA agent can be ran in training mode or evaluation mode. When running in training mode, the agent will play for a specified number of episodes while learning feature weights. The epsilon value decays gradually (default rate = 0.9999) to shift the agent from exploration to exploitation. When training completes, the agent will save the learned weights as a pickle file and generate visuailzation of the learning curve and results.
- When running in evaluation mode, the user can load previously trained weights to observe the agent's performance in the Pac-Man environment. Trained weight files are stored in the `src/sarsa/sarsa_data` directory.
- Below is an example of the command line arguments required to run the agent in training mode, be sure that you are in the `/src` directory to start:
```bash
cd src
```
```bash
python pacman.py -p SARSAAgent -n 50000 -x 50000 -l mediumClassic \
    -a epsilon=1.0,alpha=0.005,gamma=0.9,numTraining=50000,epsilonDecay=0.9999,minEpsilon=0.05
```

This command above will train the SARSA agent for 50,000 episodes on the mediumClassic layout. The parameters specify the initial epsilon, alpha (learning rate), gamma (discount factor), epsilon decay rate, and minimum epsilon value. The training progress and metrics will be logged every 100 episodes, and when training completes, the agent will save learned weights and generate a running average reward plot in the `sarsa_data` folder.

- Here is an example of the command line arguments for running the agent in evaluation mode, again ensure you are in the `src` directory:
```bash
cd src
```
```bash
python pacman.py -p SARSAAgent -n 10 -l mediumClassic \
        -a numTraining=0,weightsFile=sarsa_n50000_e0.9999_20231215_143022.pkl
```

The above command will load a trained SARSA agent and run it for 10 evaluation episodes. By setting `numTraining=0`, the agent runs in pure exploitation. The program will output performance metrics including final scores, game results, and win rate to the terminal.


#### 2. Q-Learning Agent

The second algorithm implemented was the basic Q-Learning algorithm.  Please visit [this](https://rajagopalvenkat.com/teaching/resources/AI/ch6.html#rl) link for detailed information on reinforcement learning algorithms, and more specifically the exact psuedo code that was used to implement this version of the Q-Learning algorithm.  This specific agent utilizes a partially observable environment, observing states within a specified window surrounding the agent.  It defaults to utilizing a 5x5 window with Pac-Man centered in the middle.

**File Structure**
- All trained agents utilize the `train_agent.py` file to actually undergo training.  This specific agent lives within the `/src/q_learning/q_learning.py` file, and the resulting Q-Tables and performance visualizations live within the `/src/q_learning_data` directory.

**Agent Setup Instructions**
- The Q-Learning agent can be ran in two different ways: 1) training mode, 2) evaluation mode.  When running in training mode, the agent will play for a specified number of training episodes, and the epsilon value will decay by the specified decay rate.  It will fill out a Q-Table and store it in the `/src/q_learning_data` directory.  *Note that runtimes can be particularly high for the Q-Learning agent while training.*
- The user can also run the agent in evaluation mode, in which case the user can choose from one of the many pre-created Q-Tables in the above folder for the agent to select actions from.  Each Q-Table is stored in a pickle file, with the following syntax: "Q_table_{training_episodes}_{decay_rate}_{env_window}.pickle".
- Below is an example of the command line arguments required to run the agent in training mode (ensure you are in the `/src` directory!):

```
cd src
```
```
python3 train_agent.py --agent qlearning --train --episodes 1000 --decay-rate 0.999
```

The above command line arguments will train the Q-Learning agent on 1,000 episodes with an epsilon decay rate of 0.999.  It will then save the resulting Q-Table in the aforementioned format in the `q_learning_data` folder.

- Next, here is an example of the command line arguments required to run the agent in evaluation mode (ensure you are in the `src` directory!):

```
cd src
```

```
python3 train_agent.py --agent qlearning --eval --load-model Q_table_200000_0.99997_2.pickle \
        --episodes 100 --max-steps 1000 --gui --frame-time 0.05
```

The above commands will allow you to visualize an agent trained off of 200,000 episdoes with an epsilon decay rate of 0.99997, playing for 100 episodes.  At conclusion, the program will output various performance metrics to the terminal to help the user analyze overall success & areas of weakness.  To run the agent in evaluation mode without visualizing on the GUI (*faster method*), simply omit the `--gui` and `--frame-time` arguments.


#### 3. Deep Q-Learning Agent

The third algorithm implemented is a fully-observable deep Q-learning agent. To learn more about this algorithm, we used [this](https://www.youtube.com/watch?v=EUrWGTCGzlA&t=1504s&pp=ygUpZGVlcCBxIGxlYXJuaW5nIGluIHJlaW5mb3JjZW1lbnQgbGVhcm5pbmfSBwkJJQoBhyohjO8%3D) video which gave a detailed guide on how the algorithm works on a different application. 

**File Structure**
- All trained agents utilize the `train_agent.py` file to actually undergo training.  This specific agent lives within the `/src/deep_q_learning/deep_q_learning.py` file, and the resulting Q-Tables and performance visualizations live within the `/src/deep_q_learning_data` directory. In this folder, there are reward graphs, along with output files for each training and evaluation run. 

**Agent Setup Instructions**
- The Deep Q-Learning agent can be ran in two different ways: 1) training mode, 2) evaluation mode.  When running in training mode, the agent will play for a specified number of training episodes, and the epsilon value will decay by the specified decay rate.  It will store to a Pytorch .pt file `/src/deep_q_learning_data` directory.  *Note that runtimes can be particularly high for the Deep Q-Learning agent while training.*
- The user can also run the agent in evaluation mode, in which case the user can choose from s in the above folder for the agent to select actions from.  Each policy is stored in a .pt file.
- The current .pt files in the folder are the following: 
- dqn_nors_2k.pt, dqn_nors_10k.pt, dqn_nors20k.pt are models with trained with 2k, 10k, and 20k episodes respectively, along with no reward shaping
- dqn_rs_2k.pt, dqn_rs_10k.pt are models trained with 2k and 10 episodes respectively, along with reward shaping 


- Below is an example of the command line arguments required to run the agent in training mode (ensure you are in the `/src` directory!):

```
cd src
```
```
python train_agent.py --agent dqn --train \
  --episodes 20000 \
  --learning-rate 0.0005 \
  --epsilon 1.0 \                   
  --decay-rate 0.99985 \  
  --layout mediumClassic \               
  --ghosts 2 \
  --max-steps 1000 \
  --no-reward-shaping \
  --save-model dqn_nors_20k.pt 
  
  ```

The above command trains a DQL model with 20k episodes, with a learning rate of 0.0005, on the mediumClassic layout, with two ghosts, with a maxStep count of 10000, without reward shaping and saves the model to `/src/deep_q_learning_data/` folder. `--no-reward-shaping` is a flag to disable reward shaping on the train command. See `train_agent.py` for all CLI options. 


- Next, here is an example of the command line arguments required to run the agent in evaluation mode (ensure you are in the `src` directory!):

```
cd src
```

```
python train_agent.py \
  --agent dqn \
  --eval \
  --episodes 1000 \
  --load-model dqn_nors_20k.pt \  
  --layout mediumClassic \
  --ghosts 2 \
  --max-steps 1000 --gui --frame-time 0.1
```

The above commands will allow you to visualize an agent trained off of 20k episodes, playing for 1000 episodes. At conclusion, the program will output various performance metrics to the terminal to help the user analyze overall success & areas of weakness. It will also save an evaluation file to `src/deep_q_learning_data` folder. To run the agent in evaluation mode without visualizing on the GUI (*faster method*), simply omit the `--gui` and `--frame-time` arguments. This makes the performance faster. 

### Adversarial Search Agents:

#### 1. Minimax Agent

The first adversarial search algorithm implemented was a Minimax algorithm.  Please visit [this](https://rajagopalvenkat.com/teaching/resources/AI/ch5.html#expectiminimax) link for information on adversarial search algorithms, including the high-level pseudo-code on which the Minimax agent is implemented based. As best suited for the Minimax algorithm, this agent utilizes a fully-observable, deterministic environment where Pac-Man is the maximizing player and the ghosts act as one minimizing player. In order to choose actions for Pac-Man, the algorithm assumes that Pac-Man and the ghosts are both agents that play optimally every turn (**Note:** we don't actually choose actions for the ghosts as the scope of this project is focused on comparing Pac-Man agents, rather than creating ghost agents). To choose the Minimax decision, we have defined two versions of a utility function, `basicUtility(gameState)` and `improvedUtility(gameState)`. They provide the value for that player at a given state through either using the built-in game score or a more proximity-based approach (as documented within the function), respectively. The utility function is computed when a terminal state of the game tree is reached. We selected a maximum depth of 2 for our implementation, so the terminal states are reached at the branches at that depth or when there are no legal
actions left for a player to take.

#### 2. Alpha-Beta Pruning Enhancement

The second adversarial search algorithm implemented was the Minimax algorithm with Alpha-Beta Pruning. Please visit [this](https://rajagopalvenkat.com/teaching/resources/AI/ch5.html#expectiminimax) link for more information on these algorithms and their implementations. The alpha-beta pruning algorithm is a modification of the pure Minimax algorithm where two new parameters are introduced (alpha and beta) and are used to keep track of the best value the maximizing player (Pacman) can obtain, and the best value that the minimizing player (the ghosts) can obtain, respectively. Similar to the pure Minimax algorithm, this algorithm is also recursive and assumes that Pacman and the ghosts play optimally at every turn. Using these values, the algorithm eliminates, or prunes, unnecessary exploration of subtrees or branches that will not contribute to the final decision for the next best move. This reduces the time complexity of Minimax significantly while providing similar performance, as it uses the same number of computations as Minimax to explore twice the depth of the search tree. To most accurately compare results, this agent used the same basic and improved utility functions as the pure Minimax agent, as well as a depth of 2.

**File Structure**
- Since adversarial search algorithms are based on a multi-agent philosophy, the Minimax and Alpha-Beta Pruning agent implementations both live within the `src/multiAgents.py` file.

**Agent Setup Instructions**
- Both the Minimax and Alpha-Beta Pruning agents don't require training, so they can either be run on individual games or a more extensive evaluation mode. The evaluation mode can be used to view how each agent performs and compare them using the same game configurations.

#### To run individual games with the GUI, simply run the command for the respective agent:
1. Minimax Agent
```
cd src
```
```
python pacman.py -p MinimaxAgent -l mediumClassic
``` 

2. Alpha-Beta Agent
```
cd src
```
```
python pacman.py -p AlphaBetaAgent -l mediumClassic
``` 

#### To run the evaluation mode, you can run the `src\multiAgents.py` file with the option to pass in two arguments:
 1. `agent_type`: 'Minimax' or 'AlphaBeta' (default: 'Minimax')
 2. `depth`: any integer value (default = 2, should be kept small for time/space complexity)

Below are samples of the command line arguments that can be used to run each agent in evaluation mode (make sure you are in the `/src` directory!):

1. Minimax Agent (Default Configuration)
```
cd src
```
```
python multiAgents.py
```
The above will run 1,000 games on the mediumClassic layout with 2 adversarial ghosts, using the Minimax agent with depth 2.

2. Alpha-Beta Agent
```
cd src
```
```
python multiAgents.py --agent AlphaBeta --depth 3
``` 
The above will run 1,000 games on the mediumClassic layout with 2 adversarial ghosts, using the Alpha-Beta Pruning agent with depth 3. <br>
(**Note:** Both our agents were evaluated using depth 2, this example just shows how to use the additional configurations)

Once the evaluation mode executes running, the program will output various performance metrics to the terminal so that the user can analyze the overall strengths and weaknesses of the adversarial search agents, including "Average Score", "Win Rate", and "Total Runtime".
 
## Sources

Pacman Game

The Pacman framework was developed by John DeNero and Dan Klein who are Computer Science professors at UC Berkeley. The original project can be found [here](http://ai.berkeley.edu/project_overview.html). The project was built to help teach students foundational AI concepts such as informed state-space search, probabilistic inference, and reinforcement learning. The game comes with many different layouts, but this project only used the *mediumClassic* layout as seen in the image below. The framework handles the graphics and game logistics, allowing students to focus on building the intelligent agent that navigates the map. 

All of the aforementioned algorithms seen implemented in this repository were solely developed by the authors listed at the top of this README file.

<p align="center">

| mediumClassic map                                               |
| :-------------------------------------------------------------: |
| <img width="423" height="287" src="./images/mediumClassic.png">  

</p> 
