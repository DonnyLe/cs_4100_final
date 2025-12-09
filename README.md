# Pac-Man: Through an Artificial Intelligence Lens
### CS 4100: Team 14 Final Project
This work was done for Santa Clara University's Artificial Intelligence course taught by Professor Liu

This repository was forked from @tadowney 's GitHub repo linked here.  The work seen here was collaboratively developed by John Kraunelis, Riya Roy, Donny Le, Kenneth Fan, and Swar Kewalia - Team 14 of Northeastern University's CS 4100: Artificial Intelligence course.  This course was taught by Professor Rajagopal Venkatesaramani in the Khoury College of Computer Sciences.

## Introduction

The goal of this project was to implement multiple AI agents into the classic *Pac-Man* game, then compare and contrast the results of each agent to come to general conclusions about best practices for this environment.  We tackled this problem through the lens of two different types of agents, and got more granular from there.  Below you will find information on each model we implemented, and how you can run these agents on your own to visualize the work we've done here.

<p align='center'>
  Example of Q-Learning Pac-Man Agent (Training Episodes = 200,000; Decay Rate = 0.99997)
</p>
![Pac-Man Demo](/images/pacman_winning.gif)

### Reinforcement Learning Agents:

#### 1. SARSA Agent

- Input short description of SARSA agent/algorithm here, which file(s) it lives in, and instructions on how to run the agent.

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

- Input short description of deep q-learning agent/algorithm here, which file(s) it lives in, and instructions on how to run the agent.

### Adversarial Search Agents:

#### 1. Minimax Agent

- Input short description of minimax agent/algorithm here, which file(s) it lives in, and instructions on how to run the agent.

#### 2. Alpha-Beta Pruning Enhancement

- Input short description of alpha-beta pruning agent/algorithm here, which file(s) it lives in, and instructions on how to run the agent.

## Sources

Pacman Game

The Pacman framework was developed by John DeNero and Dan Klein who are Computer Science professors at UC Berkeley. The original project can be found [here](http://ai.berkeley.edu/project_overview.html). The project was built to help teach students foundational AI concepts such as informed state-space search, probabilistic inference, and reinforcement learning. The game comes with many different layouts, but this project only used the *mediumClassic* layout as seen in the image below. The framework handles the graphics and game logistics, allowing students to focus on building the intelligent agent that navigates the map. 

All of the aforementioned algorithms seen implemented in this repository were solely developed by the authors listed at the top of this README file.

<p align="center">

| mediumClassic map                                               |
| :-------------------------------------------------------------: |
| <img width="423" height="287" src="./images/mediumClassic.png">  

</p> 
