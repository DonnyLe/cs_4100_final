"""
training_utils.py - Reusable training utilities for Pacman RL agents
"""
import os
import time
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pacman import GameState
import textDisplay
from graphicsDisplay import PacmanGraphics
import datetime


OUTPUT_DIR = 'q_learning_data'


def ensure_output_dir(agent_type=None):
    """
    Create output directory if it doesn't exist.
    If agent_type is provided, creates agent-specific subdirectory.
    """
    if agent_type == 'dqn':
        output_dir = 'deep_q_learning_data'
    else:
        output_dir = OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_output_path(filename, agent_type=None):
    """Get full path for output file."""
    output_dir = ensure_output_dir(agent_type)
    return os.path.join(output_dir, filename)


def runGamesQuiet(layout, pacman, ghosts, display, numGames, record, 
                  numTraining=0, catchExceptions=False, timeout=30, 
                  randomRewards=False, max_steps=None):
    """
    Wrapper around pacman game runner that suppresses output messages.
    Shows progress bar during training.
    """
    from pacman import ClassicGameRules
    
    rules = ClassicGameRules(timeout, randomRewards=randomRewards, maxSteps=max_steps)
    rules.quiet = True
    games = []

    use_gui = isinstance(display, PacmanGraphics)
    
    for i in tqdm(range(numGames), desc="Games", unit="game", disable=use_gui):
         # Clears the gameState from memory so we're not stacking up.

        GameState.getAndResetExplored()

         # To suppress output from the OG setup, we need to use NullGraphics

        if use_gui:
            gameDisplay = display
        else:
            gameDisplay = textDisplay.NullGraphics()
        
        game = rules.newGame(layout, pacman, ghosts, gameDisplay, True, catchExceptions)
        game.run()
        games.append(game) # Always append, unlike original runGames
    
    return games


def plot_training_rewards(reward_df, num_episodes, agent_name="Agent", agent_type=None, **kwargs):
    """
    Generate and save a training reward plot.
    
    Args:
        reward_df: DataFrame with 'episode' and 'avg_reward' columns
        num_episodes: Total number of training episodes
        agent_name: Name of agent for plot title
        agent_type: Type of agent ('dqn', 'qlearning', etc.) for directory routing
        **kwargs: Additional parameters to include in filename/title
    """
    reward_df = reward_df.sort_values('episode')

    plt.figure(figsize=(10, 6))
    plt.plot(
        reward_df['episode'],
        reward_df['avg_reward'],
        color='#0072B2',
        linewidth=2.0,
        label='Running Avg. Reward'
    )

    # Build title from kwargs
    param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    long_title = f"{agent_name} Training: Running Average Rewards ({num_episodes:,.0f} episodes"
    if param_str:
        long_title += f", {param_str}"
    long_title += ")"
    
    wrapped_title = "\n".join(textwrap.wrap(long_title, width=50))
    plt.title(wrapped_title, fontsize=14, weight="bold", pad=15)
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Running Avg. Reward", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Build filename
    filename_parts = [agent_name.lower().replace(" ", "_"), str(num_episodes)]
    filename_parts.extend([str(v) for v in kwargs.values()])
    out_name = get_output_path(f"avg_reward_plot_{'_'.join(filename_parts)}.png", agent_type=agent_type)
    
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"Saved training plot to: {out_name}")


class TrainingSession:
    """
    Manages a training session for an RL agent.
    Handles timing, statistics, and result reporting.
    """
    
    def __init__(self, agent, agent_name="Agent", agent_type=None):
        self.agent = agent
        self.agent_name = agent_name
        self.agent_type = agent_type  # 'dqn', 'qlearning', etc.
        self.start_time = None
        self.end_time = None
        self.games = []
        self.hyperparameters = {}  # Store hyperparameters for logging
        
    def run(self, layout, ghosts, display, num_episodes, 
            max_steps=None, catchExceptions=False, timeout=30, randomRewards=False):
        """
        Run training session.
        
        Returns:
            List of completed games
        """
        print(f"\n{'='*60}")
        print(f"Starting {self.agent_name} Training Session")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Layout: {layout.name if hasattr(layout, 'name') else 'Unknown'}")
        print(f"Ghosts: {len(ghosts)}")
        print(f"Max steps per episode: {max_steps if max_steps else 'None'}")
        print(f"{'='*60}\n")
        
        self.start_time = time.perf_counter()
        
        self.games = runGamesQuiet(
            layout=layout,
            pacman=self.agent,
            ghosts=ghosts,
            display=display,
            numGames=num_episodes,
            record=False,
            numTraining=num_episodes,
            catchExceptions=catchExceptions,
            timeout=timeout,
            randomRewards=randomRewards,
            max_steps=max_steps
        )
        
        self.end_time = time.perf_counter()
        
        return self.games
    
    def report_results(self, save_plot=True, **plot_kwargs):
        """
        Print training results and optionally save plots.
        
        Args:
            save_plot: Whether to generate and save reward plot
            **plot_kwargs: Additional parameters for plot filename
        """
        if not self.games or self.start_time is None or self.end_time is None:
            print("No training data to report!")
            return
        
        duration = self.end_time - self.start_time
        
        print(f"\n{'='*60}")
        print(f"{self.agent_name} Training Complete")
        print(f"{'='*60}")
        print(f"Training Runtime: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Collect statistics
        stats = {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'num_episodes': len(self.games),
            'runtime_seconds': duration,
            'runtime_minutes': duration / 60,
        }
        
        # Agent-specific statistics
        if hasattr(self.agent, 'episode_rewards') and self.agent.episode_rewards:
            rewards = self.agent.episode_rewards
            stats['avg_reward'] = np.mean(rewards)
            stats['std_reward'] = np.std(rewards)
            stats['min_reward'] = np.min(rewards)
            stats['max_reward'] = np.max(rewards)
            stats['final_reward'] = rewards[-1]
            
            print(f"Average Reward: {stats['avg_reward']:.2f}")
            print(f"Std Dev: {stats['std_reward']:.2f}")
            print(f"Min Reward: {stats['min_reward']:.2f}")
            print(f"Max Reward: {stats['max_reward']:.2f}")
            print(f"Final Reward: {stats['final_reward']:.2f}")
            
            if hasattr(self.agent, 'epsilon'):
                # Assume the agent stores its initial epsilon (you already do this in DQN)
                stats['initial_epsilon'] = getattr(self.agent, 'initial_epsilon', None)
                stats['final_epsilon'] = self.agent.epsilon
                if stats['initial_epsilon'] is not None:
                    print(f"Initial Epsilon: {stats['initial_epsilon']:.4f}")
                print(f"Final Epsilon: {stats['final_epsilon']:.4f}")
            
            if hasattr(self.agent, 'q_table'):
                stats['q_table_size'] = len(self.agent.q_table)
                print(f"Q-table Size: {stats['q_table_size']} states")
            
            # Add hyperparameters to stats
            stats.update(self.hyperparameters)
            
            # Save statistics to text file
            self._save_statistics(stats, **plot_kwargs)
            
            # Save reward data and plot
            if save_plot and len(rewards) > 0:
                running_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
                reward_df = pd.DataFrame({
                    "episode": np.arange(len(rewards)),
                    "avg_reward": running_avg,
                    "static_rewards": rewards,
                })
                
                # Save CSV
                csv_parts = [self.agent_name.lower().replace(" ", "_"), str(len(rewards))]
                csv_parts.extend([str(v) for v in plot_kwargs.values()])
                csv_name = get_output_path(
                    f"episode_rewards_{'_'.join(csv_parts)}.csv",
                    agent_type=self.agent_type
                )
                reward_df.to_csv(csv_name, index=False)
                print(f"Saved reward data to: {csv_name}")
                
                # Generate plot (running average like before)
                plot_training_rewards(
                    reward_df, len(rewards), self.agent_name,
                    agent_type=self.agent_type, **plot_kwargs
                )
        
        print(f"{'='*60}\n")

    
    def _save_statistics(self, stats, **kwargs):
        """Save detailed statistics to a text file."""
        
        # Build filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = [self.agent_name.lower().replace(" ", "_"), str(stats['num_episodes'])]
        filename_parts.extend([str(v) for v in kwargs.values()])
        filename = f"training_stats_{'_'.join(filename_parts)}_{timestamp}.txt"
        
        stats_path = get_output_path(filename, agent_type=self.agent_type)
        
        with open(stats_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"{self.agent_name} Training Statistics\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Training Configuration
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Agent Type: {stats.get('agent_type', 'N/A')}\n")
            f.write(f"Number of Episodes: {stats['num_episodes']}\n")
            f.write(f"Runtime: {stats['runtime_seconds']:.2f} seconds ({stats['runtime_minutes']:.2f} minutes)\n")
            f.write("\n")
            
            # Hyperparameters
            f.write("HYPERPARAMETERS\n")
            f.write("-" * 70 + "\n")
            hyperparams = [
                'gamma',
                'learning_rate',
                'epsilon',
                'decay_rate',
                'min_epsilon',
                'window_size',
                'batch_size',
                'discount_factor',
                'target_sync_steps',
                'replay_memory_size',
                'use_reward_shaping',
            ]
            for param in hyperparams:
                if param in stats:
                    f.write(f"{param}: {stats[param]}\n")
            f.write("\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            if 'avg_reward' in stats:
                f.write(f"Average Reward: {stats['avg_reward']:.2f}\n")
                f.write(f"Std Dev Reward: {stats['std_reward']:.2f}\n")
                f.write(f"Min Reward: {stats['min_reward']:.2f}\n")
                f.write(f"Max Reward: {stats['max_reward']:.2f}\n")
                f.write(f"Final Reward: {stats['final_reward']:.2f}\n")
            f.write("\n")
            
            # Exploration Stats
            if 'initial_epsilon' in stats:
                f.write("EXPLORATION\n")
                f.write("-" * 70 + "\n")
                f.write(f"Initial Epsilon: {stats['initial_epsilon']:.4f}\n")
                f.write(f"Final Epsilon: {stats['final_epsilon']:.4f}\n")
                decay_amount = stats['initial_epsilon'] - stats['final_epsilon']
                f.write(f"Total Decay: {decay_amount:.4f}\n")
                f.write("\n")
            
            # Q-learning specific
            if 'q_table_size' in stats:
                f.write("Q-LEARNING SPECIFIC\n")
                f.write("-" * 70 + "\n")
                f.write(f"Q-table Size: {stats['q_table_size']} states\n")
                f.write("\n")
            
            # Game Configuration
            f.write("GAME CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            game_params = ['layout', 'num_ghosts', 'ghost_type', 'max_steps', 'random_rewards']
            for param in game_params:
                if param in stats:
                    f.write(f"{param}: {stats[param]}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")
        
        print(f"Saved training statistics to: {stats_path}")

class EvaluationSession:
    """
    Manages an evaluation session for a trained RL agent.
    """
    
    def __init__(self, agent, agent_name="Agent", agent_type=None):
        self.agent = agent
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.start_time = None
        self.end_time = None
        self.games = []
        self.hyperparameters = {}  # may be filled by caller (train_agent)
        
    def run(self, layout, ghosts, display, num_games,
            max_steps=None, catchExceptions=False, timeout=30, randomRewards=False):
        """
        Run evaluation session.
        
        Returns:
            List of completed games
        """
        print(f"\n{'='*60}")
        print(f"Starting {self.agent_name} Evaluation Session")
        print(f"{'='*60}")
        print(f"Games: {num_games}")
        print(f"Layout: {layout.name if hasattr(layout, 'name') else 'Unknown'}")
        print(f"Ghosts: {len(ghosts)}")
        print(f"{'='*60}\n")
        
        self.start_time = time.perf_counter()
        
        self.games = runGamesQuiet(
            layout=layout,
            pacman=self.agent,
            ghosts=ghosts,
            display=display,
            numGames=num_games,
            record=False,
            numTraining=0,
            catchExceptions=catchExceptions,
            timeout=timeout,
            randomRewards=randomRewards,
            max_steps=max_steps
        )
        
        self.end_time = time.perf_counter()
        
        return self.games
    
    def report_results(self):
        """Print evaluation results and save to file."""
        if not self.games or self.start_time is None or self.end_time is None:
            print("No evaluation data to report!")
            return
        
        duration = self.end_time - self.start_time
        
        scores = [g.state.getScore() for g in self.games]
        wins = [g.state.isWin() for g in self.games]
        
        try:
            episode_lengths = [len(g.moveHistory) for g in self.games]
        except AttributeError:
            episode_lengths = []
        
        avg_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if scores else 0.0
        win_rate = (sum(wins) / len(wins)) if wins else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        total_eval_time = duration
        total_eval_time_min = int(total_eval_time // 60)
        eval_time_remaining_sec = int(total_eval_time % 60)
        
        print(f"\n{'='*60}")
        print(f"{self.agent_name} Evaluation Results")
        print(f"{'='*60}")
        print(f"Games Played: {len(self.games)}")
        print(f"Average Reward: {avg_score:.2f}")
        print(f"Score Std Dev: {std_score:.2f}")
        print(f"Win Rate: {win_rate:.2%} ({sum(wins)}/{len(wins)})")
        
        if episode_lengths:
            print(f"Average Episode Length: {avg_length:.2f} moves")
        
        print(f"Time taken to run {len(self.games)} evaluations: {total_eval_time:.2f} sec")
        print(f"Time taken (minutes): {total_eval_time_min}:{eval_time_remaining_sec:02d}")
        
        # If the agent exposes detailed eval stats (like QLearningAgent)
        eval_stats = None
        if hasattr(self.agent, "getEvalStats"):
            eval_stats = self.agent.getEvalStats()
            print(f"Total Actions: {eval_stats['total_actions']}")
            print(f"Q-table Actions: {eval_stats['qtable_actions']}")
            print(f"Random Actions: {eval_stats['random_actions']}")
            print(f"Percentage of Actions taken using Q-table: {eval_stats['qtable_percentage']:.2%}")
            print(f"Percentage of Random Actions Taken: {eval_stats['random_percentage']:.2%}")
            print(f"New States Encountered: {eval_stats['new_states_encountered']}")
        
        print(f"{'='*60}\n")
        
        results = {
            'num_games': len(self.games),
            'avg_score': avg_score,
            'std_score': std_score,
            'win_rate': win_rate,
            'wins': sum(wins),
            'avg_length': avg_length if episode_lengths else None,
            'duration': duration,
        }
        
        # Save evaluation results to file
        self._save_evaluation_stats(results, scores, wins, episode_lengths, eval_stats)
        
        return results
    
    def _save_evaluation_stats(self, results, scores, wins, episode_lengths, eval_stats=None):
        """Save detailed evaluation statistics to a text file."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_stats_{self.agent_name.lower().replace(' ', '_')}_{results['num_games']}_games_{timestamp}.txt"
        
        stats_path = get_output_path(filename, agent_type=self.agent_type)
        
        with open(stats_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"{self.agent_name} Evaluation Results\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Summary
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Agent Type: {self.agent_type}\n")
            f.write(f"Games Played: {results['num_games']}\n")
            f.write(f"Wins: {results['wins']}\n")
            f.write(f"Win Rate: {results['win_rate']:.2%}\n")
            f.write(f"Evaluation Runtime: {results['duration']:.2f} seconds ({results['duration']/60:.2f} minutes)\n")
            f.write("\n")
            
            # Hyperparameters / Config (if provided)
            if self.hyperparameters:
                f.write("HYPERPARAMETERS / CONFIGURATION\n")
                f.write("-" * 70 + "\n")
                for k, v in self.hyperparameters.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
            
            # Score statistics
            f.write("SCORE STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average Score: {results['avg_score']:.2f}\n")
            f.write(f"Std Dev Score: {results['std_score']:.2f}\n")
            f.write(f"Min Score: {min(scores):.2f}\n")
            f.write(f"Max Score: {max(scores):.2f}\n")
            f.write(f"Median Score: {np.median(scores):.2f}\n")
            f.write("\n")
            
            # Episode length stats
            if episode_lengths:
                f.write("EPISODE LENGTH STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Average Length: {results['avg_length']:.2f} moves\n")
                f.write(f"Min Length: {min(episode_lengths)} moves\n")
                f.write(f"Max Length: {max(episode_lengths)} moves\n")
                f.write(f"Median Length: {np.median(episode_lengths):.0f} moves\n")
                f.write("\n")
            
            # Q-learning specific evaluation stats (if available)
            if eval_stats is not None:
                f.write("Q-LEARNING EVALUATION STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total Actions: {eval_stats['total_actions']}\n")
                f.write(f"Q-table Actions: {eval_stats['qtable_actions']}\n")
                f.write(f"Random Actions: {eval_stats['random_actions']}\n")
                f.write(f"Q-table Action Percentage: {eval_stats['qtable_percentage']:.2%}\n")
                f.write(f"Random Action Percentage: {eval_stats['random_percentage']:.2%}\n")
                f.write(f"New States Encountered: {eval_stats['new_states_encountered']}\n")
                f.write("\n")
            
            # Individual game results
            f.write("INDIVIDUAL GAME RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Game':<6} {'Win':<6} {'Score':<10} {'Length':<10}\n")
            f.write("-" * 70 + "\n")
            for i, (score, win) in enumerate(zip(scores, wins), 1):
                win_str = "Yes" if win else "No"
                length_str = str(episode_lengths[i-1]) if episode_lengths else "N/A"
                f.write(f"{i:<6} {win_str:<6} {score:<10.2f} {length_str:<10}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")
        
        print(f"Saved evaluation statistics to: {stats_path}")
