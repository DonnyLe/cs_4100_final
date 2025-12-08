"""
train_agent.py - Unified training/evaluation script for Pacman RL agents

Usage:
    # Train Q-learning agent
    python train_agent.py --agent qlearning --train --episodes 25000

    # Evaluate Q-learning agent (100 games, stats only)
    python train_agent.py --agent qlearning --eval --load-model Q_table_25000.pickle --episodes 100

    # Watch Q-learning agent play with GUI
    python train_agent.py --agent qlearning --eval --load-model Q_table_25000.pickle --episodes 1 --gui

    # Train DQN agent
    python train_agent.py --agent dqn --train --episodes 5000

    # Watch DQN agent play
    python train_agent.py --agent dqn --eval --load-model dqn_model.pt --episodes 1 --gui --frame-time 0.05
"""

import sys
import argparse
from deep_q_learning.deep_q_learning import DeepQLearningAgent
import layout as pac_layout
import ghostAgents
from q_learning.q_learning import QLearningAgent
from training_utils import TrainingSession, EvaluationSession, get_output_path
from textDisplay import NullGraphics
from graphicsDisplay import PacmanGraphics


def parse_args():
    """Parse command line arguments for training/evaluation."""
    parser = argparse.ArgumentParser(
        description='Train or evaluate Pacman RL agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true',
                           help='Run in training mode')
    mode_group.add_argument('--eval', action='store_true',
                           help='Run in evaluation mode')
    
    # Agent selection
    parser.add_argument('--agent', type=str, required=True,
                       choices=['qlearning', 'dqn'],
                       help='Which RL agent to use')
    
    # Game configuration
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Pacman layout to use (default: mediumClassic)')
    parser.add_argument('--ghosts', type=int, default=2,
                       help='Number of ghosts (default: 2)')
    parser.add_argument('--ghost-type', type=str, default='RandomGhost',
                       help='Ghost agent type (default: RandomGhost)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to run (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode (default: 200)')
    
    # Display
    parser.add_argument('--gui', action='store_true',
                       help='Enable graphical display (useful for watching agent play)')
    parser.add_argument('--frame-time', type=float, default=0.001,
                       help='Frame time for GUI in seconds - higher = slower, easier to watch (default: 0.001)')
    
    # Model I/O
    parser.add_argument('--load-model', type=str, default=None,
                       help='Load model from file (required for --eval)')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Save model to file (default: auto-generate name)')
    
    # Agent-specific hyperparameters
    parser.add_argument('--gamma', type=float, default=0.9,
                       help='Discount factor (default: 0.9)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--decay-rate', type=float, default=0.99995,
                       help='Epsilon decay rate (default: 0.99995)')
    parser.add_argument('--min-epsilon', type=float, default=0.05,
                       help='Minimum exploration rate (default: 0.05)')
    parser.add_argument('--window-size', type=int, default=2,
                       help='Observation window size (Q-learning only, default: 2)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (DQN only, default: 0.001)')
    
    # Reward shaping toggle (DQN only)
    parser.add_argument(
        '--no-reward-shaping',
        action='store_true',
        help='Disable DQN reward shaping (default: shaping enabled)'
    )
    
    # Other options
    parser.add_argument('--random-rewards', action='store_true',
                       help='Enable random pellet rewards')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating training plots')
    
    return parser.parse_args()


def create_agent(args):
    """Create and configure the appropriate agent based on arguments."""
    if args.agent == 'qlearning':
        
        # Generate default filename if not provided
        save_file = args.save_model
        if args.train and save_file is None:
            save_file = f'Q_table_{args.episodes}_{args.decay_rate}_{args.window_size}.pickle'
        
        agent = QLearningAgent(
            window_size=args.window_size,
            gamma=args.gamma,
            epsilon=args.epsilon,
            decay_rate=args.decay_rate,
            min_epsilon=args.min_epsilon,
            qfile=save_file,
            load_qtable=args.load_model is not None
        )
        
        if args.load_model:
            agent.qfile = args.load_model
            agent._load_qtable()
        
        # Store hyperparameters for logging
        hyperparameters = {
            'gamma': args.gamma,
            'epsilon': args.epsilon,
            'decay_rate': args.decay_rate,
            'min_epsilon': args.min_epsilon,
            'window_size': args.window_size,
            'layout': args.layout,
            'num_ghosts': args.ghosts,
            'ghost_type': args.ghost_type,
            'max_steps': args.max_steps,
            'random_rewards': args.random_rewards,
        }
        
        return agent, save_file, hyperparameters, 'qlearning'
    
    elif args.agent == 'dqn':
        # Decide where to save
        save_file = args.save_model

        if args.train:
            # If no explicit save path, prefer continuing file or default name
            if save_file is None:
                save_file = args.load_model or f'dqn_model_{args.episodes}.pt'
        else:
            # In eval mode we don't necessarily need a save file
            save_file = args.save_model  # usually None is fine

        # Where to load from (if any)
        load_path = args.load_model

        agent = DeepQLearningAgent(
            qfile=save_file,                     # where to SAVE
            load_model=load_path is not None,   # whether to LOAD
            load_path=load_path,                # where to LOAD FROM
            use_reward_shaping=not args.no_reward_shaping,
        )

        # Override hyperparameters if provided
        agent.learning_rate = args.learning_rate
        agent.discount_factor = args.gamma
        agent.initial_epsilon = args.epsilon
        agent.epsilon = args.epsilon
        agent.min_epsilon = args.min_epsilon
        agent.epsilon_decay = args.decay_rate

        # Store hyperparameters for logging
        hyperparameters = {
            'gamma': args.gamma,
            'learning_rate': args.learning_rate,
            'epsilon': args.epsilon,
            'decay_rate': args.decay_rate,
            'min_epsilon': args.min_epsilon,
            'batch_size': agent.batch_size,
            'discount_factor': agent.discount_factor,
            'target_sync_steps': agent.target_sync_steps,
            'replay_memory_size': agent.replay_memory_size,
            'layout': args.layout,
            'num_ghosts': args.ghosts,
            'ghost_type': args.ghost_type,
            'max_steps': args.max_steps,
            'random_rewards': args.random_rewards,
            'use_reward_shaping': agent.use_reward_shaping,
        }

        return agent, save_file, hyperparameters, 'dqn'

    
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")


def main():
    """Main entry point for training/evaluation."""
    args = parse_args()
    
    # Setup display
    if args.gui:
        display = PacmanGraphics(frameTime=args.frame_time)
    else:
        display = NullGraphics()
    
    # Load layout
    layout = pac_layout.getLayout(args.layout)
    if layout is None:
        print(f"Error: Could not load layout '{args.layout}'")
        sys.exit(1)
    
    # Create ghosts
    try:
        ghost_cls = getattr(ghostAgents, args.ghost_type)
        ghosts = [ghost_cls(i + 1) for i in range(args.ghosts)]
    except AttributeError:
        print(f"Error: Unknown ghost type '{args.ghost_type}'")
        sys.exit(1)
    
    # Create agent
    agent, model_file, hyperparameters, agent_type = create_agent(args)
    agent_name = f"{args.agent.upper()} Agent"
    
    # Training mode
    if args.train:
        agent.setTraining(True)
        
        session = TrainingSession(agent, agent_name, agent_type=agent_type)
        session.hyperparameters = hyperparameters  # Store hyperparameters for logging
        session.run(
            layout=layout,
            ghosts=ghosts,
            display=display,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            randomRewards=args.random_rewards
        )
        
        # Save model
        if model_file:
            if hasattr(agent, '_save_qtable'):
                agent._save_qtable()
                print(f"Saved Q-table to: {get_output_path(model_file, agent_type=agent_type)}")
            elif hasattr(agent, '_save_model'):
                agent._save_model()
                print(f"Saved model to: {get_output_path(model_file, agent_type=agent_type)}")
        
        # Report results
        plot_kwargs = {
            'decay_rate': args.decay_rate,
        }
        if args.agent == 'qlearning':
            plot_kwargs['window_size'] = args.window_size
        
        session.report_results(save_plot=not args.no_plot, **plot_kwargs)
    
    # Evaluation mode
    elif args.eval:
        if args.load_model is None:
            print("Error: --load-model required for evaluation mode")
            sys.exit(1)
        
        agent.setTraining(False)
        session = EvaluationSession(agent, agent_name, agent_type=agent_type)
        session.hyperparameters = hyperparameters  # Store hyperparameters for logging
        session.run(
            layout=layout,
            ghosts=ghosts,
            display=display,
            num_games=args.episodes,
            max_steps=args.max_steps,
            randomRewards=args.random_rewards
        )
        
        session.report_results()


if __name__ == '__main__':
    main()
