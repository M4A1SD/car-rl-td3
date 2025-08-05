import torch
import numpy as np
import matplotlib.pyplot as plt
from td3 import TD3
from car_env import CarThrottleEnv
import os
import argparse
from typing import List, Tuple, Dict

def evaluate_model(agent: TD3, env: CarThrottleEnv, num_episodes: int = 10, render: bool = False) -> Tuple[List[float], Dict]:
    """
    Evaluate a trained TD3 agent on the car throttle environment.
    
    Args:
        agent: Trained TD3 agent
        env: Car throttle environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        Tuple of (episode_rewards, statistics)
    """
    episode_rewards = []
    episode_lengths = []
    final_speeds = []
    speed_histories = []
    action_histories = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        speed_history = []
        action_history = []
        
        while True:
            # Select action without exploration noise for evaluation
            action = agent.select_action(np.array(state), add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            speed_history.append(env.speed)
            action_history.append(action[0])
            
            if render:
                env.render()
            
            state = next_state
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_speeds.append(env.speed)
        speed_histories.append(speed_history)
        action_histories.append(action_history)
        
        if render or episode % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}, Final Speed = {env.speed:.2f} m/s")
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_final_speed': np.mean(final_speeds),
        'std_final_speed': np.std(final_speeds),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'success_rate': sum(1 for speed in final_speeds if abs(speed - env.target_speed) < 2.0) / len(final_speeds)
    }
    
    return episode_rewards, stats, speed_histories, action_histories

def plot_evaluation_results(episode_rewards: List[float], speed_histories: List[List[float]], 
                          action_histories: List[List[float]], model_name: str, target_speed: float = 25.0):
    """
    Plot evaluation results including rewards, speed trajectories, and actions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Evaluation Results for {model_name}', fontsize=16)
    
    # Plot episode rewards
    axes[0, 0].plot(episode_rewards, 'b-', linewidth=2)
    axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot speed trajectories (first 5 episodes)
    axes[0, 1].axhline(y=target_speed, color='r', linestyle='--', alpha=0.7, label=f'Target Speed: {target_speed} m/s')
    for i, speed_history in enumerate(speed_histories[:5]):
        axes[0, 1].plot(speed_history, alpha=0.7, label=f'Episode {i+1}')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].set_title('Speed Trajectories (First 5 Episodes)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot action trajectories (first 5 episodes)
    for i, action_history in enumerate(action_histories[:5]):
        axes[1, 0].plot(action_history, alpha=0.7, label=f'Episode {i+1}')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Throttle Action')
    axes[1, 0].set_title('Throttle Actions (First 5 Episodes)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot reward distribution
    axes[1, 1].hist(episode_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(episode_rewards), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'evaluation_results_{model_name.replace("/", "_").replace(".", "_")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Evaluation plots saved as {plot_filename}")
    
    return fig

def compare_models(model_names: List[str], num_episodes: int = 10):
    """
    Compare multiple models and create a comparison plot.
    """
    env = CarThrottleEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        
        # Initialize agent and load model
        agent = TD3(state_dim, action_dim, max_action, device=device)
        
        try:
            agent.load(f"./models/{model_name}")
            episode_rewards, stats, _, _ = evaluate_model(agent, env, num_episodes)
            results[model_name] = {
                'rewards': episode_rewards,
                'stats': stats
            }
            
            print(f"Results for {model_name}:")
            print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
            print(f"  Mean Final Speed: {stats['mean_final_speed']:.2f} ± {stats['std_final_speed']:.2f} m/s")
            
        except FileNotFoundError:
            print(f"Model {model_name} not found, skipping...")
            continue
    
    # Create comparison plot
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Comparison', fontsize=16)
        
        model_names_found = list(results.keys())
        mean_rewards = [results[name]['stats']['mean_reward'] for name in model_names_found]
        success_rates = [results[name]['stats']['success_rate'] for name in model_names_found]
        final_speeds = [results[name]['stats']['mean_final_speed'] for name in model_names_found]
        
        # Mean rewards comparison
        axes[0].bar(range(len(model_names_found)), mean_rewards, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Mean Reward Comparison')
        axes[0].set_xticks(range(len(model_names_found)))
        axes[0].set_xticklabels([name.replace('td3_episode_', 'Ep ').replace('td3_final', 'Final') for name in model_names_found], rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Success rate comparison
        axes[1].bar(range(len(model_names_found)), success_rates, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate Comparison')
        axes[1].set_xticks(range(len(model_names_found)))
        axes[1].set_xticklabels([name.replace('td3_episode_', 'Ep ').replace('td3_final', 'Final') for name in model_names_found], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Final speed comparison
        target_line = axes[2].axhline(y=env.target_speed, color='r', linestyle='--', alpha=0.7, label=f'Target: {env.target_speed} m/s')
        axes[2].bar(range(len(model_names_found)), final_speeds, alpha=0.7, color='orange', edgecolor='black')
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('Mean Final Speed (m/s)')
        axes[2].set_title('Final Speed Comparison')
        axes[2].set_xticks(range(len(model_names_found)))
        axes[2].set_xticklabels([name.replace('td3_episode_', 'Ep ').replace('td3_final', 'Final') for name in model_names_found], rotation=45)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved as model_comparison.png")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate TD3 Car Throttle Control Models')
    parser.add_argument('--model', type=str, default='td3_final', 
                       help='Model name to evaluate (default: td3_final)')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--render', action='store_true', 
                       help='Render the environment during evaluation')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.compare:
        # Compare all available models
        available_models = []
        models_dir = "./models"
        
        if os.path.exists(models_dir):
            # Find all unique model names (remove _actor, _critic, _optimizer suffixes)
            model_files = os.listdir(models_dir)
            model_names = set()
            for file in model_files:
                if file.endswith('_actor'):
                    model_names.add(file[:-6])  # Remove '_actor'
            
            # Sort models by episode number
            episode_models = [name for name in model_names if 'episode' in name]
            episode_models.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
            
            final_models = [name for name in model_names if 'final' in name]
            
            available_models = episode_models + final_models
            
            print(f"Found {len(available_models)} models to compare")
            compare_models(available_models, args.episodes)
        else:
            print("Models directory not found!")
            return
    
    else:
        # Evaluate single model
        env = CarThrottleEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Initialize agent
        agent = TD3(state_dim, action_dim, max_action, device=device)
        
        # Load model
        model_path = f"./models/{args.model}"
        try:
            agent.load(model_path)
            print(f"Successfully loaded model: {args.model}")
        except FileNotFoundError:
            print(f"Model {args.model} not found!")
            return
        
        # Evaluate model
        print(f"\nEvaluating model {args.model} for {args.episodes} episodes...")
        episode_rewards, stats, speed_histories, action_histories = evaluate_model(
            agent, env, args.episodes, args.render
        )
        
        # Print results
        print(f"\n=== Evaluation Results for {args.model} ===")
        print(f"Episodes: {args.episodes}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"Mean Final Speed: {stats['mean_final_speed']:.2f} ± {stats['std_final_speed']:.2f} m/s")
        print(f"Target Speed: {env.target_speed} m/s")
        print(f"Success Rate (within 2 m/s of target): {stats['success_rate']:.2%}")
        
        # Generate plots if requested
        if args.plot:
            plot_evaluation_results(episode_rewards, speed_histories, action_histories, args.model, env.target_speed)

if __name__ == "__main__":
    main()
