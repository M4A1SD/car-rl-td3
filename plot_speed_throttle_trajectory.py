import torch
import numpy as np
import matplotlib.pyplot as plt
from td3 import TD3
from car_env import CarThrottleEnv

def plot_speed_throttle_trajectory(model_name="td3_final", num_episodes=3):
    """
    Plot speed and throttle actions together over time during episodes.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CarThrottleEnv()
    
    # Load model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(state_dim, action_dim, max_action, device=device)
    agent.load(f"./models/{model_name}")
    print(f"Loaded model: {model_name}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Speed and Throttle Trajectory Analysis - {model_name}', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    all_speeds = []
    all_throttles = []
    all_times = []
    
    # Run multiple episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        speeds = [state[0]]
        throttles = []
        times = [0]
        rewards = []
        
        step = 0
        while True:
            # Get action from policy
            action = agent.select_action(np.array(state), add_noise=False)
            throttle = action[0]
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            step += 1
            speeds.append(next_state[0])
            throttles.append(throttle)
            times.append(step * env.dt)  # Convert to seconds
            rewards.append(reward)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Store for combined analysis
        all_speeds.extend(speeds)
        all_throttles.extend(throttles + [throttles[-1]])  # Match length
        all_times.extend([t + episode * max(times) for t in times])
        
        color = colors[episode % len(colors)]
        
        # Plot 1: Speed over time
        axes[0, 0].plot(times, speeds, color=color, linewidth=2, alpha=0.8, 
                       label=f'Episode {episode + 1}')
        
        # Plot 2: Throttle over time
        axes[0, 1].plot(times[:-1], throttles, color=color, linewidth=2, alpha=0.8,
                       label=f'Episode {episode + 1}')
        
        print(f"Episode {episode + 1}: {len(speeds)} steps, final speed: {speeds[-1]:.2f} m/s")
    
    # Plot 1: Speed trajectories
    axes[0, 0].axhline(y=env.target_speed, color='red', linestyle='--', alpha=0.7, 
                      label=f'Target ({env.target_speed} m/s)')
    axes[0, 0].axhspan(env.target_speed - 2, env.target_speed + 2, alpha=0.2, color='green',
                      label='Success Zone (±2 m/s)')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Speed (m/s)')
    axes[0, 0].set_title('Speed Trajectories')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Throttle trajectories
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No Throttle')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Throttle Action')
    axes[0, 1].set_title('Throttle Actions')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Speed vs Throttle (phase plot)
    # Take first episode for cleaner visualization
    state, _ = env.reset()
    episode_speeds = [state[0]]
    episode_throttles = []
    
    step = 0
    while True:
        action = agent.select_action(np.array(state), add_noise=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        episode_speeds.append(next_state[0])
        episode_throttles.append(action[0])
        
        state = next_state
        step += 1
        
        if terminated or truncated:
            break
    
    # Color by time progression
    time_colors = np.linspace(0, 1, len(episode_throttles))
    scatter = axes[1, 0].scatter(episode_speeds[:-1], episode_throttles, c=time_colors, 
                                cmap='viridis', s=20, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=env.target_speed, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Speed (m/s)')
    axes[1, 0].set_ylabel('Throttle Action')
    axes[1, 0].set_title('Speed vs Throttle (colored by time)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Time Progress')
    
    # Plot 4: Road elevation and throttle over distance
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()

    # Road elevation profile
    road_x = env.road.x
    road_y = env.road.y
    line1 = ax1.plot(road_x, road_y, color='tab:blue', linewidth=2, label='Road Elevation')
    ax1.set_xlabel('Distance along road (m)')
    ax1.set_ylabel('Elevation (m)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    # Roll one episode to collect throttle vs distance
    state, _ = env.reset()
    throttles = []
    positions = [env.position]
    while True:
        action = agent.select_action(np.array(state), add_noise=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        throttles.append(action[0])
        positions.append(env.position)
        state = next_state
        if terminated or truncated:
            break

    # Plot throttle against traveled distance (right axis)
    # Use positions[1:] to align with throttle actions taken at each step
    line2 = ax2.plot(positions[1:], throttles, color='tab:red', linewidth=2, label='Throttle')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Throttle Action', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    ax1.set_title('Road Elevation and Throttle vs Distance')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'speed_throttle_trajectory_{model_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Speed-throttle trajectory plot saved as {plot_filename}")
    
    # Print analysis
    print(f"\n=== Trajectory Analysis ===")
    final_speed = speeds[-1]
    print(f"Final speed: {final_speed:.2f} m/s (target: {env.target_speed} m/s)")
    print(f"Speed error: {abs(final_speed - env.target_speed):.2f} m/s")
    print(f"Episode duration: {times[-1]:.1f} seconds")
    print(f"Average throttle: {np.mean(throttles):.3f}")
    print(f"Throttle range: {np.min(throttles):.3f} to {np.max(throttles):.3f}")
    
    return speeds, throttles, times

def plot_single_episode_detailed(model_name="td3_final"):
    """
    Detailed plot of a single episode showing speed, throttle, and reward.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CarThrottleEnv()
    
    # Load model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(state_dim, action_dim, max_action, device=device)
    agent.load(f"./models/{model_name}")
    
    # Run one episode
    state, _ = env.reset()
    speeds = [state[0]]
    throttles = []
    rewards = []
    times = [0]
    
    step = 0
    total_reward = 0
    
    while True:
        action = agent.select_action(np.array(state), add_noise=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        step += 1
        speeds.append(next_state[0])
        throttles.append(action[0])
        rewards.append(reward)
        times.append(step * env.dt)
        total_reward += reward
        
        state = next_state
        
        if terminated or truncated:
            break
    
    # Create detailed plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Detailed Episode Analysis - {model_name}', fontsize=16)
    
    # Speed plot
    axes[0].plot(times, speeds, 'b-', linewidth=2, label='Actual Speed')
    axes[0].axhline(y=env.target_speed, color='red', linestyle='--', alpha=0.7, 
                   label=f'Target Speed ({env.target_speed} m/s)')
    axes[0].axhspan(env.target_speed - 2, env.target_speed + 2, alpha=0.2, color='green',
                   label='Success Zone (±2 m/s)')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title('Speed Trajectory')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Throttle plot
    axes[1].plot(times[:-1], throttles, 'r-', linewidth=2, label='Throttle Action')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7, label='No Throttle')
    axes[1].set_ylabel('Throttle Action')
    axes[1].set_title('Throttle Commands')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Reward plot
    axes[2].plot(times[:-1], rewards, 'g-', linewidth=2, label='Instantaneous Reward')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Reward')
    axes[2].set_title(f'Rewards (Total: {total_reward:.2f})')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save detailed plot
    plot_filename = f'detailed_episode_{model_name}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Detailed episode plot saved as {plot_filename}")
    
    return speeds, throttles, rewards, times

if __name__ == "__main__":
    print("Generating speed-throttle trajectory plots...")
    
    # Multi-episode trajectory analysis
    speeds, throttles, times = plot_speed_throttle_trajectory("td3_final", num_episodes=3)
    
    # Detailed single episode analysis
    print("\nGenerating detailed single episode analysis...")
    plot_single_episode_detailed("td3_final")
    
    plt.show()