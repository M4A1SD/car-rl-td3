import torch
import numpy as np
from td3 import TD3, ReplayBuffer
from car_env import CarThrottleEnv
import gymnasium as gym
import os
import time


def main():
    # Set device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Create environment
    env = CarThrottleEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Training parameters
    max_episodes = 1000
    max_timesteps = 1000
    batch_size = 256
    exploration_noise = 0.2  # Note: policy adds its own noise; this var is informational only
    start_timesteps = 2500  # More random exploration given large observation size
    save_frequency = 25  # Save models more frequently
    
    # Initialize agent and replay buffer
    agent = TD3(state_dim, action_dim, max_action, device=device)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Create directory for saving models
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Training loop
    episode_rewards = []
    episode_avg_speeds = []  # Track average speeds for each episode
    total_timesteps = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(1, max_episodes + 1):
        episode_reward = 0
        episode_speeds = []  # Track speeds for this episode
        state, _ = env.reset()
        
        for t in range(max_timesteps):
            total_timesteps += 1
            
            # Select action according to policy or random for exploration
            if total_timesteps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state), add_noise=True)
            
            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Track speed (first element of state is speed)
            current_speed = state[0]
            episode_speeds.append(current_speed)
            
            # Store data in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train agent after collecting enough data  
            # Train every 2 steps to allow more diverse experience collection for momentum learning
            if len(replay_buffer) > batch_size and total_timesteps % 2 == 0:
                agent.train(replay_buffer, batch_size)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Calculate and store average speed for this episode
        if episode_speeds:
            avg_speed_episode = np.mean(episode_speeds)
            episode_avg_speeds.append(avg_speed_episode)
        else:
            episode_avg_speeds.append(0.0)
        
        # Print total fuel used for this episode
        
        # Print episode statistics
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_speed_recent = np.mean(episode_avg_speeds[-10:])
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode}, Avg. Reward: {avg_reward:.2f}, Avg. Speed: {avg_speed_recent:.2f} m/s, Time: {elapsed_time:.2f}s, Total Fuel Used: {env.total_fuel_used:.3f}")
            
            # Save model periodically
            if episode % save_frequency == 0:
                agent.save(f"./models/td3_episode_{episode}")
                print(f"Model saved at episode {episode}")
    
    # Save final model
    agent.save("./models/td3_final")
    
    # Print final training summary
    final_avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    final_avg_speed = np.mean(episode_avg_speeds[-10:]) if len(episode_avg_speeds) >= 10 else np.mean(episode_avg_speeds)
    print("Training complete!")
    print(f"Final 10-episode average reward: {final_avg_reward:.2f}")
    print(f"Final 10-episode average speed: {final_avg_speed:.2f} m/s (Target: {env.target_speed:.1f} m/s)")


if __name__ == "__main__":
    main()