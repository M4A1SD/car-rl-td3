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
    max_episodes = 150
    max_timesteps = 500
    batch_size = 256
    exploration_noise = 0.2  # Increased exploration
    start_timesteps = 2000  # More random exploration steps
    save_frequency = 50  # Save models more frequently
    
    # Initialize agent and replay buffer
    agent = TD3(state_dim, action_dim, max_action, device=device)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Create directory for saving models
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Training loop
    episode_rewards = []
    total_timesteps = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(1, max_episodes + 1):
        episode_reward = 0
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
            
            # Store data in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train agent after collecting enough data
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print episode statistics
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode}, Avg. Reward: {avg_reward:.2f}, Time: {elapsed_time:.2f}s")
            
            # Save model periodically
            if episode % save_frequency == 0:
                agent.save(f"./models/td3_episode_{episode}")
                print(f"Model saved at episode {episode}")
    
    # Save final model
    agent.save("./models/td3_final")
    print("Training complete!")


if __name__ == "__main__":
    main()