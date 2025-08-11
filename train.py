import os
import time
import numpy as np
import torch
from td3 import TD3, ReplayBuffer
from car_env import CarThrottleEnv


def evaluate(env: CarThrottleEnv, agent: TD3, max_timesteps: int, episodes: int = 3):
    """Evaluate the policy without exploration noise."""
    returns = []
    fuels = []
    for _ in range(episodes):
        state, _ = env.reset()
        episode_return = 0.0
        for _ in range(max_timesteps):
            action = agent.select_action(np.array(state), add_noise=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
        fuels.append(env.fuel_usage)
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(fuels))


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
    max_episodes = 60
    # Allow enough steps to finish the ~6km route at dt=0.1s
    # At 25 m/s this is ~2400 steps; give extra margin
    max_timesteps = 3000
    # Sync environment max steps so episodes can finish by distance
    env.max_steps = max_timesteps

    batch_size = 256
    exploration_noise = 0.2  # policy adds its own noise; keep for select_action
    start_timesteps = 5000   # more random exploration for larger state
    save_frequency = 50      # Save models periodically
    
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
    best_eval_return = -np.inf
    
    for episode in range(1, max_episodes + 1):
        episode_reward = 0.0
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
        if episode % 5 == 0:
            last_n = min(10, len(episode_rewards))
            avg_reward = float(np.mean(episode_rewards[-last_n:]))
            elapsed_time = time.time() - start_time
            print(
                f"Episode {episode:4d} | AvgReward({last_n}): {avg_reward:.3f} | "
                f"Fuel(last ep): {env.fuel_usage:.4f} | Steps(last ep): {t+1} | Time: {elapsed_time:.1f}s"
            )

        # Periodic evaluation without noise and checkpointing
        if episode % save_frequency == 0:
            eval_mean, eval_std, eval_fuel = evaluate(env, agent, max_timesteps, episodes=3)
            print(
                f"Eval | Return: {eval_mean:.3f} ± {eval_std:.3f} | Fuel: {eval_fuel:.4f}"
            )
            agent.save(f"./models/td3_episode_{episode}")
            if eval_mean > best_eval_return:
                best_eval_return = eval_mean
                agent.save("./models/td3_best")
                print("Saved best model.")
    
    # Save final model
    agent.save("./models/td3_final")
    print("Training complete!")


if __name__ == "__main__":
    main()