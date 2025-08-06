import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CarThrottleEnv(gym.Env):
    def __init__(self):
        super(CarThrottleEnv, self).__init__()
        self.target_speed = 25.0  # m/s
        self.speed = 0.0
        self.max_speed = 40.0
        self.initial_route_distance = 1000
        self.route_distance = self.initial_route_distance
        self.dt = 0.1  # time per step in seconds
        self.max_steps = 500  # Maximum steps per episode

        self.observation_space = spaces.Box(low=0, high=self.max_speed, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.speed = 0.0
        self.route_distance = self.initial_route_distance
        self.step_count = 0
        return np.array([self.speed], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        throttle = np.clip(action[0], -1.0, 1.0)
        acceleration = throttle * 10 - 2  # Faster dynamics: -2=max decel, 0=drag only, 8=max accel
        v0 = self.speed
        self.speed += acceleration * self.dt
        self.speed = np.clip(self.speed, 0.0, self.max_speed)  # Prevent negative speeds

        # Calculate distance covered in this step
        distance_covered = max(0, v0 * self.dt + 0.5 * acceleration * self.dt ** 2)
        self.route_distance -= distance_covered

        # Calculate reward based on speed tracking performance
        speed_error = abs(self.speed - self.target_speed)
        
        # Base reward for maintaining target speed (higher is better)
        if speed_error < 1.0:  # Very close to target
            reward = 10.0 - speed_error
        elif speed_error < 3.0:  # Reasonably close
            reward = 5.0 - speed_error  
        else:  # Far from target
            reward = -speed_error * 2.0
        
        # Small penalty for large throttle changes (encourage smooth control)
        reward -= 0.1 * abs(throttle)
        
       
        terminated = False # reached destination
        truncated = False # reached max steps limit, but not the destination
        
        if self.route_distance <= 0:
            terminated = True
            # Bonus for completing route near target speed
            if speed_error < 2.0:
                reward += 50.0
        elif self.step_count >= self.max_steps:
            truncated = True

        return np.array([self.speed], dtype=np.float32), reward, terminated, truncated, {}

    def render(self, mode="human"):
        print(f"Speed: {self.speed:.2f} m/s")



