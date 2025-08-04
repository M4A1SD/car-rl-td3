import gym
import numpy as np


class EnvironmentWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        # TODO: Initialize environment
        pass
    
    def reset(self):
        # TODO: Reset environment and return initial state
        pass
    
    def step(self, action):
        # TODO: Execute action and return (next_state, reward, done, info)
        pass
    
    def render(self):
        # TODO: Render environment
        pass
    
    def close(self):
        # TODO: Close environment
        pass
    
    @property
    def action_space(self):
        # TODO: Return action space
        pass
    
    @property
    def observation_space(self):
        # TODO: Return observation space
        pass