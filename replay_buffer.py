import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # TODO: Initialize buffer arrays
        pass
    
    def add(self, state, action, next_state, reward, done):
        # TODO: Add experience to buffer
        pass
    
    def sample(self, batch_size):
        # TODO: Sample batch from buffer
        pass