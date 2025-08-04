import torch


class Config:
    def __init__(self):
        # Environment
        self.env_name = "Pendulum-v1"
        
        # Training
        self.max_timesteps = int(1e6)
        self.start_timesteps = 25e3
        self.batch_size = 256
        self.eval_freq = 5000
        self.max_episode_steps = 1000
        
        # TD3 hyperparameters
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        
        # Network
        self.hidden_dim = 256
        self.lr = 3e-4
        
        # Exploration
        self.expl_noise = 0.1
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"