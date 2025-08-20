# Car Throttle Control with TD3 Reinforcement Learning

A reinforcement learning project that trains an intelligent car to maintain optimal speed while driving over hilly terrain using the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.

## 🎯 Objective

Train an AI agent to control a car's throttle to:
- Maintain target speed of **25 m/s** 
- Navigate through varying road slopes (hills and valleys)
- Optimize fuel efficiency through smart momentum-based driving
- Complete a 1500-meter route successfully

## 🚗 Key Features

- **Realistic Physics**: Car dynamics with acceleration, momentum, and road slope effects
- **Complex Environment**: Hilly road profile with elevation changes up to 80 meters
- **Smart Rewards**: Balances speed control, fuel efficiency, and route completion
- **Advanced Algorithm**: TD3 with twin critics for stable continuous control
- **Comprehensive Evaluation**: Performance analysis, speed trajectories, and model comparison

## 🛠️ Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy>=1.21.0 gymnasium matplotlib>=3.5.0

# Or use requirements file
pip install -r requirements.txt
```

## 🚀 Usage

### Training
```bash
python train.py
```

### Evaluation
```bash
# Evaluate final model
python evaluate.py 
```

### Plotting Trajectories
```bash
python plot_speed_throttle_trajectory.py
```

## 📁 Project Structure

- `car_env.py` - Car environment with physics and road interaction
- `road.py` - Road elevation profile generator 
- `td3.py` - TD3 algorithm implementation (Actor-Critic networks)
- `train.py` - Training script with hyperparameters
- `evaluate.py` - Model evaluation and comparison tools
- `plot_speed_throttle_trajectory.py` - Visualization utilities

## 🎛️ Key Parameters

- **Target Speed**: 25 m/s
- **Route Length**: 1500 meters  
- **Max Episodes**: 1000
- **Road Profile**: Single hill with 80m peak height
- **Fuel Penalty**: Encourages efficient throttle usage
- **Success Criteria**: Finish within 2 m/s of target speed

## 📊 Results

The trained agent learns to:
- Achieve consistent target speed with minimal fuel consumption
- Successfully complete the challenging hilly route

## 🔧 Environment Details

- **State Space**: Current speed + forward-looking slope information
- **Action Space**: Continuous throttle control [-1, 1] 
- **Reward Function**: Speed tracking + fuel efficiency + completion bonus
- **Physics**: Realistic acceleration, momentum, and slope effects

---

*Built with PyTorch, Gymnasium, and TD3 reinforcement learning*
