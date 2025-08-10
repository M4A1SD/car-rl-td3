import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.interpolate import make_interp_spline
# import road


class CarThrottleEnv(gym.Env):
    def __init__(self):
        super(CarThrottleEnv, self).__init__()
        # Targets and limits
        self.target_speed = 25.0  # m/s
        self.max_speed = 40.0
        self.initial_route_distance = 6000.0  # meters
        self.dt = 0.1  # s
        self.max_steps = 500

        # Road profile (meters vs elevation)
        x = np.array([0,300,600,1000,1200,1600,1800,2000,2500,2700,3000,3100,3500,4000,4300,4600,5000,6000], dtype=float)
        y = np.array([0,20,20,80,80,100,100,80,80,60,80,80,120,20,30,20,20,60], dtype=float)
        self.spline = make_interp_spline(x, y)
        self.x_new = np.linspace(x.min(), x.max(), 300)  # ~20 m resolution
        self.y_new = self.spline(self.x_new)
        # Slope dy/dx; smooth and clamp via tanh for bounded features
        raw_slopes = np.gradient(self.y_new, self.x_new)  # rise/run
        self.road_slope = np.tanh(raw_slopes)  # in [-1,1]

        # Road preview configuration
        self.preview_distance = 2000.0  # meters ahead to preview
        self.preview_points = 20        # number of samples in preview vector

        # State variables
        self.speed = 0.0
        self.s = 0.0  # current distance along route [0, initial_route_distance]
        self.route_distance = self.initial_route_distance
        self.step_count = 0
        self.fuel_usage = 0.0  # total fuel used so far

        # Observation: [speed] + [current slope] + [preview slopes vector]
        self.obs_dim = 1 + 1 + self.preview_points
        low = np.concatenate((
            np.array([0.0]),                      # speed
            np.array([-1.0]),                     # current slope (tanh)
            -np.ones(self.preview_points)         # preview slopes
        )).astype(np.float32)
        high = np.concatenate((
            np.array([self.max_speed]),
            np.array([1.0]),
            np.ones(self.preview_points)
        )).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action: throttle in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Physics/consumption coefficients (simplified)
        self.throttle_accel_coeff = 4.0  # throttle [-1,1] -> accel contribution [-4,4] m/s^2
        self.slope_accel_scale = 3.8     # slope effect on accel via tanh slope in [-1,1] -> [-3,3]
        self.fuel_rate_base = 0.0
        self.fuel_rate_throttle = 0.015  # fuel per unit positive throttle
        self.fuel_rate_uphill = 0.012 #was 0.010    # extra fuel when uphill and throttling

        # Reward shaping weights: prioritize fuel, allow speed flexibility within a margin
        self.fuel_reward_scale = 1000.0  # scales per-step fuel to comparable magnitude
        self.speed_margin = 4.0          # you can go up to 15 km/h without consequences haha
        self.speed_penalty_weight = 10  # penalty per m/s outside margin (small)
        self.finish_bonus = 4000.0        # encourage finishing the route

    def _index_at_s(self, s_m):
        s_clamped = np.clip(s_m, 0.0, self.initial_route_distance)
        return int(np.searchsorted(self.x_new, s_clamped, side='left').clip(0, len(self.x_new)-1))

    def _road_preview(self):
        # Build a fixed-size vector of future slope samples over preview_distance
        if self.preview_points == 0:
            return np.zeros(0, dtype=np.float32)
        dists = np.linspace(0.0, self.preview_distance, self.preview_points+1)[1:]  # exclude 0
        idxs = [self._index_at_s(self.s + d) for d in dists]
        preview = self.road_slope[idxs]
        return preview.astype(np.float32)

    def _obs(self):
        idx = self._index_at_s(self.s)
        current_slope = self.road_slope[idx]
        preview = self._road_preview()
        obs = np.concatenate(([self.speed], [current_slope], preview)).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.speed = 0.0
        self.s = 0.0
        self.route_distance = self.initial_route_distance
        self.step_count = 0
        self.fuel_usage = 0.0
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        throttle = float(np.clip(action[0], -1.0, 1.0))

        # Road-induced accel from current slope
        idx = self._index_at_s(self.s)
        slope_tanh = self.road_slope[idx]  # [-1,1], positive = uphill, negative = downhill
        # Uphill should reduce acceleration; downhill should increase it -> flip sign
        external_acc = -slope_tanh * self.slope_accel_scale  # [-3,3]

        # Vehicle acceleration: engine/brake contribution plus gravity along slope
        accel = throttle * self.throttle_accel_coeff + external_acc  # approx accel range [-7.8, 7.8]

        # Fuel usage (incremental)
        # Only positive throttle burns fuel; more uphill (positive slope_tanh) burns extra
        pos_throttle = max(0.0, throttle) # dont burn fuel when braking
        fuel_rate = self.fuel_rate_base \
                    + self.fuel_rate_throttle * pos_throttle \
                    + self.fuel_rate_uphill * pos_throttle * max(0.0, slope_tanh)
        # fuel_rate = 0.015 * throttle + 0.01 * slope_tanh * throttle
        # up hill
        # fuel_rate = throttle ( 0.015 + 0.01 * slope_tanh)
        # down hill
        # fuel_rate = throttle ( 0.015 )
        # fuel_rate  [+- 0.315]
        fuel_used = fuel_rate * self.dt
        self.fuel_usage += fuel_used

        # Kinematics
        v0 = self.speed
        self.speed = np.clip(self.speed + accel * self.dt, 0.0, self.max_speed)

        # Distance advance
        distance_covered = max(0.0, v0 * self.dt + 0.5 * accel * self.dt ** 2)
        self.s += distance_covered
        self.route_distance = max(0.0, self.initial_route_distance - self.s)

        # Reward: strongly penalize fuel, lightly penalize speed only outside a margin
        speed_error = abs(self.speed - self.target_speed)
        reward = 0.0
        reward -= self.fuel_reward_scale * fuel_used
        excess = max(0.0, speed_error - self.speed_margin)
        # 2 levels of excess
        if speed_error < 1.0: # [24,26]
            reward -= self.speed_penalty_weight * excess * 0.1 
        elif speed_error < 3.0: # [22,28]
            reward -= self.speed_penalty_weight * excess * 0.4
        else: # [0,22] [28,40]
            reward -= self.speed_penalty_weight * excess 

        # Termination
        terminated = False
        truncated = False
        if self.route_distance <= 0.0:
            terminated = True
            # Bonus for finishing the route
            reward += self.finish_bonus
        elif self.step_count >= self.max_steps:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, {}

    def render(self, mode="human"):
        idx = self._index_at_s(self.s)
        print(f"s={self.s:.1f} m  Speed={self.speed:.2f} m/s  Slope(tanh)={self.road_slope[idx]:.3f}  Fuel={self.fuel_usage:.4f}")



