import gymnasium as gym
from gymnasium import spaces
import numpy as np
from road import RoadProfile

class CarThrottleEnv(gym.Env):
    def __init__(self):
        super(CarThrottleEnv, self).__init__()
        self.target_speed = 25.0  # m/s
        self.speed = 0.0
        self.position = 0.0  # meters along the road
        self.max_speed = 40.0
        self.dt = 0.1  # time per step in seconds
        self.max_steps = 1000  # Maximum steps per episode (aligns better with route length)

        # Road profile providing external acceleration from slope
        self.road = RoadProfile(total_length=1500.0, num_points=1500, peak_height=80.0, bump_width_fraction=0.4)

        # Route distance aligned with road length
        self.initial_route_distance = float(self.road.total_length)
        self.route_distance = self.initial_route_distance
        self.slope_map = self.road.get_slope_map()  # tanh-normalized slopes in [-1, 1]
        self.num_slope_points = len(self.slope_map)
        # How many future meters of slope to expose (window ahead). Set to full route to "see" all incoming slopes
        self.lookahead_meters = self.road.total_length  # expose entire remaining route by default
        

        # Observation: speed followed by an aligned, forward-looking slope vector
        # Speed is in [0, max_speed]; slopes are in [-1, 1]
        low = np.concatenate(([0.0], np.full(self.num_slope_points, -1.0, dtype=np.float32))).astype(np.float32)
        high = np.concatenate(([self.max_speed], np.full(self.num_slope_points, 1.0, dtype=np.float32))).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(1 + self.num_slope_points,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Fuel usage penalty settings (only positive throttle consumes fuel)
        self.fuel_penalty_coeff = 8  # penalty per unit fuel used
        self.total_fuel_used = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.speed = 0.0
        self.position = 0.0
        self.route_distance = self.initial_route_distance
        self.step_count = 0
        self.total_fuel_used = 0.0
        obs = self._build_observation()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        throttle = np.clip(action[0], -1.0, 1.0)

        # External acceleration from road slope at current position
        external_acc = self.road.get_external_acc(self.position)

        # Vehicle dynamics: engine accel + external
        engine_acc = throttle * 4 #- 2  # baseline drag modeled in engine term
        acceleration = engine_acc + external_acc

        v0 = self.speed
        self.speed += acceleration * self.dt
        self.speed = np.clip(self.speed, 0.0, self.max_speed)  # Prevent negative speeds

        # Distance covered (trapezoidal integration of velocity + accel effect)
        distance_covered = max(0.0, v0 * self.dt + 0.5 * acceleration * self.dt ** 2)
        self.position += distance_covered
        self.route_distance -= distance_covered

        # Reward based on speed tracking
        speed_error = abs(self.speed - self.target_speed)

        # Fuel penalty: only positive throttle uses fuel, zero or braking does not
        throttle_pos = max(0.0, float(throttle))
        fuel_used = throttle_pos * self.dt
        self.total_fuel_used += fuel_used
        
        # Base reward for maintaining target speed (higher is better)
        if speed_error < 1.0:  # Very close to target
            reward = 10.0 - speed_error # ~10
        elif speed_error < 3.0:  # Reasonably close
            reward = 5.0 - speed_error  # ~5
        else:  # Far from target
            reward = -speed_error * 4.0

  
        # reward -= self.fuel_penalty_coeff * fuel_used
        # reward -= -0.5 *0.5 * [-1,1] -> [-4,4]


       
        terminated = False # reached destination
        truncated = False # reached max steps limit, but not the destination
        
        if self.route_distance <= 0:
            terminated = True
            # Bonus for completing route near target speed
            if speed_error < 2.0:
                reward += 50.0
            # reward -= 10.0 * self.total_fuel_used
            reward += 1000 * np.exp(-self.total_fuel_used/2)
        elif self.step_count >= self.max_steps:
            truncated = True

        obs = self._build_observation()
        info = {"fuel_used": float(fuel_used), "total_fuel_used": float(self.total_fuel_used)}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Speed: {self.speed:.2f} m/s, Position: {self.position:.1f} m, ExtAcc: {self.road.get_external_acc(self.position):.2f} m/s^2")

    def _build_observation(self) -> np.ndarray:
        # Convert current position to index in slope map
        route_pos = float(np.clip(self.position, 0.0, self.road.total_length))
        frac = route_pos / self.road.total_length if self.road.total_length > 0 else 0.0
        idx = int(np.clip(int(round(frac * max(self.num_slope_points - 1, 0))), 0, max(self.num_slope_points - 1, 0)))

        # Create a forward-looking slope vector starting from current index wrapping with zeros after route end
        remaining = self.slope_map[idx:]
        if remaining.size < self.num_slope_points:
            pad = np.zeros(self.num_slope_points - remaining.size, dtype=np.float32)
            slope_obs = np.concatenate((remaining.astype(np.float32), pad), axis=0)
        else:
            slope_obs = remaining.astype(np.float32)

        obs = np.concatenate((np.array([self.speed], dtype=np.float32), slope_obs), axis=0)
        return obs



