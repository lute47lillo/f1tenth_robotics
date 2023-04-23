import gym
import numpy as np

from gym import spaces
import drivers

driver = drivers.FollowTheGap()

"""
    Helper Function
    Converts value(s) from range to another ranges ---> [min, max]
"""
def convert_range(value, input_range, output_range):
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment.
    """

    def __init__(self, env, init_step, finish_step, init_max_reward, finish_max_reward=None):
        super().__init__(env)

        # normalised action space, steer and speed
        self.action_space = spaces.Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float)

        # normalised observations, just take the lidar scans
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1080,), dtype=np.float)

        # store allowed steering/speed/lidar ranges for normalisation
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = 0.0
        self.v_max = 7.0
        self.lidar_min = 0
        self.lidar_max = 40  # see ScanSimulator2D max_range

        # store car dimensions and some track info
        self.car_length = self.env.params['length']
        self.car_width = self.env.params['width']
        self.track_width = 3.2

        # radius of circle where car can start on track, relative to a centerpoint
        self.start_radius = (self.track_width / 2) - \
            ((self.car_length + self.car_width) / 2)  # just extra wiggle room

        self.step_count = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100
        self.count = 0
        
        # Reward attributes
        self.end_step = finish_step
        self.start_step = init_step
        self.start_max_reward = init_max_reward
        
        # set finishing maximum reward to be maximum possible speed by default
        if finish_max_reward is None:
            self.end_max_reward = self.v_max
        else:
            self.end_max_reward = finish_max_reward

        # calculate slope for reward changing over time (steps)
        self.reward_slope = (self.end_max_reward - self.start_max_reward) / (self.end_step - self.start_step)

    def step(self, action):
        
        # convert normalised actions (from RL algorithms) back to actual actions for simulator
        action_convert = self.un_normalise_actions(action)
        
        #print(action_convert)
        obs, _, _, _ = self.env.step(np.array([action_convert]))
        ranges_scan = obs['scans'][0]
        
        # Process LiDAR scan to obtain reward based on adaptive method Follow the Gap
        actions = []    
        speed, steer = driver.process_lidar(ranges_scan, action_convert)
        reward = 0
        
        actions.append([steer, speed])
        actions = np.array(actions)
        
        observation, lidar_reward, done, info = self.env.step(actions)
        reward += lidar_reward

        self.step_count += 1

        #eoins reward function
        vel_magnitude = np.linalg.norm(
            [observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        reward = vel_magnitude
            
        # Negative reward for collision
        if observation['collisions'][0]:
            self.count = 0
            reward = -2

        # End episode if car is spinning
        if abs(observation['poses_theta'][0]) > self.max_theta:
            done = True

        # penalise changes in car angular orientation (reward smoothness)
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 0.75:
            reward += -ang_magnitude/10
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 5:
            reward = -(vel_magnitude/10)

        # Reward the car for completing laps
        if self.env.lap_counts[0] > 0:
            self.count = 0
            reward += 1
            if self.env.lap_counts[0] > 1:
                reward += 5
                self.env.lap_counts[0] = 0
                
        lidar_ranges = observation['scans'][0]
        obs = self.normalise_observations(lidar_ranges)
        return obs, reward, bool(done), info

    def reset(self, start_xy=None, direction=None):

        # start from origin if no pose input
        if start_xy is None:
            start_xy = np.zeros(2)
            
        # start in random direction if no direction input
        if direction is None:
            direction = np.random.uniform(0, 2 * np.pi)
            
        # get slope perpendicular to track direction
        slope = np.tan(direction + np.pi / 2)
        
        # get magintude of slope to normalise parametric line
        magnitude = np.sqrt(1 + np.power(slope, 2))
        
        # get random point along line of width track
        rand_offset = np.random.uniform(-1, 1)
        rand_offset_scaled = rand_offset * self.start_radius

        # convert position along line to position between walls at current point
        x, y = start_xy + rand_offset_scaled * np.array([1, slope]) / magnitude
        
        starting_angle = 1.5708
        
        # reset car with chosen pose
        observation, _, _, _ = self.env.reset(poses=np.array([[x, y, starting_angle]]))
        
        return self.normalise_observations(observation['scans'][0])

    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])

    def seed(self, seed):
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")
        
    def reward(self, reward):
        # Base Case
        if self.step_count < self.start_step:
            return min(reward, self.start_max_reward)
        
        # For 800k steps
        elif self.step_count > self.end_step:
            return min(reward, self.end_max_reward)
        
        # 200k last steps, proportional reward between two step endpoints
        else:
            proportional_reward = self.start_max_reward + (self.step_count - self.start_step) * self.reward_slope
            return min(reward, proportional_reward)