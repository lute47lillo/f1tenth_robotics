import gym
import numpy as np

from gym import spaces
import drivers
import f110_gym.envs.laser_models as laser

RACETRACK = 'TRACK_2'

driver = drivers.FollowTheGap()


def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car.
    """

    def __init__(self, env):
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
        self.v_min = self.env.params['v_min']
        #self.v_max = self.env.params['v_max']
        self.v_max = 8.0
        self.lidar_min = 0
        self.lidar_max = 30  # see ScanSimulator2D max_range

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

    def step(self, action):
        
        # convert normalised actions (from RL algorithms) back to actual actions for simulator
        action_convert = self.un_normalise_actions(action)
        
        #print(action_convert)
        obs, _, _, _ = self.env.step(np.array([action_convert]))
        ranges_scan = obs['scans'][0]
        
        #print(observation['scans'][0])
        # reward = lidar_reward
        
        # Process LiDAR scan to obtain reward based on adaptive method Follow the Gap
        actions = []    
        speed, steer = driver.process_lidar(ranges_scan, action_convert)
        reward = 0
        
        actions.append([steer, speed])
        actions = np.array(actions)
        
        # actions.append(steer)
        # actions.append(speed)
        
        # self.un_normalise_actions(actions)
        # print(actions)
    
        observation, lidar_reward, done, info = self.env.step(actions)
        reward += lidar_reward

        self.step_count += 1

        #eoins reward function
        vel_magnitude = np.linalg.norm(
            [observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        reward = vel_magnitude #/10 maybe include if speed is having too much of an effect

        # Reward function that returns percent of lap left
        # globwaypoints = np.genfromtxt(f"maps/Catalunya/Catalunya_centerline.csv", delimiter=',')
        
        # if self.count < len(globwaypoints):
        #     wx, wy = globwaypoints[self.count][:2]
        #     X, Y = observation['poses_x'][0], observation['poses_y'][0]
        #     dist = np.sqrt(np.power((X - wx), 2) + np.power((Y - wy), 2))
        #     if dist > 2:
        #         self.count += 1
        #         complete = (self.count/len(globwaypoints)) * 0.5
        #         #print("Percent complete: ", int(complete*100))
        #         reward += complete
        # else:
        #     self.count = 0
            
        if observation['collisions'][0]:
            self.count = 0
            reward = -1 # Might try changing it to more negative reward

        # end episode if car is spinning
        if abs(observation['poses_theta'][0]) > self.max_theta:
            done = True

        # penalise changes in car angular orientation (reward smoothness)
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 0.75:
            reward += -ang_magnitude/10
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 5:
            reward = -(vel_magnitude/10)

        if self.env.lap_counts[0] > 0:
            self.count = 0
            reward += 1
            if self.env.lap_counts[0] > 1:
                reward += 2
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

    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None

    def seed(self, seed):
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")