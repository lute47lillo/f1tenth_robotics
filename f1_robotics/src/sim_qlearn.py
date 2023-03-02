import numpy as np
import gym
import yaml
import time
import simulator
import concurrent.futures
from gym import spaces


# Set Track and Agent drivers
RACETRACK = 'TRACK_1'
drivers = [simulator.AnotherDriver()]

# Set up map configuration
if __name__ == '__main__':
    with open('maps/{}.yaml'.format(RACETRACK)) as map_conf_file:
        map_conf = yaml.load(map_conf_file, Loader=yaml.FullLoader)
scale = map_conf['resolution'] / map_conf['default_resolution']
starting_angle = map_conf['starting_angle']



# Create Gym environment
env = gym.make('f110_gym:f110-v0', map="maps/{}".format(RACETRACK),
            map_ext=".png", num_agents=len(drivers))
poses = np.array([[-1.25*scale + (i * 0.75*scale), 0., starting_angle] for i in range(len(drivers))])


observation, reward, done, info = env.reset(poses=poses)
#print(observation)

laptime = 0.0
start_time = time.time()


last_time_steps = np.ndarray(0)

# Normalize action space
action_space = spaces.Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float)

total_episodes = 10000


for x in range(total_episodes):
    done = False
    cumulated_reward = 0
    observation, reward, done, info = env.reset(poses=poses) # Reset the environment to init position
 
    state = ''.join(map(str, observation))
    print(state)
    
    for i in range(150):
        
        actions = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, driver in enumerate(drivers):
                futures.append(executor.submit(
                    driver.process_lidar,
                    observation['scans'][i])
                )
        for future in futures:
            speed, steer = future.result()
            actions.append([steer, speed])
        actions = np.array(actions)
        
        # actions are [[desired steering angle, desired velocity]]
        print("Sending: ", actions)

        # # Execute the action and get feedback
        observation, reward, done, info = env.step(actions)
        
        # Negative reward for collisions
        if observation['collisions'][0]:
            # reward = -100
            reward = -100
       
        cumulated_reward += reward #For QLearn

        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        nextState = ''.join(map(str, observation))
        
        if not(done):
            state = nextState
        else:
            last_time_steps = np.append(last_time_steps, [int(i + 1)])
            break 
        
        
        env.render(mode='human')
        
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)

    