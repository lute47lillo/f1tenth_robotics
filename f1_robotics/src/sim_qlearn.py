import numpy as np
import gym
import yaml
import time
import simulator
import concurrent.futures
from gym import spaces

def parse_observation(obs):
    # Create state with lap_times, lap_counts, collisions
   
    poses_x = obs['poses_x'][0]
    poses_y = obs['poses_y'][0]
    lin_vel_x = obs['linear_vels_x'][0]
    lin_vel_y = obs['linear_vels_y'][0]
    coll = obs['collisions'][0]
    lap_times = obs['lap_times'][0]
    lap_counts = obs['lap_counts'][0]
    
    state = [poses_x, poses_y, lin_vel_x, lin_vel_y, coll, lap_times, lap_counts]
    return state


# def chooseAction(state, q_table):
#     action = q_table[state]
        
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

# Set initial position of car
poses = np.array([[-1.25*scale + (i * 0.75*scale), 0., starting_angle] for i in range(len(drivers))])

# Reset to initial position
observation, reward, done, info = env.reset(poses=poses)


# Create Q-Table 11x2
states_space = 1500
action_space = 2

q_table = np.zeros((states_space, action_space))

laptime = 0.0
start = time.time()

# Hyperparameters
alpha = 0.7 #learning rate                 
discount_factor = 0.618               
epsilon = 1                  
max_epsilon = 1
min_epsilon = 0.01         
decay = 0.01

train_episodes = 2000    
test_episodes = 100          
max_steps = 100



    
training_rewards = []  
epsilons = []

for episode in range(train_episodes):
    #Reseting the environment each time as per requirement
    state = env.reset(poses=poses)  

    #Starting the tracker for the rewards
    total_training_rewards = 0
    
    for step in range(100):
        #Choosing an action given the states based on a random number
        exp_exp_tradeoff = np.random.uniform(0, 1) 
        
        
        ### STEP 2: SECOND option for choosing the initial action - exploit     
        #If the random number is larger than epsilon: employing exploitation 
        #and selecting best action 
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])      
            
        ### STEP 2: FIRST option for choosing the initial action - explore       
        #Otherwise, employing exploration: choosing a random action 
        else:
            futures = []
            actions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, driver in enumerate(drivers):
                    futures.append(executor.submit(
                        driver.process_lidar,
                        state[0]['scans'][i])
                    )
            for future in futures:
                speed, steer = future.result()
                actions.append([steer, speed])
            action = np.array(actions)
            
            ### STEPs 3 & 4: performing the action and getting the reward     
        #Taking the action and getting the reward and outcome state
        new_state, reward, done, info = env.step(action)
        

        ### STEP 5: update the Q-table
        #Updating the Q-table using the Bellman equation
        
        
        
        
        
        q_table[state, action] = q_table[state, action] + alpha * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]) 
        #Increasing our total reward and updating the state
        total_training_rewards += reward      
        state = new_state         
        
        #Ending the episode
        if done == True:
            #print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
            break
    
    #Cutting down on exploration by reducing the epsilon 
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)
    
    #Adding the total reward and reduced epsilon values
    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)
            







# epochs = 10000
# for x in range(epochs):
#     done = False
#     observation1, reward, done, info = env.reset(poses=poses) # Reset the environment to init position
    
#     # Create list of observation attributes
#     state = parse_observation(observation1)

#     while not done:
            
#         actions = []
#         futures = []
#         epoch = 0
        
#         # Choose actions based on LiDAR information
#         # with concurrent.futures.ThreadPoolExecutor() as executor:
#         #     for i, driver in enumerate(drivers):
#         #         futures.append(executor.submit(
#         #             driver.process_lidar,
#         #             observation1['scans'][i]), 
#         #         )
#         # for future in futures:
#         #     speed, steer = future.result()
#         #     actions.append([steer, speed])
#         # actions = np.array(actions)
#         actions = chooseAction(state)
        
        
#         obs, step_reward, done, info = env.step(actions)
#         laptime += step_reward
        # env.render(mode='human')
        # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)