import gym
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from wrapper import F110_Wrapped
from stable_baselines3.common.monitor import Monitor
TRAIN_STEPS = 10 * np.power(10, 5)

MIN_EVAL_EPISODES = 100
#MAP_PATH = "maps/Austin/Austin_map"
MAP_PATH = "maps/Catalunya/Catalunya_map"
#MAP_PATH = "maps/TRACK_1"
#MAP_PATH = "maps/TRACK_2"
MAP_EXTENSION = ".png"

def evaluate():

        # Create evaluation environment (same as train environment in this case)
        eval_env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                        map_ext=MAP_EXTENSION, num_agents=1)

        # Wrap evaluation environment
        eval_env = F110_Wrapped(eval_env, 0, int(0.85 * TRAIN_STEPS), 1)
        eval_env.seed(np.random.randint(pow(2, 31) - 1))
        
        # Wrap environment on Monitor environment for helper functions
        eval_env = Monitor(eval_env)
        
        model = PPO.load("train/ppo_work/1MPPO11_Working.zip")
        #model = TRPO.load("train/trpo/1MTRPO.zip")
       

        # Use Helper function from sb3 library to understand the evaluation
        policy_evaluation = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                            deterministic=False, render=True, callback=None,
                                            reward_threshold=None, return_episode_rewards=True, warn=True)
        
        print(policy_evaluation)
        
        '''
        Mean reward per episode, std of reward per episode. Returns ([float], [int]) when return_episode_rewards is True,
        first list containing per-episode rewards and second containing per-episode lengths (in number of steps).
        '''

        
        # Simulate a few episodes and render them, ctrl-c to cancel an episode
        
        # episode = 0
        # while episode < MIN_EVAL_EPISODES:
        #     try:
        #         episode += 1
        #         obs = eval_env.reset()
        #         done = False
        #         while not done:
        #             action, _ = model.predict(obs)
        #             obs, reward, done, _ = eval_env.step(action)
        #             eval_env.render()
        #     except KeyboardInterrupt:
        #         break
                
evaluate()