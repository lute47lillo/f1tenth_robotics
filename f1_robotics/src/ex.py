import gym
import time
import numpy as np
import torch
import os
import argparse
import yaml
from typing import Callable
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from wrapper import F110_Wrapped
#from code.schedulers import linear_schedule

TRAIN_DIRECTORY = "./train"
TRAIN_STEPS = 1.5 * np.power(10, 5)    # for reference, it takes about one sec per 500 steps
SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 10)
MIN_EVAL_EPISODES = 100
NUM_PROCESS = 1
MAP_PATH = "./f1tenth_racetracks/Austin/Austin_map"
MAP_EXTENSION = ".png"

RACETRACK = 'TRACK_2'

#adaptive learning rate
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def main():

    #       #
    # TRAIN #
    #       #

    # prepare the environment
    def wrap_env():
        # starts F110 gym        
        env = gym.make('f110_gym:f110-v0', map="maps/{}".format(RACETRACK),
                       map_ext=".png", num_agents=1)
    
        # wrap basic gym with RL functions
        env = F110_Wrapped(env)
      
        return env

    # vectorise environment (parallelise)
    envs = make_vec_env(wrap_env,
                        n_envs=NUM_PROCESS,
                        seed=np.random.randint(pow(2, 31) - 1),
                        vec_env_cls=SubprocVecEnv)

    # choose RL model and policy here
    """eval_env = gym.make("f110_gym:f110-v0",map=MAP_PATH,map_ext=MAP_EXTENSION,num_agents=1)
    eval_env = F110_Wrapped(eval_env)
    eval_env = RandomF1TenthMap(eval_env, 500)
    eval_env.seed(np.random.randint(pow(2, 31) - 1))"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #RuntimeError: CUDA error: out of memory whenever I use gpu
    model = PPO("MlpPolicy", envs,  learning_rate=linear_schedule(0.0003), gamma=0.99, gae_lambda=0.95, verbose=1, device='cpu')
    
    
    eval_callback = EvalCallback(envs, best_model_save_path='./train_test/',
                             log_path='./train_test/', eval_freq=5000,
                             deterministic=True, render=False)

    # train model and record time taken
    start_time = time.time()
    model.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback)
    print(f"Training time {time.time() - start_time:.2f}s")
    print("Training cycle complete.")

    # save model with unique timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model.save(f"./train/ppo-f110-{timestamp}")




    #          #
    # EVALUATE #
    #          #

    # create evaluation environment (same as train environment in this case)
    eval_env = gym.make('f110_gym:f110-v0', map="maps/{}".format(RACETRACK),
                       map_ext=".png", num_agents=1)

    # wrap evaluation environment
    eval_env = F110_Wrapped(eval_env)
    eval_env.seed(np.random.randint(pow(2, 31) - 1))
    model = model.load("./train_test/best_model")

    # simulate a few episodes and render them, ctrl-c to cancel an episode
    episode = 0
    while episode < MIN_EVAL_EPISODES:
        try:
            episode += 1
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = eval_env.step(action)
                eval_env.render()
        except KeyboardInterrupt:
            pass

# necessary for Python multi-processing
if __name__ == "__main__":
    main()