import gym
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from wrapper import F110_Wrapped

MIN_EVAL_EPISODES = 100
MAP_PATH = "maps/Austin/Austin_map"

def evaluate():

        # Create evaluation environment (same as train environment in this case)
        eval_env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                        map_ext=".png", num_agents=1)

        # Wrap evaluation environment
        eval_env = F110_Wrapped(eval_env)
        eval_env.seed(np.random.randint(pow(2, 31) - 1))

        # TODO: Add command arguments to test evaluation for different models based on algorithms
        #model = A2C.load("train_testA2C/best_model.zip")
        model = PPO.load("train_test/best_model.zip")

            # Simulate a few episodes and render them, ctrl-c to cancel an episode
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
                break
                
evaluate()