import gym
import time
import numpy as np
import torch
from typing import Callable
from datetime import datetime
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from wrapper import F110_Wrapped
from rewards import ThrottleMaxSpeedReward

TRAIN_DIRECTORY = "./train"
TRAIN_STEPS = 1.5 * np.power(10, 5)
SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 10)
MIN_EVAL_EPISODES = 100
NUM_PROCESS = 4

MAP_PATH = "maps/Catalunya/Catalunya_map"
MAP_EXTENSION = ".png"

# "maps/{}".format(RACETRACK)
RACETRACK = 'TRACK_2'

class A2C_F1Tenth():
    
    #adaptive learning rate
    def linear_schedule(self, initial_value: float) -> Callable[[float], float]:
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

    # prepare the environment
    def wrap_env(self):
        # Starts F110 gym        
        env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                    map_ext=".png", num_agents=1)
        
        # Wrap basic gym with RL functions
        env = F110_Wrapped(env)
        env = ThrottleMaxSpeedReward(env, 0, int(0.75 * TRAIN_STEPS), 2.5)
        return env
    
    def evaluate(self, model):

        # Create evaluation environment (same as train environment in this case)
        eval_env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                        map_ext=".png", num_agents=1)

        # Wrap evaluation environment
        eval_env = F110_Wrapped(eval_env)
        eval_env.seed(np.random.randint(pow(2, 31) - 1))
        
        model = model.load("train_testA2C/best_model")

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
            
    def train(self):
        
        # Vectorize environment by parallelizing in n_proc
        envs = make_vec_env(self.wrap_env,
                            n_envs=NUM_PROCESS,
                            seed=np.random.randint(pow(2, 31) - 1),
                            vec_env_cls=SubprocVecEnv)

        # Choose RL model and policy here
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #RuntimeError: CUDA error: out of memory whenever I use gpu
        model = A2C("MlpPolicy", envs, learning_rate=self.linear_schedule(0.0001), gamma=0.99, gae_lambda=0.95, ent_coef=0.1,
                    vf_coef=0.5, max_grad_norm=0.5, verbose=1, device='cpu')
        
        # Create Evaluation Callback to save model
        eval_callback = EvalCallback(envs, best_model_save_path='./train_testA2C/',
                                log_path='./train_testA2C/', eval_freq=10000,
                                deterministic=True, render=False)

        # Train model and record time taken
        start_time = time.time()
        model.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback)
        print(f"Training time {time.time() - start_time:.2f}s")
        print("Training cycle complete.")

        # Save model with unique timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        model.save(f"./train/a2c-f110-{timestamp}")
        
        return model
            
    def main(self):
        
        # Train the model
        trained_model = self.train()
        
        # Evaluate the trained model
        self.evaluate(trained_model)

# necessary for Python multi-processing
if __name__ == "__main__":
    init = A2C_F1Tenth()
    init.main()