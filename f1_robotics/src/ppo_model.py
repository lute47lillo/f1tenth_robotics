import gym
import time
import numpy as np
import torch
from typing import Callable
from datetime import datetime
from stable_baselines3 import PPO
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from wrapper import F110_Wrapped
from rewards import ThrottleMaxSpeedReward

# Activate the environment: source robotics/bin/activate

TRAIN_DIRECTORY = "./train"
TRAIN_STEPS = 4 * np.power(10, 5)

#TRAIN_STEPS = 1000
SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 50)
MIN_EVAL_EPISODES = 100
NUM_PROCESS = 4

#MAP_PATH = "maps/Catalunya/Catalunya_map"
MAP_PATH = "maps/TRACK_2"
MAP_EXTENSION = ".png"

# "maps/{}".format(RACETRACK)
RACETRACK = 'TRACK_2'

class PPO_F1Tenth():
    
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
        env = ThrottleMaxSpeedReward(env, 0, int(0.75 * TRAIN_STEPS), 1)
        return env
    
    def evaluate_lstm(self, model):
        # Create evaluation environment (same as train environment in this case)
        eval_env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                        map_ext=".png", num_agents=1)

        # Wrap evaluation environment
        eval_env = F110_Wrapped(eval_env)
        eval_env.seed(np.random.randint(pow(2, 31) - 1))
        model = model.load("train_test/best_model")
              # cell and hidden state of the LSTM
        lstm_states = None
        num_envs = 1
        obs = eval_env.reset()
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)
        while True:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, info = eval_env.step(action)
            episode_starts = dones
            eval_env.render()
        
    def evaluate(self, model):

        # Create evaluation environment (same as train environment in this case)
        eval_env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                        map_ext=".png", num_agents=1)

        # Wrap evaluation environment
        eval_env = F110_Wrapped(eval_env)
        eval_env.seed(np.random.randint(pow(2, 31) - 1))
        model = model.load("train_test/best_model")

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
       
        #model = TRPO("MlpPolicy", envs, verbose=1, learning_rate=self.linear_schedule(0.0003), gamma=0.99, gae_lambda=0.935, 
        #            device='cpu', tensorboard_log="ppo_log/", target_kl=0.035)
        
        model = PPO("MlpPolicy", envs, learning_rate=self.linear_schedule(0.0005), gamma=0.98, n_steps=4096,
                    gae_lambda=0.925, ent_coef=0.005, vf_coef=1, max_grad_norm=0.85, clip_range=0.3,
                    normalize_advantage=True, verbose=1, tensorboard_log="ppo_log/", device='cpu', target_kl=0.25)
       
        
        # Create Evaluation Callback to save model
        eval_callback = EvalCallback(envs, best_model_save_path='./train_test/',
                                log_path='./train_test/', eval_freq=5000,
                                deterministic=True, render=False) # Changed deterministis to False

        # Train model and record time taken
        start_time = time.time()
        model.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback)
        print(f"Training time {time.time() - start_time:.2f}s")
        print("Training cycle complete.")

        # Save model with unique timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        model.save(f"./train/pp0_rew-f110-{timestamp}")
        #model.save(f"./train/trp0-f110-{timestamp}")
        
        return model
            
    def main(self):
        
        # Train the model
        trained_model = self.train()
        
        # Evaluate the trained model
        #self.evaluate_lstm(trained_model)
        self.evaluate(trained_model)

# necessary for Python multi-processing
if __name__ == "__main__":
    init = PPO_F1Tenth()
    init.main()