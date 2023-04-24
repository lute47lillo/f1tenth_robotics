import gym
import time
import numpy as np
import torch
from typing import Callable
from datetime import datetime
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from wrapper import F110_Wrapped

# Activate the environment: source robotics/bin/activate

TRAIN_DIRECTORY = "./train/ppo_work"
TRAIN_STEPS = 10 * np.power(10, 5)

SAVE_CHECK_FREQUENCY = int(TRAIN_STEPS / 100)
MIN_EVAL_EPISODES = 10
NUM_PROCESS = 4

MAP_PATH = "maps/Catalunya/Catalunya_map"
#MAP_PATH = "maps/TRACK_1"
MAP_EXTENSION = ".png"

class TRPO_F1Tenth():
    
    # Adaptive learning rate
    """
            Linear learning rate schedule.
            Progress will decrease from 1 (beginning) to 0.
            Returns the current lr
    """
    def linear_schedule(self, initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    # prepare the environment
    def wrap_env(self):
        # Starts F110 gym        
        env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                    map_ext=".png", num_agents=1)
        
        # Wrap basic gym with RL functions
        env = F110_Wrapped(env, 0, int(0.85 * TRAIN_STEPS), 1)
        return env
        
    def evaluate(self, model):

        # Create evaluation environment (same as train environment in this case)
        eval_env = gym.make('f110_gym:f110-v0', map=MAP_PATH,
                        map_ext=".png", num_agents=1)

        # Wrap evaluation environment
        eval_env = F110_Wrapped(eval_env, 0, int(0.85 * TRAIN_STEPS), 1)
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
        
        model = TRPO("MlpPolicy", envs, learning_rate = self.linear_schedule(0.0007), n_steps = 4096,
                     gamma = 0.97, gae_lambda = 0.925, target_kl=0.28, normalize_advantage=True, verbose=1,
                     tensorboard_log="ppo_log/trpo/", device='cpu', sub_sampling_factor=2, n_critic_updates=5)
        
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
        model.save(f"./train/ppo_work/trp0_itr-f110-{timestamp}")

        
        return model
            
    def main(self):
        
        # Train the model
        trained_model = self.train()
        
        # Evaluate the trained model
        self.evaluate(trained_model)

# necessary for Python multi-processing
if __name__ == "__main__":
    init = TRPO_F1Tenth()
    init.main()