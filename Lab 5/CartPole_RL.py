import os
import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

environment_name = "CartPole-v0"


class CartPoleModel:

    def __init__(self, log_path=None, ppo_path=None):
        if log_path is None:
            log_path = os.path.join("Trainings", "Logs")
        if ppo_path is None:
            ppo_path = os.path.join("Trainings", "Saved_Models", "PPO_CartPole_Model")

        gym_env = gym.make(environment_name)
        self.env = DummyVecEnv([lambda: gym_env])
        self.model = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log=log_path)

        if len(os.listdir("Trainings/Saved_Models")) == 0:
            self.train_model(ppo_path)

        self.loaded_model = self.model.load(ppo_path)

    def train_model(self, ppo_path):
        self.model.learn(total_timesteps=20000)
        self.model.save(ppo_path)

        ev_policy = evaluate_policy(self.model, self.env, n_eval_episodes=10, render=True)
        print(f"Evaluated Policy: {ev_policy}")

    def run_model(self):
        episodes = 5

        for episode in range(1, episodes + 1):
            observation = self.env.reset()
            done = False
            score = 0

            while not done:
                self.env.render()
                time.sleep(0.1)

                action, state = self.loaded_model.predict(observation)
                observation, reward, done, info = self.env.step(action)
                score += reward

            print(f"Episode: {episode},Score: {score}")
            self.env.close()

        self.env.reset()


cartPole_model = CartPoleModel()
cartPole_model.run_model()
