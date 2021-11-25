import os
import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

environment_name = "CartPole-v0"


class CartPoleModel:

    def __init__(self, log_path: str = None, ppo_path: str = None, re_train=False):
        self.log_path = os.path.join("Trainings", "Logs") if log_path is None else os.path.join("Trainings", log_path)

        self.ppo_path = os.path.join("Trainings", "Saved_Models", "PPO_CartPole_Model") if ppo_path is None else \
            os.path.join("Trainings", "Saved_Models", ppo_path)

        gym_env = gym.make(environment_name)
        self.env = DummyVecEnv([lambda: gym_env])

        self.algorithm = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log=self.log_path)

        if (not os.listdir("Trainings/Saved_Models").__contains__(self.ppo_path.split("/").pop() + ".zip")) or re_train:
            self.train_model(self.ppo_path)

        self.loaded_model = self.algorithm.load(self.ppo_path)

    def train_model(self, ppo_path, callback=None, algo=None):

        if algo is not None:
            self.algorithm = algo

        self.algorithm.learn(total_timesteps=20000, callback=callback)

        self.algorithm.save(ppo_path)

        ev_policy = evaluate_policy(self.algorithm, self.env, n_eval_episodes=10, render=True)
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


if __name__ == '__main__':
    cartPole_model = CartPoleModel()
    cartPole_model.run_model()
