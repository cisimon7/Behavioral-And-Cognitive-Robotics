import os.path

from stable_baselines3 import PPO, DQN
from CartPole_RL import CartPoleModel
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def training_callback():
    cart_pole_model = CartPoleModel()

    # Callback Function
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    # To be triggered on each training round
    eval_callback = EvalCallback(
        eval_env=cart_pole_model.env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        best_model_save_path=os.path.join("Trainings", "Saved_Models"),
        verbose=1
    )

    cart_pole_model.train_model(cart_pole_model.ppo_path, callback=eval_callback)
    cart_pole_model.run_model()


def arch_net():
    net_arch = [dict(pl=[128, 128, 128, 128], vf=[128, 128, 128, 128])]

    cartPole_model = CartPoleModel(ppo_path="PPO_CartPole_Model_ArchNet", re_train=True)
    m_policy = PPO("MlpPolicy", cartPole_model.env, verbose=1, tensorboard_log=cartPole_model.log_path,
                   policy_kwargs={'net_arch': net_arch})
    cartPole_model.train_model(cartPole_model.ppo_path, algo=m_policy)
    cartPole_model.run_model()


def dqn_model():
    cart_pole = CartPoleModel(ppo_path="DQN_CartPole_Model_ArchNet", re_train=True)
    dqn_algo = DQN("MlpPolicy", cart_pole.env, verbose=1, tensorboard_log=cart_pole.log_path)
    cart_pole.train_model(cart_pole.ppo_path, algo=dqn_algo)
    cart_pole.run_model()


if __name__ == '__main__':
    dqn_model()
