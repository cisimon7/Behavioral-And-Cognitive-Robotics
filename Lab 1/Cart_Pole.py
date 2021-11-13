import os
import time
import gym


def cart_pole():

    env = gym.make('CartPole-v0')
    observation = env.reset()

    done = False
    fitness = 0

    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        fitness += reward

        if done:
            time.sleep(5)

        env.render()
        time.sleep(0.1)

    env.close()
    print(fitness)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['LANG'] = 'en_US'
    cart_pole()
