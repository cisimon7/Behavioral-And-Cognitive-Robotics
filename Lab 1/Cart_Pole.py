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
    cart_pole()


"""
    Observation Vector:
        An array of size 4, containing:
        Idx     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
        
        
    Action Vector:
        A set of two distinct discrete numbers
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        
        
    How reward is calculated:
        The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the 
        cart's velocity. For each step taken, that is the robot is not fallen, the reward is a constant: 1
    
    
    How initial conditions are varied by the reset method:
        An array of size 4 is generated using a uniform random distribution bounded between -0.05 amd 0.05
        
        
    What are the termination conditions:
        Fallen state, i.e. Angle and distance at which to fail the episode, is defined at initialization of system. 
        At this angle or distance, the pendulum is taken as fallen. Environment terminates if observation vector 
        violates any of this limit.
"""