import gym
import time
import numpy as np


class Network:
    def __init__(self, env, n_hidden):
        """
            n_hidden : number of internal neurons
        """

        self.env = env

        p_variance = 2  # variance of initial parameters
        pp_variance = 5  # variance of perturbations

        self.n_inputs = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.box.Box):
            self.n_outputs = env.action_space.shape[0]
        else:
            self.n_outputs = env.action_space.n

        self.sensory = np.zeros(shape=(self.n_inputs, 1))
        self.internal, self.motor = np.zeros(shape=(n_hidden, 1)), np.zeros(shape=(self.n_outputs, 1))

        self.w1 = np.random.randn(n_hidden, self.n_inputs) * p_variance  # First connection Layer
        self.w2 = np.random.randn(self.n_outputs, n_hidden) * pp_variance  # Second connection Layer
        self.b1 = np.zeros(shape=(n_hidden, 1))  # Bias internal neurons
        self.b2 = np.zeros(shape=(self.n_outputs, 1))  # Bias output neurons

    def update(self, observation):
        w1, w2, b1, b2 = self.w1, self.w2, self.b1, self.b2
        n_inputs, n_outputs = self.n_inputs, self.n_outputs

        observation.resize(n_inputs, 1)  # Current system vector

        z1 = np.dot(w1, observation) + b1
        a1 = np.tanh(z1)  # internal

        z2 = np.dot(w2, a1) + b2
        a2 = np.tanh(z2)  # motor

        if isinstance(self.env.action_space, gym.spaces.box.Box):
            action = a2
        else:
            action = np.argmax(a2)

        self.sensory = observation
        self.internal, self.motor = a1, a2

        return action

    def evaluate(self, n_episodes, show_print=False, show_render=False):
        n_fitness = []
        env = self.env

        for i in range(n_episodes):
            observation = env.reset()
            done, fitness = False, 0
            j = 0
            while not done:
                action = self.update(observation)
                observation, reward, done, _ = env.step(action)
                fitness += reward

                if show_render:
                    env.render()
                    time.sleep(0.1)

                if show_print:
                    print(f"Episode {i} step {j}")
                    print(f"sensory:\t {np.ravel(self.sensory)}, \ninternal:\t {np.ravel(self.internal)}, "
                          f"\nmotors:\t\t {np.ravel(self.motor)}\n")
                j += 1

            n_fitness.append(fitness)

        return np.mean(n_fitness)

    def n_parameters(self):
        s, i, m = len(self.sensory), len(self.internal), len(self.motor)
        n_parameters = (s * i) + (i * m) + i + m
        return n_parameters

    def set_parameters(self, genotype):
        s, i, m = len(self.sensory), len(self.internal), len(self.motor)
        d1, d2, d3, d4 = (s * i), (i * m), i, m

        self.w1 = np.asarray(genotype[0:d1]).reshape((s, i)).T
        self.w2 = np.asarray(genotype[d1:d1 + d2]).reshape((i, m)).T
        self.b1 = np.asarray(genotype[d1 + d2:d1 + d2 + d3]).reshape((i, 1))
        self.b2 = np.asarray(genotype[d1 + d2 + d3:d1 + d2 + d3 + d4]).reshape((m, 1))


class EvolutionStrategy:
    def __init__(self):
        self.pop_size = 10
        self.gen_range = 0.1
        self.mut_range = 0.02
        self.n_episodes = 3
        self.n_generations = 100

    def run(self, network: Network, n_episodes: int, show_render=False):
        n_parameters = network.n_parameters()
        half_pop = int(self.pop_size / 2)
        # population = np.random.randn(self.pop_size, n_parameters) * self.gen_range
        population = np.tile(np.concatenate((network.w1, network.w2, network.b1, network.b2), axis=None),
                             (n_parameters, 1))
        fitness2idx = []
        fitness_plot_data = []
        best_fitness = 0
        best_parameters = 0

        for gen in range(self.n_generations):
            for i in range(self.pop_size):
                network.set_parameters(population[i])
                i_fitness = network.evaluate(n_episodes)
                fitness2idx.append((i_fitness, i))
            fitness2idx.sort(key=lambda y: y[0])

            fitnesses = list(map(lambda fit: fit[0], np.asarray(fitness2idx)))
            fitness_plot_data.append(fitnesses)
            print(f"generation {gen}: best {np.max(fitnesses)} average {np.mean(fitnesses)}")

            if np.max(best_fitness) < np.max(fitnesses):
                best_fitness = fitnesses
                best_parameters = population[-1]

            for i in range(half_pop):
                population[fitness2idx[i][1]] = population[fitness2idx[i + half_pop][1]] + (
                        np.random.randn(n_parameters) * self.mut_range)

            fitness2idx = []

        if show_render:
            network.set_parameters(best_parameters)
            network.evaluate(1, show_print=False, show_render=True)


def test_network():
    environ = gym.make("CartPole-v0")
    network = Network(environ, 5)

    n_episodes = 10
    avg_fitness = network.evaluate(n_episodes, show_print=True, show_render=True)

    print(f"Fitness (averaged over {n_episodes} episodes): {avg_fitness}")


def test_evolution_strategy():
    e_strategy = EvolutionStrategy()
    environ = gym.make("CartPole-v0")
    network = Network(environ, 5)
    n_episodes = 10

    e_strategy.run(network, n_episodes, show_render=True)


if __name__ == '__main__':
    # test_network()
    test_evolution_strategy()
