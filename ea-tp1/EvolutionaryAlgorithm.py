import random
import numpy as np
from numpy.ma.core import cumsum
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class EvolutionaryAlgorithm:

    def __init__(self, generations, chromosome_length, population_length, target_fn, maximize, elitism, x_min, x_max,crossover_rate,mutation_rate, epsilon=0.001):
        self.generations = generations
        self.chromosome_length = chromosome_length
        self.population_length = population_length
        self.target_fn = target_fn
        self.maximize = maximize
        self.epsilon = epsilon
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation_best = []
        self.x_min = x_min
        self.x_max = x_max

    def initialize_population(self):
        return [self.generate_chromosome(self.chromosome_length) for _ in range(self.population_length)]

    def selection_algorithm(self):
        raise NotImplementedError

    def get_progenitors(self):
        return [self.selection_algorithm() for _ in range(self.population_length)]

    def get_descendants(self, progenitors):
        grouped_progenitors = zip(progenitors[0::2], progenitors[1::2])
        unflattened_descendants = [self.crossover(p1,p2) for p1,p2 in grouped_progenitors]
        return sum(unflattened_descendants, [])

    def mutate(self, chromosome):
        assert(type(chromosome) == str)
        mutated_chromosome = [int(not int(bit)) if random.uniform(0, 1) < self.mutation_rate else bit for bit in chromosome]
        return "".join(map(str, mutated_chromosome))

    def get_mutations(self, descendants):
        return [self.mutate(descendant) for descendant in descendants]

    def apply_elitism(self, mutations):
        self.population.sort(key=self.fitness)
        mutations.sort(key=self.fitness, reverse=True)
        for i in range(len(mutations)):
            if self.fitness(mutations[i]) > self.fitness(self.population[i]):
                self.population[i] = mutations[i]

    def bin_to_dec(self, chromosome):
        decimal = int(chromosome, 2)
        x = self.x_min + decimal * (self.x_max - self.x_min) / ((2 ** self.chromosome_length) - 1)
        return x

    def dec_to_bin(self, decimal):
        number = (decimal-self.x_min) * (2**self.chromosome_length - 1)/(self.x_max - self.x_min)
        int_number = round(number)
        return format(int_number, f'0{self.chromosome_length}b')

    def fitness(self, chromosome):
        x = self.bin_to_dec(chromosome)
        if self.maximize:
            return self.target_fn(x)
        else:
            return 1/(self.target_fn(x) + self.epsilon)


    def execute(self):
        self.population = self.initialize_population()
        for generation in range(self.generations):
            progenitors = self.get_progenitors()
            descendants = self.get_descendants(progenitors)
            mutations = self.get_mutations(descendants)
            if self.elitism:
                self.apply_elitism(mutations)
            else:
                self.population = mutations

            best_individual = max(self.population, key=self.fitness)
            self.generation_best.append(best_individual)

        return max(self.population, key=self.fitness)

    def get_best_individual(self):
        best = self.generation_best[self.generations-1]
        return best, self.bin_to_dec(best), self.fitness(best)

    def one_point_crossover(self, progenitor_1, progenitor_2):
        if random.uniform(0, 1) < self.crossover_rate:
            crossover_point = random.randint(1, len(progenitor_1) - 1)
            descendant_1 = progenitor_1[:crossover_point] + progenitor_2[crossover_point:]
            descendant_2 = progenitor_2[:crossover_point] + progenitor_1[crossover_point:]
        else:
            descendant_1, descendant_2 = progenitor_1, progenitor_2
        return [descendant_1, descendant_2]

    def crossover(self, p1,p2):
        return self.one_point_crossover(p1,p2)

    def generate_chromosome(self, chromosome_length):
        random_bits = [random.randint(0, 1) for _ in range(chromosome_length)]
        chromosome = "".join(map(str, random_bits))
        return chromosome


class Roulette(EvolutionaryAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'Roulette'

    def selection_algorithm(self):
        fitness_total = sum(self.fitness(chromosome) for chromosome in self.population)
        probabilities = [self.fitness(individual) / fitness_total for individual in self.population]
        cumulative_probs = np.cumsum(probabilities)
        r = random.uniform(0, 1)
        i = next(i for i, cumulative_prob in enumerate(cumulative_probs) if r <= cumulative_prob)
        return self.population[i]

class LinearRanking(EvolutionaryAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'Linear Ranking'

    def selection_algorithm(self):
        # se calcula la aptitud de cada individuo
        fitness_list = [self.fitness(individual) for individual in self.population]
        sorted_population = sorted(zip(self.population, fitness_list), key=lambda x: x[1])

        # se calcula probabilidades segun el ranking lineal
        N = len(self.population)
        s = 1.7  # Factor de seleccion comunmente usado
        probabilities = [(2 - s) / N + (2 * i * (s - 1)) / (N * (N - 1)) for i in range(N)]

        # se selecciona un progenitor basado en las probabilidades
        r = random.uniform(0, 1)
        cumulative_probs = np.cumsum(probabilities)
        i = next(i for i, cumulative_prob in enumerate(cumulative_probs) if r <= cumulative_probs[i])

        return sorted_population[i][0]

class Tournament(EvolutionaryAlgorithm):

    def __init__(self, tournament_size, **kwargs):
        super().__init__(**kwargs)
        self.tournament_size = tournament_size
        self.model_name = 'Tournament'

    def selection_algorithm(self):
        candidates = random.sample(self.population, self.tournament_size)
        progenitor = max(candidates, key=self.fitness)
        return progenitor


class BivariateTargetFun:
    def __init__(self, x1_min, x1_max, x2_min, x2_max, decimal_places):
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max
        # self.decimal_places = decimal_places
        self.x1_bits = math.ceil(np.log2((x1_max - x1_min) * (10 ** decimal_places)))
        self.x2_bits = math.ceil(np.log2((x2_max - x2_min) * (10 ** decimal_places)))
        self.total_bits = self.x1_bits + self.x2_bits

    def dec_to_bin(self, decimal, bits):
        return format(int(decimal), f'0{bits}b')

    def target_fn(self, x1, x2):
        return 7.7 + 0.15 * x1 + 0.22 * x2 - 0.05 * (x1 ** 2) - 0.016 * (x2 ** 2) - 0.007 * x1 * x2

    def bin_to_dec(self, chromosome, range_min, range_max, bits_needed):
        decimal = int(chromosome, 2)
        return range_min + decimal * (range_max - range_min) / ((2 ** bits_needed) - 1)

    def split_answer(self, chromosome):
        idx = self.x1_bits
        x1 = self.bin_to_dec(chromosome[:idx], self.x1_min, self.x1_max, self.x1_bits)
        x2 = self.bin_to_dec(chromosome[idx:], self.x2_min, self.x2_max, self.x2_bits)
        return x1, x2

    def target_fun(self, decimal):
        chromosome = self.dec_to_bin(decimal, self.total_bits)
        x1, x2 = self.split_answer(chromosome)
        return self.target_fn(x1, x2)

    def plot_function(self):
        # Create a grid of x and y values
        x_vals = np.linspace(self.x1_min, self.x1_max, 100)
        y_vals = np.linspace(self.x2_min, self.x2_max, 100)
        x, y = np.meshgrid(x_vals, y_vals)

        # Compute the corresponding z values
        z = self.target_fn(x, y)

        # Plotting the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()