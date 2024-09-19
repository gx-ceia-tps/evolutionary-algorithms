import random
import numpy as np
from numpy.ma.core import cumsum


class EvolutionaryAlgorithm:

    def __init__(self, generations, chromosome_length, population_length, target_fn, maximize, elitism, x_min, x_max,crossover_rate,mutation_rate, epsilon=1e-3):
        self.generations = generations
        self.chromosome_length = chromosome_length
        self.population_length = population_length
        self.target_fn = target_fn
        self.maximize = maximize
        self.epsilon = epsilon
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation_info = {}
        self.x_min = x_min
        self.x_max = x_max
        self.best = [None for i in range(generations)]

    # def generate_chromosome(self, chromosome_length):
    #     raise NotImplementedError

    def initialize_population(self):
        return [self.generate_chromosome(self.chromosome_length) for _ in range(self.population_length)]

    def selection_algorithm(self):
        raise NotImplementedError

    def get_progenitors(self):
        return [self.selection_algorithm() for _ in range(self.population_length)]

    # def crossover(self, p1,p2):
    #     raise NotImplementedError

    def get_descendants(self, progenitors):
        grouped_progenitors = zip(progenitors[0::2], progenitors[1::2])
        unflattened_descendants = [self.crossover(p1,p2) for p1,p2 in grouped_progenitors]
        return sum(unflattened_descendants, [])

    def mutate(self, chromosome):
        mutated_chromosome = [int(not int(bit)) if random.uniform(0, 1) < self.mutation_rate else bit for bit in chromosome]
        return "".join(map(str, mutated_chromosome))

    def get_mutations(self, descendants):
        return [self.mutate(descendant) for descendant in descendants]

    def apply_elitism(self, mutations):
        # esto asume q queremos fitnesse mas grande o menos grande, depende de si maximizamos o minimizamos...
        self.population.sort(key=self.fitness)
        mutations.sort(key=self.fitness, reverse=True)
        for i in range(len(mutations)):
            if self.fitness(mutations[i]) > self.fitness(self.population[i]):
                self.population[i] = mutations[i]

    def bin_to_dec(self, chromosome):
        decimal = int(chromosome, 2)
        x = self.x_min + decimal * (self.x_max - self.x_min) / ((2 ** self.chromosome_length) - 1)
        return x

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

            best_individual = max(self.population, key=self.fitness)
            self.generation_info[generation] = {"best_individual": best_individual}

    def get_best_individual(self):
        best = self.generation_info[self.generations-1]['best_individual']
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
    def selection_algorithm(self):
        fitness_total = sum(self.fitness(chromosome) for chromosome in self.population)
        probabilities = [self.fitness(individual) / fitness_total for individual in self.population]
        cumulative_probs = np.cumsum(probabilities)
        r = random.uniform(0, 1)
        i = next(i for i, cumulative_prob in enumerate(cumulative_probs) if r <= cumulative_prob)
        return self.population[i]

class LinearRanking(EvolutionaryAlgorithm):
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

    def selection_algorithm(self):
        progenitors = []
        for _ in range(len(self.population)):
            candidates = random.sample(self.population, self.tournament_size)
            progenitor = max(candidates, key=self.fitness)
            progenitors.append(progenitor)
        return progenitors


#
# # model_args = {generations:10, chromosome_length:10,population_length:4, maximize:True, target_fn=lambda x:x**2, elitism=True,crossover_rate=0.92, x_min=-31,x_max=31, mutation_rate=0.1}
#
# num_gens = 10
#
# ea = Roulette(
#     generations=num_gens,
#     chromosome_length=10,
#     population_length=4,
#     maximize=False,
#     target_fn=lambda x:x**2,
#     elitism=True,
#     crossover_rate=0.92,
#     x_min=-31,
#     x_max=31,
#     epsilon=0.001,
#     mutation_rate=0.1
# )
#
# random.seed(42)
# ea.execute()
# # podria terminar antes, no necesariamente es el mejor
# goat = ea.generation_info[num_gens-1]["best_individual"]
#
# print('Mejor solucion: ', ea.bin_to_dec(max(ea.population, key=lambda x: ea.fitness(x))))
# print(ea.fitness(goat))

# print(ea.bin_to_dec(ea.generation_info[9]["best_individual"]))