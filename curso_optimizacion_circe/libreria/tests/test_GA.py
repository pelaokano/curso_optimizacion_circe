import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from MetaOpt import GeneticAlg

# Define a simple fitness function (maximize the sum of values)
def fitness_fn(solution):
    return np.sum(solution)

# Define a simple mutation function (randomly alter a solution)
def mutation_fn(solution):
    idx = np.random.randint(len(solution))
    solution[idx] = np.random.random()
    return solution

# Define a simple crossover function (one-point crossover)
def crossover_fn(parents):
    crossover_point = np.random.randint(1, len(parents[0]))
    child1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
    child2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
    return child1, child2

# Define a simple selection function (select the top 50% based on fitness)
def selection_fn(fitness):
    fitness = np.array(fitness)
    fitness_prob = fitness / np.sum(fitness)
    return fitness_prob

# Create an initial population of 10 solutions, each with 5 genes
initial_population = np.random.rand(10, 5)

# Initialize the Genetic Algorithm
ga = GeneticAlg(
    initial_population=initial_population,
    fitness_fn=fitness_fn,
    mutation_fn=mutation_fn,
    crossover_fn=crossover_fn,
    selection_fn=selection_fn,
    keep_parents=2,
    maximize=True,
    callback=None
)

# Run the genetic algorithm for 20 generations, logging every 5 generations
ga.run(num_generations=20, verbose=True, log_freq=5)

# Print the best solution and its fitness
print(f'Best solution: {ga.best_solution}')
print(f'Best fitness: {ga.best_fitness}')