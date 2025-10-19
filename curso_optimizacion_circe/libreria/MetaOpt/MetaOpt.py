import time
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import copy
import math
import random
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Algoritmo Genético (GA)

# Este es un algoritmo evolutivo de tipo poblacion, es bueno para explorar el espacio de soluciones

# Inicialización:
# 1. Se crea una población inicial de soluciones aleatorias (cromosomas).
# 2. Se evalúa la aptitud (fitness) de cada individuo en la población.
# 3. Se seleccionan los mejores individuos de la población inicial para la siguiente generación.

# Selección:
# Durante la selección, se eligen los individuos que serán padres de la siguiente generación.
# Usualmente, se utilizan métodos como la selección por torneo o la ruleta para elegir a los padres.

# Cruce (Crossover):
# En esta etapa, se realiza el cruce entre dos padres seleccionados para producir un hijo.
# El cruce combina las características de ambos padres para generar nuevas soluciones.
# Ejemplo:
# crossover(parent1, parent2):
#    offspring1 = parent1[0:split_point] + parent2[split_point:]
#    offspring2 = parent2[0:split_point] + parent1[split_point:]
# La posición de corte (split_point) determina cómo se mezclan los padres.

# Mutación:
# La mutación introduce cambios aleatorios en el cromosoma para evitar la convergencia prematura.
# Esto ayuda a explorar nuevas soluciones y aumenta la diversidad genética.
# Ejemplo:
# mutation(offspring):
#    if random() < mutation_rate:
#        offspring[random_index] = new_value
# La tasa de mutación (mutation_rate) define la probabilidad de mutación en un cromosoma.

# Actualización de la población:
# Después del cruce y la mutación, se reemplaza la población actual con la nueva generación de hijos.
# Se puede usar un enfoque de reemplazo completo o de reemplazo por elitismo (mantener los mejores individuos).

# Evaluación de la aptitud (fitness):
# En cada iteración, se evalúa la aptitud de cada individuo (cromosoma) en la nueva generación.
# El algoritmo sigue evolucionando hasta que se alcanza un criterio de parada, como el número máximo de generaciones o una solución óptima.

# Criterio de parada:
# El algoritmo termina cuando se alcanza uno de los siguientes criterios:
# - El número máximo de generaciones.
# - La aptitud de la mejor solución alcanza un umbral predefinido.
# - La mejora en la aptitud entre generaciones es mínima.

class GeneticAlg():
    """
    A general implementation of a genetic algorithm to optimize a population of solutions.

    The class supports basic genetic algorithm operations such as selection, mutation, crossover, 
    and elitism. It can handle both maximization and minimization tasks based on the provided 
    fitness function.

    Attributes:
        population (numpy.ndarray): The population of solutions represented as a numpy array.
        fitness_fn (function): The fitness function that evaluates the population.
        mutation_fn (function): The mutation function that modifies solutions in the population.
        crossover_fn (function): The crossover function that combines two solutions into a new one.
        selection_fn (function): The selection function that determines the probability of selecting 
                                individuals from the population based on their fitness.
        keep_parents (int): The number of parents to retain for elitism selection.
        maximize (bool): If True, the algorithm will maximize the fitness function; otherwise, it will minimize.
        callback (list of functions): A list of callback functions to call after each generation.
        fitness_hist (list): A history of fitness values at each generation.
        best_fitness (float): The fitness value of the best solution found.
        best_solution (numpy.ndarray): The best solution found during the evolution.
        num_generations (int): The number of generations run so far.
        fitness (list): The fitness values for the current population.
        
    Methods:
        __init__(self, initial_population=None, fitness_fn=None, mutation_fn=None, crossover_fn=None,
                selection_fn=None, keep_parents=0, maximize=True, callback=None):
            Initializes the genetic algorithm with the given parameters.

        score(self):
            Computes the fitness of the population and updates the best solution found.

        generate_new_pop(self):
            Generates a new population by selecting parents, applying crossover, and mutation, 
            and retaining elites based on the fitness values.

        run(self, num_generations, verbose=True, log_freq=1, plot_fitness=False):
            Runs the genetic algorithm for a specified number of generations. 
            It logs the progress and can plot the fitness distribution.

        save(self, dir):
            Saves the current state of the genetic algorithm, including the population, fitness, 
            best solution, and fitness history, to the specified directory.

    Args:
        initial_population (numpy.ndarray, optional): The initial population of solutions.
        fitness_fn (function, optional): The function used to evaluate the fitness of each solution.
        mutation_fn (function, optional): The function used to mutate a solution.
        crossover_fn (function, optional): The function used to apply crossover between two solutions.
        selection_fn (function, optional): The function used to select parents based on their fitness.
        keep_parents (int, optional): The number of parents to keep for elitism selection. Defaults to 0.
        maximize (bool, optional): If True, the algorithm will try to maximize the fitness. Defaults to True.
        callback (list of functions, optional): A list of callback functions to call after each generation.

    Raises:
        ValueError: If any of the required functions (fitness_fn, mutation_fn, crossover_fn, or selection_fn) 
                    is not defined or is not callable.
    """
    
    def __init__(self, initial_population=None, fitness_fn=None, mutation_fn=None, crossover_fn=None,
                 selection_fn=None, keep_parents=0, maximize=True, callback=None):
        
        if initial_population is None or len(initial_population) < 2:
            raise ValueError('The object is not defined -> initial_population')
        
        if not fitness_fn or not callable(fitness_fn):
            raise ValueError("The function 'fitness_fn' is either not defined or not callable.")
        
        if not mutation_fn or not callable(mutation_fn):
            raise ValueError("The function 'mutation_fn' is either not defined or not callable.")
        
        if not crossover_fn or not callable(crossover_fn):
            raise ValueError("The function 'crossover_fn' is either not defined or not callable.")
        
        if not selection_fn or not callable(selection_fn):
            raise ValueError("The function 'selection_fn' is either not defined or not callable.")
        
        self.population=initial_population
        self.fitness_fn=fitness_fn
        self.mutation_fn=mutation_fn
        self.crossover_fn=crossover_fn
        self.selection_fn=selection_fn
        self.keep_parents=keep_parents
        self.fitness_hist=[]
        self.best_fitness=None
        self.best_solution=None
        self.num_generations=0
        self.fitness=None
        self.maximize = maximize
        self.callback=callback
        
        
    def score(self):
        '''Compute fitness and save best solution (supports both maximization and minimization)'''
                
        # We calculate the fitness of the entire population.
        self.fitness = list(map(self.fitness_fn, self.population))
        current_best = np.amax(self.fitness) if self.maximize else np.amin(self.fitness)

        # If it's the first evaluation, we initialize the values.
        if self.best_fitness is None:
            self.best_fitness = current_best
            self.best_solution = self.population[np.argmax(self.fitness)] if self.maximize else self.population[np.argmin(self.fitness)]
        
        # If we find a better solution (based on maximizing or minimizing), we update it.
        elif (self.maximize and current_best > self.best_fitness) or (not self.maximize and current_best < self.best_fitness):
            self.best_fitness = current_best
            self.best_solution = self.population[np.argmax(self.fitness)] if self.maximize else self.population[np.argmin(self.fitness)]
        
        # We store the metrics history
        self.fitness_hist.append((self.num_generations, current_best, 
                                np.mean(self.fitness), np.std(self.fitness)))

    def generate_new_pop(self):
        '''Generate a new population of solutions'''
        pop_size=len(self.population)
        parents_p=self.selection_fn(self.fitness)
        elites=[]
        #The parents with the highest probability are propagated to the nex generation
        elites_indices=np.argsort(parents_p)[-self.keep_parents:]
        
        for idx in elites_indices:
            elites.append(self.population[idx])
        children=[]
        for j in range(0, pop_size-self.keep_parents, 2):
            parents_idx=np.random.choice(pop_size, size=2, 
                                     replace=False, p=parents_p)
            parents=[self.population[parents_idx[0]], self.population[parents_idx[1]]]
            child1, child2 =self.crossover_fn(parents)
            child1=self.mutation_fn(child1)
            child2=self.mutation_fn(child2)
            children.append(child1)
            children.append(child2)
        self.population=elites+children
         
         
    def run(self, num_generations, verbose=True, log_freq=1, plot_fitness=False):
        '''Run the genetic algorithm'''
        counter=1
        if self.fitness is None:
            self.score() #score once before the training
        start=time.time()
        for i in range(num_generations):
            self.generate_new_pop()    
            self.score()
            #compute time passed and expected
            now=time.time()
            elapsed = now - start
            estimated = (elapsed / counter) * (num_generations - counter)
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            he, rem_e = divmod(int(estimated), 3600)
            me, se = divmod(rem_e, 60)
            
            counter += 1
            self.num_generations += 1
            
            #print log
            if verbose and self.num_generations % log_freq == 0:
                logging.info("-" * 79)
                logging.info(f'Generation {self.num_generations}')
                logging.info("-" * 79)
                logging.info(f'Population size: {len(self.population)}')
                logging.info(f'Fitness size: {len(self.fitness)}')
                logging.info(f'Current best fitness: {self.fitness_hist[-1][1]},\n'
                             f'Fitness mean: {self.fitness_hist[-1][2]},\n'
                             f'Fitness std: {self.fitness_hist[-1][3]},\n'
                             f'All-time best fitness: {self.best_fitness}.')
                
                elapsed_hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
                estimated_hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(estimated))
                logging.info(f'Elapsed Time: {elapsed_hms}, Estimated to completion: {estimated_hms}')
                
                if plot_fitness:
                    plt.hist(self.fitness)
                    plt.xlabel('Fitness')
                    plt.title('Population fitness histogram')
                    plt.show()
            if self.callback is not None:
                for call in self.callback:
                    call(self)

    def save(self, dir):
        if not os.path.exists(dir+'/saved_model'):
            os.makedirs(dir+'/saved_model')
        np.save(dir+'/saved_model/population', self.population)
        np.save(dir+'/saved_model/fitness', self.fitness)
        np.save(dir+'/saved_model/generation', self.num_generations)
        np.save(dir+'/saved_model/hist', self.fitness_hist)
        np.save(dir+'/saved_model/best_fitness', self.best_fitness)
        np.save(dir+'/saved_model/best_sol', self.best_solution)
        print('Model saved')

# ======================================================================================================================================================
        
# Algoritmo de PSO (Particle Swarm Optimization)

# Este es un algoritmo evolutivo de tipo poblacion, es bueno para explorar el espacio de soluciones

# Inicialización:
# 1. Se inicializan las partículas con posiciones y velocidades aleatorias dentro de un espacio de búsqueda predefinido.
# 2. Se asigna a cada partícula su pbest como su posición inicial.
# 3. Se define un gbest que es la mejor posición encontrada por todas las partículas.

# Actualización de velocidades y posiciones:
# En cada iteración, las partículas actualizan su velocidad y posición según la siguiente fórmula:

# Nueva velocidad de la partícula i
# v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
# Donde:
# - v_i(t+1) es la nueva velocidad de la partícula i.
# - w es el factor de inercia, que controla la influencia de la velocidad anterior.
# - c1 y c2 son los factores de aceleración, que controlan la atracción hacia el pbest y el gbest.
# - r1 y r2 son números aleatorios entre 0 y 1.
# - pbest_i es la mejor posición personal de la partícula i.
# - gbest es la mejor posición global.

# Luego, la posición de la partícula se actualiza usando su velocidad:
# x_i(t+1) = x_i(t) + v_i(t+1)
# Donde:
# - x_i(t+1) es la nueva posición de la partícula i.
# - x_i(t) es la posición actual de la partícula i.
# - v_i(t+1) es la nueva velocidad de la partícula i.


class PSOAlg():
    
    """
    Particle Swarm Optimization (PSO) Algorithm.

    This class implements the Particle Swarm Optimization (PSO) algorithm, a heuristic optimization technique
    inspired by the social behavior of birds and fish. PSO is commonly used for continuous optimization problems.

    Parameters:
    -----------
    initial_particles_position : np.ndarray
        Initial positions of the particles in the search space. The shape of the array should be (num_particles, num_dimensions).
        
    initial_particles_velocity : np.ndarray
        Initial velocities of the particles. The shape should match the initial positions' shape.
        
    fitness_fn : callable
        A function that calculates the fitness of a given particle position. It should take a single argument
        (the position vector) and return a scalar fitness value.
        
    r1_r2_fn : callable, optional
        A function that generates two random matrices (r1, r2) used in the velocity update equation. The default function
        generates random values between 0 and 1.
        
    c1_c2_fn : callable, optional
        A function that computes the cognitive and social acceleration coefficients (c1 and c2) based on the current iteration.
        The default function linearly decreases these coefficients as the algorithm progresses.
        
    w_fn : callable, optional
        A function that computes the inertia weight based on the current iteration. The default function linearly decreases
        the inertia weight over the generations.
        
    maximize : bool, optional
        If True, the algorithm will attempt to maximize the fitness function. If False, the algorithm will minimize the fitness.
        Default is True.
        
    callback : list of callable, optional
        A list of functions to be called at each iteration. Each function will be passed the current PSOAlg instance as an argument.

    Attributes:
    -----------
    particles_position : np.ndarray
        The current positions of the particles in the search space.
        
    particles_velocity : np.ndarray
        The current velocities of the particles.
        
    pbest_position : np.ndarray
        The best position found by each particle so far.
        
    pbest_value : np.ndarray
        The fitness value of the best position found by each particle so far.
        
    gbest_position : np.ndarray
        The best position found by any particle (global best position).
        
    gbest_value : float
        The fitness value of the global best position.
        
    iteration : int
        The current iteration of the algorithm.
        
    num_generations : int
        The total number of generations for the optimization process.
        
    history : list of tuples
        A list that stores the fitness history over the iterations. Each entry is a tuple of the form (iteration, gbest_value, fitness_mean, fitness_std).
        
    cognitive : np.ndarray
        The cognitive component of the particle's velocity update (based on the personal best).
        
    social : np.ndarray
        The social component of the particle's velocity update (based on the global best).

    Methods:
    --------
    update_c1_c2(c1_max=2.5, c1_min=0.5, c2_max=2.5, c2_min=0.5):
        Calculates and returns the cognitive and social coefficients based on the current iteration.
        
    update_w(w_max=0.9, w_min=0.2):
        Calculates and returns the inertia weight based on the current iteration.
        
    update_r1_r2():
        Generates random values for r1 and r2, used in the velocity update equation.
        
    score():
        Computes the fitness of all particles and updates the personal best (pbest) and global best (gbest).
        
    run(num_generations, verbose=True, log_freq=1, plot_fitness=False):
        Runs the PSO algorithm for a given number of generations, updating particle positions, velocities, and the best solutions.
        It logs progress and optionally plots the fitness distribution.
        
    save(dir):
        Saves the current state of the algorithm, including particle positions, velocities, best solutions, history, and number of generations,
        to the specified directory.
    
    Notes:
    ------
    - The algorithm uses a swarm of particles that move through the solution space. Each particle has a position and a velocity.
    - The particles update their positions based on their own experience (personal best) and the experience of the whole swarm (global best).
    - The parameters c1 (cognitive coefficient) and c2 (social coefficient) control the influence of the personal and global best experiences on the particle's velocity.
    - The inertia weight (w) controls how much of the previous velocity is retained in the new velocity.
    - The algorithm is flexible and allows users to define their own functions for the random values (r1, r2), cognitive and social coefficients, and inertia weight.
    - It supports both maximization and minimization optimization problems.
    """
        
    def __init__(self, initial_particles_position=None, initial_particles_velocity=None, fitness_fn=None, r1_r2_fn=None, c1_c2_fn=None, w_fn=None, maximize=True, callback=None):
                
        if initial_particles_position is None or not isinstance(initial_particles_position, np.ndarray) or len(initial_particles_position) < 2:
            raise ValueError("The object 'initial_particles_position' is not defined or is not npndarray or lenght < 2")
        
        if initial_particles_velocity is None or not isinstance(initial_particles_velocity, np.ndarray) or len(initial_particles_velocity) < 2:
            raise ValueError("The object 'initial_particles_velocity' is not defined or is not npndarray or lenght < 2")
        
        if not fitness_fn or not callable(fitness_fn):
            raise ValueError("The function 'fitness_fn' is either not defined or not callable.")
            
        if r1_r2_fn and not callable(r1_r2_fn):
            raise ValueError("The function 'r1_r2_fn' is not callable.")
        elif not r1_r2_fn:
            self.r1_r2_fn = self.update_r1_r2  # Asigna la función por defecto
        else:
            self.r1_r2_fn = r1_r2_fn  # Asigna la función pasada como argumento
        
        if c1_c2_fn and not callable(c1_c2_fn):
            raise ValueError("The function 'c1_c2_fn' not callable.")
        elif not c1_c2_fn:
            self.c1_c2_fn = self.update_c1_c2
        else:
            self.c1_c2_fn = c1_c2_fn            
        
        if w_fn and not callable(w_fn):
            raise ValueError("The function 'w_fn' not callable.")
        elif not w_fn:
            self.w_fn = self.update_w
        else:
             self.w_fn =  w_fn 
        
        self.particles_position = initial_particles_position
        self.particles_velocity = initial_particles_velocity
        self.fitness_fn = fitness_fn
        self.pbest_position = copy.deepcopy(self.particles_position)
        self.iteration = 0
        self.num_generations = 0
        np.array([])
        self.maximize = maximize
        self.gbest_position = 0
        self.cognitive = 0
        self.social = 0
        self.callback=callback
        
        
        if self.maximize:
            self.pbest_value = np.full(len(self.particles_position), -np.inf)
            self.gbest_value = -np.inf
        else:
            self.pbest_value = np.full(len(self.particles_position), np.inf)
            self.gbest_value = np.inf
            
        self.history = []
        
    def update_c1_c2(self, c1_max=2.5, c1_min=0.5, c2_max=2.5, c2_min=0.5):
        c1 = c1_max - (c1_max - c1_min) * (self.iteration / self.num_generations)
        c2 = c2_min + (c2_max - c2_min) * (self.iteration / self.num_generations)
        return c1, c2
    
    def update_w(self,  w_max = 0.9, w_min = 0.2):
        return w_max - (self.iteration / self.num_generations) * (w_max - w_min)
    
    def update_r1_r2(self):
        n_row, n_col = self.particles_position.shape
        return np.random.rand(n_row, n_col), np.random.rand(n_row, n_col)
    
    def score(self):
        '''Compute fitness and save best solution (supports both maximization and minimization)'''
        self.fitness = np.array([self.fitness_fn(ind) for ind in self.particles_position])
        if self.maximize:
            better_mask = self.fitness > self.pbest_value
        else:
            better_mask = self.fitness < self.pbest_value

        # Actualizar los valores personales de mejor aptitud (pbest)
        self.pbest_value[better_mask] = self.fitness[better_mask]
        self.pbest_position[better_mask] = self.particles_position[better_mask]

        # Encontrar el índice de la mejor partícula según maximize
        if self.maximize:
            best_fitness_idx = np.argmax(self.pbest_value)  # Índice del máximo
            if self.pbest_value[best_fitness_idx] > self.gbest_value:
                self.gbest_value = self.pbest_value[best_fitness_idx]
                self.gbest_position = self.pbest_position[best_fitness_idx]
        else:
            best_fitness_idx = np.argmin(self.pbest_value)  # Índice del mínimo
            if self.pbest_value[best_fitness_idx] < self.gbest_value:
                self.gbest_value = self.pbest_value[best_fitness_idx]
                self.gbest_position = self.pbest_position[best_fitness_idx]
        
        self.history.append((self.iteration, self.gbest_value, np.mean(self.fitness), np.std(self.fitness)))
    
    def run(self, num_generations, verbose=True, log_freq=1, plot_fitness=False):
        self.num_generations = num_generations
        start=time.time()
        counter=1
        for self.iteration in range(self.num_generations):
            
            C1, C2 = self.c1_c2_fn()
            W = self.w_fn()
            r1, r2 = self.r1_r2_fn()
            
            self.score()
            
            self.cognitive = C1 * r1 * (self.pbest_position - self.particles_position)
            self.social = C2 * r2 * (self.gbest_position - self.particles_position)
            self.particles_velocity = W * self.particles_velocity + self.cognitive + self.social
            self.particles_position += self.particles_velocity
        
            now=time.time()
            elapsed = now - start
            estimated = (elapsed / counter) * (num_generations - counter)            
            counter+=1
        
            if verbose and self.iteration % log_freq == 0:
                logging.info("-" * 79)
                logging.info(f'Generation {self.iteration}')
                logging.info("-" * 79)
                logging.info(f'Particles position size: {len(self.particles_position)}')
                logging.info(f'Particles velocity size: {len(self.particles_velocity)}')
                logging.info(f'Fitness size: {len(self.fitness)}')
                logging.info(f'Current best fitness: {self.history[-1][1]},\n'
                             f'Fitness mean: {self.history[-1][2]},\n'
                             f'Fitness std: {self.history[-1][3]}.')
                
                elapsed_hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
                estimated_hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(estimated))
                logging.info(f'Elapsed Time: {elapsed_hms}, Estimated to completion: {estimated_hms}')
                
                if plot_fitness:
                    plt.hist(self.fitness)
                    plt.xlabel('Fitness')
                    plt.title('Population fitness histogram')
                    plt.show()
            if self.callback is not None:
                for call in self.callback:
                    call(self)
                    
    def save(self, dir):
        if not os.path.exists(dir + '/saved_model'):
            os.makedirs(dir + '/saved_model')
        
        # Guardar las posiciones y velocidades de las partículas
        np.save(dir + '/saved_model/particles_position', self.particles_position)
        np.save(dir + '/saved_model/particles_velocity', self.particles_velocity)
        
        # Guardar el mejor valor de fitness (global) y la mejor posición (global)
        np.save(dir + '/saved_model/gbest_value', self.gbest_value)
        np.save(dir + '/saved_model/gbest_position', self.gbest_position)
        
        # Guardar el historial de fitness (se puede ajustar según el formato que desees)
        np.save(dir + '/saved_model/history', self.history)
        
        # Guardar el número de generaciones
        np.save(dir + '/saved_model/generation', self.num_generations)
        
        print('Model saved')
        
# ======================================================================================================================================================        
# Algoritmo de Recacalento (Simulated Annealing)
# Este es un algoritmo evolutivo de tipo trayectoria, es bueno para buscar maximos o minimos locales

# Inicialización:
# 1. Se inicializa una solución candidata de manera aleatoria (denotada como `x`).
# 2. Se define una temperatura inicial `T0`, que controla la probabilidad de aceptar soluciones peores a medida que se reduce la temperatura.
# 3. Se establece una función de costo (o energía) que se desea minimizar (denotada como `E(x)`).
# 4. Se define la tasa de enfriamiento `alpha`, que controla la disminución de la temperatura en cada iteración.
# 5. Se establece un número máximo de iteraciones por cada temperatura (`max_iter`).

# Proceso de Recacalento:
# 1. A cada iteración:
#    - Se genera una vecina `x'` de la solución actual `x`.
#    - Se calcula el cambio en la energía: `delta_E = E(x') - E(x)`.

# 2. Regla de aceptación:
#    - Si `delta_E < 0`, acepta la nueva solución `x'` (es una mejora).
#    - Si `delta_E >= 0`, acepta la nueva solución con una probabilidad determinada por `P = exp(-delta_E / T)`, donde `T` es la temperatura actual.

# 3. La temperatura `T` se disminuye en cada iteración según la tasa de enfriamiento: `T = alpha * T`.

class SimulatedAnnealingAlg:
    """
    Implementation of the Simulated Annealing (SA) algorithm for optimization problems.

    Simulated Annealing is a probabilistic technique for approximating the global 
    optimum of a given function. This implementation allows for solving both 
    minimization and maximization problems by iteratively exploring neighboring 
    solutions and accepting new solutions based on a probability function 
    that depends on a decreasing temperature parameter.

    Parameters
    ----------
    initial_solution : Any
        The initial solution for the optimization process. Must be a valid input 
        for the cost function and the neighbor function.
    cost_fn : callable
        The cost function to be optimized. If `minimize` is True, the algorithm 
        will attempt to minimize this function; otherwise, it will maximize it.
    neighbor_fn : callable
        A function that generates a neighboring solution given a current solution.
    T0 : float, optional
        The initial temperature, which controls the probability of accepting 
        worse solutions. Default is 1000.
    alpha : float, optional
        The cooling rate, determining how fast the temperature decreases. 
        Should be between 0 and 1. Default is 0.9.
    max_iter : int, optional
        The maximum number of iterations per temperature level. Default is 100.
    minimize : bool, optional
        If True, the algorithm minimizes the cost function; if False, it maximizes it.
        Default is True.
    callback : list of callables, optional
        A list of functions to be executed at each iteration. These functions receive
        the instance of SimulatedAnnealingAlg as input.

    Attributes
    ----------
    best_solution : Any
        The best solution found during the optimization process.
    best_cost : float
        The best cost found during the optimization process.
    current_solution : Any
        The current solution being explored in the algorithm.
    current_cost : float
        The cost of the current solution.
    history : list
        A record of optimization progress, storing tuples (iteration, best_cost).
    iteration : int
        The current iteration number.
    temperature : float
        The current temperature value.

    Methods
    -------
    update_temperature()
        Decreases the temperature according to the cooling rate.
    score()
        Updates the best solution found if the current solution is better.
    run(num_iterations, verbose=True, plot_fitness=False, log_freq=10)
        Runs the simulated annealing algorithm for a given number of iterations.
    save(dir)
        Saves the model state, including the best solution, best cost, and history, 
        to a specified directory.
    """
    def __init__(self, initial_solution=None, cost_fn=None, neighbor_fn=None, T0=1000, alpha=0.9, max_iter=100, minimize=True, callback=None):
        if initial_solution is None:
            raise ValueError("The variable 'initial_solution' is not defined")
        
        if cost_fn is None or not callable(cost_fn):
            raise ValueError("The object 'cost_fn' is not defined or is not callable.")
        
        if neighbor_fn is None or not callable(neighbor_fn):
            raise ValueError("The object 'neighbor_fn' is not defined or is not callable.")
        
        self.initial_solution = initial_solution
        self.cost_fn = cost_fn
        self.neighbor_fn = neighbor_fn
        self.T0 = T0
        self.alpha = alpha
        self.max_iter = max_iter
        self.minimize = minimize
        self.callback = callback
        
        # Variable Initialization
        self.best_solution = initial_solution
        self.best_cost = cost_fn(initial_solution)
        self.current_solution = initial_solution
        self.current_cost = self.best_cost
        self.history = []
        self.iteration = 0
        self.temperature = T0
        
    def update_temperature(self):
        """Decreases the temperature according to the cooling rate"""
        self.temperature *= self.alpha

    def score(self):
        """calculates and stores the best solution found in each iteration"""
        if self.minimize:
            if self.current_cost < self.best_cost:
                self.best_solution = self.current_solution
                self.best_cost = self.current_cost
        else:
            if self.current_cost > self.best_cost:
                self.best_solution = self.current_solution
                self.best_cost = self.current_cost
            
    def run(self, num_iterations, verbose=True, plot_fitness=False, log_freq=10):
        """
        Runs the simulated annealing algorithm.

        Args:
        - num_iterations: Number of iterations of the algorithm.
        """
        start = time.time()
        counter = 1
        
        for self.iteration in range(num_iterations):
            for _ in range(self.max_iter):  # Performs max_iter iterations per temperature
                # Generates a neighboring solution
                neighbor_solution = self.neighbor_fn(self.current_solution)
                neighbor_cost = self.cost_fn(neighbor_solution)

                # Calculates the change in energy/cost
                delta_E = neighbor_cost - self.current_cost

                # Acceptance rule
                if delta_E < 0:  # If the new solution is better, we accept it.
                    self.current_solution = neighbor_solution
                    self.current_cost = neighbor_cost
                else:
                    # If the new solution is worse, we accept it with a probability based on the temperature.
                    acceptance_probability = np.exp(-delta_E / self.temperature)
                    if np.random.rand() < acceptance_probability:
                        self.current_solution = neighbor_solution
                        self.current_cost = neighbor_cost

                # Updates the best solution found
                self.score()
            
            # Decreases the temperature
            self.update_temperature()
            self.history.append((self.iteration, self.best_cost))

            # Estimated execution time
            now = time.time()
            elapsed = now - start
            estimated = (elapsed / counter) * (num_iterations - counter)
            counter += 1
            
            if verbose and self.iteration % log_freq == 0:
                logging.info("-" * 80)
                logging.info(f'Iteration {self.iteration} - Temperature: {self.temperature:.4f}')
                logging.info(f'Current best cost: {self.best_cost}')
                logging.info(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}')
                logging.info(f'Estimated remaining time: {time.strftime("%H:%M:%S", time.gmtime(estimated))}')
                
            if plot_fitness:
                costs = [self.cost_fn(self.neighbor_fn(self.current_solution)) for _ in range(100)]
                plt.figure(figsize=(6, 4))
                plt.hist(costs, bins=20, alpha=0.75, color='b', edgecolor='black')
                plt.xlabel('Cost')
                plt.ylabel('Frequency')
                plt.title(f'Cost Histogram - Iteration {self.iteration}')
                plt.show()

            if self.callback is not None:
                for call in self.callback:
                    call(self)

        logging.info("Process completed")
    
    def save(self, dir):
        """Save the modelGuarda el modelo entrenado en un directorio específico"""
        if not os.path.exists(dir + '/saved_model'):
            os.makedirs(dir + '/saved_model')
        
        np.save(dir + '/saved_model/best_solution', self.best_solution)
        np.save(dir + '/saved_model/best_cost', self.best_cost)
        np.save(dir + '/saved_model/history', self.history)
        
        print('Saved model.')

# ======================================================================================================================================================        
# Algoritmo Genético Multiobjetivo NSGA-II (Non-dominated Sorting Genetic Algorithm II)

# El algoritmo NSGA-II es una extensión de los algoritmos genéticos clásicos que está diseñado para 
# optimizar múltiples objetivos simultáneamente, sin requerir la ponderación de esos objetivos. 
# NSGA-II utiliza el concepto de dominancia de Pareto para encontrar un conjunto de soluciones óptimas 
# en lugar de una única solución.

# Inicialización:
# 1. Se genera una población inicial de soluciones aleatorias (cromosomas).
# 2. Se evalúan múltiples funciones objetivo para cada individuo.
# 3. Se realizan las siguientes operaciones para cada generación:
#    - Dominancia de Pareto: Se identifican las soluciones no dominadas, que representan el frente de Pareto.
#    - Distancia de aglomeramiento: Se calcula la distancia de aglomeramiento para mantener la diversidad en la población.

# Dominancia de Pareto:
# - Un individuo A domina a otro B si A es mejor en al menos un objetivo y no es peor en los demás.
# - Las soluciones no dominadas forman el frente de Pareto, representando soluciones óptimas en el sentido multiobjetivo.

# Operaciones:
# 1. Selección: 
#    - Se realiza un torneo binario basado en la dominancia de Pareto y las distancias de aglomeramiento.
# 2. Cruce:
#    - Se utiliza el cruce "Blend Crossover" para generar dos descendientes a partir de dos padres.
# 3. Mutación:
#    - Se introduce variabilidad aleatoria a los descendientes mediante una mutación controlada.

# Población combinada:
# Se combina la población de padres y descendientes y se aplica el procedimiento de clasificación por dominancia 
# de Pareto y la distancia de aglomeramiento para seleccionar los mejores individuos para la siguiente generación.

# Evaluación y actualización de la población:
# 1. Se evalúa la aptitud de los individuos en términos de los múltiples objetivos.
# 2. Se identifican las soluciones no dominadas y se calcula la distancia de aglomeramiento.
# 3. La población se actualiza con los mejores individuos, manteniendo la diversidad.

# Criterio de parada:
# El algoritmo finaliza cuando se cumple alguna de las siguientes condiciones:
# - Se alcanza el número máximo de generaciones.
# - No hay mejoras significativas en el frente de Pareto.
# - Se cumple un criterio de convergencia predefinido.

# ======================================================================================================================================================

class AGMO_NSGAII():
    """
    Implementation of the NSGA-II (Non-dominated Sorting Genetic Algorithm II) 
    for multi-objective optimization.

    This class provides an evolutionary algorithm to optimize a population 
    based on multiple objectives using non-dominated sorting, crowding distance, 
    selection, crossover, and mutation operators.

    Attributes:
        population (list): The initial population of individuals.
        maximizes (tuple): A tuple indicating whether each objective should be maximized.
        callback (function, optional): A function to be called at each iteration.
        performance_population (list): Stores the performance values of the population.
        non_dominated_sorted_solution (list): List of Pareto fronts from non-dominated sorting.
        ranks_solution (list): List of ranks assigned to individuals.
        crowding_distance_values (list): List of crowding distance values for individuals.
        children (list): Stores offspring generated in each generation.
        combined_population (list): Stores the combined parent and offspring population.
        performance_combined_population (list): Stores the performance values of the combined population.
        non_dominated_sorted_solution2 (list): List of Pareto fronts after combining populations.
        ranks_solution2 (list): Ranks for the combined population.
        crowding_distance_values2 (list): Crowding distance values for the combined population.
        new_population (list): The new population selected for the next generation.
        individuals_best_front_hist (list): History of best individuals from the first Pareto front.
        performance_best_front_hist (list): Performance values of the best front over generations.

    Methods:
        fast_non_dominated_sort(values1, values2, maximization_flags):
            Performs fast non-dominated sorting to identify Pareto fronts.

        index_of(a, list):
            Returns the index of an element in a list or -1 if not found.

        sort_by_values(list1, values):
            Sorts a list of indices based on their corresponding values.

        crowding_distance(values1, values2, front):
            Calculates the crowding distance for individuals in a Pareto front.

        binary_tournament(population, fronts, crowding_distances):
            Performs binary tournament selection based on front rank and crowding distance.

        blend_crossover(parent1, parent2, alpha=0.5):
            Applies BLX-α crossover to generate two offspring.

        mutate(individual, mutation_rate=0.1, mutation_strength=0.5):
            Applies mutation to an individual by modifying its genes with a given probability.

        run(num_generations, verbose=True, log_freq=1, plot_fitness=False):
            Runs the NSGA-II algorithm for a specified number of generations.
    """
    
    def __init__(self, initial_population, performance_fn=None, mutation_fn=None, crossover_fn=None, selection_fn=None, maximizes=(True, True), callback=None):
        if initial_population is None or len(initial_population) < 2:
            raise ValueError('The object is not defined -> initial_population')
        
        if performance_fn is None: 
            self.performance_fn = self.performance_example
        elif not callable(performance_fn):
            raise ValueError("The function 'fitness_fn' is not callable.")
        else:
            self.performance_fn=performance_fn
            
        if mutation_fn is None:
            self.mutation_fn=self.mutate
        elif not callable(mutation_fn):
            raise ValueError("The function 'mutation_fn' is not callable.")
        else:
            self.mutation_fn=mutation_fn
        
        if crossover_fn is None: 
            self.crossover_fn=self.blend_crossover
        elif not callable(crossover_fn):
            raise ValueError("The function 'crossover_fn' is not callable.")
        else:
            self.crossover_fn=crossover_fn
        
        if selection_fn is None:
            self.selection_fn = self.binary_tournament
        elif not callable(selection_fn):
            raise ValueError("The function 'selection_fn' is not callable.")
        else:
            self.selection_fn=selection_fn
        
        self.population=initial_population
        self.maximizes = maximizes
        self.callback=callback
        
        self.performance_population = None       
        self.non_dominated_sorted_solution = [] 
        self.ranks_solution = []
        self.crowding_distance_values = []  
        
        self.children = [] 
        self.combined_population = [] 
        self.performance_combined_population = []
        self.non_dominated_sorted_solution2 = []
        self.ranks_solution2 = []
        self.crowding_distance_values2 = [] 
        self.new_population = []
        
        self.individuals_best_front_hist = []
        self.performance_best_front_hist = []
    
    @staticmethod    
    def fast_non_dominated_sort(values1, values2, maximization_flags= (True, True)):
        # Function to check dominancy based on maximization/minimization flags
        def dominates(i, j):
            if maximization_flags[0] and maximization_flags[1]:  # Both maximization
                return (values1[i] >= values1[j] and values2[i] > values2[j]) or \
                    (values1[i] > values1[j] and values2[i] >= values2[j])
            elif not maximization_flags[0] and not maximization_flags[1]:  # Both minimization
                return (values1[i] <= values1[j] and values2[i] < values2[j]) or \
                    (values1[i] < values1[j] and values2[i] <= values2[j])
            elif maximization_flags[0] and not maximization_flags[1]:  # Mixed (max for first, min for second)
                return (values1[i] >= values1[j] and values2[i] <= values2[j]) or \
                    (values1[i] > values1[j] and values2[i] < values2[j])
            elif not maximization_flags[0] and maximization_flags[1]:  # Mixed (min for first, max for second)
                return (values1[i] <= values1[j] and values2[i] >= values2[j]) or \
                    (values1[i] < values1[j] and values2[i] > values2[j])
        
        S = [[] for _ in range(len(values1))]
        front = [[]]
        n = [0 for _ in range(len(values1))]
        rank = [0 for _ in range(len(values1))]

        for p in range(len(values1)):
            S[p] = []
            n[p] = 0
            for q in range(len(values1)):
                if dominates(p, q):  # If solution p dominates solution q
                    S[p].append(q)
                elif dominates(q, p):  # If solution q dominates solution p
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p) 

        i = 0
        while front[i]:
            Q = []
            for p in front[ i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i += 1
            front.append(Q)
        del front[-1]
        return front, rank

    @staticmethod
    def index_of(a, list):
        # Function to find the index of a value in a list
        try:
            return list.index(a)
        except ValueError:
            return -1    

    def sort_by_values(self, list1, values):
        # Function to sort by values
        sorted_list = []
        values_copy = values[:]
        while len(sorted_list) != len(list1):
            min_index = self.index_of(min(values_copy), values_copy)
            if min_index in list1:
                sorted_list.append(min_index)
            values_copy[min_index] = math.inf
        return sorted_list
    
    def crowding_distance(self, values1, values2, front):
        distance = [0 for _ in range(len(front))]
        sorted1 = self.sort_by_values(front, values1[:])
        sorted2 = self.sort_by_values(front, values2[:])
        distance[0] = distance[-1] = float('inf')
        for k in range(1, len(front) - 1):
            distance[k] += (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (max(values1) - min(values1))
            distance[k] += (values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
        return distance

    @staticmethod
    def binary_tournament(population, fronts, crowding_distances):        
        i1, i2 = random.sample(range(len(population)), 2)

        front1 = next(f for f in range(len(fronts)) if i1 in fronts[f])
        front2 = next(f for f in range(len(fronts)) if i2 in fronts[f])
        
        distance_front1 = crowding_distances[front1]
        distance_front2 = crowding_distances[front2]

        if front1 < front2:
            return population[i1]
        elif front2 < front1:
            return population[i2]
        else:
            # if crowding_distances[i1] > crowding_distances[i2]:  
            if distance_front1[fronts[front1].index(i1)] > distance_front2[fronts[front2].index(i2)]:  
                return population[i1]
            else:
                return population[i2]

    @staticmethod
    def blend_crossover(parent1, parent2, alpha=0.5):
        child1, child2 = [], []
        for i in range(len(parent1)):
            d = abs(parent1[i] - parent2[i])  # Diferencia entre genes
            lower = min(parent1[i], parent2[i]) - alpha * d
            upper = max(parent1[i], parent2[i]) + alpha * d
            child1.append(random.uniform(lower, upper))  # Generar valores aleatorios en el rango
            child2.append(random.uniform(lower, upper))
        return tuple(child1), tuple(child2)

    @staticmethod
    def mutate(individual, mutation_rate=0.1, mutation_strength=0.5):
        mutated = []
        for gene in individual:
            if random.random() < mutation_rate:  # Se decide mutar con probabilidad mutation_rate
                mutation = random.uniform(-mutation_strength, mutation_strength)
                mutated.append(gene + mutation)
            else:
                mutated.append(gene)
        return tuple(mutated)    
    
    def run(self, num_generations, verbose=True, log_freq=1, plot_fitness=False):
        
        num_generation = 1
        start=time.time()
        
        for _ in range(num_generations):
                        
            self.performance_population = []
            self.non_dominated_sorted_solution = []
            self.ranks_solution = []
            self.crowding_distance_values = []    
            values1, values2, values12, values22 = None, None, None, None
            self.children = [] 
            self.combined_population = [] 
            self.performance_combined_population = []
            self.non_dominated_sorted_solution2 = []
            self.ranks_solution2 = []
            self.crowding_distance_values2 = [] 
            self.new_population = []
            individuals_best_front = []
            performance_best_front = []
            
            self.performance_population = self.performance_fn(self.population)
            values1, values2 = zip(*self.performance_population)
            values1, values2 = list(values1), list(values2)
                        
            self.non_dominated_sorted_solution, self.ranks_solution = self.fast_non_dominated_sort(values1[:], values2[:], self.maximizes)
            for i in range(len(self.non_dominated_sorted_solution)):
                self.crowding_distance_values.append(self.crowding_distance(values1[:], values2[:], self.non_dominated_sorted_solution[i][:]))
                
            self.children = [] 
            self.combined_population = []   
            parent1, parent2, child1, child2 = None, None, None, None
            for i in range(0,len(self.population),2):   
                parent1 = self.binary_tournament(self.population, self.non_dominated_sorted_solution, self.crowding_distance_values)
                parent2 = self.binary_tournament(self.population, self.non_dominated_sorted_solution, self.crowding_distance_values)
                child1, child2 = self.crossover_fn(parent1, parent2)    
                child1, child2 = self.mutation_fn(child1), self.mutation_fn(child2)
                self.children.append(child1)
                self.children.append(child2)
            
            self.combined_population = self.population + self.children
            self.performance_combined_population = self.performance_fn(self.combined_population)
            
            values12, values22 = zip(*self.performance_combined_population)
            values12, values22 = list(values12), list(values22)
            
            self.non_dominated_sorted_solution2, self.ranks_solution2 = self.fast_non_dominated_sort(values12[:], values22[:], self.maximizes)
            for i in range(len(self.non_dominated_sorted_solution2)):
                self.crowding_distance_values2.append(self.crowding_distance(values12[:], values22[:], self.non_dominated_sorted_solution2[i][:]))
            
            individuals_best_front = [self.combined_population[i] for i in self.non_dominated_sorted_solution2[0]]
            performance_best_front = self.performance_fn(individuals_best_front)
            
            self.individuals_best_front_hist.append(individuals_best_front)
            self.performance_best_front_hist.append(performance_best_front)
            
            self.new_population = []
            for i, front in enumerate(self.non_dominated_sorted_solution2):
                if len(self.new_population) + len(front) <= len(self.population):
                    self.new_population.extend([self.combined_population[j] for j in front])
                else:
                    distances = self.crowding_distance_values2[i]
                    sorted_front = [front[j] for j in sorted(range(len(front)), key=lambda x: distances[x], reverse=True)]
                    self.new_population.extend([self.combined_population[k] for k in sorted_front[:len(self.population) - len(self.new_population)]])
                    break
                
            now=time.time()
            elapsed = now - start
            estimated = (elapsed / num_generation) * (num_generations - num_generation)            
            
            if verbose and num_generation % log_freq == 0:
                logging.info(f'Numero generacion: {num_generation}')
                logging.info("-" * 79)
                logging.info(f'largo poblacion: {len(self.population)}')
                logging.info(f'largo funcion objetivo: {len(self.performance_population)}')
                
                logging.info(f'largo de la poblacion combinada: {len(self.combined_population)}')
                logging.info(f'largo funcion objetivo de la poblacion combinada: {len(self.performance_combined_population)}')

                logging.info(f'largo de la poblacion final de la iteracion: {len(self.new_population)}')
                logging.info(f'el mejor frente de pareto de la iteracion:')
                logging.info(individuals_best_front)
                elapsed_hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
                estimated_hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(estimated))
                logging.info(f'Elapsed Time: {elapsed_hms}, Estimated to completion: {estimated_hms}')
                
            self.population = self.new_population
            
            num_generation += 1
            
        if plot_fitness:
            # graficar los mejores frentes
            plt.figure(figsize=(15, 10))
            for i, values in enumerate(self.performance_best_front_hist):
                if i % 10 == 0 or i == len(self.performance_best_front_hist) - 1:
                    x = [v[0] for v in values]
                    y = [v[1] for v in values]
        
                plt.scatter(x, y, color='blue', alpha=0.7)
                for j, (xi, yi) in enumerate(zip(x, y)):
                    plt.text(xi, yi, str(i+1), fontsize=9, color='blue', ha='center', va='center')

            plt.xlabel('objective function 1')
            plt.ylabel('objective function 2')
            plt.title('The best fronts')
            plt.show()
    
    def save(self, dir):
        """Save the modelGuarda el modelo entrenado en un directorio específico"""
        if not os.path.exists(dir + '/saved_model'):
            os.makedirs(dir + '/saved_model')
        
        with open(dir + '/saved_model/history_fronts.pkl', 'wb') as f:
            pickle.dump(self.individuals_best_front_hist, f)
            
        with open(dir + '/saved_model/history_performace.pkl', 'wb') as f:
            pickle.dump(self.performance_best_front_hist, f)
        
        np.save(dir + '/saved_model/best_front', self.individuals_best_front_hist[-1])
        
        print('Saved model.')
