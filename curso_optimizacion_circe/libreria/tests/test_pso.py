import numpy as np
from MetaOpt import PSOAlg

# Define the fitness function
def fitness_fn(position):
    """
    Simple fitness function: Sphere function (sum of squares).
    The objective is to minimize the sum of squared values of the position vector.
    """
    return np.sum(position**2)

# Initialize particle positions and velocities (Example)
num_particles = 30
num_dimensions = 2  # For a 2D search space

initial_particles_position = np.random.uniform(-5, 5, (num_particles, num_dimensions))  # Random positions in the range [-5, 5]
initial_particles_velocity = np.random.uniform(-1, 1, (num_particles, num_dimensions))  # Random initial velocity

# Initialize PSO Algorithm
pso = PSOAlg(
    initial_particles_position=initial_particles_position,
    initial_particles_velocity=initial_particles_velocity,
    fitness_fn=fitness_fn,
    maximize=False  # Minimize the fitness function
)

# Run the PSO for 100 generations
pso.run(num_generations=100, verbose=True, log_freq=10)

# After running, the best solution found will be stored in the global best position
print(f'Global Best Position: {pso.gbest_position}')
print(f'Global Best Fitness: {pso.gbest_value}')

