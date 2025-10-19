import numpy as np
from MetaOpt import SimulatedAnnealingAlg

# 1. Definir la función de costo (a minimizar)
def cost_fn(x):
    return x**2  # Queremos minimizar x^2, cuyo mínimo es en x = 0

# 2. Definir la función de vecinos (para generar una nueva solución cercana)
def neighbor_fn(x):
    return x + np.random.uniform(-0.5, 0.5)  # Añadir un pequeño valor aleatorio para explorar el espacio

# 3. Configuración del algoritmo Simulated Annealing
initial_solution = np.random.uniform(-10, 10)  # Iniciar en un valor aleatorio entre -10 y 10

# Crear el objeto del algoritmo SA
sa = SimulatedAnnealingAlg(
    initial_solution=initial_solution,
    cost_fn=cost_fn,
    neighbor_fn=neighbor_fn,
    T0=1000,  # Temperatura inicial
    alpha=0.9,  # Tasa de enfriamiento
    max_iter=100,  # Máximo número de iteraciones por temperatura
    minimize=True,  # Queremos minimizar la función
    callback=None  # Sin función de callback por ahora
)

# 4. Ejecutar el algoritmo
sa.run(num_iterations=200, verbose=True, plot_fitness=False)

# 5. Ver el mejor resultado
print(f"Mejor solución encontrada: {sa.best_solution}")
print(f"Mejor costo encontrado: {sa.best_cost}")