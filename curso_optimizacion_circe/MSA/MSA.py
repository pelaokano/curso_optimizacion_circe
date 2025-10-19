import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable


class MSAOptimizer:
    """
    Implementacion del Algoritmo del Espiritu Musical (MSA)
    Una metaheuristica bio-inspirada para optimizacion continua
    """
    
    def __init__(self, 
                 objective_function: Callable,
                 n_variables: int,
                 lower_bounds: List[float],
                 upper_bounds: List[float],
                 n_musicians: int = 5,
                 memory_size: int = 10,
                 max_iterations: int = 1000,
                 par_min: float = 0.1,
                 par_max: float = 0.9,
                 bw_min: float = 0.01,
                 bw_max: float = 1.0):
        """
        Inicializa el optimizador MSA
        
        Args:
            objective_function: Funcion a minimizar
            n_variables: Numero de variables de decision
            lower_bounds: Limites inferiores de las variables
            upper_bounds: Limites superiores de las variables
            n_musicians: Numero de musicos (PMN)
            memory_size: Tamaño de la memoria musical (PMS)
            max_iterations: Numero maximo de iteraciones
            par_min: Valor minimo del Pitch Adjusting Rate
            par_max: Valor maximo del Pitch Adjusting Rate
            bw_min: Valor minimo del Bandwidth
            bw_max: Valor maximo del Bandwidth
        """
        self.objective_function = objective_function
        self.n_variables = n_variables
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.n_musicians = n_musicians
        self.memory_size = memory_size
        self.max_iterations = max_iterations
        self.par_min = par_min
        self.par_max = par_max
        self.bw_min = bw_min
        self.bw_max = bw_max
        
        # Memoria Musical: almacena las mejores melodias encontradas
        self.musical_memory = []
        
        # Historiales para tracking del algoritmo
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def _generate_random_melody(self) -> np.ndarray:
        """
        Genera una melodia (solucion) aleatoria dentro de los limites
        
        Returns:
            Melodia aleatoria como array numpy
        """
        melody = np.random.uniform(
            self.lower_bounds, 
            self.upper_bounds, 
            size=self.n_variables
        )
        return melody
    
    def _evaluate_melody(self, melody: np.ndarray) -> float:
        """
        Evalua la aptitud (fitness) de una melodia
        
        Args:
            melody: Melodia a evaluar
            
        Returns:
            Valor de fitness (menor es mejor)
        """
        return self.objective_function(melody)
    
    def _initialize_musical_memory(self):
        """
        Inicializa la Memoria Musical con melodias aleatorias
        Fase 1 del algoritmo MSA
        """
        print("Inicializando Memoria Musical...")
        
        # Generar melodias aleatorias para llenar la memoria
        for _ in range(self.memory_size):
            melody = self._generate_random_melody()
            fitness = self._evaluate_melody(melody)
            self.musical_memory.append((melody.copy(), fitness))
        
        # Ordenar la memoria por fitness (mejores primero)
        self.musical_memory.sort(key=lambda x: x[1])
        
        # Actualizar mejor solucion global
        self.best_solution = self.musical_memory[0][0].copy()
        self.best_fitness = self.musical_memory[0][1]
        
        print(f"Memoria inicializada. Mejor fitness inicial: {self.best_fitness:.6f}")
    
    def _calculate_par(self, iteration: int) -> float:
        """
        Calcula el Pitch Adjusting Rate para la iteracion actual
        PAR aumenta linealmente con las iteraciones
        
        Args:
            iteration: Iteracion actual
            
        Returns:
            Valor de PAR para esta iteracion
        """
        # Normalizacion de la iteracion entre 0 y 1
        t_normalized = iteration / self.max_iterations
        
        # Formula del documento: PARGen = PARmin + (PARnormalized * t)
        par = self.par_min + (self.par_max - self.par_min) * t_normalized
        
        return min(par, self.par_max)  # Asegurar que no exceda el maximo
    
    def _calculate_bandwidth(self, iteration: int) -> float:
        """
        Calcula el Bandwidth para la iteracion actual
        Bandwidth disminuye exponencialmente con las iteraciones
        
        Args:
            iteration: Iteracion actual
            
        Returns:
            Valor de bandwidth para esta iteracion
        """
        # Normalizacion de la iteracion entre 0 y 1
        t_normalized = iteration / self.max_iterations
        
        # Formula del documento: bwGen = bwmax * e^(bwnormalized*t)
        # Usamos factor negativo para que disminuya con el tiempo
        bw = self.bw_max * np.exp(-3.0 * t_normalized)
        
        return max(bw, self.bw_min)  # Asegurar que no sea menor al minimo
    
    def _improvise_new_melody(self, iteration: int) -> np.ndarray:
        """
        Genera una nueva melodia mediante improvisacion
        Fase 2 del algoritmo MSA (SIS/GIS)
        
        Args:
            iteration: Iteracion actual
            
        Returns:
            Nueva melodia improvisada
        """
        # Calcular parametros dinamicos
        par = self._calculate_par(iteration)
        bandwidth = self._calculate_bandwidth(iteration)
        
        # Crear nueva melodia
        new_melody = np.zeros(self.n_variables)
        
        for i in range(self.n_variables):
            if random.random() < par:
                # Copiar de la memoria musical (con probabilidad PAR)
                # Seleccionar una melodia aleatoria de la memoria
                selected_melody_idx = random.randint(0, len(self.musical_memory) - 1)
                base_value = self.musical_memory[selected_melody_idx][0][i]
                
                # Aplicar ajuste fino con bandwidth
                adjustment = random.uniform(-bandwidth, bandwidth)
                new_melody[i] = base_value + adjustment
                
            else:
                # Generar valor completamente aleatorio
                new_melody[i] = random.uniform(
                    self.lower_bounds[i], 
                    self.upper_bounds[i]
                )
            
            # Asegurar que este dentro de los limites
            new_melody[i] = np.clip(
                new_melody[i], 
                self.lower_bounds[i], 
                self.upper_bounds[i]
            )
        
        return new_melody
    
    def _update_musical_memory(self, new_melody: np.ndarray, new_fitness: float):
        """
        Actualiza la Memoria Musical si la nueva melodia es mejor
        Fases 3 y 4 del algoritmo MSA (Comparacion y Actualizacion)
        
        Args:
            new_melody: Nueva melodia candidata
            new_fitness: Fitness de la nueva melodia
        """
        # Comparar con la peor melodia en la memoria
        worst_fitness = self.musical_memory[-1][1]
        
        if new_fitness < worst_fitness:
            # La nueva melodia es mejor que la peor, reemplazarla
            self.musical_memory[-1] = (new_melody.copy(), new_fitness)
            
            # Reordenar la memoria para mantener orden ascendente
            self.musical_memory.sort(key=lambda x: x[1])
            
            # Actualizar mejor solucion global si es necesario
            if new_fitness < self.best_fitness:
                self.best_solution = new_melody.copy()
                self.best_fitness = new_fitness
    
    def _calculate_statistics(self):
        """
        Calcula estadisticas de la memoria musical para tracking
        """
        fitnesses = [melody[1] for melody in self.musical_memory]
        
        best_fitness = min(fitnesses)
        mean_fitness = np.mean(fitnesses)
        
        self.best_fitness_history.append(best_fitness)
        self.mean_fitness_history.append(mean_fitness)
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el algoritmo MSA completo
        
        Args:
            verbose: Si imprimir informacion de progreso
            
        Returns:
            Tupla con (mejor_solucion, mejor_fitness)
        """
        print("Iniciando optimizacion con MSA...")
        print(f"Parametros: {self.n_variables} variables, {self.memory_size} memoria, {self.max_iterations} iteraciones")
        
        # Fase 1: Inicializacion
        self._initialize_musical_memory()
        
        # Ciclo principal de optimizacion
        for iteration in range(self.max_iterations):
            # Fase 2: Improvisacion - generar nueva melodia
            new_melody = self._improvise_new_melody(iteration)
            new_fitness = self._evaluate_melody(new_melody)
            
            # Fases 3 y 4: Comparacion y Actualizacion
            self._update_musical_memory(new_melody, new_fitness)
            
            # Calcular estadisticas para tracking
            self._calculate_statistics()
            
            # Imprimir progreso cada 100 iteraciones
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteracion {iteration + 1}: Mejor fitness = {self.best_fitness:.6f}")
        
        print(f"\nOptimizacion completada!")
        print(f"Mejor solucion encontrada: {self.best_solution}")
        print(f"Mejor fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness
    
    def plot_convergence(self):
        """
        Grafica la convergencia del algoritmo
        """
        plt.figure(figsize=(10, 6))
        
        iterations = range(len(self.best_fitness_history))
        
        plt.plot(iterations, self.best_fitness_history, 'r-', 
                label='Mejor Fitness', linewidth=2)
        plt.plot(iterations, self.mean_fitness_history, 'b-', 
                label='Fitness Promedio', linewidth=2)
        
        plt.xlabel('Iteracion')
        plt.ylabel('Fitness')
        plt.title('Convergencia del Algoritmo MSA')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Ejemplo de uso: funcion objetivo del documento
def objective_function(x):
    """
    Funcion objetivo del documento: f(x) = (x1-10)^3 + (x2-20)^3
    Solucion optima: x1=10, x2=20, f(x*)=0
    
    Args:
        x: Vector de variables [x1, x2]
        
    Returns:
        Valor de la funcion objetivo
    """
    return (x[0] - 10)**3 + (x[1] - 20)**3


if __name__ == "__main__":
    # Configuracion del problema
    n_variables = 2
    lower_bounds = [0, 0]    # Limites inferiores
    upper_bounds = [30, 30]  # Limites superiores
    
    # Crear optimizador MSA
    msa = MSAOptimizer(
        objective_function=objective_function,
        n_variables=n_variables,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_musicians=5,
        memory_size=10,
        max_iterations=500,
        par_min=0.1,
        par_max=0.9,
        bw_min=0.01,
        bw_max=2.0
    )
    
    # Ejecutar optimizacion
    best_solution, best_fitness = msa.optimize(verbose=True)
    
    # Mostrar resultados
    print(f"\nResultados finales:")
    print(f"Solucion optima teorica: [10.0, 20.0]")
    print(f"Solucion encontrada: [{best_solution[0]:.6f}, {best_solution[1]:.6f}]")
    print(f"Error en x1: {abs(best_solution[0] - 10.0):.6f}")
    print(f"Error en x2: {abs(best_solution[1] - 20.0):.6f}")
    print(f"Fitness final: {best_fitness:.6f}")
    
    # Graficar convergencia
    msa.plot_convergence()