import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class Individual:
    """
    Representa un individuo en la poblacion de NSGA-II
    Cada individuo tiene genes, valores de fitness, rank y distancia de crowding
    """
    genes: np.ndarray  # Variables de decision (cromosoma)
    objectives: np.ndarray  # Valores de las funciones objetivo
    rank: int = 0  # Rango de Pareto (frente al que pertenece)
    crowding_distance: float = 0.0  # Distancia de crowding
    
    def __post_init__(self):
        # Inicializar objetivos si no se proporcionan
        if self.objectives is None:
            self.objectives = np.array([])


class NSGA2:
    """
    Implementacion del algoritmo NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    para optimizacion multiobjetivo
    """
    
    def __init__(self,
                 objective_functions: List[Callable],
                 n_variables: int,
                 lower_bounds: List[float],
                 upper_bounds: List[float],
                 population_size: int = 100,
                 max_generations: int = 250,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1):
        """
        Inicializa el algoritmo NSGA-II
        
        Args:
            objective_functions: Lista de funciones objetivo a optimizar
            n_variables: Numero de variables de decision
            lower_bounds: Limites inferiores de las variables
            upper_bounds: Limites superiores de las variables
            population_size: Tamaño de la poblacion (debe ser par)
            max_generations: Numero maximo de generaciones
            crossover_rate: Probabilidad de cruce
            mutation_rate: Probabilidad de mutacion
        """
        self.objective_functions = objective_functions
        self.n_objectives = len(objective_functions)
        self.n_variables = n_variables
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        # Asegurar que el tamaño de poblacion sea par
        if self.population_size % 2 != 0:
            self.population_size += 1
            
        # Almacenar el historial del frente de Pareto
        self.pareto_history = []
        
    def _create_individual(self, genes: np.ndarray = None) -> Individual:
        """
        Crea un individuo con genes aleatorios o especificados
        
        Args:
            genes: Genes especificos (opcional)
            
        Returns:
            Nuevo individuo con objetivos evaluados
        """
        if genes is None:
            # Generar genes aleatorios dentro de los limites
            genes = np.random.uniform(
                self.lower_bounds, 
                self.upper_bounds, 
                size=self.n_variables
            )
        
        # Evaluar funciones objetivo
        objectives = np.array([func(genes) for func in self.objective_functions])
        
        return Individual(genes=genes, objectives=objectives)
    
    def _initialize_population(self) -> List[Individual]:
        """
        Inicializa una poblacion aleatoria
        
        Returns:
            Lista de individuos que forman la poblacion inicial
        """
        population = []
        for _ in range(self.population_size):
            individual = self._create_individual()
            population.append(individual)
        
        return population
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Verifica si ind1 domina a ind2 segun Pareto
        
        Criterios de dominancia:
        1. ind1 es al menos tan bueno como ind2 en todos los objetivos
        2. ind1 es estrictamente mejor que ind2 en al menos un objetivo
        
        Args:
            ind1: Primer individuo
            ind2: Segundo individuo
            
        Returns:
            True si ind1 domina a ind2, False en caso contrario
        """
        # Verificar si ind1 es al menos tan bueno en todos los objetivos
        better_or_equal = np.all(ind1.objectives <= ind2.objectives)
        
        # Verificar si ind1 es estrictamente mejor en al menos uno
        strictly_better = np.any(ind1.objectives < ind2.objectives)
        
        return better_or_equal and strictly_better
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Algoritmo rapido de clasificacion no dominada
        Clasifica la poblacion en frentes de Pareto
        
        Args:
            population: Poblacion a clasificar
            
        Returns:
            Lista de frentes, donde cada frente es una lista de individuos
        """
        n = len(population)
        
        # Inicializar estructuras de datos
        fronts = [[]]  # Lista de frentes
        domination_count = [0] * n  # Numero de individuos que dominan a cada individuo
        dominated_solutions = [[] for _ in range(n)]  # Individuos dominados por cada uno
        
        # Para cada individuo en la poblacion
        for i in range(n):
            # Comparar con todos los otros individuos
            for j in range(n):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        # i domina a j
                        dominated_solutions[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        # j domina a i
                        domination_count[i] += 1
            
            # Si no es dominado por nadie, pertenece al primer frente
            if domination_count[i] == 0:
                population[i].rank = 1
                fronts[0].append(population[i])
        
        # Construir los siguientes frentes
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []
            
            # Para cada individuo en el frente actual
            for ind_p in fronts[front_index]:
                # Encontrar el indice de ind_p en la poblacion
                p_index = -1
                for idx, ind in enumerate(population):
                    if ind is ind_p:  # Usar 'is' para comparacion de identidad
                        p_index = idx
                        break
                
                if p_index == -1:
                    continue  # No se encontro el individuo, continuar
                
                # Para cada individuo dominado por ind_p
                for q_index in dominated_solutions[p_index]:
                    domination_count[q_index] -= 1
                    
                    # Si ya no es dominado por nadie, pertenece al siguiente frente
                    if domination_count[q_index] == 0:
                        population[q_index].rank = front_index + 2
                        next_front.append(population[q_index])
            
            front_index += 1
            fronts.append(next_front)
        
        # Remover el ultimo frente vacio
        if len(fronts[-1]) == 0:
            fronts.pop()
        
        return fronts
    
    def _calculate_crowding_distance(self, front: List[Individual]):
        """
        Calcula la distancia de crowding para todos los individuos en un frente
        
        La distancia de crowding mide que tan denso esta el espacio alrededor
        de cada individuo. Se prefieren individuos con mayor distancia de crowding
        para mantener diversidad en el frente de Pareto.
        
        Args:
            front: Lista de individuos en el mismo frente de Pareto
        """
        if len(front) == 0:
            return
        
        # Inicializar distancias de crowding a cero
        for individual in front:
            individual.crowding_distance = 0.0
        
        # Calcular distancia para cada objetivo
        for obj_index in range(self.n_objectives):
            # Ordenar por el objetivo actual
            front.sort(key=lambda x: x.objectives[obj_index])
            
            # Asignar distancia infinita a los extremos
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calcular rango del objetivo
            obj_min = front[0].objectives[obj_index]
            obj_max = front[-1].objectives[obj_index]
            obj_range = obj_max - obj_min
            
            # Evitar division por cero
            if obj_range == 0:
                continue
            
            # Calcular distancia de crowding para individuos intermedios
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[obj_index] - 
                           front[i - 1].objectives[obj_index]) / obj_range
                front[i].crowding_distance += distance
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Seleccion por torneo binario usando rank de Pareto y distancia de crowding
        
        Criterios de seleccion:
        1. Menor rank de Pareto (mejor frente)
        2. En caso de empate, mayor distancia de crowding (mas diversidad)
        
        Args:
            population: Poblacion de donde seleccionar
            
        Returns:
            Individuo ganador del torneo
        """
        # Seleccionar dos individuos aleatoriamente
        ind1 = random.choice(population)
        ind2 = random.choice(population)
        
        # Comparar usando los criterios de NSGA-II
        if ind1.rank < ind2.rank:
            return ind1
        elif ind1.rank > ind2.rank:
            return ind2
        else:
            # Mismo rank, usar distancia de crowding
            if ind1.crowding_distance > ind2.crowding_distance:
                return ind1
            else:
                return ind2
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Operador de cruce binario (BLX-alpha crossover)
        
        Genera dos hijos combinando los genes de los padres
        usando la formula del documento:
        y1 = alpha * x1 + (1 - alpha) * x2
        y2 = alpha * x2 + (1 - alpha) * x1
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Tupla con dos hijos generados
        """
        # Generar factor de cruce aleatorio para cada gen
        alpha = np.random.random(self.n_variables)
        
        # Crear genes de los hijos
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = alpha * parent2.genes + (1 - alpha) * parent1.genes
        
        # Asegurar que esten dentro de los limites
        child1_genes = np.clip(child1_genes, self.lower_bounds, self.upper_bounds)
        child2_genes = np.clip(child2_genes, self.lower_bounds, self.upper_bounds)
        
        # Crear individuos hijos
        child1 = self._create_individual(child1_genes)
        child2 = self._create_individual(child2_genes)
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        Operador de mutacion uniforme
        
        Selecciona aleatoriamente un gen y le asigna un nuevo valor
        aleatorio dentro de los limites del problema
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Nuevo individuo mutado
        """
        # Copiar genes del individuo original
        mutated_genes = individual.genes.copy()
        
        # Para cada gen, decidir si mutar
        for i in range(self.n_variables):
            if random.random() < self.mutation_rate:
                # Mutar el gen con un valor aleatorio
                mutated_genes[i] = random.uniform(
                    self.lower_bounds[i], 
                    self.upper_bounds[i]
                )
        
        # Crear nuevo individuo con genes mutados
        return self._create_individual(mutated_genes)
    
    def _environmental_selection(self, 
                                combined_population: List[Individual]) -> List[Individual]:
        """
        Seleccion ambiental para formar la siguiente generacion
        
        Este es el corazon de NSGA-II:
        1. Clasifica la poblacion combinada en frentes de Pareto
        2. Copia frentes completos hasta llenar la poblacion
        3. Si el ultimo frente no cabe completo, usa distancia de crowding
        
        Args:
            combined_population: Poblacion padres + hijos (tamaño 2N)
            
        Returns:
            Nueva poblacion de tamaño N
        """
        # Clasificacion no dominada
        fronts = self._fast_non_dominated_sort(combined_population)
        
        new_population = []
        front_index = 0
        
        # Copiar frentes completos hasta que no quepan mas
        while (len(new_population) + len(fronts[front_index])) <= self.population_size:
            # Calcular distancia de crowding para el frente actual
            self._calculate_crowding_distance(fronts[front_index])
            
            # Agregar todo el frente a la nueva poblacion
            new_population.extend(fronts[front_index])
            front_index += 1
            
            # Si ya se llenó la poblacion o no hay mas frentes
            if len(new_population) == self.population_size or front_index >= len(fronts):
                break
        
        # Si necesitamos individuos del ultimo frente parcial
        if len(new_population) < self.population_size and front_index < len(fronts):
            remaining_slots = self.population_size - len(new_population)
            last_front = fronts[front_index]
            
            # Calcular distancia de crowding para el ultimo frente
            self._calculate_crowding_distance(last_front)
            
            # Ordenar por distancia de crowding descendente
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            
            # Tomar los mejores individuos del ultimo frente
            new_population.extend(last_front[:remaining_slots])
        
        return new_population
    
    def _extract_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """
        Extrae el primer frente de Pareto de la poblacion
        
        Args:
            population: Poblacion completa
            
        Returns:
            Lista de individuos que forman el frente de Pareto
        """
        fronts = self._fast_non_dominated_sort(population)
        return fronts[0] if len(fronts) > 0 else []
    
    def optimize(self, verbose: bool = True) -> Tuple[List[Individual], List[List[Individual]]]:
        """
        Ejecuta el algoritmo NSGA-II completo
        
        Args:
            verbose: Si imprimir informacion de progreso
            
        Returns:
            Tupla con (frente_pareto_final, historial_pareto_por_generacion)
        """
        if verbose:
            print("Iniciando optimizacion con NSGA-II...")
            print(f"Poblacion: {self.population_size}, Generaciones: {self.max_generations}")
            print(f"Variables: {self.n_variables}, Objetivos: {self.n_objectives}")
        
        # Inicializar poblacion
        population = self._initialize_population()
        
        # Evaluar poblacion inicial
        fronts = self._fast_non_dominated_sort(population)
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Guardar frente inicial
        pareto_front = self._extract_pareto_front(population)
        self.pareto_history.append([ind for ind in pareto_front])
        
        if verbose:
            print(f"Frente de Pareto inicial: {len(pareto_front)} soluciones")
        
        # Ciclo evolutivo principal
        for generation in range(self.max_generations):
            # Generar poblacion de descendientes
            offspring = []
            
            while len(offspring) < self.population_size:
                # Seleccion de padres
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Cruce
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    # Si no hay cruce, copiar padres
                    child1 = self._create_individual(parent1.genes.copy())
                    child2 = self._create_individual(parent2.genes.copy())
                
                # Mutacion
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Asegurar tamaño exacto de descendientes
            offspring = offspring[:self.population_size]
            
            # Combinar padres y descendientes
            combined_population = population + offspring
            
            # Seleccion ambiental
            population = self._environmental_selection(combined_population)
            
            # Extraer y guardar frente de Pareto actual
            pareto_front = self._extract_pareto_front(population)
            self.pareto_history.append([ind for ind in pareto_front])
            
            # Imprimir progreso
            if verbose and (generation + 1) % 50 == 0:
                print(f"Generacion {generation + 1}: {len(pareto_front)} soluciones en frente de Pareto")
        
        final_pareto_front = self._extract_pareto_front(population)
        
        if verbose:
            print(f"\nOptimizacion completada!")
            print(f"Frente de Pareto final: {len(final_pareto_front)} soluciones")
        
        return final_pareto_front, self.pareto_history
    
    def plot_pareto_front(self, 
                         pareto_front: List[Individual] = None,
                         show_evolution: bool = False,
                         generation_step: int = 50):
        """
        Grafica el frente de Pareto para problemas de 2 objetivos
        
        Args:
            pareto_front: Frente de Pareto especifico a graficar
            show_evolution: Si mostrar la evolucion del frente
            generation_step: Cada cuantas generaciones mostrar
        """
        if self.n_objectives != 2:
            print("La graficacion solo esta disponible para 2 objetivos")
            return
        
        plt.figure(figsize=(12, 8))
        
        if show_evolution and len(self.pareto_history) > 1:
            # Mostrar evolucion del frente de Pareto
            generations_to_show = range(0, len(self.pareto_history), generation_step)
            
            for i, gen in enumerate(generations_to_show):
                if gen < len(self.pareto_history):
                    front = self.pareto_history[gen]
                    if len(front) > 0:
                        objectives = np.array([ind.objectives for ind in front])
                        alpha = 0.3 + 0.7 * (i / len(generations_to_show))
                        plt.scatter(objectives[:, 0], objectives[:, 1], 
                                  alpha=alpha, s=30, 
                                  label=f'Gen {gen}' if i % 3 == 0 else "")
        
        # Mostrar frente final
        if pareto_front is None:
            pareto_front = self.pareto_history[-1] if self.pareto_history else []
        
        if len(pareto_front) > 0:
            final_objectives = np.array([ind.objectives for ind in pareto_front])
            plt.scatter(final_objectives[:, 0], final_objectives[:, 1], 
                       c='red', s=60, marker='o', alpha=0.8, 
                       label='Frente Final', edgecolors='black', linewidth=1)
            
            # Ordenar para dibujar linea del frente
            sorted_indices = np.argsort(final_objectives[:, 0])
            sorted_objectives = final_objectives[sorted_indices]
            plt.plot(sorted_objectives[:, 0], sorted_objectives[:, 1], 
                    'r--', alpha=0.7, linewidth=2)
        
        plt.xlabel('Objetivo 1 (minimizar)')
        plt.ylabel('Objetivo 2 (minimizar)')
        plt.title('Frente de Pareto - NSGA-II')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Funciones objetivo de ejemplo: Problema ZDT1 modificado
def objective_1(x):
    """
    Primera funcion objetivo: f1(x) = x1
    Favorece valores pequeños de la primera variable
    """
    return x[0]

def objective_2(x):
    """
    Segunda funcion objetivo: f2(x) = g(x) * h(f1, g)
    donde g(x) = 1 + 9 * sum(x[1:]) / (n-1)
    y h(f1, g) = 1 - sqrt(f1/g)
    
    Esta funcion crea conflicto con f1
    """
    n = len(x)
    g = 1 + 9 * np.sum(x[1:]) / (n - 1) if n > 1 else 1
    f1 = objective_1(x)
    h = 1 - np.sqrt(f1 / g) if g > 0 else 1
    return g * h


if __name__ == "__main__":
    # Configuracion del problema
    n_variables = 3  # Numero de variables de decision
    lower_bounds = [0.0] * n_variables  # Limites inferiores
    upper_bounds = [1.0] * n_variables  # Limites superiores
    
    # Funciones objetivo
    objective_functions = [objective_1, objective_2]
    
    # Crear optimizador NSGA-II
    nsga2 = NSGA2(
        objective_functions=objective_functions,
        n_variables=n_variables,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        population_size=100,
        max_generations=200,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    
    # Ejecutar optimizacion
    pareto_front, pareto_history = nsga2.optimize(verbose=True)
    
    # Mostrar algunas soluciones del frente de Pareto
    print(f"\nAlgunas soluciones del frente de Pareto:")
    print("Variables -> [Obj1, Obj2]")
    print("-" * 40)
    for i, individual in enumerate(pareto_front[:10]):  # Mostrar primeras 10
        print(f"Sol {i+1}: {individual.genes} -> {individual.objectives}")
    
    # Graficar resultados
    print(f"\nGraficando frente de Pareto...")
    nsga2.plot_pareto_front(pareto_front, show_evolution=True, generation_step=40)
    
    # Graficar solo el frente final
    nsga2.plot_pareto_front(pareto_front, show_evolution=False)