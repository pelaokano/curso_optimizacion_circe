import random
import math
import copy

class AlgoritmoGeneticoReal:
    def __init__(self, num_variables, rango_min=-5.12, rango_max=5.12, 
                 tamano_poblacion=100, tasa_mutacion=0.1, tasa_cruce=0.8):
        self.num_variables = num_variables
        self.rango_min = rango_min
        self.rango_max = rango_max
        self.tamano_poblacion = tamano_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        
    def funcion_rastrigin(self, x):
        """
        Funcion de Rastrigin - funcion de prueba multimodal
        Minimo global en (0,0,...,0) con valor 0
        Tiene muchos minimos locales que dificultan la optimizacion
        """
        A = 10
        n = len(x)
        suma = sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)
        return A * n + suma
    
    def funcion_objetivo(self, individuo):
        """
        Wrapper de la funcion objetivo
        Como buscamos minimizar, devolvemos el negativo para maximizar fitness
        """
        return -self.funcion_rastrigin(individuo)
    
    def crear_individuo(self):
        """
        Crea un individuo aleatorio como vector de numeros reales
        """
        return [random.uniform(self.rango_min, self.rango_max) 
                for _ in range(self.num_variables)]
    
    def crear_poblacion_inicial(self):
        """
        Genera poblacion inicial con individuos aleatorios
        """
        return [self.crear_individuo() for _ in range(self.tamano_poblacion)]
    
    def calcular_fitness(self, individuo):
        """
        Calcula fitness del individuo
        Transforma el valor de la funcion objetivo a un valor positivo
        """
        valor_funcion = self.funcion_rastrigin(individuo)
        # Transformamos para tener fitness positivo
        # Mientras menor sea el valor de Rastrigin, mayor el fitness
        return 1.0 / (1.0 + valor_funcion)
    
    def seleccion_ruleta(self, poblacion, fitness_scores):
        """
        Seleccion por ruleta ponderada por fitness
        """
        fitness_total = sum(fitness_scores)
        if fitness_total == 0:
            return random.choice(poblacion)
        
        punto_seleccion = random.uniform(0, fitness_total)
        suma_acumulada = 0
        
        for individuo, fitness in zip(poblacion, fitness_scores):
            suma_acumulada += fitness
            if suma_acumulada >= punto_seleccion:
                return individuo
        
        return poblacion[-1]
    
    def cruce_aritmetico(self, padre1, padre2):
        """
        Cruce aritmetico: combina linealmente los genes de los padres
        hijo = alpha * padre1 + (1-alpha) * padre2
        """
        if random.random() > self.tasa_cruce:
            return copy.deepcopy(padre1), copy.deepcopy(padre2)
        
        alpha = random.random()
        hijo1 = []
        hijo2 = []
        
        for i in range(len(padre1)):
            gen1 = alpha * padre1[i] + (1 - alpha) * padre2[i]
            gen2 = (1 - alpha) * padre1[i] + alpha * padre2[i]
            hijo1.append(gen1)
            hijo2.append(gen2)
        
        return hijo1, hijo2
    
    def mutacion_gaussiana(self, individuo):
        """
        Mutacion gaussiana: añade ruido gaussiano a cada gen
        """
        individuo_mutado = []
        sigma = (self.rango_max - self.rango_min) * 0.1  # Desviacion estandar
        
        for gen in individuo:
            if random.random() < self.tasa_mutacion:
                # Añadir ruido gaussiano
                ruido = random.gauss(0, sigma)
                nuevo_gen = gen + ruido
                # Mantener dentro del rango valido
                nuevo_gen = max(self.rango_min, min(self.rango_max, nuevo_gen))
                individuo_mutado.append(nuevo_gen)
            else:
                individuo_mutado.append(gen)
        
        return individuo_mutado
    
    def evolucionar(self, max_generaciones=1000, tolerancia=1e-6):
        """
        Ejecuta el algoritmo genetico principal
        """
        poblacion = self.crear_poblacion_inicial()
        mejor_fitness_historico = float('-inf')
        mejor_individuo_historico = None
        generaciones_sin_mejora = 0
        
        print(f"Optimizando funcion de Rastrigin con {self.num_variables} variables")
        print(f"Rango de busqueda: [{self.rango_min}, {self.rango_max}]")
        print(f"Minimo global esperado: (0, 0, ..., 0) con valor 0")
        print("-" * 60)
        
        for generacion in range(max_generaciones):
            # Evaluar fitness de la poblacion
            fitness_scores = [self.calcular_fitness(ind) for ind in poblacion]
            
            # Encontrar el mejor de esta generacion
            mejor_fitness_actual = max(fitness_scores)
            mejor_indice = fitness_scores.index(mejor_fitness_actual)
            mejor_individuo_actual = poblacion[mejor_indice]
            
            # Actualizar el mejor historico
            if mejor_fitness_actual > mejor_fitness_historico:
                mejor_fitness_historico = mejor_fitness_actual
                mejor_individuo_historico = copy.deepcopy(mejor_individuo_actual)
                generaciones_sin_mejora = 0
            else:
                generaciones_sin_mejora += 1
            
            # Mostrar progreso
            valor_funcion = self.funcion_rastrigin(mejor_individuo_actual)
            if generacion % 100 == 0 or generacion < 10:
                print(f"Gen {generacion:4d}: Mejor valor = {valor_funcion:.6f}, "
                      f"Fitness = {mejor_fitness_actual:.6f}")
                print(f"         Solucion: {[round(x, 4) for x in mejor_individuo_actual]}")
            
            # Criterio de convergencia
            if valor_funcion < tolerancia:
                print(f"\nSolucion encontrada en generacion {generacion}")
                break
            
            # Criterio de estancamiento
            if generaciones_sin_mejora > 200:
                print(f"\nAlgoritmo estancado por {generaciones_sin_mejora} generaciones")
                break
            
            # Crear nueva poblacion
            nueva_poblacion = []
            
            # Elitismo: mantener los mejores individuos
            indices_elite = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:5]
            for i in indices_elite:
                nueva_poblacion.append(copy.deepcopy(poblacion[i]))
            
            # Generar resto de la poblacion
            while len(nueva_poblacion) < self.tamano_poblacion:
                # Seleccionar padres
                padre1 = self.seleccion_ruleta(poblacion, fitness_scores)
                padre2 = self.seleccion_ruleta(poblacion, fitness_scores)
                
                # Cruzar
                hijo1, hijo2 = self.cruce_aritmetico(padre1, padre2)
                
                # Mutar
                hijo1 = self.mutacion_gaussiana(hijo1)
                hijo2 = self.mutacion_gaussiana(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            # Ajustar tamaño de poblacion
            poblacion = nueva_poblacion[:self.tamano_poblacion]
        
        # Resultados finales
        valor_final = self.funcion_rastrigin(mejor_individuo_historico)
        print(f"\n{'='*60}")
        print(f"RESULTADO FINAL:")
        print(f"Mejor valor encontrado: {valor_final:.8f}")
        print(f"Solucion: {[round(x, 6) for x in mejor_individuo_historico]}")
        print(f"Error respecto al optimo: {abs(valor_final):.8f}")
        
        return mejor_individuo_historico, valor_final

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar el problema
    num_variables = 5  # Dimension del problema
    
    # Crear y ejecutar el algoritmo genetico
    ag = AlgoritmoGeneticoReal(
        num_variables=num_variables,
        rango_min=-5.12,
        rango_max=5.12,
        tamano_poblacion=150,
        tasa_mutacion=0.15,
        tasa_cruce=0.85
    )
    
    # Ejecutar optimizacion
    mejor_solucion, mejor_valor = ag.evolucionar(max_generaciones=2000, tolerancia=1e-5)