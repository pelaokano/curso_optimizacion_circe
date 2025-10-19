import random
import math

class AlgoritmoGeneticoSimple:
    def __init__(self, num_variables=3, tamano_poblacion=50, tasa_mutacion=0.1):
        self.num_variables = num_variables
        self.tamano_poblacion = tamano_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.rango_min = -10.0
        self.rango_max = 10.0
        
    def funcion_objetivo(self, x):
        """
        Funcion cuadratica simple: f(x) = suma(xi^2)
        Minimo global en (0,0,0) con valor 0
        """
        return sum(xi**2 for xi in x)
    
    def crear_individuo(self):
        """
        Crea un individuo como lista de numeros reales aleatorios
        """
        return [random.uniform(self.rango_min, self.rango_max) 
                for _ in range(self.num_variables)]
    
    def crear_poblacion(self):
        """
        Crea la poblacion inicial
        """
        return [self.crear_individuo() for _ in range(self.tamano_poblacion)]
    
    def calcular_fitness(self, individuo):
        """
        Calcula fitness transformando el valor de la funcion objetivo
        Como queremos minimizar, usamos 1/(1+f(x))
        """
        valor = self.funcion_objetivo(individuo)
        return 1.0 / (1.0 + valor)
    
    def seleccion_ruleta(self, poblacion, fitness_lista):
        """
        Seleccion por ruleta: probabilidad proporcional al fitness
        """
        # Calcular suma total de fitness
        suma_fitness = sum(fitness_lista)
        
        # Si todos tienen fitness 0, seleccionar al azar
        if suma_fitness == 0:
            return random.choice(poblacion)
        
        # Generar numero aleatorio entre 0 y suma_fitness
        punto_ruleta = random.uniform(0, suma_fitness)
        
        # Encontrar individuo correspondiente
        suma_acumulada = 0
        for i, fitness in enumerate(fitness_lista):
            suma_acumulada += fitness
            if suma_acumulada >= punto_ruleta:
                return poblacion[i]
        
        # Por seguridad, retornar el ultimo
        return poblacion[-1]
    
    def cruzar(self, padre1, padre2):
        """
        Cruce de un punto: intercambia segmentos entre padres
        """
        if len(padre1) <= 1:
            return padre1[:], padre2[:]
        
        punto = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        
        return hijo1, hijo2
    
    def mutar(self, individuo):
        """
        Mutacion simple: cambiar genes con cierta probabilidad
        """
        individuo_mutado = []
        for gen in individuo:
            if random.random() < self.tasa_mutacion:
                # Añadir pequeña perturbacion aleatoria
                perturbacion = random.uniform(-1.0, 1.0)
                nuevo_gen = gen + perturbacion
                # Mantener en rango valido
                nuevo_gen = max(self.rango_min, min(self.rango_max, nuevo_gen))
                individuo_mutado.append(nuevo_gen)
            else:
                individuo_mutado.append(gen)
        
        return individuo_mutado
    
    def evolucionar(self, generaciones=200):
        """
        Ejecuta el algoritmo genetico principal
        """
        # Crear poblacion inicial
        poblacion = self.crear_poblacion()
        
        print(f"Optimizando funcion f(x) = suma(xi^2) con {self.num_variables} variables")
        print(f"Poblacion: {self.tamano_poblacion}, Mutacion: {self.tasa_mutacion}")
        print(f"Objetivo: encontrar minimo en (0, 0, 0) con valor 0")
        print("-" * 50)
        
        for gen in range(generaciones):
            # Calcular fitness de toda la poblacion
            fitness_lista = [self.calcular_fitness(ind) for ind in poblacion]
            
            # Encontrar el mejor individuo actual
            mejor_fitness = max(fitness_lista)
            mejor_indice = fitness_lista.index(mejor_fitness)
            mejor_individuo = poblacion[mejor_indice]
            mejor_valor = self.funcion_objetivo(mejor_individuo)
            
            # Mostrar progreso cada 25 generaciones
            if gen % 25 == 0:
                print(f"Generacion {gen:3d}: Valor = {mejor_valor:.4f}, "
                      f"Solucion = {[round(x, 3) for x in mejor_individuo]}")
            
            # Crear nueva poblacion usando seleccion por ruleta
            nueva_poblacion = []
            
            # Mantener al mejor (elitismo simple)
            nueva_poblacion.append(mejor_individuo[:])
            
            # Generar resto de la poblacion
            while len(nueva_poblacion) < self.tamano_poblacion:
                # Seleccionar padres usando ruleta
                padre1 = self.seleccion_ruleta(poblacion, fitness_lista)
                padre2 = self.seleccion_ruleta(poblacion, fitness_lista)
                
                # Cruzar padres
                hijo1, hijo2 = self.cruzar(padre1, padre2)
                
                # Aplicar mutacion
                hijo1 = self.mutar(hijo1)
                hijo2 = self.mutar(hijo2)
                
                # Añadir hijos a nueva poblacion
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < self.tamano_poblacion:
                    nueva_poblacion.append(hijo2)
            
            # Reemplazar poblacion antigua
            poblacion = nueva_poblacion
        
        # Resultado final
        fitness_final = [self.calcular_fitness(ind) for ind in poblacion]
        mejor_fitness_final = max(fitness_final)
        mejor_indice_final = fitness_final.index(mejor_fitness_final)
        mejor_solucion = poblacion[mejor_indice_final]
        mejor_valor_final = self.funcion_objetivo(mejor_solucion)
        
        print(f"\n{'='*50}")
        print(f"RESULTADO FINAL:")
        print(f"Mejor valor: {mejor_valor_final:.6f}")
        print(f"Mejor solucion: {[round(x, 4) for x in mejor_solucion]}")
        print(f"Error desde optimo: {mejor_valor_final:.6f}")
        
        return mejor_solucion

# Ejemplo de uso
if __name__ == "__main__":
    # Crear algoritmo genetico
    ag = AlgoritmoGeneticoSimple(
        num_variables=3,
        tamano_poblacion=50,
        tasa_mutacion=0.15
    )
    
    # Ejecutar optimizacion
    resultado = ag.evolucionar(generaciones=150)
    