import random
import string

class AlgoritmoGenetico:
    def __init__(self, objetivo, tamano_poblacion=100, tasa_mutacion=0.01):
        self.objetivo = objetivo
        self.tamano_poblacion = tamano_poblacion
        self.tasa_mutacion = tasa_mutacion
        self.longitud_cromosoma = len(objetivo)
        self.genes_posibles = string.ascii_letters + string.digits + ' '
        
    def crear_individuo(self):
        """
        Crea un individuo aleatorio representado como una cadena de caracteres
        """
        return ''.join(random.choice(self.genes_posibles) for _ in range(self.longitud_cromosoma))
    
    def crear_poblacion_inicial(self):
        """
        Genera la poblacion inicial con individuos aleatorios
        """
        return [self.crear_individuo() for _ in range(self.tamano_poblacion)]
    
    def calcular_fitness(self, individuo):
        """
        Calcula la aptitud del individuo basado en coincidencias con el objetivo
        Mayor fitness significa mejor adaptacion
        """
        coincidencias = sum(1 for i, j in zip(individuo, self.objetivo) if i == j)
        return coincidencias / len(self.objetivo)
    
    def seleccionar_padres(self, poblacion, fitness_scores):
        """
        Selecciona dos padres usando seleccion por torneo
        """
        def torneo():
            candidatos = random.sample(list(zip(poblacion, fitness_scores)), 3)
            return max(candidatos, key=lambda x: x[1])[0]
        
        padre1 = torneo()
        padre2 = torneo()
        return padre1, padre2
    
    def cruzar(self, padre1, padre2):
        """
        Realiza cruzamiento de un punto entre dos padres
        """
        if len(padre1) <= 1:
            return padre1, padre2
            
        punto_cruce = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
        hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
        return hijo1, hijo2
    
    def mutar(self, individuo):
        """
        Aplica mutacion aleatoria a un individuo
        """
        individuo_mutado = list(individuo)
        for i in range(len(individuo_mutado)):
            if random.random() < self.tasa_mutacion:
                individuo_mutado[i] = random.choice(self.genes_posibles)
        return ''.join(individuo_mutado)
    
    def evolucionar(self, max_generaciones=1000):
        """
        Ejecuta el algoritmo genetico principal
        """
        poblacion = self.crear_poblacion_inicial()
        
        for generacion in range(max_generaciones):
            # Evaluar fitness de toda la poblacion
            fitness_scores = [self.calcular_fitness(ind) for ind in poblacion]
            
            # Verificar si encontramos la solucion
            mejor_fitness = max(fitness_scores)
            if mejor_fitness == 1.0:
                mejor_individuo = poblacion[fitness_scores.index(mejor_fitness)]
                print(f"Solucion encontrada en generacion {generacion}: {mejor_individuo}")
                return mejor_individuo
            
            # Mostrar progreso cada 50 generaciones
            if generacion % 1 == 0:
                mejor_individuo = poblacion[fitness_scores.index(mejor_fitness)]
                print(f"Generacion {generacion}: {mejor_individuo} (fitness: {mejor_fitness:.3f})")
            
            # Crear nueva poblacion
            nueva_poblacion = []
            
            # Mantener al mejor individuo (elitismo)
            mejor_indice = fitness_scores.index(mejor_fitness)
            nueva_poblacion.append(poblacion[mejor_indice])
            
            # Generar resto de la poblacion
            while len(nueva_poblacion) < self.tamano_poblacion:
                padre1, padre2 = self.seleccionar_padres(poblacion, fitness_scores)
                hijo1, hijo2 = self.cruzar(padre1, padre2)
                
                hijo1 = self.mutar(hijo1)
                hijo2 = self.mutar(hijo2)
                
                nueva_poblacion.extend([hijo1, hijo2])
            
            # Ajustar tamano si es necesario
            poblacion = nueva_poblacion[:self.tamano_poblacion]
        
        # Retornar mejor solucion encontrada
        fitness_scores = [self.calcular_fitness(ind) for ind in poblacion]
        mejor_fitness = max(fitness_scores)
        mejor_individuo = poblacion[fitness_scores.index(mejor_fitness)]
        print(f"Mejor solucion encontrada: {mejor_individuo} (fitness: {mejor_fitness:.3f})")
        return mejor_individuo

# Ejemplo de uso
if __name__ == "__main__":
    # Definir el objetivo a evolucionar
    objetivo = "Hello World"
    
    # Crear y ejecutar el algoritmo genetico
    ag = AlgoritmoGenetico(objetivo, tamano_poblacion=200, tasa_mutacion=0.02)
    resultado = ag.evolucionar(max_generaciones=500)
    
    print(f"Objetivo: {objetivo}")
    print(f"Resultado: {resultado}")