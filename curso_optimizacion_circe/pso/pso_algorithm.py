import random
import math
import copy

class Particula:
    """
    Clase que representa una particula individual en el enjambre
    Cada particula tiene:
    - Posicion actual en el espacio de busqueda
    - Velocidad actual (direccion y magnitud del movimiento)
    - Mejor posicion personal encontrada hasta ahora (pbest)
    - Mejor valor de fitness personal encontrado (pbest_fitness)
    """
    def __init__(self, dimensiones, rango_min, rango_max):
        # Inicializar posicion aleatoria dentro del rango valido
        self.posicion = [random.uniform(rango_min, rango_max) for _ in range(dimensiones)]
        
        # Inicializar velocidad aleatoria (pequena para evitar movimientos bruscos)
        rango_velocidad = (rango_max - rango_min) * 0.1  # 10% del rango de posicion
        self.velocidad = [random.uniform(-rango_velocidad, rango_velocidad) for _ in range(dimensiones)]
        
        # La mejor posicion personal inicialmente es la posicion actual
        self.mejor_posicion_personal = self.posicion[:]
        
        # El mejor fitness personal se inicializa como infinito negativo
        # (sera actualizado en la primera evaluacion)
        self.mejor_fitness_personal = float('-inf')

class PSO:
    """
    Clase principal que implementa el algoritmo de Optimizacion por Enjambre de Particulas
    
    PSO esta inspirado en el comportamiento social de bandadas de pajaros o cardumenes de peces
    Las particulas se mueven por el espacio de busqueda influenciadas por:
    1. Su propia experiencia (componente cognitivo)
    2. La experiencia del grupo (componente social)
    3. Su momentum actual (componente de inercia)
    """
    
    def __init__(self, num_particulas=30, dimensiones=2, rango_min=-10, rango_max=10):
        """
        Inicializa el enjambre de particulas
        
        Parametros:
        - num_particulas: cantidad de particulas en el enjambre
        - dimensiones: numero de variables del problema de optimizacion
        - rango_min, rango_max: limites del espacio de busqueda
        """
        self.num_particulas = num_particulas
        self.dimensiones = dimensiones
        self.rango_min = rango_min
        self.rango_max = rango_max
        
        # Parametros del algoritmo PSO (valores tipicos)
        self.w = 0.7        # Factor de inercia: controla la influencia de la velocidad anterior
        self.c1 = 1.5       # Coeficiente cognitivo: atraccion hacia la mejor posicion personal
        self.c2 = 1.5       # Coeficiente social: atraccion hacia la mejor posicion global
        
        # Crear el enjambre de particulas
        self.particulas = [Particula(dimensiones, rango_min, rango_max) 
                          for _ in range(num_particulas)]
        
        # Mejor posicion global encontrada por todo el enjambre
        self.mejor_posicion_global = None
        
        # Mejor fitness global encontrado por todo el enjambre
        self.mejor_fitness_global = float('-inf')
        
        # Historial para analisis del algoritmo
        self.historial_fitness = []
    
    def funcion_objetivo(self, posicion):
        """
        Funcion de Ackley - una funcion multimodal desafiante para optimizacion
        
        Caracteristicas:
        - Tiene un minimo global en (0, 0, ..., 0) con valor 0
        - Tiene muchos minimos locales que pueden atrapar al algoritmo
        - Es una buena funcion de prueba para algoritmos de optimizacion
        
        Formula: f(x) = -20*exp(-0.2*sqrt(1/n * sum(xi^2))) - exp(1/n * sum(cos(2*pi*xi))) + 20 + e
        """
        x = posicion
        n = len(x)
        
        # Primer termino: -20 * exp(-0.2 * sqrt(1/n * suma(xi^2)))
        suma_cuadrados = sum(xi**2 for xi in x)
        termino1 = -20 * math.exp(-0.2 * math.sqrt(suma_cuadrados / n))
        
        # Segundo termino: -exp(1/n * suma(cos(2*pi*xi)))
        suma_cosenos = sum(math.cos(2 * math.pi * xi) for xi in x)
        termino2 = -math.exp(suma_cosenos / n)
        
        # Resultado final
        resultado = termino1 + termino2 + 20 + math.e
        
        # Como PSO maximiza por defecto, devolvemos el negativo para minimizar
        return -resultado
    
    def evaluar_particula(self, particula):
        """
        Evalua el fitness de una particula y actualiza su mejor posicion personal
        si es necesario
        """
        # Calcular fitness de la posicion actual
        fitness_actual = self.funcion_objetivo(particula.posicion)
        
        # Si este fitness es mejor que el mejor personal, actualizar
        if fitness_actual > particula.mejor_fitness_personal:
            particula.mejor_fitness_personal = fitness_actual
            particula.mejor_posicion_personal = particula.posicion[:]
            
            # Si tambien es mejor que el mejor global, actualizar
            if fitness_actual > self.mejor_fitness_global:
                self.mejor_fitness_global = fitness_actual
                self.mejor_posicion_global = particula.posicion[:]
    
    def actualizar_velocidad(self, particula):
        """
        Actualiza la velocidad de una particula segun la formula de PSO
        
        La nueva velocidad tiene tres componentes:
        1. Inercia: fraccion de la velocidad anterior (exploracion)
        2. Cognitivo: atraccion hacia la mejor posicion personal (explotacion personal)
        3. Social: atraccion hacia la mejor posicion global (explotacion colectiva)
        
        Formula: v_nueva = w*v_actual + c1*r1*(pbest - posicion) + c2*r2*(gbest - posicion)
        """
        for i in range(self.dimensiones):
            # Componente de inercia: mantiene la direccion de movimiento anterior
            inercia = self.w * particula.velocidad[i]
            
            # Componente cognitivo: atraccion hacia la mejor posicion personal
            r1 = random.random()  # Numero aleatorio entre 0 y 1
            cognitivo = self.c1 * r1 * (particula.mejor_posicion_personal[i] - particula.posicion[i])
            
            # Componente social: atraccion hacia la mejor posicion global
            r2 = random.random()  # Numero aleatorio entre 0 y 1
            social = self.c2 * r2 * (self.mejor_posicion_global[i] - particula.posicion[i])
            
            # Calcular nueva velocidad como suma de los tres componentes
            nueva_velocidad = inercia + cognitivo + social
            
            # Limitar velocidad para evitar movimientos muy grandes
            velocidad_maxima = (self.rango_max - self.rango_min) * 0.2  # 20% del rango
            if nueva_velocidad > velocidad_maxima:
                nueva_velocidad = velocidad_maxima
            elif nueva_velocidad < -velocidad_maxima:
                nueva_velocidad = -velocidad_maxima
            
            particula.velocidad[i] = nueva_velocidad
    
    def actualizar_posicion(self, particula):
        """
        Actualiza la posicion de una particula basada en su velocidad
        
        Formula simple: posicion_nueva = posicion_actual + velocidad
        """
        for i in range(self.dimensiones):
            # Calcular nueva posicion
            nueva_posicion = particula.posicion[i] + particula.velocidad[i]
            
            # Mantener la particula dentro de los limites del espacio de busqueda
            if nueva_posicion < self.rango_min:
                nueva_posicion = self.rango_min
                particula.velocidad[i] = 0  # Detener velocidad en esa dimension
            elif nueva_posicion > self.rango_max:
                nueva_posicion = self.rango_max
                particula.velocidad[i] = 0  # Detener velocidad en esa dimension
            
            particula.posicion[i] = nueva_posicion
    
    def optimizar(self, max_iteraciones=500):
        """
        Ejecuta el algoritmo PSO principal
        
        El proceso es iterativo:
        1. Evaluar todas las particulas
        2. Actualizar mejores posiciones personales y global
        3. Actualizar velocidades basadas en atraccion cognitiva y social
        4. Actualizar posiciones basadas en velocidades
        5. Repetir hasta convergencia o maximo de iteraciones
        """
        print(f"Iniciando PSO con {self.num_particulas} particulas")
        print(f"Dimensiones: {self.dimensiones}, Rango: [{self.rango_min}, {self.rango_max}]")
        print(f"Parametros: w={self.w}, c1={self.c1}, c2={self.c2}")
        print(f"Optimizando funcion de Ackley (minimo global en origen)")
        print("-" * 60)
        
        # Evaluacion inicial: evaluar todas las particulas y encontrar la mejor
        for particula in self.particulas:
            self.evaluar_particula(particula)
        
        # Bucle principal del algoritmo
        for iteracion in range(max_iteraciones):
            # Actualizar cada particula en el enjambre
            for particula in self.particulas:
                # Paso 1: Actualizar velocidad basada en atraccion cognitiva y social
                self.actualizar_velocidad(particula)
                
                # Paso 2: Mover la particula segun su nueva velocidad
                self.actualizar_posicion(particula)
                
                # Paso 3: Evaluar nueva posicion y actualizar mejores si es necesario
                self.evaluar_particula(particula)
            
            # Guardar el mejor fitness de esta iteracion para analisis
            valor_funcion_real = -self.mejor_fitness_global  # Convertir de vuelta a valor de Ackley
            self.historial_fitness.append(valor_funcion_real)
            
            # Mostrar progreso cada 50 iteraciones
            if iteracion % 10 == 0 or iteracion < 10:
                print(f"Iteracion {iteracion:3d}: Mejor valor = {valor_funcion_real:.6f}")
                print(f"               Mejor posicion = {[round(x, 4) for x in self.mejor_posicion_global]}")
            
            # Criterio de convergencia: si estamos muy cerca del optimo, parar
            if valor_funcion_real < 1e-6:
                print(f"\nConvergencia alcanzada en iteracion {iteracion}")
                break
        
        # Resultados finales
        valor_final = -self.mejor_fitness_global
        print(f"\n{'='*60}")
        print(f"RESULTADOS FINALES:")
        print(f"Mejor valor encontrado: {valor_final:.8f}")
        print(f"Mejor posicion: {[round(x, 6) for x in self.mejor_posicion_global]}")
        print(f"Error desde optimo global: {valor_final:.8f}")
        print(f"Numero de evaluaciones: {iteracion * self.num_particulas}")
        
        return self.mejor_posicion_global, valor_final

# Ejemplo de uso y demostracion
if __name__ == "__main__":
    print("DEMOSTRACION DEL ALGORITMO PSO")
    print("=" * 50)
    
    # Crear instancia de PSO con problema 2D (facil de visualizar)
    pso = PSO(
        num_particulas=25,    # Enjambre pequeno para ver comportamiento claramente
        dimensiones=2,        # Problema 2D
        rango_min=-5,         # Limite inferior del espacio de busqueda
        rango_max=5           # Limite superior del espacio de busqueda
    )
    
    # Ejecutar optimizacion
    mejor_solucion, mejor_valor = pso.optimizar(max_iteraciones=300)
    
    print(f"\n{'-'*50}")
    print("SEGUNDA EJECUCION CON MAS DIMENSIONES:")
    
    # Probar con problema mas complejo (mas dimensiones)
    pso2 = PSO(
        num_particulas=40,    # Mas particulas para problema mas complejo
        dimensiones=5,        # Problema 5D
        rango_min=-3,
        rango_max=3
    )
    
    mejor_solucion2, mejor_valor2 = pso2.optimizar(max_iteraciones=400)
    
    print(f"\n{'-'*50}")
    print("COMPARACION DE RENDIMIENTO:")
    print(f"Problema 2D: Error final = {mejor_valor:.6f}")
    print(f"Problema 5D: Error final = {mejor_valor2:.6f}")