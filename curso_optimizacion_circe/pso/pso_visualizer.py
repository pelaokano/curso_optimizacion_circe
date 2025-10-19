import matplotlib.pyplot as plt
import numpy as np
import copy
import time

# Importar las clases PSO y Particula del archivo anterior
# from pso_algorithm import PSO, Particula

class VisualizadorPSO:
    """
    Clase para visualizar el comportamiento del algoritmo PSO en 2D
    Muestra como las particulas se mueven por el espacio de busqueda
    y convergen hacia el optimo global
    
    Funcionalidades:
    - Mapa de contorno de la funcion objetivo
    - Visualizacion en tiempo real del movimiento de particulas
    - Trayectorias de las particulas
    - Seguimiento de la mejor posicion global
    """
    
    def __init__(self, pso_instance):
        """
        Inicializa el visualizador con una instancia de PSO
        Solo funciona para problemas 2D (dimensiones = 2)
        """
        self.pso = pso_instance
        
        # Verificar que sea un problema 2D
        if self.pso.dimensiones != 2:
            raise ValueError("El visualizador solo funciona para problemas 2D (dimensiones = 2)")
        
        # Configurar matplotlib
        plt.style.use('default')
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Datos para almacenar el historial de la optimizacion
        self.historial_posiciones = []     # [iteracion][particula][x, y]
        self.historial_gbest = []          # [iteracion] = [x, y]
        self.historial_fitness = []       # [iteracion] = valor_fitness
        
        # Elementos graficos que se actualizaran
        self.scatter_particulas = None
        self.scatter_gbest = None
        self.lineas_trayectoria = []
        
    def crear_mapa_funcion(self, resolucion=100):
        """
        Crea un mapa de contorno de la funcion objetivo para visualizar el paisaje
        de optimizacion. Esto ayuda a entender la dificultad del problema.
        """
        print("Creando mapa de contorno de la funcion objetivo...")
        
        # Crear grilla de puntos para evaluar la funcion
        x = np.linspace(self.pso.rango_min, self.pso.rango_max, resolucion)
        y = np.linspace(self.pso.rango_min, self.pso.rango_max, resolucion)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar funcion en cada punto de la grilla
        Z = np.zeros_like(X)
        for i in range(resolucion):
            for j in range(resolucion):
                # Convertir de fitness (que maximizamos) a valor real de funcion (que minimizamos)
                valor_fitness = self.pso.funcion_objetivo([X[i, j], Y[i, j]])
                Z[i, j] = -valor_fitness  # Convertir de vuelta a valor de Ackley
        
        return X, Y, Z
    
    def configurar_grafico_inicial(self):
        """
        Configura el grafico inicial con el mapa de contorno y elementos basicos
        """
        # Limpiar grafico anterior
        self.ax.clear()
        
        # Crear y mostrar mapa de contorno de la funcion objetivo
        X, Y, Z = self.crear_mapa_funcion(resolucion=80)
        
        # Dibujar contornos de la funcion (paisaje de optimizacion)
        contornos = self.ax.contour(X, Y, Z, levels=15, colors='gray', alpha=0.4, linewidths=0.5)
        contornos_rellenos = self.ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.2)
        
        # Añadir barra de colores para interpretar los valores
        cbar = plt.colorbar(contornos_rellenos, ax=self.ax, shrink=0.8)
        cbar.set_label('Valor de la Funcion Objetivo', rotation=270, labelpad=20)
        
        # Marcar el optimo global (punto objetivo)
        self.ax.plot(0, 0, 'r*', markersize=20, markeredgecolor='black', 
                    markeredgewidth=2, label='Optimo Global (0,0)', zorder=10)
        
        # Configurar limites y etiquetas
        self.ax.set_xlim(self.pso.rango_min, self.pso.rango_max)
        self.ax.set_ylim(self.pso.rango_min, self.pso.rango_max)
        self.ax.set_xlabel('Dimension X1', fontsize=12)
        self.ax.set_ylabel('Dimension X2', fontsize=12)
        self.ax.set_title('Optimizacion PSO - Particulas buscando el minimo global', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        
        # Configurar leyenda
        self.ax.legend(loc='upper right')
        
        plt.tight_layout()
    
    def actualizar_grafico(self, iteracion):
        """
        Actualiza el grafico con las posiciones actuales de las particulas
        """
        # Obtener posiciones actuales de todas las particulas
        posiciones_x = [p.posicion[0] for p in self.pso.particulas]
        posiciones_y = [p.posicion[1] for p in self.pso.particulas]
        
        # Si es la primera vez, crear los elementos graficos
        if self.scatter_particulas is None:
            # Particulas actuales
            self.scatter_particulas = self.ax.scatter(posiciones_x, posiciones_y, 
                                                    c='blue', s=60, alpha=0.7, 
                                                    edgecolors='navy', linewidth=1,
                                                    label='Particulas', zorder=5)
            
            # Mejor posicion global
            if self.pso.mejor_posicion_global:
                self.scatter_gbest = self.ax.scatter(self.pso.mejor_posicion_global[0], 
                                                   self.pso.mejor_posicion_global[1],
                                                   c='orange', s=150, marker='s',
                                                   edgecolors='red', linewidth=2,
                                                   label='Mejor Global', zorder=8)
        else:
            # Actualizar posiciones existentes
            self.scatter_particulas.set_offsets(np.column_stack((posiciones_x, posiciones_y)))
            
            # Actualizar mejor posicion global
            if self.pso.mejor_posicion_global and self.scatter_gbest:
                self.scatter_gbest.set_offsets([[self.pso.mejor_posicion_global[0], 
                                               self.pso.mejor_posicion_global[1]]])
        
        # Actualizar titulo con informacion de la iteracion
        valor_actual = -self.pso.mejor_fitness_global if self.pso.mejor_fitness_global != float('-inf') else float('inf')
        self.ax.set_title(f'PSO - Iteracion {iteracion} | Mejor valor: {valor_actual:.6f}', fontsize=14)
        
        # Actualizar leyenda si es necesario
        if iteracion == 0:
            self.ax.legend(loc='upper right')
    
    def dibujar_trayectorias(self, max_trayectorias=5):
        """
        Dibuja las trayectorias de algunas particulas para mostrar su recorrido
        """
        if len(self.historial_posiciones) < 2:
            return
            
        # Seleccionar algunas particulas para mostrar sus trayectorias
        num_particulas_mostrar = min(max_trayectorias, len(self.pso.particulas))
        indices_particulas = np.linspace(0, len(self.pso.particulas)-1, num_particulas_mostrar, dtype=int)
        
        colores = ['cyan', 'magenta', 'yellow', 'lime', 'orange']
        
        for i, idx_particula in enumerate(indices_particulas):
            # Extraer trayectoria de esta particula
            trayectoria_x = []
            trayectoria_y = []
            
            for posiciones_iteracion in self.historial_posiciones:
                if idx_particula < len(posiciones_iteracion):
                    trayectoria_x.append(posiciones_iteracion[idx_particula][0])
                    trayectoria_y.append(posiciones_iteracion[idx_particula][1])
            
            # Dibujar trayectoria
            if len(trayectoria_x) > 1:
                self.ax.plot(trayectoria_x, trayectoria_y, 
                           color=colores[i % len(colores)], 
                           alpha=0.6, linewidth=1.5, linestyle='--',
                           label=f'Trayectoria P{idx_particula}' if i < 3 else "")
    
    def optimizar_con_visualizacion(self, max_iteraciones=200, intervalo_actualizacion=5, 
                                  mostrar_trayectorias=True, pausa_entre_frames=0.2,
                                  factor_velocidad=1.0):
        """
        Ejecuta el algoritmo PSO con visualizacion en tiempo real
        
        Parametros:
        - max_iteraciones: numero maximo de iteraciones
        - intervalo_actualizacion: cada cuantas iteraciones actualizar el grafico (menor = mas fluido)
        - mostrar_trayectorias: si mostrar las trayectorias de las particulas
        - pausa_entre_frames: tiempo de pausa entre actualizaciones graficas (segundos)
        - factor_velocidad: multiplicador para la velocidad de las particulas (menor = mas lento)
        """
        print(f"Iniciando PSO con visualizacion...")
        print(f"Particulas: {self.pso.num_particulas}, Iteraciones: {max_iteraciones}")
        print(f"Actualizacion grafica cada {intervalo_actualizacion} iteraciones")
        print(f"Pausa entre frames: {pausa_entre_frames}s, Factor velocidad: {factor_velocidad}")
        print("-" * 60)
        
        # Ajustar velocidad de las particulas si se especifica
        if factor_velocidad != 1.0:
            for particula in self.pso.particulas:
                for i in range(len(particula.velocidad)):
                    particula.velocidad[i] *= factor_velocidad
        
        # Configurar grafico inicial
        self.configurar_grafico_inicial()
        
        # Evaluacion inicial de todas las particulas
        for particula in self.pso.particulas:
            self.pso.evaluar_particula(particula)
        
        # Bucle principal de optimizacion
        for iteracion in range(max_iteraciones):
            # Guardar posiciones actuales para el historial
            posiciones_actuales = [[p.posicion[0], p.posicion[1]] for p in self.pso.particulas]
            self.historial_posiciones.append(copy.deepcopy(posiciones_actuales))
            
            # Guardar mejor posicion global actual
            if self.pso.mejor_posicion_global:
                self.historial_gbest.append(self.pso.mejor_posicion_global[:])
                self.historial_fitness.append(-self.pso.mejor_fitness_global)
            
            # Actualizar grafico periodicamente
            if iteracion % intervalo_actualizacion == 0:
                self.actualizar_grafico(iteracion)
                
                # Mostrar trayectorias si esta habilitado
                if mostrar_trayectorias and iteracion > 20:
                    self.dibujar_trayectorias()
                
                plt.pause(pausa_entre_frames)  # Pausa personalizable para controlar velocidad
            
            # Mostrar progreso en consola
            if iteracion % 25 == 0:
                valor_actual = -self.pso.mejor_fitness_global if self.pso.mejor_fitness_global != float('-inf') else float('inf')
                print(f"Iteracion {iteracion:3d}: Valor = {valor_actual:.6f}, "
                      f"Posicion = [{self.pso.mejor_posicion_global[0]:.4f}, {self.pso.mejor_posicion_global[1]:.4f}]")
            
            # Criterio de convergencia
            if self.pso.mejor_fitness_global != float('-inf'):
                valor_actual = -self.pso.mejor_fitness_global
                if valor_actual < 1e-6:
                    print(f"\nConvergencia alcanzada en iteracion {iteracion}")
                    break
            
            # Ejecutar una iteracion del algoritmo PSO
            for particula in self.pso.particulas:
                self.pso.actualizar_velocidad(particula)
                
                # Aplicar factor de velocidad para controlar el movimiento
                if factor_velocidad != 1.0:
                    for i in range(len(particula.velocidad)):
                        particula.velocidad[i] *= factor_velocidad
                
                self.pso.actualizar_posicion(particula)
                self.pso.evaluar_particula(particula)
        
        # Actualizacion final del grafico
        self.actualizar_grafico(iteracion)
        if mostrar_trayectorias:
            self.dibujar_trayectorias()
        
        # Mostrar resultados finales
        valor_final = -self.pso.mejor_fitness_global
        print(f"\n{'='*60}")
        print(f"RESULTADOS FINALES:")
        print(f"Mejor valor encontrado: {valor_final:.8f}")
        print(f"Mejor posicion: [{self.pso.mejor_posicion_global[0]:.6f}, {self.pso.mejor_posicion_global[1]:.6f}]")
        print(f"Error desde optimo: {valor_final:.8f}")
        
        plt.title(f'PSO Completado - Mejor valor: {valor_final:.6f}', fontsize=14)
        plt.show()
        
        return self.pso.mejor_posicion_global, valor_final
    
    def optimizar_modo_lento(self, max_iteraciones=300):
        """
        Modo predefinido para visualizacion lenta y detallada
        Ideal para observar el comportamiento paso a paso
        """
        return self.optimizar_con_visualizacion(
            max_iteraciones=max_iteraciones,
            intervalo_actualizacion=2,      # Actualizar cada 2 iteraciones
            mostrar_trayectorias=True,
            pausa_entre_frames=0.5,         # Pausa larga entre frames
            factor_velocidad=0.3            # Velocidad muy reducida
        )
    
    def optimizar_modo_medio(self, max_iteraciones=250):
        """
        Modo predefinido para velocidad media
        Balance entre observacion y tiempo de ejecucion
        """
        return self.optimizar_con_visualizacion(
            max_iteraciones=max_iteraciones,
            intervalo_actualizacion=3,      # Actualizar cada 3 iteraciones
            mostrar_trayectorias=True,
            pausa_entre_frames=0.2,         # Pausa media
            factor_velocidad=0.6            # Velocidad reducida
        )
    
    def optimizar_modo_rapido(self, max_iteraciones=200):
        """
        Modo predefinido para visualizacion rapida
        Para obtener resultados rapidamente
        """
        return self.optimizar_con_visualizacion(
            max_iteraciones=max_iteraciones,
            intervalo_actualizacion=8,      # Actualizar cada 8 iteraciones  
            mostrar_trayectorias=False,     # Sin trayectorias para mayor velocidad
            pausa_entre_frames=0.05,        # Pausa minima
            factor_velocidad=1.0            # Velocidad normal
        )
    
    def graficar_convergencia(self):
        """
        Crea un grafico separado mostrando la convergencia del algoritmo
        """
        if not self.historial_fitness:
            print("No hay datos de convergencia para graficar")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.historial_fitness, 'b-', linewidth=2, alpha=0.8)
        plt.xlabel('Iteracion')
        plt.ylabel('Mejor Valor de Funcion Objetivo')
        plt.title('Convergencia del Algoritmo PSO')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala logaritmica para ver mejor la convergencia
        plt.tight_layout()
        plt.show()

# Ejemplo de uso del visualizador
if __name__ == "__main__":
    # Primero necesitamos importar o definir las clases PSO y Particula
    # En un uso real, descomentarias la linea de import al inicio
    
    print("Para usar este visualizador:")
    print("1. Asegurate de tener las clases PSO y Particula disponibles")
    print("2. Importa: from pso_algorithm import PSO, Particula")
    print("3. Crea una instancia de PSO 2D")
    print("4. Crea una instancia del VisualizadorPSO")
    print("5. Llama a optimizar_con_visualizacion() o usa los modos predefinidos")
    print()
    print("Ejemplo de codigo - Control de velocidad personalizado:")
    print("""
    from pso_algorithm import PSO, Particula
    
    # Crear PSO para problema 2D
    pso = PSO(num_particulas=20, dimensiones=2, rango_min=-5, rango_max=5)
    
    # Crear visualizador
    visualizador = VisualizadorPSO(pso)
    
    # OPCION 1: Control manual de velocidad
    mejor_pos, mejor_val = visualizador.optimizar_con_visualizacion(
        max_iteraciones=150,
        intervalo_actualizacion=3,     # Actualizar cada 3 iteraciones
        mostrar_trayectorias=True,
        pausa_entre_frames=0.4,        # Pausa de 0.4 segundos entre frames
        factor_velocidad=0.4           # Velocidad reducida al 40%
    )
    
    # OPCION 2: Modos predefinidos (mas facil)
    # mejor_pos, mejor_val = visualizador.optimizar_modo_lento()    # Muy lento y detallado
    # mejor_pos, mejor_val = visualizador.optimizar_modo_medio()    # Velocidad equilibrada  
    # mejor_pos, mejor_val = visualizador.optimizar_modo_rapido()   # Rapido
    
    # Mostrar grafico de convergencia
    visualizador.graficar_convergencia()
    """)
    print()
    print("CONTROLES DE VELOCIDAD:")
    print("- intervalo_actualizacion: menor numero = mas fluido (ej: 1-10)")
    print("- pausa_entre_frames: tiempo entre actualizaciones (ej: 0.1-1.0 segundos)")
    print("- factor_velocidad: multiplicador de velocidad (ej: 0.1-1.0)")
    print("- Modos predefinidos: optimizar_modo_lento(), optimizar_modo_medio(), optimizar_modo_rapido()")