from pso_algorithm import PSO, Particula
from pso_visualizer import VisualizadorPSO

# Crear PSO para problema 2D
pso = PSO(num_particulas=25, dimensiones=2, rango_min=-5, rango_max=5)

# Crear visualizador
visualizador = VisualizadorPSO(pso)

# ELEGIR UNO DE ESTOS MODOS:

# Modo muy lento (ideal para aprender)
# mejor_posicion, mejor_valor = visualizador.optimizar_modo_lento()

# Modo medio (equilibrado)
mejor_posicion, mejor_valor = visualizador.optimizar_modo_medio()

# Modo rápido (resultados rápidos)
# mejor_posicion, mejor_valor = visualizador.optimizar_modo_rapido()

# Mostrar gráfico de convergencia
visualizador.graficar_convergencia()