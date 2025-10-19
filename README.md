# Metaheuristicas y Optimizacion: Implementaciones en Python

Este repositorio contiene implementaciones en Python de diversas metaheuristicas y algoritmos evolutivos utilizados para resolver problemas de optimizacion continua, discreta y multiobjetivo.

---

## Estructura del Proyecto

El repositorio esta organizado por el nombre del algoritmo principal implementado, facilitando la navegacion a cada tecnica de optimizacion.

| Carpeta | Descripcion | Problemas Resueltos (Ejemplos) |
| :--- | :--- | :--- |
| pso | Algoritmo de Optimizacion por Enjambre de Particulas (Particle Swarm Optimization). | Minimizacion de funciones multimodales (e.g., Funcion de Ackley). |
| GA | Algoritmos Geneticos (Genetic Algorithms) para optimizacion mono-objetivo. | Minimizacion de funciones simples (suma x_i^2) y busqueda de cadenas. |
| MSA | Algoritmo del Espiritu Musical (Musical Spirit Algorithm). | Optimizacion continua de funciones (e.g., funciones polinomicas simples). |
| NSGA2 | Algoritmo Genetico de Clasificacion No Dominada II (Non-dominated Sorting Genetic Algorithm II). | Optimizacion Multiobjetivo (MOO) y problemas de Frente de Pareto (e.g., ZDT1). |
| otros | Problemas especificos de optimizacion exacta. | Programacion Lineal y Entera (e.g., Problema de Expansion de Capacidad del Sector Electrico). |
| libreria | Modulos o scripts auxiliares compartidos entre implementaciones (si aplica). | -- |

---

## Instalacion y Requisitos

Para ejecutar los scripts de optimizacion, necesitas tener Python instalado (se recomienda Python 3.8+).

### Dependencias

La mayoria de los scripts se basan en librerias cientificas estandar de Python:

```bash
pip install numpy matplotlib pandas
```

Para los ejemplos de Programacion Entera (carpeta otros), se requiere el uso de librerias de modelado especificas:

```bash
pip install pulp scipy
```

-----

## Uso Basico

Cada carpeta contiene scripts independientes que demuestran la aplicacion del algoritmo a un problema de prueba especifico.

1.  **Clonar el Repositorio:**

    ```bash
    git clone https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories
    cd [Nombre del Repositorio]
    ```

2.  **Ejecutar un Algoritmo:**
    Para ejecutar, por ejemplo, el PSO:

    ```bash
    python pso/pso_algorithm.py
    ```

3.  **Visualizacion:**
    Muchos scripts (como NSGA2.py y MSA.py) generan graficas de convergencia o frentes de Pareto utilizando matplotlib para visualizar el rendimiento del algoritmo.

-----

## Contribuciones

Las contribuciones son bienvenidas, especialmente para:

  * Anadir nuevos algoritmos metaheuristicos (e.g., colonia de hormigas, recocido simulado).
  * Implementar nuevos problemas de prueba estandar.
  * Mejorar la eficiencia y claridad del codigo existente.

Por favor, crea un Issue o envia un Pull Request para cualquier cambio propuesto.

-----

## Licencia

Este proyecto esta licenciado bajo la Licencia MIT - ver el archivo [LICENSE] para mas detalles.
