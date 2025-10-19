from pathlib import Path
import pandas as pd
import numpy as np
import multiprocessing
from typing import Callable, List, Tuple, Optional

class AlgoritmoGenetico:
    
    def __init__(
        self,
        crear_individuo_fn: Callable,
        calculo_aptitud_fn: Callable,
        crear_pareja_fn: Callable,
        selection_fn: Callable,
        temperatura_fn: Callable,
        tamano_poblacion: int = 300,
        num_generaciones: int = 200,
        prob_mutacion: float = 0.01,
        num_elitistas: int = 0,
        max_sin_mejora: int = 100,
        usar_multiprocessing: bool = True,
        verbose: bool = True
    ):
        """
        Inicializa el algoritmo genético.
        
        Args:
            crear_individuo_fn: Función que crea un individuo ()
            calculo_aptitud_fn: Función que calcula aptitud (perfil_carga, perfil_descarga, individuo)
            crear_pareja_fn: Función que crea pareja de hijos (poblacion, probs, prob_mut)
            selection_fn: Función de selección (aptitudes, temperatura)
            temperatura_fn: Función de temperatura (generacion)
            tamano_poblacion: Tamaño de la población
            num_generaciones: Número de generaciones
            prob_mutacion: Probabilidad de mutación
            num_elitistas: Número de individuos elitistas
            max_sin_mejora: Generaciones máximas sin mejora
            usar_multiprocessing: Si usar procesamiento paralelo
            verbose: Si imprimir información
        """
        self.crear_individuo_fn = crear_individuo_fn
        self.calculo_aptitud_fn = calculo_aptitud_fn
        self.crear_pareja_fn = crear_pareja_fn
        self.selection_fn = selection_fn
        self.temperatura_fn = temperatura_fn
        
        self.tamano_poblacion = tamano_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.num_elitistas = num_elitistas
        self.max_sin_mejora = max_sin_mejora
        self.usar_multiprocessing = usar_multiprocessing
        self.verbose = verbose
        
        # Para multiprocessing
        self.num_procesos = max(1, (multiprocessing.cpu_count() * 3) // 4) if usar_multiprocessing else 1
        self.pool = None
    
    def crear_poblacion_inicial(self) -> List:
        """Crea la población inicial."""
        if self.usar_multiprocessing:
            return self.pool.starmap(self.crear_individuo_fn, [() for _ in range(self.tamano_poblacion)])
        else:
            return [self.crear_individuo_fn() for _ in range(self.tamano_poblacion)]
    
    def calcular_aptitudes(self, poblacion: List, perfil_carga: List, perfil_descarga: List) -> List[float]:
        """Calcula las aptitudes de toda la población."""
        if self.usar_multiprocessing:
            return self.pool.starmap(
                self.calculo_aptitud_fn,
                [(perfil_carga, perfil_descarga, ind) for ind in poblacion]
            )
        else:
            return [self.calculo_aptitud_fn(perfil_carga, perfil_descarga, ind) for ind in poblacion]
    
    def seleccionar_elitistas(self, poblacion: List, aptitudes: List[float]) -> List:
        """Selecciona los mejores individuos (elitismo)."""
        if self.num_elitistas == 0:
            return []
        
        mejores_indices = np.argsort(aptitudes)[(-1) * self.num_elitistas:]
        return [poblacion[i] for i in mejores_indices]
    
    def generar_nueva_poblacion(self, poblacion: List, probabilidades: List[float]) -> List:
        """Genera nueva población mediante cruce y mutación."""
        if self.usar_multiprocessing:
            nuevos_hijos = self.pool.starmap(
                self.crear_pareja_fn,
                [(poblacion, probabilidades, self.prob_mutacion) for _ in range(0, self.tamano_poblacion, 2)]
            )
        else:
            nuevos_hijos = [
                self.crear_pareja_fn(poblacion, probabilidades, self.prob_mutacion)
                for _ in range(0, self.tamano_poblacion, 2)
            ]
        
        # Aplanar la lista de parejas
        return [hijo for pareja in nuevos_hijos for hijo in pareja]
    
    def evolucionar(
        self,
        perfil_carga: List[float],
        perfil_descarga: List[float],
        generaciones_verbose: Optional[List[int]] = None
    ) -> Tuple[any, float, int]:
        """
        Ejecuta el algoritmo genético para un día.
        
        Args:
            perfil_carga: Perfil de precios de carga
            perfil_descarga: Perfil de precios de descarga
            generaciones_verbose: Lista de generaciones a imprimir (ej: [1,2,3,4,5,100,200])
        
        Returns:
            (mejor_individuo, mejor_aptitud, generacion_final)
        """
        if generaciones_verbose is None:
            generaciones_verbose = list(range(1, 6)) + [100, 200]
        
        # Crear población inicial
        poblacion = self.crear_poblacion_inicial()
        
        mejor_aptitud = -float('inf')
        mejor_individuo = None
        generaciones_sin_mejora = 0
        
        for generacion in range(self.num_generaciones):
            # Calcular aptitudes
            aptitudes = self.calcular_aptitudes(poblacion, perfil_carga, perfil_descarga)
            
            # Verbose
            if self.verbose and ((generacion + 1) % 100 == 0 or (generacion + 1) in generaciones_verbose):
                print(f"Generación {generacion + 1}")
                print(f"  Tamaño población: {len(poblacion)}")
                print(f"  Tamaño aptitudes: {len(aptitudes)}")
                print(f"  Aptitud máxima: {max(aptitudes):.4f}")
            
            # Encontrar mejor individuo de esta generación
            aptitud_maxima = max(aptitudes)
            indice_mejor = aptitudes.index(aptitud_maxima)
            mejor_individuo_gen = poblacion[indice_mejor]
            
            # Actualizar mejor global
            if aptitud_maxima == mejor_aptitud:
                generaciones_sin_mejora += 1
            elif aptitud_maxima > mejor_aptitud:
                generaciones_sin_mejora = 0
                mejor_aptitud = aptitud_maxima
                mejor_individuo = mejor_individuo_gen
            else:
                generaciones_sin_mejora = 0
            
            # Criterio de parada
            if generaciones_sin_mejora >= self.max_sin_mejora:
                if self.verbose:
                    print(f"Parada temprana: {self.max_sin_mejora} generaciones sin mejora")
                break
            
            # Selección
            mejores_individuos = self.seleccionar_elitistas(poblacion, aptitudes)
            temperatura = self.temperatura_fn(generacion)
            probabilidades = self.selection_fn(aptitudes, temperatura)
            
            # Generar nueva población
            nueva_poblacion = self.generar_nueva_poblacion(poblacion, probabilidades)
            
            # Agregar elitistas
            if mejores_individuos:
                nueva_poblacion = mejores_individuos + nueva_poblacion[:self.tamano_poblacion - len(mejores_individuos)]
            
            poblacion = nueva_poblacion
        
        # Evaluar población final
        aptitudes_finales = self.calcular_aptitudes(poblacion, perfil_carga, perfil_descarga)
        mejor_individuo_final = poblacion[aptitudes_finales.index(max(aptitudes_finales))]
        mejor_aptitud_final = max(aptitudes_finales)
        
        return mejor_individuo_final, mejor_aptitud_final, generacion + 1
    
    def ejecutar_multiples_dias(
        self,
        perfiles_carga: List[List[float]],
        perfiles_descarga: List[List[float]],
        carpeta_resultados: Path,
        nombre_archivo_final: str = "resultados_individuos.xlsx"
    ) -> pd.DataFrame:
        """
        Ejecuta el AG para múltiples días y guarda resultados.
        
        Args:
            perfiles_carga: Lista de perfiles de carga (uno por día)
            perfiles_descarga: Lista de perfiles de descarga (uno por día)
            carpeta_resultados: Carpeta donde guardar resultados
            nombre_archivo_final: Nombre del archivo Excel final
        
        Returns:
            DataFrame con todos los resultados
        """
        carpeta_resultados.mkdir(exist_ok=True)
        todos_los_dias = []
        
        # Iniciar pool si se usa multiprocessing
        if self.usar_multiprocessing:
            self.pool = multiprocessing.Pool(self.num_procesos)
        
        try:
            for dia, (perfil_c, perfil_d) in enumerate(zip(perfiles_carga, perfiles_descarga), start=1):
                if self.verbose:
                    print(f'\n{"="*50}')
                    print(f'DÍA {dia} del año')
                    print(f'{"="*50}')
                
                # Evolucionar
                mejor_individuo, mejor_aptitud, gen_final = self.evolucionar(perfil_c, perfil_d)
                
                if self.verbose:
                    print(f"\nMejor individuo: {mejor_individuo}")
                    print(f"Aptitud máxima: {mejor_aptitud:.4f}")
                    print(f"Generación final: {gen_final}")
                
                # Guardar resultados del día
                dic_resultados = {
                    'dia': [dia] * len(mejor_individuo),
                    'aptitud': [mejor_aptitud] * len(mejor_individuo),
                    'individuo': mejor_individuo,
                    'perfil_carga': perfil_c,
                    'perfil_descarga': perfil_d
                }
                
                df_dia = pd.DataFrame(dic_resultados)
                df_dia.to_csv(carpeta_resultados / f'{dia}.csv', index=False)
                todos_los_dias.append(df_dia)
        
        finally:
            # Cerrar pool
            if self.usar_multiprocessing and self.pool:
                self.pool.close()
                self.pool.join()
        
        # Consolidar resultados
        df_final = pd.concat(todos_los_dias, ignore_index=True)
        df_final.to_excel(carpeta_resultados / nombre_archivo_final, index=False)
        
        if self.verbose:
            print(f"\nResultados guardados en {carpeta_resultados / nombre_archivo_final}")
        
        return df_final

    
    