import random
from itertools import accumulate
import pandas as pd
from pathlib import Path
import math
import numpy as np
import multiprocessing
import logging
from utilis_ga import AlgoritmoGenetico
from functools import partial

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def cambiar_signo_lista(func):
    def wrapper(*args, **kwargs):
        individuo_nuevo = list(map(lambda x: -x, func(*args, **kwargs)))
        return individuo_nuevo
    return wrapper

def generar_individuo(carga_maxima_energia=3, potencia_nominal = 1, periodo_horas = 24, paso_min = 15, semilla=None):
    if semilla is not None:
        random.seed(semilla)
    
    def calcular_estado():
        return random.choice([True, False])

    def calcular_posible_valor(valor_maximo):
        return round(random.choice([0, valor_maximo, random.uniform(0, valor_maximo)]), 3)
    
    # el individuo inicial tiene un largo de 24 horas
    potencia_paso = potencia_nominal / (60 / paso_min)
    individuo = [0] * periodo_horas * int((60 / paso_min))
    carga_actual = 0
    comportamiento_carga = []
    pasos_descarga = carga_maxima_energia / potencia_paso
    
    for paso_tiempo in range(len(individuo)):
        valor_actual = 0
        # pasos de tiempo sin incluir el ultimo paso
        if paso_tiempo <= (len(individuo) - (pasos_descarga + 1)):
            if carga_actual == 0:
                # si la carga es cero la bateria puede cargar o no hacer nada
                valor_actual = calcular_posible_valor(potencia_paso)
                individuo[paso_tiempo] = valor_actual
                carga_actual += valor_actual
                comportamiento_carga.append(carga_actual)
                
            elif 0 < carga_actual <= carga_maxima_energia:
                if calcular_estado():
                    # si es verdadero carga o no hace nada
                    carga_remanente = carga_maxima_energia - carga_actual
                    if carga_remanente >= potencia_paso:
                        valor_actual = calcular_posible_valor(potencia_paso)
                        individuo[paso_tiempo] = valor_actual
                        carga_actual += valor_actual
                        comportamiento_carga.append(carga_actual)
                        
                    else:
                        valor_actual = calcular_posible_valor(carga_remanente)
                        individuo[paso_tiempo] = valor_actual
                        carga_actual += valor_actual
                        comportamiento_carga.append(carga_actual)
                        
                else:
                    # si es falso descarga o no hace nada
                    if carga_actual >= potencia_paso:
                        valor_actual = calcular_posible_valor(potencia_paso)
                        individuo[paso_tiempo] = (-1)*valor_actual
                        carga_actual -= valor_actual
                        comportamiento_carga.append(carga_actual)
                    else:
                        valor_actual = calcular_posible_valor(carga_actual)
                        individuo[paso_tiempo] = (-1)*valor_actual
                        carga_actual -= valor_actual
                        comportamiento_carga.append(carga_actual)
                        
                    
        elif paso_tiempo > (len(individuo) - (pasos_descarga + 1)):
            potencia_max_restante = potencia_paso * (len(individuo) - paso_tiempo)
            # print(f'potencia maxima restante: {potencia_max_restante}')
            # print(f'carga actual: {carga_actual}')
            if carga_actual == potencia_max_restante:
                # debe descargar para cerrar el dia con energia 0 en la bateria
                individuo[paso_tiempo] = (-1)*potencia_paso
                carga_actual -= potencia_paso
                comportamiento_carga.append(carga_actual)
            elif carga_actual >= (potencia_max_restante - potencia_paso):
                # debe descargar para cerrar el dia con energia 0
                individuo[paso_tiempo] = (-1)*round((carga_actual - (potencia_max_restante - potencia_paso)),3)
                carga_actual -= (carga_actual - (potencia_max_restante - potencia_paso))
                comportamiento_carga.append(carga_actual)
            elif carga_actual < (potencia_max_restante - potencia_paso):  
                # puede cargar dado que todavia tiene espacio para descargar antes de terminar el dia
                
                if ((potencia_max_restante - potencia_paso) - carga_actual) >= potencia_paso:
                    valor = calcular_posible_valor(potencia_paso)
                else:
                    valor = calcular_posible_valor(((potencia_max_restante - potencia_paso) - carga_actual))
                    
                individuo[paso_tiempo] = valor 
                carga_actual += valor
                comportamiento_carga.append(carga_actual)   
            comportamiento_carga.append(carga_actual)

    # return individuo
    # agregamos cambio de signo
    return list(map(lambda x: -x, individuo))

def calcular_posible_valor(valor_maximo):
    return round(random.choice([0, valor_maximo, random.uniform(0, valor_maximo)]), 3)

def ajustar_carga_final_dia(individuo, carga_maxima_energia=3, potencia_nominal = 1, paso_min = 15):
    
    potencia_paso = potencia_nominal / (60 / paso_min)
    pasos_descarga = int(carga_maxima_energia / potencia_paso)
    carga_actual = round(sum(individuo[:len(individuo)-pasos_descarga]), 3)
    valor_maximo = 0.25
    
    # print(f'potencia paso: {potencia_paso}')
    # print(f'pasos_descarga: {pasos_descarga}')
    # print(f'carga_actual: {carga_actual}')
    # print(f'valor_maximo: {valor_maximo}')
    
    for paso_tiempo in range(len(individuo) - pasos_descarga, len(individuo)):
        
        # print(f'paso tiempo: {paso_tiempo}')
        potencia_max_restante = potencia_paso * (len(individuo) - paso_tiempo)
        # print(f'potencia_max_restante: {potencia_max_restante}')
        # print(f'carga_actual_antes de cambio: {carga_actual}')
        
        if paso_tiempo < len(individuo)-1:
            
            if carga_actual == potencia_max_restante:
                individuo[paso_tiempo] = -potencia_paso
                carga_actual -= potencia_paso
                # print(f'cambio if1: {-potencia_paso}')
                # print(f'carga_actual: {carga_actual}')
                
            elif carga_actual >= (potencia_max_restante - potencia_paso):
                
                descarga = round(random.uniform(carga_actual - (potencia_max_restante - potencia_paso), valor_maximo), 3)
                individuo[paso_tiempo] = -descarga
                carga_actual -= descarga
                
                # print(f'cambio if2: {-descarga}')
                # print(f'carga_actual: {carga_actual}')
                
            elif carga_actual < (potencia_max_restante - potencia_paso):
                espacio_restante = (potencia_max_restante - potencia_paso) - carga_actual
                valor = calcular_posible_valor(potencia_paso if espacio_restante >= potencia_paso else espacio_restante)
                individuo[paso_tiempo] = valor
                carga_actual += valor
                # print(f'cambio if3: {valor}')
                # print(f'carga_actual: {carga_actual}')
        
        elif paso_tiempo == len(individuo)-1:
            # ultimo paso
            if 0 < carga_actual <= potencia_paso:
                valor = carga_actual
                individuo[paso_tiempo] = (-1)*valor
                carga_actual -= valor
                # print(f'cambio if4 ultimo paso: {-valor}')
                # print(f'carga_actual: {carga_actual}')
        
        carga_actual = round(carga_actual, 3)
    
    return individuo

def ajustar_carga_final_dia_v3(serie, energia_max = 3, potencia_max = 1, periodo_calculo = 24, pasos_hora = 4):

    energia_max = 3
    potencia_max = 1
    periodo_calculo = 24 # horas
    pasos_hora = 4
    potencia_paso_max = potencia_max / pasos_hora
    pasos_descarga_max = int(energia_max / potencia_paso_max)
    pasos_totales = periodo_calculo * pasos_hora
    
    serie_nueva = serie.copy()

    # primeros valores de la serie hasta los pasos_descarga_max
    # a[:len(a)-pasos_descarga_max]
    # suma de los primeros valores de la serie hasta los pasos_descarga_max
    # sum(a[:len(a)-pasos])

    # ultimos pasos_descarga_max de la serie
    # a[-pasos_descarga_max:]

    # esta es la entrada para este algoritmo, es el acumulado antes del ajuste
    carga_max_acumulada_antes_ajuste = -energia_max
    # print(f'carga_max_acumulada_antes_ajuste: {carga_max_acumulada_antes_ajuste}')

    carga_actual_antes_ajuste = sum(serie_nueva[:len(serie_nueva)-pasos_descarga_max])
    # print(f'carga_actual_antes_ajuste: {carga_actual_antes_ajuste}')
    
    # si la carga previa es igual a la carga_max_acumulada_antes_ajuste entonces los ultimos pasos de la serie deben ser igual a 
    # la lista_descarga_max
    lista_descarga_max = [potencia_paso_max for _ in range(pasos_descarga_max)]

    acumulado_max = [carga_max_acumulada_antes_ajuste + lista_descarga_max[0]] + [
        carga_max_acumulada_antes_ajuste + sum(lista_descarga_max[:i+1]) for i in range(1, len(lista_descarga_max))]

    # en caso de que la carga acumulada antes de ajuste sea menor a la carga_max_acumulada_antes_ajuste se debe calcular en cada paso 
    # la posible carga o descarga

    acumulado_real = []
    serie_final_bateria = []

    for i in range(pasos_descarga_max):  
            if i == 0:
                acumulado_actual_max = acumulado_max[i] # acumulado maximo en este paso
                margen_acumulado = acumulado_actual_max - carga_actual_antes_ajuste
                limite_superior = round(min(-carga_actual_antes_ajuste, 0.25), 4)
                limite_inferior = round(max(margen_acumulado, -0.25), 4)
                
                if limite_superior == limite_inferior:
                    valor_bateria = limite_inferior
                elif limite_inferior > limite_superior:
                    valor_bateria = round(random.uniform(limite_superior, limite_inferior), 4)
                else:
                    valor_bateria = round(random.uniform(limite_inferior, limite_superior), 4)
                    
                # valor_bateria = round(random.uniform(limite_inferior, limite_superior), 4)
                acumulado_real.append(carga_actual_antes_ajuste + valor_bateria)
                serie_final_bateria.append(valor_bateria)
                                    
            else:
                acumulado_actual_max = acumulado_max[i]
                margen_acumulado = acumulado_actual_max - acumulado_real[i-1]
                limite_superior = round(min(-acumulado_real[i-1], 0.25), 4)
                limite_inferior = round(max(margen_acumulado, -0.25), 4)
                
                if limite_superior == limite_inferior:
                    valor_bateria = limite_inferior
                elif limite_inferior > limite_superior:
                    valor_bateria = round(random.uniform(limite_superior, limite_inferior), 4)
                else:
                    valor_bateria = round(random.uniform(limite_inferior, limite_superior), 4)
                                
                # valor_bateria = round(random.uniform(limite_inferior, limite_superior), 4)
                acumulado_real.append(acumulado_real[i-1] + valor_bateria)
                serie_final_bateria.append(valor_bateria)
                
    #         print(f'acumulado_actual_max: {acumulado_actual_max}') 
    #         print(f'margen_acumulado: {margen_acumulado}')  
    #         print(f'limite_inferior: {limite_inferior}')  
    #         print(f'limite_superior: {limite_superior}')  
    #         print(f'valor_bateria: {valor_bateria}')  
    #         print(f'acumulado_real: {acumulado_real[i]}')  
            
    # print('serie_final_bateria')
    # print(serie_final_bateria)  
    # print('acumulado_real')  
    # print(acumulado_real)  
    
    serie_nueva = serie_nueva[:len(serie_nueva)-pasos_descarga_max] + serie_final_bateria
    return serie_nueva

def verificar_limite(valor, limite_superior=0.25, limite_inferior=-0.25):
    valor_aux = valor
    if valor < limite_inferior:
        valor_aux = limite_inferior
    elif valor > limite_superior:
        valor_aux = limite_superior
    return valor_aux

def revisar_individuo(individuo, limite_superior=0.25, limite_inferior=-0.25, energia_max=-3, energia_min=-1e-3):  
    acumulado = 0
    individuo_ajustado1 = []
    for valor in individuo:
        if (acumulado + valor) < energia_max:
            valor = energia_max - acumulado
        elif (acumulado + valor) > energia_min:
            valor = -acumulado
        acumulado += valor
        individuo_ajustado1.append(round(valor, 6))
        
    individuo_ajustado2 = ajustar_carga_final_dia_v3(individuo_ajustado1)
    individuo_ajustado3 = [verificar_limite(i, limite_superior, limite_inferior) for i in individuo_ajustado2]
                   
    return individuo_ajustado3

def verificar_negativo(individuo):
    primer_no_cero = next((x for x in individuo if x != 0), None)  # Encuentra el primer número distinto de cero
    if primer_no_cero < 0:
        return True
    else:
        return False

def verificar_individuo(individuo, energia_max = -3, limite_paso = 0.25):
    suma = sum(individuo)
    if -0.0009 <= suma <= 0.0009:
        suma = 0
    cantidad_positivos = sum([1 if i > 0 else 0 for i in individuo])
    cantidad_negativos = sum([1 if i < 0 else 0 for i in individuo])
    largo = len(individuo)
    verificar_acumulado = all([True if round(x, 4) >= energia_max else False for x in accumulate(individuo)])
    verificar_pasos = all([True if -limite_paso <= x <= limite_paso else False for x in individuo])
    
    return {'suma_total':suma,
            'positivos': cantidad_positivos,
            'negativos': cantidad_negativos,
            'largo': largo,
            'verificar_acumulado': verificar_acumulado,
            'verificar_pasos': verificar_pasos, 
            'verificar_negativo': verificar_negativo(individuo)}

def verificar_condiciones(individuo):
    condicion_1 = -0.01 < sum(individuo) < 0.01
    condicion_2 = all(x < 3.01 for x in accumulate(individuo))
    condicion_3 = all(x < 0.25 or x > -0.25 for x in individuo)
    return condicion_1 and condicion_2 and condicion_3

def seleccion_por_rank(poblacion, aptitudes):
    clasificados = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i], reverse=True)
    rangos = [len(poblacion) - i for i in range(len(poblacion))]  # Asignar rangos
    total_rango = sum(rangos)
    seleccionados = []
    for _ in range(len(poblacion)):
        punto_seleccion = random.uniform(0, total_rango)
        suma = 0
        for i, rango in enumerate(rangos):
            suma += rango
            if suma >= punto_seleccion:
                seleccionados.append(poblacion[clasificados[i]])
                break
    return seleccionados

def cruzamiento(padre1, padre2):
    if len(padre1) == len(padre2):
        punto_cruce = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
        hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
        return hijo1, hijo2

def gaussian_mutation(array, sigma=0.05, mutation_rate=1.0, lower_bound=-0.25, upper_bound=0.25):
    mutated_array = array.copy()
    for i in range(len(mutated_array)):
        if random.random() < mutation_rate:
            mutated_array[i] += random.gauss(0, sigma)
            if mutated_array[i] < lower_bound:
                mutated_array[i] = lower_bound
            elif mutated_array[i] > upper_bound:
                mutated_array[i] = upper_bound
    return mutated_array

def calculo_aptitud(perfil_carga, perfil_descarga, individuo):
    '''
    perfil_carga: perfil asociado con la carga
    perfil_descarga: perfil asociado con la descarga  
    '''
    def evaluar_perfil(valor_perfil_carga, valor_perfil_descarga, valor_individuo):
        # Los valores negativos del individuo implican carga (-)
        # Los valores positivos del individuo implican descarga (+)
        if valor_individuo > 0:
            return valor_perfil_descarga
        elif valor_individuo < 0:
            return valor_perfil_carga
        else:
            return 0
    precios = [evaluar_perfil(perfil_carga[i], perfil_descarga[i], individuo[i]) for i in range(len(individuo))]   
    return sum(map(lambda x: x[0]*x[1], zip(individuo, precios)))

def calculate_temperature(n_gen):
    return max(3 * 2 ** (-n_gen / 50), 0.3)

def boltzmann_selection(fitness, T):
    # Calculamos la exponencial de cada fitness dividido por T.
    exp_values = [math.exp(f / T) for f in fitness]
    # Sumamos todos los valores exponenciales.
    norm = sum(exp_values)
    # Dividimos cada valor exponencial por la suma total para obtener la probabilidad.
    probabilities = [value / norm for value in exp_values]
    return probabilities

def selection_fn(fitness, temperature):
    n = len(fitness)
    # Calcular la media.
    mean_val = sum(fitness) / n
    # Calcular la varianza (usando ddof=0) y luego la desviación estándar.
    variance = sum((x - mean_val) ** 2 for x in fitness) / n
    std_val = math.sqrt(variance)
    
    # Evitar división por cero en caso de que todos los fitness sean iguales.
    if std_val == 0:
        normalized_fitness = [0 for _ in fitness]
    else:
        normalized_fitness = [(x - mean_val) / std_val for x in fitness]
    
    return boltzmann_selection(normalized_fitness, T=temperature)

def crear_pareja_hijos(poblacion, probabilidad_padres, p_mutacion):      
    index_padres = list(np.random.choice(len(poblacion), size=2, replace=False, p=probabilidad_padres))            
    hijo1, hijo2 = cruzamiento(poblacion[index_padres[0]], poblacion[index_padres[1]])
    hijo1 = gaussian_mutation(hijo1, mutation_rate=p_mutacion)
    hijo2 = gaussian_mutation(hijo2, mutation_rate=p_mutacion)
    hijo1 = revisar_individuo(hijo1)
    hijo2 = revisar_individuo(hijo2)
    return hijo1, hijo2

def dividir_perfiles_diarios(perfil, hora_inicio=0, hora_fin=24, intervalo_min=15):
    periodos_por_dia = (24 * 60) // intervalo_min  # Cantidad de periodos en un día
    inicio_idx = (hora_inicio * 60) // intervalo_min
    fin_idx = (hora_fin * 60) // intervalo_min

    dias = len(perfil) // periodos_por_dia  # Número de días en el año
    perfiles_diarios = []

    for i in range(dias):
        inicio = i * periodos_por_dia + inicio_idx
        fin = i * periodos_por_dia + fin_idx
        perfiles_diarios.append(perfil[inicio:fin])  # Guardar solo el rango horario deseado

    return perfiles_diarios

def dividir_perfiles_diarios_v2(perfil, hora_inicio=0, hora_fin=24, intervalo_min=15):
    periodos_por_dia = (24 * 60) // intervalo_min  # Cantidad de periodos en un día
    inicio_idx = (hora_inicio * 60) // intervalo_min
    fin_idx = (hora_fin * 60) // intervalo_min

    dias = len(perfil) // periodos_por_dia  # Número de días completos en el perfil
    perfiles_diarios = []

    for i in range(dias):
        if inicio_idx < fin_idx:
            # Caso normal: el intervalo está dentro del mismo día
            inicio = i * periodos_por_dia + inicio_idx
            fin = i * periodos_por_dia + fin_idx
            perfiles_diarios.append(perfil[inicio:fin])
        else:
            # Caso especial: el intervalo cruza la medianoche
            inicio = i * periodos_por_dia + inicio_idx
            fin = (i + 1) * periodos_por_dia + fin_idx

            if inicio < len(perfil):
                parte_1 = perfil[inicio:(i + 1) * periodos_por_dia]  # Desde la hora de inicio hasta medianoche
            else:
                parte_1 = []

            if fin <= len(perfil):
                parte_2 = perfil[(i + 1) * periodos_por_dia:fin]  # Desde medianoche hasta la hora final
            else:
                parte_2 = []

            perfiles_diarios.append(parte_1 + parte_2)  # Combinar las dos partes

    return perfiles_diarios


if __name__ == '__main__':

    carpeta_resultados = Path("resultados")
    carpeta_resultados.mkdir(exist_ok=True)
        
    todos_los_dias = []
    
    ruta_base = Path(__file__).resolve().parent
    
    data = pd.read_csv(ruta_base / 'Precios compraventa MD.csv', sep=';', decimal=',')
    perfil_carga = data['MD compra'].to_list()
    perfil_descarga = data['MD venta'].to_list()
    
    # 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    # perfil_carga_dias = dividir_perfiles_diarios_v2(perfil_carga, hora_inicio=7, hora_fin=18, intervalo_min=60)
    # perfil_descarga_dias = dividir_perfiles_diarios_v2(perfil_descarga, hora_inicio=7, hora_fin=18, intervalo_min=60)

    perfil_carga_dias = dividir_perfiles_diarios_v2(perfil_carga, hora_inicio=0, hora_fin=24, intervalo_min=60)
    perfil_descarga_dias = dividir_perfiles_diarios_v2(perfil_descarga, hora_inicio=0, hora_fin=24, intervalo_min=60)
    
    # create_individual_solo_carga = partial(generar_individuo, 3, 1, 11, 60, None)
    create_individual_solo_carga = partial(generar_individuo, 3, 1, 24, 60, None)

    ag = AlgoritmoGenetico(
        crear_individuo_fn=create_individual_solo_carga,
        calculo_aptitud_fn=calculo_aptitud,
        crear_pareja_fn=crear_pareja_hijos,
        selection_fn=selection_fn,
        temperatura_fn=calculate_temperature,
        tamano_poblacion=5000,
        num_generaciones=500,
        prob_mutacion=0.01,
        num_elitistas=10,
        max_sin_mejora=300,
        usar_multiprocessing=True,
        verbose=True
    )
    
    df_resultados = ag.ejecutar_multiples_dias(
        perfiles_carga=perfil_carga_dias,
        perfiles_descarga=perfil_descarga_dias,
        carpeta_resultados=carpeta_resultados,
        nombre_archivo_final="resultados_individuos.xlsx"
    )
