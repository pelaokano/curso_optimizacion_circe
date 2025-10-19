"""
Problema de Expansión de la Capacidad (GEP) del Sector Eléctrico
Implementación en Python puro con scipy y pulp
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
import pulp
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class EnergyOptimizer:
    """
    Clase para optimizar la expansión de capacidad del sector eléctrico
    """
    
    def __init__(self, capava_file: str, exicap_file: str, plants_file: str):
        """
        Inicializa el optimizador con los datos de entrada
        
        Args:
            capava_file: Archivo CSV con capacidad disponible
            exicap_file: Archivo CSV con capacidad existente
            plants_file: Archivo CSV con características de plantas
        """
        self.load_data(capava_file, exicap_file, plants_file)
        self.setup_parameters()
        
    def load_data(self, capava_file: str, exicap_file: str, plants_file: str):
        """Carga los datos desde archivos CSV"""
        try:
            self.capava = pd.read_csv(capava_file)
            self.exicap = pd.read_csv(exicap_file)
            self.plants = pd.read_csv(plants_file)
            print("Datos cargados exitosamente")
        except FileNotFoundError as e:
            print(f"Error al cargar archivos: {e}")
            # Crear datos de ejemplo si no se encuentran los archivos
            self.create_sample_data()
    
    def create_sample_data(self):
        """Crea datos de ejemplo basados en el notebook"""
        # Datos de capacidad disponible
        capava_data = {
            'Region': ['Rio_Escondido', 'Nuevo_Laredo', 'Reynosa', 'Matamoros', 'Monterrey',
                      'Saltillo', 'Valles', 'Huasteca', 'Tamazunchale', 'Gomez'],
            'Carboelectrica': [2729, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Ciclo_combinado': [260, 0, 359, 1610, 8835, 298, 0, 2857, 2300, 0],
            'Combustion_interna': [91, 43, 0, 0, 92, 34, 0, 62, 17, 0],
            'Eolica': [2158, 0, 5492, 168, 2829, 1799, 0, 0, 0, 2297],
            'Hidroelectrica': [66, 33, 0, 0, 0, 0, 19, 0, 0, 0],
            'Solar': [20, 0, 0, 0, 1152, 4216, 0, 0, 0, 20],
            'Termoelectrica': [16, 0, 300, 0, 79, 0, 0, 663, 0, 0],
            'Turbogas': [70, 0, 20, 0, 875, 209, 62, 297, 0, 0]
        }
        
        # Datos de capacidad existente
        exicap_data = {
            'Region': ['Rio_Escondido', 'Nuevo_Laredo', 'Reynosa', 'Matamoros', 'Monterrey',
                      'Saltillo', 'Valles', 'Huasteca', 'Tamazunchale', 'Gomez'],
            'Carboelectrica': [2600, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Ciclo_combinado': [260, 0, 359, 1490, 3687, 298, 0, 2857, 1236, 0],
            'Combustion_interna': [24, 43, 0, 0, 77, 34, 0, 62, 17, 0],
            'Eolica': [0, 0, 306, 0, 22, 340, 0, 0, 0, 249],
            'Hidroelectrica': [66, 33, 0, 0, 0, 0, 19, 0, 0, 0],
            'Solar': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Termoelectrica': [16, 0, 300, 0, 79, 0, 0, 663, 0, 0],
            'Turbogas': [70, 0, 20, 0, 251, 209, 62, 262, 0, 0]
        }
        
        # Datos de plantas
        plants_data = {
            'Tecnologia': ['Carboelectrica', 'Ciclo_combinado', 'Combustion_interna', 'Eolica',
                          'Hidroelectrica', 'Solar', 'Termoelectrica', 'Turbogas'],
            'MeanCap': [1792.7867, 338.3587, 6.3598, 93.3106, 147.0033, 9.2928, 212.6368, 39.204],
            'NewCapCos': [1425508, 1013205, 2877291, 1423023, 1931246, 1260000, 2045088, 813156],
            'GenCos': [2.0, 3.587, 3.4737, 0.0, 0.0, 0.0, 3.1266, 4.984],
            'GHGEmi': [0.5498, 0.3933, 0.6194, 0.0, 0.0, 0.0, 0.6005, 0.7904],
            'MaxGen': [0.8, 0.75, 0.7, 0.333962, 0.61, 0.152453, 0.6, 0.6]
        }
        
        self.capava = pd.DataFrame(capava_data)
        self.exicap = pd.DataFrame(exicap_data)
        self.plants = pd.DataFrame(plants_data)
        print("Datos de ejemplo creados")
    
    def setup_parameters(self):
        """Configura los parámetros del modelo"""
        self.n_regions = len(self.capava)  # 10 regiones
        self.n_technologies = len(self.plants)  # 8 tecnologías
        self.demand = 89917974  # MWh demanda 2032
        self.hours_year = 8760  # horas en un año
        
        # Extraer datos como matrices numpy para optimización
        self.existing_capacity = self.exicap.iloc[:, 1:].values
        self.available_capacity = self.capava.iloc[:, 1:].values
        self.mean_capacity = self.plants['MeanCap'].values
        self.max_generation = self.plants['MaxGen'].values
        self.ghg_emissions = self.plants['GHGEmi'].values
        self.new_cap_cost = self.plants['NewCapCos'].values
        self.generation_cost = self.plants['GenCos'].values
        
    def solve_with_scipy_emissions(self) -> Dict:
        """
        Resuelve el problema de minimización de emisiones usando scipy
        """
        print("Resolviendo con scipy - Minimización de emisiones...")
        
        # Variables: NewCap (n_regions x n_technologies), Gen (n_regions x n_technologies)
        n_vars_newcap = self.n_regions * self.n_technologies
        n_vars_gen = self.n_regions * self.n_technologies
        n_vars_total = n_vars_newcap + n_vars_gen
        
        # Función objetivo: minimizar emisiones de GEI
        c = np.zeros(n_vars_total)
        # Los coeficientes de NewCap son 0 para emisiones
        # Los coeficientes de Gen son los factores de emisión repetidos por región
        for i in range(self.n_regions):
            start_idx = n_vars_newcap + i * self.n_technologies
            end_idx = start_idx + self.n_technologies
            c[start_idx:end_idx] = self.ghg_emissions
        
        # Restricciones de igualdad y desigualdad
        A_ub, b_ub = self._build_inequality_constraints()
        A_eq, b_eq = self._build_equality_constraints()
        
        # Límites de variables (todas no negativas, NewCap enteras se manejan después)
        bounds = [(0, None) for _ in range(n_vars_total)]
        
        # Resolver relajación lineal
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            newcap_values = result.x[:n_vars_newcap].reshape(self.n_regions, self.n_technologies)
            gen_values = result.x[n_vars_newcap:].reshape(self.n_regions, self.n_technologies)
            
            # Redondear NewCap a enteros (aproximación simple)
            newcap_values = np.round(newcap_values)
            
            return {
                'status': 'Optimal',
                'objective_value': result.fun,
                'newcap': newcap_values,
                'generation': gen_values,
                'total_cost': self._calculate_total_cost(newcap_values, gen_values),
                'solver': 'scipy'
            }
        else:
            return {'status': 'Failed', 'message': result.message}
    
    def solve_with_scipy_cost(self) -> Dict:
        """
        Resuelve el problema de minimización de costos usando scipy
        """
        print("Resolviendo con scipy - Minimización de costos...")
        
        n_vars_newcap = self.n_regions * self.n_technologies
        n_vars_gen = self.n_regions * self.n_technologies
        n_vars_total = n_vars_newcap + n_vars_gen
        
        # Función objetivo: minimizar costos
        c = np.zeros(n_vars_total)
        
        # Coeficientes para NewCap: NewCapCos * MeanCap
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                idx = i * self.n_technologies + t
                c[idx] = self.new_cap_cost[t] * self.mean_capacity[t]
        
        # Coeficientes para Gen: GenCos
        for i in range(self.n_regions):
            start_idx = n_vars_newcap + i * self.n_technologies
            end_idx = start_idx + self.n_technologies
            c[start_idx:end_idx] = self.generation_cost
        
        # Restricciones
        A_ub, b_ub = self._build_inequality_constraints()
        A_eq, b_eq = self._build_equality_constraints()
        
        bounds = [(0, None) for _ in range(n_vars_total)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            newcap_values = result.x[:n_vars_newcap].reshape(self.n_regions, self.n_technologies)
            gen_values = result.x[n_vars_newcap:].reshape(self.n_regions, self.n_technologies)
            newcap_values = np.round(newcap_values)
            
            return {
                'status': 'Optimal',
                'objective_value': result.fun,
                'newcap': newcap_values,
                'generation': gen_values,
                'total_emissions': self._calculate_total_emissions(gen_values),
                'solver': 'scipy'
            }
        else:
            return {'status': 'Failed', 'message': result.message}
    
    def solve_with_pulp_emissions(self) -> Dict:
        """
        Resuelve el problema de minimización de emisiones usando pulp
        """
        print("Resolviendo con pulp - Minimización de emisiones...")
        
        # Crear el modelo
        model = pulp.LpProblem("Minimizar_Emisiones", pulp.LpMinimize)
        
        # Variables de decisión
        newcap_vars = {}
        gen_vars = {}
        totcap_vars = {}
        
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                newcap_vars[(i,t)] = pulp.LpVariable(f"NewCap_{i+12}_{t+1}", 
                                                    lowBound=0, cat='Integer')
                gen_vars[(i,t)] = pulp.LpVariable(f"Gen_{i+12}_{t+1}", 
                                                lowBound=0, cat='Continuous')
                totcap_vars[(i,t)] = pulp.LpVariable(f"TotCap_{i+12}_{t+1}", 
                                                   lowBound=0, cat='Continuous')
        
        # Función objetivo: minimizar emisiones
        model += pulp.lpSum(self.ghg_emissions[t] * gen_vars[(i,t)] 
                           for i in range(self.n_regions) 
                           for t in range(self.n_technologies))
        
        # Restricción de demanda
        model += pulp.lpSum(gen_vars[(i,t)] 
                           for i in range(self.n_regions) 
                           for t in range(self.n_technologies)) >= self.demand
        
        # Restricciones por región y tecnología
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                # Capacidad total = capacidad existente + nueva capacidad
                model += (totcap_vars[(i,t)] == 
                         self.existing_capacity[i,t] + 
                         self.mean_capacity[t] * newcap_vars[(i,t)])
                
                # Capacidad total <= capacidad disponible
                model += totcap_vars[(i,t)] <= self.available_capacity[i,t]
                
                # Generación <= capacidad * factor de generación * horas
                model += (gen_vars[(i,t)] <= 
                         self.hours_year * self.max_generation[t] * totcap_vars[(i,t)])
        
        # Resolver
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if model.status == pulp.LpStatusOptimal:
            # Extraer resultados
            newcap_result = np.zeros((self.n_regions, self.n_technologies))
            gen_result = np.zeros((self.n_regions, self.n_technologies))
            
            for i in range(self.n_regions):
                for t in range(self.n_technologies):
                    newcap_result[i,t] = newcap_vars[(i,t)].varValue
                    gen_result[i,t] = gen_vars[(i,t)].varValue
            
            return {
                'status': 'Optimal',
                'objective_value': pulp.value(model.objective),
                'newcap': newcap_result,
                'generation': gen_result,
                'total_cost': self._calculate_total_cost(newcap_result, gen_result),
                'solver': 'pulp'
            }
        else:
            return {'status': 'Failed', 'pulp_status': pulp.LpStatus[model.status]}
    
    def solve_with_pulp_cost(self) -> Dict:
        """
        Resuelve el problema de minimización de costos usando pulp
        """
        print("Resolviendo con pulp - Minimización de costos...")
        
        model = pulp.LpProblem("Minimizar_Costos", pulp.LpMinimize)
        
        # Variables de decisión
        newcap_vars = {}
        gen_vars = {}
        totcap_vars = {}
        
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                newcap_vars[(i,t)] = pulp.LpVariable(f"NewCap_{i+12}_{t+1}", 
                                                    lowBound=0, cat='Integer')
                gen_vars[(i,t)] = pulp.LpVariable(f"Gen_{i+12}_{t+1}", 
                                                lowBound=0, cat='Continuous')
                totcap_vars[(i,t)] = pulp.LpVariable(f"TotCap_{i+12}_{t+1}", 
                                                   lowBound=0, cat='Continuous')
        
        # Función objetivo: minimizar costos
        model += pulp.lpSum(self.new_cap_cost[t] * self.mean_capacity[t] * newcap_vars[(i,t)] +
                           self.generation_cost[t] * gen_vars[(i,t)]
                           for i in range(self.n_regions) 
                           for t in range(self.n_technologies))
        
        # Restricciones (iguales al modelo de emisiones)
        model += pulp.lpSum(gen_vars[(i,t)] 
                           for i in range(self.n_regions) 
                           for t in range(self.n_technologies)) >= self.demand
        
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                model += (totcap_vars[(i,t)] == 
                         self.existing_capacity[i,t] + 
                         self.mean_capacity[t] * newcap_vars[(i,t)])
                model += totcap_vars[(i,t)] <= self.available_capacity[i,t]
                model += (gen_vars[(i,t)] <= 
                         self.hours_year * self.max_generation[t] * totcap_vars[(i,t)])
        
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if model.status == pulp.LpStatusOptimal:
            newcap_result = np.zeros((self.n_regions, self.n_technologies))
            gen_result = np.zeros((self.n_regions, self.n_technologies))
            
            for i in range(self.n_regions):
                for t in range(self.n_technologies):
                    newcap_result[i,t] = newcap_vars[(i,t)].varValue
                    gen_result[i,t] = gen_vars[(i,t)].varValue
            
            return {
                'status': 'Optimal',
                'objective_value': pulp.value(model.objective),
                'newcap': newcap_result,
                'generation': gen_result,
                'total_emissions': self._calculate_total_emissions(gen_result),
                'solver': 'pulp'
            }
        else:
            return {'status': 'Failed', 'pulp_status': pulp.LpStatus[model.status]}
    
    def _build_inequality_constraints(self):
        """Construye las restricciones de desigualdad para scipy"""
        n_vars_newcap = self.n_regions * self.n_technologies
        n_vars_gen = self.n_regions * self.n_technologies
        n_vars_total = n_vars_newcap + n_vars_gen
        
        constraints = []
        bounds = []
        
        # Restricción de demanda: sum(Gen) >= demand
        # Para scipy: -sum(Gen) <= -demand
        demand_constraint = np.zeros(n_vars_total)
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                gen_idx = n_vars_newcap + i * self.n_technologies + t
                demand_constraint[gen_idx] = -1
        constraints.append(demand_constraint)
        bounds.append(-self.demand)
        
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                newcap_idx = i * self.n_technologies + t
                gen_idx = n_vars_newcap + i * self.n_technologies + t
                
                # TotCap <= CapAva
                # TotCap = ExiCap + MeanCap * NewCap
                # ExiCap + MeanCap * NewCap <= CapAva
                # MeanCap * NewCap <= CapAva - ExiCap
                constraint1 = np.zeros(n_vars_total)
                constraint1[newcap_idx] = self.mean_capacity[t]
                constraints.append(constraint1)
                bounds.append(self.available_capacity[i,t] - self.existing_capacity[i,t])
                
                # Gen <= MaxGen * TotCap * 8760
                # Gen <= MaxGen * (ExiCap + MeanCap * NewCap) * 8760
                # Gen - MaxGen * MeanCap * NewCap * 8760 <= MaxGen * ExiCap * 8760
                constraint2 = np.zeros(n_vars_total)
                constraint2[gen_idx] = 1
                constraint2[newcap_idx] = -self.hours_year * self.max_generation[t] * self.mean_capacity[t]
                constraints.append(constraint2)
                bounds.append(self.hours_year * self.max_generation[t] * self.existing_capacity[i,t])
        
        return np.array(constraints), np.array(bounds)
    
    def _build_equality_constraints(self):
        """Construye las restricciones de igualdad para scipy"""
        n_vars_newcap = self.n_regions * self.n_technologies
        n_vars_gen = self.n_regions * self.n_technologies
        n_vars_total = n_vars_newcap + n_vars_gen
        
        # No hay restricciones de igualdad en este modelo
        # Todas las restricciones son de desigualdad
        return np.empty((0, n_vars_total)), np.empty(0)
    
    def _calculate_total_cost(self, newcap: np.ndarray, generation: np.ndarray) -> float:
        """Calcula el costo total"""
        installation_cost = 0
        operation_cost = 0
        
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                installation_cost += (self.new_cap_cost[t] * 
                                    self.mean_capacity[t] * 
                                    newcap[i,t])
                operation_cost += self.generation_cost[t] * generation[i,t]
        
        return installation_cost + operation_cost
    
    def _calculate_total_emissions(self, generation: np.ndarray) -> float:
        """Calcula las emisiones totales"""
        total_emissions = 0
        for i in range(self.n_regions):
            for t in range(self.n_technologies):
                total_emissions += self.ghg_emissions[t] * generation[i,t]
        return total_emissions
    
    def print_results(self, result: Dict):
        """Imprime los resultados de manera formateada"""
        if result['status'] != 'Optimal':
            print(f"Solución no óptima: {result}")
            return
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS - Solver: {result['solver'].upper()}")
        print(f"{'='*60}")
        print(f"Estado: {result['status']}")
        print(f"Valor objetivo: {result['objective_value']:,.2f}")
        
        if 'total_cost' in result:
            print(f"Costo total: ${result['total_cost']:,.2f}")
        if 'total_emissions' in result:
            print(f"Emisiones totales: {result['total_emissions']:,.2f} toneladas")
        
        print("\nPlantas nuevas por tecnología:")
        tech_names = self.plants['Tecnologia'].values
        for t in range(self.n_technologies):
            total_new = np.sum(result['newcap'][:, t])
            print(f"{tech_names[t]}: {total_new:.0f} plantas")
        
        print(f"\nGeneración total: {np.sum(result['generation']):,.2f} MWh")
        print(f"Demanda requerida: {self.demand:,.2f} MWh")
        
    def run_all_optimizations(self) -> Dict:
        """Ejecuta todas las optimizaciones y compara resultados"""
        results = {}
        
        # Scipy
        try:
            results['scipy_emissions'] = self.solve_with_scipy_emissions()
        except Exception as e:
            print(f"Error en scipy emisiones: {e}")
            results['scipy_emissions'] = {'status': 'Error', 'message': str(e)}
        
        try:
            results['scipy_cost'] = self.solve_with_scipy_cost()
        except Exception as e:
            print(f"Error en scipy costos: {e}")
            results['scipy_cost'] = {'status': 'Error', 'message': str(e)}
        
        # Pulp
        try:
            results['pulp_emissions'] = self.solve_with_pulp_emissions()
        except Exception as e:
            print(f"Error en pulp emisiones: {e}")
            results['pulp_emissions'] = {'status': 'Error', 'message': str(e)}
        
        try:
            results['pulp_cost'] = self.solve_with_pulp_cost()
        except Exception as e:
            print(f"Error en pulp costos: {e}")
            results['pulp_cost'] = {'status': 'Error', 'message': str(e)}
        
        return results


def main():
    """Función principal para ejecutar las optimizaciones"""
    
    # Crear instancia del optimizador
    # Los archivos CSV deben estar en el mismo directorio
    optimizer = EnergyOptimizer('CapAva.csv', 'ExiCap.csv', 'Plants.csv')
    
    print("Iniciando optimizaciones del sector energético...")
    print("="*60)
    
    # Ejecutar todas las optimizaciones
    results = optimizer.run_all_optimizations()
    
    # Mostrar resultados
    for method, result in results.items():
        print(f"\n{method.upper().replace('_', ' - ')}")
        optimizer.print_results(result)
    
    # Comparación de resultados
    print(f"\n{'='*60}")
    print("COMPARACIÓN DE RESULTADOS")
    print(f"{'='*60}")
    
    if (results['pulp_emissions']['status'] == 'Optimal' and 
        results['pulp_cost']['status'] == 'Optimal'):
        
        emissions_obj = results['pulp_emissions']['objective_value']
        cost_obj = results['pulp_cost']['objective_value']
        emissions_cost = results['pulp_emissions']['total_cost']
        cost_emissions = results['pulp_cost']['total_emissions']
        
        print(f"Minimizando emisiones:")
        print(f"  - Emisiones: {emissions_obj:,.2f} toneladas")
        print(f"  - Costo: ${emissions_cost:,.2f}")
        
        print(f"\nMinimizando costos:")
        print(f"  - Costo: ${cost_obj:,.2f}")
        print(f"  - Emisiones: {cost_emissions:,.2f} toneladas")
        
        print(f"\nTrade-offs:")
        cost_increase = ((emissions_cost - cost_obj) / cost_obj) * 100
        emission_reduction = ((cost_emissions - emissions_obj) / cost_emissions) * 100
        
        print(f"  - Incremento en costo para reducir emisiones: {cost_increase:.1f}%")
        print(f"  - Reducción en emisiones con mayor costo: {emission_reduction:.1f}%")


if __name__ == "__main__":
    main()