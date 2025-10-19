import random
from MetaOpt import AGMO_NSGAII

def performance_example(population):
    def cubic2(x, y):
        return - (x**3 + y**3 - 3*x*y)

    def quadratic(x, y):
        return - (x**2 + y**2 - 4*x - 6*y)

    return [(cubic2(x, y), quadratic(x, y)) for x, y in population]

if __name__ == '__main__':
    
    population = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(100)]
    algoritmo = AGMO_NSGAII(initial_population=population,
                            performance_fn= performance_example,
                            maximizes=(True, True))
    
    algoritmo.run(100, verbose=True, plot_fitness=True)
    
    algoritmo.save(r'C:\Users\AdrianAlarconBecerra\Documents\proyectos_python\libreria_GA')