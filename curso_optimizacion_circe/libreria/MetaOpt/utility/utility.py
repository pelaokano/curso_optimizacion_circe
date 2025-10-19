import numpy as np

def calculate_temperature(n_gen):
    return max(3 * 2 ** (-n_gen / 50), 0.3)

def boltzmann_selection(fitness, T):
    f=np.array(fitness)
    norm=np.sum(np.exp(f/T))
    p=np.exp(f/T)/norm
    return p

def selection_fn(fitness, temperature):
    #Normalize fitnesses before passing to exponential, this will avoid scale dependence
    fitness=(fitness-np.mean(fitness))/(np.std(fitness)) 
    return boltzmann_selection(fitness, T=temperature)

def random_mutation(sol, mutation_rate, range):
    l=len(sol)
    sol=np.array(sol)
    N=np.random.binomial(l, mutation_rate)
    mutation_idx=np.random.choice(l, replace=False, size=N)
    additions=np.random.rand(N)*(range[1]-range[0])+range[0]
    sol[mutation_idx]+=additions
    sol=sol.tolist()
    return sol

def gaussian_mutation(sol, mutation_rate, std):
    l=len(sol)
    sol=np.array(sol)
    N=np.random.binomial(l, mutation_rate)
    # print(N)
    mutation_idx=np.random.choice(l, replace=False, size=N)
    additions=np.random.normal(loc=0., scale=std, size=N)
    sol[mutation_idx]+=additions
    sol=sol.tolist()
    return sol

def single_point_crossover(parents, crossover_p):
    if np.random.random()<crossover_p:
        l=len(parents[0])
        point=np.random.choice(l)
        child=np.concatenate((parents[0][:point],parents[1][point:]))
    else:
        child=parents[0]
    return child

def initialize_speeds(model, percent=0.2):
    limit = lambda x: np.sqrt(1/x) * percent
    data = [(name, param.shape) for name, param in model.named_parameters()]
    paired_data = [(data[i], data[i+1]) for i in range(0, len(data) - 1, 2)]
    param_list = []
    for x in paired_data:
        lim1, dim1, dim2, dim3 = x[0][1][1], x[0][1][0], x[0][1][1], x[1][1][0]
        limit_value = limit(lim1)
        random_values = np.random.uniform(-limit_value, limit_value, size=(dim1*dim2+dim3))
        param_list.append(random_values)
    data = np.concatenate(param_list)
    return data
