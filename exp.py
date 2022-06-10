import numpy as np
from beamforming import *
import time
from enumeration import optimal_enumeration
import pickle

np.random.seed(400)

N_list = [1,2,3,4,5]
users = [1,2,3,4,5]

kappa_coeff = 100
M = 4
num_egs = 15

timeout = 50000

FILEPATH = 'data/result_N3.pkl'
result = {'size':[], 'data':[], 'time':[], 'opt_obj': [], 'smt_obj':[]}
for N in N_list: 
    print("N", N)
    result['size'].append((N,M))
    avg_time = 0
    avg_smt_solution = 0
    avg_opt_solution = 0
    
    optimals = []
    solutions = []
    objectives = []
    solution_time = []
    for eg in range(num_egs):
        H = np.random.randn(N,M)
        min_power, optimal_w, optimal_z = optimal_enumeration(H)
        kappa = kappa_coeff*min_power
        optimals.append(min_power)
        # kappa = 1000

        print('now computing smt')
        t1 = time.time()
        x_sol, z_sol = beamforming(H, kappa, timeout=timeout)
        time_taken = time.time() - t1
        
        solutions.append(x_sol)
        if x_sol is None:
            print('Problem is not satisfiable', 'time', time_taken)
            objectives.append(None)
        else:
            power = np.linalg.norm(x_sol, 2)**2
            objectives.append(power)
            print('power', power, 'time', time_taken)

        solution_time.append(time_taken)
        
    result['time'].append(solution_time)
    result['opt_obj'].append(optimals)
    result['smt_obj'].append(objectives)
    
    with open(FILEPATH, 'wb') as handle:
        pickle.dump(result, handle)
