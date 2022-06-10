import cvxpy as cp
import itertools
import numpy as np



def optimal_enumeration(H):
    N, M = H.shape
    min_power = np.inf
    best_w = None
    best_z = None
    for L in range(M+1):
        for subset in itertools.combinations(range(M), L):
            z = np.ones(M)
            z[list(subset)] = -1
            w, power = solve_subproblem(H,z)
            if w is not None:
                if power < min_power:
                    min_power = power
                    best_w = w.copy()
                    best_z = z.copy()
    return min_power, best_w.copy(), best_z.copy()
    

def solve_subproblem(H,z):
    N,M = H.shape
    w = cp.Variable(N)
    objective = cp.Minimize(cp.square(cp.norm(w)))

    constraints = [cp.multiply(z, H.T @ w) >= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    p = w.value
    return w.value, objective.value
