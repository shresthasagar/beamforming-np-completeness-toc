import numpy as np
# from z3 import *
import z3
from decimal import *


def to_float(x_rational):
    x_frac = x_rational.as_fraction()
    return float(Decimal(x_frac.numerator)/Decimal(x_frac.denominator))


def beamforming(H, kappa, print_log=False, timeout=120000):
    N, M = H.shape
    x = [z3.Real('x_{}'.format(i)) for i in range(N)]
    z = [z3.Bool('z_{}'.format(i)) for i in range(M)]

    s = z3.Solver()
    s.set("timeout", timeout)

    for m, p in enumerate(z):
        multiplier = z3.If(p, 1, -1)
        snr = 0
        for n in range(N):
            snr = snr + multiplier*x[n]*H[n,m]
        s.add(snr >= 1)

    objective = 0
    for n in range(N):
        objective = objective + x[n]*x[n]
    s.add(objective < kappa )
    r = s.check()

    if r!=z3.sat:
        print('is not sat')
        return None, None
    # assert r==z3.sat, "the problem is not SAT"

    m = s.model()
    
    if print_log:
        for a in x:
            print(a, ">>", to_float(m[a]))

        for b in z:
            print(b, ">>", bool((m[b]))*2 - 1)

    x_sol = [to_float(m[a]) for a in x]
    z_sol = [bool(m[b])*2-1 for b in z]

    return np.array(x_sol), np.array(z_sol)

if __name__ == '__main__':
    antennas = [1,2,3,4,5]
    users = [1,2,3,4,5]
    
    N, M = 3,2
    kappa = 50
    H = np.random.rand(N,M)
    x_sol, z_sol = beamforming(H, kappa)
    if x_sol is None:
        print('Problem is not satisfiable')
    else:
        power = np.linalg.norm(x_sol, 2)
        print('x_sol', x_sol)
        print('z_sol', z_sol)
        print('power', power)
