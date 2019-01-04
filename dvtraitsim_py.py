import numpy as np
import platform
import sys
if platform.system() == 'Windows':
    sys.path.append('C:/Liang/GPU_Python')
elif platform.system() == 'Darwin':
    sys.path.append('/Users/dudupig/Documents/GitHub/Code/Pro2/Python_p2')

from dvtraitsim_shared import DVTreeData, DVParam
import dvtraitsim_cpp as dvcpp


def competition_functions(a, zi, nj):
    """ competition functions.

    returns beta = Sum_j( exp(-a(zi-zj)^2) * Nj)
            sigma = Sum_j( 2a * (zi-zj) * exp(-a(zi-zj)^2) * Nj)
            sigmaSqr = Sum_j( 4a^2 * (zi-zj)^2 * exp(-a(zi-zj)^2) * Nj)
    """
    T = zi[:, np.newaxis] - zi  # trait-distance matrix (via 'broadcasting')
    t1 = np.exp(-a * T ** 2) * nj
    t2 = (2 * a) * T
    beta = np.sum(t1, axis=1)
    sigma = np.sum(t2 * t1, axis=1)
    sigmasqr = np.sum(t2 ** 2 * t1, axis=1)
    return beta, sigma, sigmasqr


def DVSim(td, param):
    # parameters from DVParam
    gamma = param[0]
    a = param[1]
    K = param[2]
    nu = param[3]
    r = param[4]
    theta = param[5]
    Vmax = param[6]
    inittrait = param[7]
    initpop = param[8]
    initpop_sigma = param[9]
    break_on_mu = bool(param[10])

    sim_evo_time = td.sim_evo_time
    events = td.sim_events

    # Initialize trait evolution and population evolution matrices
    trait_RI_dr = np.zeros((sim_evo_time + 1, td.total_species))  # trait
    population_RI_dr = np.zeros((sim_evo_time + 1, td.total_species)).astype(np.int32)  # population
    V = np.zeros((sim_evo_time + 1, td.total_species))  # trait variance

    #  initialize condition for species trait and population
    trait_RI_dr[0, (0, 1)] = inittrait  # trait for species
    population_RI_dr[0, (0, 1)] = np.random.normal(initpop, initpop_sigma, 2).astype(np.int32)
    V[0] = (1 / td.total_species)   # <----- why?
    existing_species = td.traittable
    node = 0;
    next_event = events[node];
    idx = np.where(existing_species[node] == 1)[0]    # existing species

    # trait-population coevolution model
    for i in range(sim_evo_time):
        # pull current state
        Ni = population_RI_dr[i, idx]
        Vi = V[i, idx]
        zi = trait_RI_dr[i, idx]
        Ki = K
        dtz = theta - zi
        beta, sigma, sigmasqr = competition_functions(a, zi, Ni)

        # update
        var_trait = Vi / (2.0 * Ni)
        trait_RI_dr[i + 1, idx] = zi + Vi * (2.0 * gamma * dtz + 1 / Ki * sigma) + np.random.normal(0.0, var_trait)
        mu = Ni * r * np.exp(-gamma * dtz**2 + (1 - beta / Ki)) # un-truncated mean
        if np.any(mu <= 1.0):       # mu < 1.0 + 1.11e-16
            if (break_on_mu):
                print(i, "invalid mean population size")
                break
        ztp_lambda = dvcpp.ztp_lambda_from_untruncated_mean(mu)
        population_RI_dr[i + 1, idx] = dvcpp.ztpoisson(ztp_lambda)
        V[i + 1, idx] = Vi / 2.0 + 2.0 * Ni * nu * Vmax / (1.0 + 4.0 * Ni * nu) \
                        + Vi**2 * (
                            -2.0 * gamma + 4.0 * gamma**2 * dtz**2 +
                                1.0 / Ki * (2.0 * a * beta - sigmasqr) + 4.0 * gamma / Ki *
                                dtz * sigma + sigma**2 / Ki**2
                            )
        # events
        while (i + 1) == next_event[0]:
            daughter = next_event[2]
            if (daughter == -1):
                # extinction
                extinct_species = next_event[1]
                V[i + 1, extinct_species] = None
                trait_RI_dr[i + 1, extinct_species] = None
                population_RI_dr[i + 1, extinct_species] = 0
            else:
                # speciation
                parent = next_event[1]
                parentN = population_RI_dr[i + 1, parent]
                if parentN <= 1:
                    print(i, "attempt to split singleton")
                    # results in split <- 0, will be trapped by sanity check below  
                split = dvcpp.split_binomial50(parentN)
                population_RI_dr[i + 1, daughter] = parentN - split
                population_RI_dr[i + 1, parent] = split
                V[i + 1, parent] *= 0.5
                V[i + 1, daughter] = V[i + 1, parent]
                trait_RI_dr[i + 1, daughter] = trait_RI_dr[i + 1, parent]
            # advance to next event/node
            node = node + 1
            next_event = events[node];
            idx = np.where(existing_species[node] == 1)[0]

        # sanity check
        if np.any(population_RI_dr[i + 1, idx] < 1):
            print(i, 'Inconsistent extinction')
            break
        if np.any(V[i + 1, idx] < 0.0) or np.any(V[i + 1, idx] > 100000.0):
            print(i, 'runaway variance')
            break

    row_ext = np.where(population_RI_dr == 0)[0]
    col_ext = np.where(population_RI_dr == 0)[1]
    V[row_ext, col_ext] = None
    trait_RI_dr[row_ext, col_ext] = None
    return { 'sim_time': i + 1, 'N': population_RI_dr, 'Z': trait_RI_dr, 'V': V }
