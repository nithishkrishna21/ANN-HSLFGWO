import numpy as np
import random 

def find_alpha_beta_delta(pop_weights_vector, errors):

    # we are looking for solutions that minimize the error
    # hence we take the solutions with least 3 errors

    idx1, idx2, idx3 = np.argsort(errors)[:3]
    x_alpha_score, x_beta_score, x_delta_score = np.sort(errors)[:3]
    X_alpha = pop_weights_vector[idx1]
    X_beta = pop_weights_vector[idx2]
    X_delta = pop_weights_vector[idx3]

    return X_alpha, X_beta, X_delta, x_alpha_score, x_beta_score, x_delta_score


def GWO(population_vectors, X_alpha, X_beta, X_delta, a):

    num_solutions, num_dimensions = population_vectors.shape
    X_new = np.zeros((num_solutions, num_dimensions))

    for i in range(num_solutions):
        
        A1 = a * (2 * np.random.random(num_dimensions) - 1)
        A2 = a * (2 * np.random.random(num_dimensions) - 1)
        A3 = a * (2 * np.random.random(num_dimensions) - 1)
        C1 = 2 * np.random.random(num_dimensions)
        C2 = 2 * np.random.random(num_dimensions)
        C3 = 2 * np.random.random(num_dimensions)
        X1 = X_alpha - (A1 * np.abs((C1 * X_alpha) - population_vectors[i]))
        X2 = X_beta - (A2 * np.abs((C2 * X_beta) - population_vectors[i]))
        X3 = X_delta - (A3 * np.abs((C3 * X_delta) - population_vectors[i]))
        pos_new = (X1 + X2 + X3) / 3.0
        
        # clip the new solution
        pos_new = np.clip(pos_new, -10, 10)
        X_new[i] = pos_new
        
    return X_new  


def CGWO(population_vectors, X_alpha, X_beta, X_delta, a, 
         ran1, ran2, ran3, ran4, ran5, ran6):

    num_solutions, num_dimensions = population_vectors.shape
    X_new = np.zeros((num_solutions, num_dimensions))

    for i in range(num_solutions):
        
        A1 = a * (2 * ran1[i, :] - 1)
        A2 = a * (2 * ran2[i, :] - 1)
        A3 = a * (2 * ran3[i, :] - 1)
        C1 = 2 * ran4[i, :]
        C2 = 2 * ran5[i, :]
        C3 = 2 * ran6[i, :]
        X1 = X_alpha - (A1 * np.abs((C1 * X_alpha) - population_vectors[i]))
        X2 = X_beta - (A2 * np.abs((C2 * X_beta) - population_vectors[i]))
        X3 = X_delta - (A3 * np.abs((C3 * X_delta) - population_vectors[i]))
        pos_new = (X1 + X2 + X3) / 3.0
        
        # clip the new solution
        # pos_new = np.clip(pos_new, -10, 10)
        X_new[i] = pos_new
        
    return X_new


def IGWO(population_vectors, X_alpha, X_beta, X_delta, a_alpha, a_delta):

    # Improved GWO
    num_solutions, num_dimensions = population_vectors.shape
    X_new = np.zeros((num_solutions, num_dimensions))

    a_beta = (a_alpha + a_delta) * 0.5

    for i in range(num_solutions):

        A1 = a_alpha * (2 * np.random.random(num_dimensions) - 1)
        A2 = a_beta * (2 * np.random.random(num_dimensions) - 1)
        A3 = a_delta * (2 * np.random.random(num_dimensions) - 1)
        C1 = 2 * np.random.random(num_dimensions)
        C2 = 2 * np.random.random(num_dimensions)
        C3 = 2 * np.random.random(num_dimensions)
        X1 = X_alpha - (A1 * np.abs((C1 * X_alpha) - population_vectors[i]))
        X2 = X_beta - (A2 * np.abs((C2 * X_beta) - population_vectors[i]))
        X3 = X_delta - (A3 * np.abs((C3 * X_delta) - population_vectors[i]))
        pos_new = (X1 + X2 + X3) / 3.0

        X_new[i] = pos_new

    return X_new 


def HGWO(population_vectors, X_alpha, X_beta, X_delta, a_alpha, a_beta, a_delta):

    # Hyperbolic GWO
    num_solutions, num_dimensions = population_vectors.shape
    X_new = np.zeros((num_solutions, num_dimensions))

    for i in range(num_solutions):

        A1 = a_alpha * (2 * np.random.random(num_dimensions) - 1)
        A2 = a_beta * (2 * np.random.random(num_dimensions) - 1)
        A3 = a_delta * (2 * np.random.random(num_dimensions) - 1)
        C1 = 2 * np.random.random(num_dimensions)
        C2 = 2 * np.random.random(num_dimensions)
        C3 = 2 * np.random.random(num_dimensions)
        X1 = X_alpha - (A1 * np.abs((C1 * X_alpha) - population_vectors[i]))
        X2 = X_beta - (A2 * np.abs((C2 * X_beta) - population_vectors[i]))
        X3 = X_delta - (A3 * np.abs((C3 * X_delta) - population_vectors[i]))
        pos_new = (X1 + X2 + X3) / 3.0
#         pos_new = np.clip(pos_new, -10, 10)
        X_new[i] = pos_new

    return X_new 


def MHGWO(population_vectors, X_alpha, X_beta, X_delta, a_alpha, a_beta, a_delta, R1, R2, R3):
    
    # Modified Search with Hyperbolic GWO
    num_solutions, num_dimensions = population_vectors.shape
    X_new = np.zeros((num_solutions, num_dimensions))

    for i in range(num_solutions):

        A1 = a_alpha * (2 * np.random.rand(num_dimensions) - 1)
        A2 = a_beta * (2 * np.random.rand(num_dimensions) - 1)
        A3 = a_delta * (2 * np.random.rand(num_dimensions) - 1)
        C1 = 2 * np.random.rand(num_dimensions)
        C2 = 2 * np.random.rand(num_dimensions)
        C3 = 2 * np.random.rand(num_dimensions)
        X1 = X_alpha - (A1 * np.abs((C1 * X_alpha) - population_vectors[i]))
        X2 = X_beta - (A2 * np.abs((C2 * X_beta) - population_vectors[i]))
        X3 = X_delta - (A3 * np.abs((C3 * X_delta) - population_vectors[i]))
        
        pos_new = (R1 * X1) + (R2 * X2) + (R3 * X3)
#         pos_new = np.clip(pos_new, -10, 10)
        X_new[i] = pos_new

    return X_new 