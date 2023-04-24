import math
import random
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

GENERATIONS = 10
POPULATION_SIZE = 200
OPT_PARAM_VECTOR_SIZE = 20
BINARY_CODE_SIZE = 4
MUTATION_PROBABILITY = 0.05
GAMMA_BOUNDS = (-0.524, 0.524)
BETA_BOUNDS = (-5, 5)
K = 200

x0 = 0.0
y0 = 8.0
alpha0 = 0.0
v0 = 0.0
xf = 0.0
yf = 0.0
alphaf = 0.0
vf = 0.0

def initialize_population():
    population = []
    binary_population = []
    for i in range(POPULATION_SIZE):
        binary_code = np.random.randint(0, 2, size=(OPT_PARAM_VECTOR_SIZE * BINARY_CODE_SIZE))
        binary_population.append(binary_code)
        gamma_beta_values = []
        for j in range(0, OPT_PARAM_VECTOR_SIZE, 2):
            gamma_binary = binary_code[j * BINARY_CODE_SIZE: (j + 1) * BINARY_CODE_SIZE]
            beta_binary = binary_code[(j + 1) * BINARY_CODE_SIZE: (j + 1) * BINARY_CODE_SIZE + 2]
            decimal_gamma = int(''.join(str(bit) for bit in gamma_binary), 2)
            decimal_beta = int(''.join(str(bit) for bit in beta_binary), 2)
            gamma = GAMMA_BOUNDS[0] + (GAMMA_BOUNDS[1] - GAMMA_BOUNDS[0]) * (decimal_gamma / (2 ** BINARY_CODE_SIZE - 1))
            beta = BETA_BOUNDS[0] + (BETA_BOUNDS[1] - BETA_BOUNDS[0]) * (decimal_beta / (2 ** BINARY_CODE_SIZE - 1))
            gamma_beta_values.append(gamma)
            gamma_beta_values.append(beta)
        population.append(np.asarray(gamma_beta_values))
    return population, binary_population

def ode_system(state_space, beta_t, gamma_t):
    x, y, alpha, v = state_space
    alpha_dot = gamma_t
    v_dot = beta_t
    x_dot = v * np.cos(alpha)
    y_dot = v * np.sin(alpha)
    return [x_dot, y_dot, alpha_dot, v_dot]

def checking(x, y):
    if (x <= -4) and (y > 3):
        return True
    elif (x > -4 and x < 4) and y > -1:
        return True
    elif (x >= 4) and (y > 3):
        return True
    else:
        return False

def cost_function(individual):
    gamma = individual[0:20:2]
    t = np.linspace(0, 10, 10)
    time_new = np.linspace(0, 10, 100)
    f_gamma = interp1d(t, gamma, kind='linear', bounds_error=False)
    gamma_new = f_gamma(time_new)
    beta = individual[1:20:2]
    f_beta = interp1d(t, beta, kind='linear', bounds_error=False)
    beta_new = f_beta(time_new)

    dt = time_new[1] - time_new[0]
    z = np.array([x0, y0, alpha0, v0])
    z_history = np.zeros((len(time_new), len(z)))
    z_history[0] = z

    for i in range(1, len(time_new)):
        z_dot = ode_system(z, beta_new[i], gamma_new[i])
        z += np.array(z_dot)*dt
        z_history[i] = z

    cost_values = []
    for i in range(len(time_new)):
        condition = abs(z_history[i, 0] - xf) + abs(z_history[i, 1] - yf) + abs(z_history[i, 2] - alphaf) + abs(z_history[i, 3] - vf)
        #feasible
        if checking(z_history[i, 0], z_history[i, 1]):
            cost_values.append(condition)
        #infeasible
        else:
            cost_values.append(K)

    J = np.asarray(cost_values)
    return np.mean(J)

def fitness_function(cost):
    return 1 / (1 + cost)

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]), axis=0)
    return child1, child2

def mutation(child):
    mutated_child = list(child)
    for i in range(len(mutated_child)):
        if random.random() < MUTATION_PROBABILITY:
            mutated_child[i] = 1 - mutated_child[i]
    return np.array(mutated_child)


def main():
    individuals, binary_population = initialize_population()
    for gen in range(GENERATIONS):
        new_population = []
        cost_values = []
        population = []
        for i in range(POPULATION_SIZE):
            cost = cost_function(individuals[i])
            cost_values.append(cost)
            population.append([binary_population[i], cost_values[i]])

        fitness_ratios = []
        for i in range(POPULATION_SIZE):
            fitness_ratios.append(fitness_function(cost_values[i]))

        elites = sorted(population, key=lambda x: x[1])[0:2]
        min_cost_value = [i[1] for i in elites]
        print(f"Generation {gen}: {min_cost_value[0]}")

        fitness_ratios = sorted(fitness_ratios, reverse=True)
        max_fitness_values = [i for i in fitness_ratios][:2]

        for i in range(len(elites)):
            new_population.append(elites[i])
            population.pop(0)
            fitness_ratios.pop(0)

        while len(new_population) <= (POPULATION_SIZE-2):
            parents = random.choices(population, weights=fitness_ratios, k=2)
            new_parents = crossover(parents[0][0], parents[1][0])
            child1 = mutation(new_parents[0])
            child2 = mutation(new_parents[1])

            new_population.append([child1, None])
            new_population.append([child2, None])

        population = new_population

    # print(population)




if __name__ == '__main__':
    main()
