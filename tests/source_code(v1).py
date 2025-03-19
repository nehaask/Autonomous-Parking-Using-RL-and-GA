import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

GAMMA_BOUNDS = (-0.524, 0.524)
BETA_BOUNDS = (-5, 5)
POPULATION_SIZE = 200
OPT_PARAM_VECTOR_SIZE = 20
BINARY_CODE_SIZE = 7
MUTATION_RATE = 0.05
K = 200

COST_TOL = 0.1
MAX_POP_SIZE = 500
MAX_GEN = 1200
MAX_TIME = 7 * 60

x0 = 0.0
y0 = 8.0
alpha0 = 0.0
v0 = 0.0
xf = 0.0
yf = 0.0
alphaf = 0.0
vf = 0.0

def convert_binary_to_gray(binary_list):
    gray = np.zeros_like(binary_list)
    gray[..., 0] = binary_list[..., 0]
    for i in range(1, binary_list.shape[-1]):
        gray[..., i] = np.bitwise_xor(binary_list[..., i - 1], binary_list[..., i])
    return gray

def gamma_beta(binary_code):
    gamma_list = []
    beta_list = []
    for i in range(len(binary_code)):
        gamma_temp = []
        beta_temp = []
        for j in range(0, OPT_PARAM_VECTOR_SIZE, 2):
            gamma_binary = binary_code[i][j * BINARY_CODE_SIZE: (j + 1) * BINARY_CODE_SIZE]
            beta_binary = binary_code[i][(j + 1) * BINARY_CODE_SIZE: (j + 1) * BINARY_CODE_SIZE + 7]
            decimal_gamma = int(''.join(str(bit) for bit in gamma_binary), 2)
            decimal_beta = int(''.join(str(bit) for bit in beta_binary), 2)
            gamma = GAMMA_BOUNDS[0] + (GAMMA_BOUNDS[1] - GAMMA_BOUNDS[0]) * (
                        decimal_gamma / (2 ** BINARY_CODE_SIZE - 1))
            beta = BETA_BOUNDS[0] + (BETA_BOUNDS[1] - BETA_BOUNDS[0]) * (
                        decimal_beta / (2 ** BINARY_CODE_SIZE - 1))
            gamma_temp.append(gamma)
            beta_temp.append(beta)
        gamma_list.append(gamma_temp)
        beta_list.append(beta_temp)

    return np.array(gamma_list), np.array(beta_list)

def checking(x, y):
    if (x <= -4) and (y > 3):
        return True
    elif (x > -4 and x < 4) and y > -1:
        return True
    elif (x >= 4) and (y > 3):
        return True
    else:
        return False

def interpolate_gamma_beta(pop, gamma, beta):
    t = np.linspace(0, 10, 10)
    time_new = np.linspace(0, 10, 100)
    interp_gamma = np.zeros((len(gamma), len(time_new)))
    interp_beta = np.zeros((len(beta), len(time_new)))

    for individual in range(len(pop)):
        f_gamma = interp1d(t, gamma[individual], kind='linear', bounds_error=False)
        gamma_new = f_gamma(time_new)
        interp_gamma[individual] = gamma_new

        f_beta = interp1d(t, beta[individual], kind='linear',bounds_error=False)
        beta_new = f_beta(time_new)
        interp_beta[individual] = beta_new

    return interp_gamma, interp_beta

def ode_system(state_space, beta_t, gamma_t):
    x, y, alpha, v = state_space
    alpha_dot = gamma_t
    v_dot = beta_t
    x_dot = v * np.cos(alpha)
    y_dot = v * np.sin(alpha)
    return [x_dot, y_dot, alpha_dot, v_dot]


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]), axis=0)
    return child1, child2

def mutation(child):
    mutated_child = list(child)
    for i in range(len(mutated_child)):
        if random.random() < MUTATION_RATE:
            if mutated_child[i] == 1:
                mutated_child[i] = 0
            else:
                mutated_child[i] = 1
    return np.array(mutated_child)

def euler(gamma, beta, ind):
    time_new = np.linspace(0, 10, 100)
    dt = time_new[1] - time_new[0]
    z = np.array([x0, y0, alpha0, v0])
    z_history = np.zeros((len(time_new), len(z)))
    z_history[0] = z

    flag = True
    for i in range(1, len(time_new)):
        z_dot = ode_system(z, beta[ind][i], gamma[ind][i])
        z += np.array(z_dot) * dt
        z_history[i] = z
        if not checking(z_history[i, 0], z_history[i, 1]):
            flag = False

    if flag:
        cost = math.sqrt(abs(z[0] - xf) ** 2 + abs(z[1] - yf) ** 2) + abs(z[2] - alphaf) + abs(z[3] - vf)
    else:
        cost = K
    return cost, z

def fitness_function(cost):
    return 1 / (1 + cost)

def plot_the_plots(state, x_plots, y_plots):
    x1 = np.linspace(-16, -4, 100)
    y1 = np.ones_like(x1) * 3
    y2 = np.linspace(3, -1, 100)
    x2 = np.ones_like(y2) * -4
    x4 = np.linspace(-4, 4, 100)
    y4 = np.ones_like(x4) * -1
    y5 = np.linspace(3, -1, 100)
    x5 = np.ones_like(y5) * 4
    x3 = np.linspace(4, 15, 100)
    y3 = np.ones_like(x3) * 3
    fig, ax = plt.subplots()
    ax.plot(x1, y1, color='blue')
    ax.plot(x2, y2, color='red')
    ax.plot(x4, y4, color='green')
    ax.plot(x5, y5, color='purple')
    ax.plot(x3, y3, color='orange')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Line Segments')
    plt.show()

def main():
    pop = np.random.randint(2, size=(POPULATION_SIZE, OPT_PARAM_VECTOR_SIZE * BINARY_CODE_SIZE))
    grey = convert_binary_to_gray(pop)
    gamma, beta = gamma_beta(grey)
    interpolated_gamma, interpolated_beta = interpolate_gamma_beta(pop, gamma, beta)
    cost = []
    x_plot = []
    y_plot = []
    for individual in range(0, 200):
        e, state = euler(interpolated_gamma, interpolated_beta, individual)
        x_plot.append(state[0])
        y_plot.append(state[1])

        if e < 0.1:
            quit()
        cost.append(e)

    plt.plot(x_plot, y_plot)
    fitness_ratios = []
    for i in range(POPULATION_SIZE):
        fitness_ratios.append(fitness_function(cost[i]))

    elites = np.argsort(fitness_ratios)[-2:][::-1]
    min_cost_value = min(cost[elites[0]], cost[elites[1]])
    print(f"Generation 0: {min_cost_value}")

    new_population = []
    for i in range(len(elites)):
        new_population.append(grey[elites[i]])
    while len(new_population) <= (POPULATION_SIZE - 2):
        parents = random.choices(grey, weights=fitness_ratios, k=2)
        child1, child2 = crossover(parents[0], parents[1])
        mutated_child1 = mutation(child1)
        mutated_child2 = mutation(child2)

        new_population.append(mutated_child1)
        new_population.append(mutated_child2)

    grey = new_population
    for gen in range(1, MAX_GEN):
        gamma, beta = gamma_beta(grey)
        interpolated_gamma, interpolated_beta = interpolate_gamma_beta(pop, gamma, beta)
        cost = []
        x_plot = [[]]
        y_plot = [[]]
        for individual in range(0, 200):
            x_plot.append([])
            y_plot.append([])
            e, state = euler(interpolated_gamma, interpolated_beta, individual)
            x_plot[individual].append(state[0])
            y_plot[individual].append(state[1])
            if e < 0.1:
                print("Final state values:")
                print("x_f = ", state[0])
                print("y_f = ", state[1])
                print("alpha_f = ", state[2])
                print("v_f = ", state[3])
                plot_the_plots(state, x_plot, y_plot)
                quit()
            cost.append(e)
        fitness_ratios = []
        for i in range(POPULATION_SIZE):
            fitness_ratios.append(fitness_function(cost[i]))

        elites = np.argsort(fitness_ratios)[-2:][::-1]
        min_cost_value = min(cost[elites[0]], cost[elites[1]])
        print(f"Generation {gen}: {min_cost_value}")

        new_population = []
        for i in range(len(elites)):
            new_population.append(grey[elites[i]])
        while len(new_population) <= (POPULATION_SIZE - 2):
            parents = random.choices(grey, weights=fitness_ratios, k=2)
            child1, child2 = crossover(parents[0], parents[1])
            mutated_child1 = mutation(child1)
            mutated_child2 = mutation(child2)

            new_population.append(mutated_child1)
            new_population.append(mutated_child2)

        grey = new_population


if __name__ == '__main__':
    main()
