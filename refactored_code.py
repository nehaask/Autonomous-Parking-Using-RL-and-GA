import math
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

GENERATIONS = 10
POPULATION_SIZE = 200
OPT_PARAM_VECTOR_SIZE = 20
BINARY_CODE_SIZE = 7
MUTATION_PROBABILITY = 0.05
GAMMA_BOUNDS = (-0.524, 0.524)
BETA_BOUNDS = (-5, 5)
MUTATION_RATE = 0.005
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
        gray[..., i] = np.bitwise_xor(
            binary_list[..., i - 1], binary_list[..., i])
    return gray


def gamma_beta(binary_code):
    gamma_list = []
    beta_list = []
    for i in range(len(binary_code)):
        gamma_temp = []
        beta_temp = []
        for j in range(0, OPT_PARAM_VECTOR_SIZE, 2):
            gamma_binary = binary_code[i][j *
                                          BINARY_CODE_SIZE: (j + 1) * BINARY_CODE_SIZE]
            beta_binary = binary_code[i][(
                j + 1) * BINARY_CODE_SIZE: (j + 2) * (BINARY_CODE_SIZE)]
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


def plot_the_plots(x_plot, y_plot, alpha_plot, v_plot, gamma_plot, beta_plot):
    x1 = np.linspace(-16, -4, 1000)
    y1 = 3 * np.ones_like(x1)
    y2 = np.linspace(3, -1, 1000)
    x2 = -4 * np.ones_like(y2)
    x3 = np.linspace(4, 15, 1000)
    y3 = 3 * np.ones_like(x3)
    x4 = 4 * np.ones_like(y2)
    x5 = np.linspace(-4, 4, 1000)
    y5 = -1 * np.ones_like(x5)

    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.plot(x3, y3)
    ax.plot(x4, y2)
    ax.plot(x5, y5)
    ax.plot(x_plot, y_plot)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("TRAJECTORY PATH")

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), gamma_plot)
    plt.grid()
    plt.xlabel("TIME")
    plt.ylabel("GAMMA")
    plt.title("GAMMA VS TIME")

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), beta_plot)
    plt.grid()
    plt.xlabel("TIME")
    plt.ylabel("BETA")
    plt.title("BETA VS TIME")

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), x_plot)
    plt.grid()
    plt.xlabel("TIME")
    plt.ylabel("X")
    plt.title("X VS TIME")

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), y_plot)
    plt.grid()
    plt.xlabel("TIME")
    plt.ylabel("Y")
    plt.title("Y VS TIME")

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), alpha_plot)
    plt.grid()
    plt.xlabel("TIME")
    plt.ylabel("ALPHA")
    plt.title("ALPHA VS TIME")

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), v_plot)
    plt.grid()
    plt.xlabel("TIME")
    plt.ylabel("V")
    plt.title("V VS TIME")
    plt.show()


def interpolate_gamma_beta(pop, gamma, beta):
    t = np.linspace(0, 9, 10)
    time_new = np.linspace(0, 9, 1000)
    interp_gamma = np.zeros((len(gamma), len(time_new)))
    interp_beta = np.zeros((len(beta), len(time_new)))

    for individual in range(len(pop)):
        f_gamma = interp1d(t, gamma[individual],
                           kind='linear', bounds_error=False)
        gamma_new = f_gamma(time_new)
        interp_gamma[individual] = gamma_new

        f_beta = interp1d(t, beta[individual],
                          kind='linear', bounds_error=False)
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


def euler(gamma, beta, ind):
    time_new = np.linspace(0, 9, 1000)
    z_history = np.zeros((1000, 4))
    z = np.array([x0, y0, alpha0, v0])
    z_history[0] = z

    x_plot, y_plot, alpha_plot, v_plot = [], [], [], []
    x_plot.append(x0)
    y_plot.append(y0)
    alpha_plot.append(alpha0)
    v_plot.append(v0)

    flag = True
    for i in range(1, len(time_new)):
        z_dot = ode_system(z, beta[ind][i], gamma[ind][i])
        z += np.array(z_dot) * 0.01
        z_history[i] = z
        x_plot.append(z[0])
        y_plot.append(z[1])
        alpha_plot.append(z[2])
        v_plot.append(z[3])
        if not checking(z_history[i, 0], z_history[i, 1]):
            flag = False

    if flag:
        cost = math.sqrt((abs(z[0] - xf) ** 2 + abs(z[1] - yf)
                         ** 2)) + abs(z[2] - alphaf) + abs(z[3] - vf)
    else:
        cost = K
    return cost, z, x_plot, y_plot, alpha_plot, v_plot


def fitness_function(cost):
    return 1 / (1 + cost)


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate(
        (parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    child2 = np.concatenate(
        (parent2[:crossover_point], parent1[crossover_point:]), axis=0)
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


def main():
    initial_population = np.random.randint(
        2, size=(POPULATION_SIZE, OPT_PARAM_VECTOR_SIZE * BINARY_CODE_SIZE))
    grey = convert_binary_to_gray(initial_population)
    gamma, beta = gamma_beta(grey)
    interpolated_gamma, interpolated_beta = interpolate_gamma_beta(
        initial_population, gamma, beta)

    cost = []
    for individual in range(POPULATION_SIZE):
        c, state, x_plot, y_plot, alpha_plot, v_plot = euler(
            interpolated_gamma, interpolated_beta, individual)
        x_plot.append(state[0])
        y_plot.append(state[1])
        alpha_plot.append(state[2])
        v_plot.append(state[3])

        if c < COST_TOL:
            plot_the_plots(x_plot, y_plot, alpha_plot, v_plot,
                           interpolated_gamma[individual][0], interpolated_beta[individual][0])
            quit()
        cost.append(c)

    fitness_ratios = []
    for i in range(POPULATION_SIZE):
        fitness_ratios.append(fitness_function(cost[i]))

    elites = np.argsort(fitness_ratios)[-2:][::-1]
    min_cost_value = min(cost[elites[0]], cost[elites[1]])
    print(f"Generation 0: J = {min_cost_value}")

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
    start = time.time()
    for gen in range(1, MAX_GEN):
        gamma, beta = gamma_beta(grey)
        interpolated_gamma, interpolated_beta = interpolate_gamma_beta(
            grey, gamma, beta)
        cost = []
        control_var = []
        x_plot, y_plot, alpha_plot, v_plot = [[]], [[]], [[]], [[]]
        for individual in range(POPULATION_SIZE):
            x_plot.append([])
            y_plot.append([])
            alpha_plot.append([])
            v_plot.append([])
            c, state, x_coords, y_coords, alpha_coords, v_coords = euler(
                interpolated_gamma, interpolated_beta, individual)
            x_plot[individual].append(state[0])
            y_plot[individual].append(state[1])
            alpha_plot[individual].append(state[2])
            v_plot[individual].append(state[3])
            cost.append(c)
            control_var.append(state)

            if c < COST_TOL:
                print("Final state values:")
                print("x_f = ", state[0])
                print("y_f = ", state[1])
                print("alpha_f = ", state[2])
                print("v_f = ", state[3])
                plot_the_plots(x_coords, y_coords, alpha_coords, v_coords,
                               interpolated_gamma[gen], interpolated_beta[gen])
                controls_text = []
                for idx in range(len(gamma[individual])):
                    controls_text.append(gamma[individual][idx])
                    controls_text.append(beta[individual][idx])
                arr = np.asarray(controls_text)
                np.savetxt('controls.dat', arr, delimiter='\n')
                quit()

        fitness_ratios = []
        for i in range(POPULATION_SIZE):
            fitness_ratios.append(fitness_function(cost[i]))

        elites = np.argsort(fitness_ratios)[-2:][::-1]
        min_cost_value = min(cost[elites[0]], cost[elites[1]])
        print(f"Generation {gen}: J = {min_cost_value}")

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
