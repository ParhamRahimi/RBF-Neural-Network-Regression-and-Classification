#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

train = []
test = []
number_of_groups = 10
dimension = 2
classification = True
# number_of_groups = 10
# dimension = 3
# classification = False

IND_SIZE = number_of_groups * dimension + number_of_groups
MIN_VALUE = -1
MAX_VALUE = 1
MIN_STRATEGY = 0.01
MAX_STRATEGY = 0.5

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")


# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


# def e(ind):
#     error


def accuracy(ind):
    if classification:
        G = []
        W = []
        V = np.array(ind[0:-number_of_groups]).reshape([number_of_groups, dimension])
        for point in test:
            g_row = []
            for group_num in range(number_of_groups):
                difference = np.array(point)[:-1] - V[group_num]
                pow = -ind[number_of_groups * dimension + group_num] * (np.matmul(np.transpose(difference), difference))
                element = np.math.e ** pow
                g_row.append(element)
            G.append(g_row)
        G = np.array(G)
        W = np.array(W)
        Gt = np.transpose(G)
        a = np.array(test)[:, -1]
        a = list(map(int, a))
        a = np.array(a)
        y = np.zeros((len(test), number_of_groups))
        y[np.arange(len(test)), a] = 1
        try:
            W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Gt, G)), Gt), y)
        except:
            W = np.random.ranf((number_of_groups, number_of_groups))
        y_hat = np.matmul(G, W)
        a = np.zeros((len(test), dimension + 1))
        for i in range(len(test)):
            if test[i][-1] != np.argmax(y_hat, axis=1)[i]:
                a[i] = test[i]
        green = np.full(len(test), 'green')
        red = np.full(len(test), 'red')
        test_all = test[:, -1] == np.argmax(y_hat, axis=1)
        colors = np.where(test_all, green, red)
        plot_data(test, a, V, colors, ind[-number_of_groups:])
        return np.sum(test_all)
    else:
        G = []
        W = []
        V = np.array(ind[0:-number_of_groups]).reshape([number_of_groups, dimension])
        for point in test:
            g_row = []
            for group_num in range(number_of_groups):
                difference = np.array(point)[:-1] - V[group_num]
                pow = -ind[number_of_groups * dimension + group_num] * (np.matmul(np.transpose(difference), difference))
                element = np.math.e ** pow
                g_row.append(element)
            G.append(g_row)
        G = np.array(G)
        W = np.array(W)
        Gt = np.transpose(G)
        y = np.array(test)[:, -1]
        W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Gt, G)), Gt), y)
        y_hat = np.matmul(G, W)
        a = np.zeros((len(test), 2))
        for i in range(len(test)):
            a[i][0] = i
            a[i][1] = y_hat[i]
        b = np.zeros((len(test), 2))
        for i in range(len(test)):
            b[i][0] = i
            b[i][1] = test[i][-1]
        scatter_data(b, a)
        return y_hat


def error(ind):
    if classification:
        G = []
        W = []
        V = np.array(ind[0:-number_of_groups]).reshape([number_of_groups, dimension])
        for point in train:
            g_row = []
            for group_num in range(number_of_groups):
                difference = np.array(point)[:-1] - V[group_num]
                pow = -ind[number_of_groups * dimension + group_num] * (np.matmul(np.transpose(difference), difference))
                element = np.math.e ** pow
                g_row.append(element)
            G.append(g_row)
        G = np.array(G)
        W = np.array(W)
        Gt = np.transpose(G)
        y = np.array(train)[:, -1]

        a = list(map(int, y))
        a = np.array(a)
        wy = np.zeros((len(train), number_of_groups))
        wy[np.arange(len(train)), a] = 1

        W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Gt, G)), Gt), wy)
        y_hat = np.matmul(G, W)
        sub = np.subtract(np.argmax(y_hat, axis=1), y)
        return 0.5 * np.matmul(np.transpose(sub), sub),
    else:
        G = []
        W = []
        V = np.array(ind[0:-number_of_groups]).reshape([number_of_groups, dimension])
        for point in train:
            g_row = []
            for group_num in range(number_of_groups):
                difference = np.array(point)[:-1] - V[group_num]
                pow = -ind[number_of_groups * dimension + group_num] * (np.matmul(np.transpose(difference), difference))
                element = np.math.e ** pow
                g_row.append(element)
            G.append(g_row)
        G = np.array(G)
        W = np.array(W)
        Gt = np.transpose(G)
        y = np.array(train)[:, -1]

        W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(Gt, G)), Gt), y)
        y_hat = np.matmul(G, W)
        sub = np.subtract(y_hat, y)
        return 0.5 * np.matmul(np.transpose(sub), sub),


toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", error)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def main():
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=20, stats=stats, halloffame=hof)

    with open("ind.csv", "w+") as ind_csv:
        for i in range(number_of_groups * dimension + number_of_groups):
            ind_csv.write(str(pop[0][i]) + ',')

    print("Accuracy= ", (accuracy(ind=pop[0]) / len(test)))

    return pop, logbook, hof


def plot_data(data, data2, data3, colors=None, radius=None):




    # x_min, x_max = data[:, 0].min(), data[:, 0].max()
    # y_min, y_max = data[:, 1].min(), data[:, 1].max()
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # x_range = np.arange(x_min, x_max, 0.1)
    # y_range = np.arange(y_min, y_max, 0.1)
    # xx, yy = np.meshgrid(x_range, y_range)
    # cmap = plt.get_cmap('Paired')
    # zz = np.zeros(xx.shape)
    #
    # for i in range(zz.shape[0]):
    #     for j in range(zz.shape[1]):
    #         x_vector = np.array([xx[i][j], yy[i][j]])
    #         network_answer = my_rbf.apply_network(x_vector)
    #         cls = np.argmax(network_answer)
    #         zz[i][j] = int(cls)
    #
    # plt.pcolormesh(xx, yy, zz, cmap=cmap)





    if colors is None:
        plt.plot(column(data, 0), column(data, 1), 'go', column(data2, 0), column(data2, 1), 'ro', column(data3, 0),
                 column(data3, 1), 'bo')
        plt.yscale('linear')
        plt.xscale('linear')
        plt.grid(True)
        plt.show()
    else:
        plt.clf()
        ax = plt.gca()
        ax.cla()
        plt.scatter(data[:, 0], data[:, 1], color=colors)
        plt.scatter(data[:, 0], data[:, 1], color=colors)
        plt.scatter(data3[:, 0], data3[:, 1], color='blue', s=5)
        for idx in range(len(radius)):
            circle = plt.Circle(data3[idx], radius[idx], color='blue', fill=False)
            ax.add_artist(circle)
        plt.show()


def plot_data2(data, data2, data3, data4):
    plt.plot(column(data, 0), column(data, 1), 'go', column(data2, 0), column(data2, 1), 'ro', column(data3, 0),
             column(data3, 1), 'bo')
    plt.yscale('linear')
    plt.xscale('linear')
    plt.grid(True)
    plt.show()


def scatter_data(data, data2):
    plt.scatter(column(data, 0), column(data, 1), color='blue', s=5)
    plt.yscale('linear')
    plt.xscale('linear')
    plt.grid(True)
    plt.pause(5)
    plt.scatter(column(data2, 0), column(data2, 1), color='red', s=5)
    plt.show()


def column(matrix, i):
    return [row[i] for row in matrix]


def make_it_to_float(my_list):
    # return list(map(float, my_list))
    return np.array(my_list).astype(np.float)

def initial():
    if classification:
        first = True
        with open("5clstrain1500.csv", "r") as my_csv:
            data = []
            reader = csv.reader(my_csv)
            # data = list(csv.reader(my_csv))
            for row in reader:
                if first:
                    first = False
                    continue
                data.append(np.array(row).astype(np.float))
        # data = list(map(make_it_to_float, data))
        data = np.array(data)
        data[:, :-1] /= np.max(data[:, :-1])
        # data[:, :-1] /= np.maximum(np.argmax(np.array(data)[:, :-1]), np.absolute(np.argmin(np.argmax(np.array(data)[:, :-1]))))
        global train
        train = data
        with open("5clstest5000.csv", "r") as my_csv:
            # data2 = list(csv.reader(my_csv))
            data2 = []
            reader = csv.reader(my_csv)
            first = True
            for row in reader:
                if first:
                    first = False
                    continue
                data2.append(np.array(row).astype(np.float))
        data2 = list(map(make_it_to_float, data2))
        data2 = np.array(data2)
        data2[:, :-1] /= np.max(data2[:, :-1])
        plt.scatter(data2[0:999, 0], data2[0:999, 1], color='purple')
        plt.scatter(data2[1000:1999, 0], data2[1000:1999, 1], color='cyan')
        plt.scatter(data2[2000:2999, 0], data2[2000:2999, 1], color='blue')
        plt.scatter(data2[3000:3999, 0], data2[3000:3999, 1], color='orange')
        plt.scatter(data2[4000:, 0], data2[4000:, 1], color='yellow')
        # plt.scatter(data2[0:249, 0], data2[0:249, 1], color='red')
        # plt.scatter(data2[250:499, 0], data2[250:499, 1], color='blue')
        # plt.scatter(data2[500:749, 0], data2[500:749, 1], color='yellow')
        # plt.scatter(data2[750:999, 0], data2[750:999, 1], color='green')
        # plt.scatter(data2[1000:, 0], data2[1000:, 1], color='purple')
        plt.show()
        # plt.clf()
        # data2[:, :-1] /= np.maximum(np.argmax(np.array(data)[:, :-1]), np.absolute(np.argmin(np.argmax(np.array(data)[:, :-1]))))
        global test
        test = data2
    else:
        with open("reg.csv", "r") as my_csv:
        #     data = list(csv.reader(my_csv))
        # data = list(map(make_it_to_float, data))
            reader = csv.reader(my_csv)
            first = True
            data = []
            for row in reader:
                if first:
                    first = False
                    continue
                data.append(np.array(row).astype(np.float))
        data = np.array(data)
        data[:, :-1] /= np.max(data[:, :-1])

        mask = np.random.choice([False, True], len(data), p=[0.4, 0.6])

        global test
        test = data
        print(len(test))

        global train
        train = data[mask]
        print(len(train))


if __name__ == "__main__":
    initial()
    main()
