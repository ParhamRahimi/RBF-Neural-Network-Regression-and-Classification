import random
import numpy as np

cluster_size = 50
dimension = 2
scale = 1
cluster_count = 5


if __name__ == "__main__":
    with open("train.csv", "w+") as train_csv:
        with open("test.csv", "w+") as test_csv:
            for j in range(cluster_count):
                x = random.random() * 10
                y = random.random() * 10
                a = np.random.normal(size=[cluster_size, dimension], loc=[x, y], scale=scale)
                b = np.random.normal(size=[int(cluster_size * 5), dimension], loc=[x, y], scale=scale)
                for i in a:
                    train_csv.write(str(i[0]) + ',' + str(i[1]) + ',' + str(j) + '\n')
                for i in b:
                    test_csv.write(str(i[0]) + ',' + str(i[1]) + ',' + str(j) + '\n')