import sys
import numpy as np

x_train, y_train, x_test, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
x_train_lines = np.loadtxt(x_train, delimiter=",")
x_test_lines = np.loadtxt(x_test, delimiter=",")
y_train_lines = np.loadtxt(y_train)


# A function that returns the class with the most instances
def maximum(class_0, class_1, class_2):
    if (class_0 >= class_1) and (class_0 >= class_2):
        return 0
    elif (class_1 >= class_0) and (class_1 >= class_2):
        return 1
    return 2


def min_max_normalization(vector_to_normalize, train_vector):
    np_vector = np.array(train_vector)
    normalized_vector = vector_to_normalize.copy()

    for i, line in enumerate(vector_to_normalize):
        for j, value in enumerate(line):
            min_value = min(np_vector[:, j])
            max_value = max(np_vector[:, j])
            new_value = (value - min_value) / (max_value - min_value)
            normalized_vector[i][j] = new_value
    return normalized_vector


# KNN Algorithm
def knn():
    k = 9
    normalized_x_train = min_max_normalization(x_train_lines, x_train_lines)
    normalized_x_test = min_max_normalization(x_test_lines, x_train_lines)
    labels = []
    for test in normalized_x_test:
        dists_list = list()
        index = 0
        for train in normalized_x_train:
            # euclidean distance between test point and train point
            dist = np.linalg.norm(test - train)
            dists_list.insert(len(dists_list), (index, dist))
            index += 1

        # sort by distances
        dists_list.sort(key=lambda t: t[1])
        # counters for the number of instances of each class
        class_0, class_1, class_2 = 0, 0, 0
        # k nearest neighbors
        for i in range(k):
            int_class = y_train_lines[dists_list[i][0]]
            if int_class == 0:
                class_0 += 1
            elif int_class == 1:
                class_1 += 1
            else:
                class_2 += 1

        # the class with the most instances
        label = maximum(class_0, class_1, class_2)
        labels.insert(len(labels), label)
    return labels


def z_score_normalization(vector_to_normalize, train_vector):
    np_vector = np.array(train_vector)
    normalized_vector = vector_to_normalize.copy()

    for i, line in enumerate(vector_to_normalize):
        for j, value in enumerate(line):
            mean_value = np.mean(np_vector[:, j])
            stand_dev_value = np.std(np_vector[:, j])
            new_value = (value - mean_value) / stand_dev_value
            normalized_vector[i][j] = new_value

    return normalized_vector


# prediction with weights
def predict(weights):
    normalized_x_test = z_score_normalization(x_test_lines, x_train_lines)
    normalized_x_test = np.insert(normalized_x_test, 0, np.ones(normalized_x_test.shape[0]), axis=1)
    labels = []
    for test in normalized_x_test:
        y_hat = np.argmax(np.dot(weights, test))
        labels.insert(len(labels), y_hat)
    return labels


# estimate Perceptron weights
def train_perceptron():
    normalized_x_train = z_score_normalization(x_train_lines, x_train_lines)
    normalized_x_train = np.insert(normalized_x_train, 0, np.ones(normalized_x_train.shape[0]), axis=1)
    epochs = 22
    weights = [[0] * 6 for i in range(3)]
    for i in range(epochs):
        for x, y in zip(normalized_x_train, y_train_lines):
            y_hat = np.argmax(np.dot(weights, x))
            if y_hat != y:
                weights[int(y)] += x * 0.1
                weights[y_hat] -= x * 0.1
    return weights


# Perceptron Algorithm
def perceptron():
    weights = train_perceptron()
    return predict(weights)


# estimate Passive Aggressive weights
def train_passive_aggressive():
    normalized_x_train = z_score_normalization(x_train_lines, x_train_lines)
    normalized_x_train = np.insert(normalized_x_train, 0, np.ones(normalized_x_train.shape[0]), axis=1)
    epochs = 5
    weights = [[0] * 6 for i in range(3)]
    for i in range(epochs):
        for x, y in zip(normalized_x_train, y_train_lines):
            y_hat = np.argmax(np.dot(weights, x))
            if y_hat != y:
                tau = max(0, 1 - np.dot(weights[int(y)], x) + np.dot(weights[int(y_hat)], x))\
                      / (2 * pow(np.linalg.norm(x), 2))
                weights[int(y)] += x * tau
                weights[int(y_hat)] -= x * tau
    return weights


# Passive Aggressive Algorithm
def passive_aggressive():
    weights = train_passive_aggressive()
    return predict(weights)


# estimate SVM weights
def train_svm():
    normalized_x_train = z_score_normalization(x_train_lines, x_train_lines)
    normalized_x_train = np.insert(normalized_x_train, 0, np.ones(normalized_x_train.shape[0]), axis=1)
    weights = np.array([[0] * 6 for i in range(3)])
    epochs = 10
    for i in range(epochs):
        for x, y in zip(normalized_x_train, y_train_lines):
            dot_result = np.array(np.dot(weights, x))
            dot_result[int(y)] = float('-inf')
            y_hat = np.argmax(dot_result)
            tau = max(0, 1 - np.dot(weights[int(y)], x) + np.dot(weights[int(y_hat)], x))
            eta = 0.1
            lamda = 0.0002
            if tau > 0:
                weights = (1 - lamda * eta) * weights
                weights[int(y)] = weights[int(y)] + eta * x
                weights[int(y_hat)] = weights[int(y_hat)] - eta * x
            else:
                weights = (1 - lamda * eta) * weights
    return weights


# SVM Algorithm
def svm():
    weights = train_svm()
    return predict(weights)


knn = knn()
perceptron = perceptron()
passive_aggressive = passive_aggressive()
svm = svm()

size = len(x_test_lines)

out_file = open(output_file, "w")
for n in range(size):
    out_file.write(f"knn: {knn[n]}, perceptron: {perceptron[n]}, svm: {svm[n]}, pa: {passive_aggressive[n]}\n")
out_file.close()


