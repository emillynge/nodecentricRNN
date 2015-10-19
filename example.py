__author__ = 'emil'
from nodecentricRNN import recipes
import numpy as np
np.random.seed(0)

""" small XOR example """
N = 20
X_train = np.matrix(np.random.randint(0, high=2, size=N), dtype=float).reshape((N, 1))
Y_train = np.matrix(np.logical_xor(X_train[1:, :], X_train[:-1, :]), dtype=float)

network = recipes.SRNN(X_train, Y_train, 3)  # use the recipe Simple Recurrent Neural Network
network.make_train_net()  # build a network using supplied X and Y

print("Input Nodes:\n", network.input_nodes)  # input = X
print("Ground Truth:\n", network.ground_truth_node)  # ground truth = Y

print('Train network')
network.train()  # train theta parameters using X and Y

Y_hat = network.predict(X_train)   # estimate output
print('Train performance')
print('Estimate:\n', np.round(Y_hat).T)
print('Correct:\n', Y_train.T)
print('Difference:\n', Y_train.T - np.round(Y_hat).T)


X_test = np.matrix([1, 1, 1, 0, 1]).reshape((5, 1))  # test data
Y_test = np.matrix([0, 0, 1, 1]).reshape((4, 1))  # test data

Y_hat = network.predict(X_test)   # estimate output

print('Test performance')
print('Estimate:\n', np.round(Y_hat))
print('Correct:\n', Y_test)
print('Difference:\n', Y_test.T - np.round(Y_hat).T)

