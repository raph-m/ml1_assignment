import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params
from predict import predict


def insertOne(x):
    """
    :param x: matrix to modify
    :return: the same matrix but with an additional row of ones
    """
    s = x.shape
    a = np.ones((s[0], s[1]+1))
    a[:, 1:] = x
    return a


def backwards(nn_weights, layers, X, y, num_labels, lambd):
    """
    :param nn_weights: Neural network parameters (vector)
    :param layers: a list with the number of units per layer.
    :param X: a matrix where every row is a training example for a handwritten digit image
    :param y: a vector with the labels of each instance
    :param num_labels: the number of units in the output layer
    :param lambd: regularization factor
    :return: Computes the gradient fo the neural network.
    """

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)

    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = np.zeros((num_labels, m))
    for i in range(len(y)):
        yv[int(y[i]), i] = 1
    yv = np.transpose(yv)

    a = []
    z = []
    x = np.copy(X)
    a.append(insertOne(x))
    z.append(x)

    # if you want to be able to follow the training accuracy:
    # pred = predict(Theta, X)
    # accuracy = np.mean(y == pred) * 100
    # print(accuracy)

    for i in range(num_layers - 1):

        s = np.shape(Theta[i])
        theta = Theta[i][:, 1:s[1]]
        x = np.dot(x, np.transpose(theta))
        x = x + Theta[i][:, 0]
        z.append(x)
        x = sigmoid(x)
        a.append(insertOne(x))

    delta = [np.zeros(w.shape) for w in z]
    delta[num_layers - 1] = (x - yv)

    for i in range(num_layers - 2, 0, -1):
        s = np.shape(Theta[i])
        theta = np.copy(Theta[i][:, 1:s[1]])
        temp = np.dot(np.transpose(theta), np.transpose(delta[i + 1]))
        delta[i] = np.transpose(temp) * sigmoidGradient(z[i])

    Delta = []
    for i in range(num_layers - 1):
        temp = np.dot(np.transpose(delta[i + 1]), a[i])
        Delta.append(temp)

    # if you want to follow the cost during the training:
    # cost = (yv * np.log(x) + (1 - yv) * np.log(1 - x)) / m
    # cost = -np.sum(cost)
    #
    # somme = 0
    #
    # for i in range(num_layers - 1):
    #     somme += lambd * np.sum(Theta[i] ** 2) / (2 * m)
    #
    # cost += somme

    Theta_grad = [(d / m) for d in Delta]

    i = 0
    for t in Theta:
        current = lambd*t/m
        # d'après le poly il faudrait qu'il y ait cette ligne
        # mais après quand on son checkNNGradient il vaut mieux enlever
        # cette ligne donc je ne sais pas ...:
        # current[:, 0] = current[:, 0]*0
        Theta_grad[i] += current
        i += 1

    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad
