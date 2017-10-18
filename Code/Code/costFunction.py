import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = np.zeros((num_labels, m))
    for i in range(len(y)):
        yv[int(y[i])] = 1  # TODO: the int conversion is maybe not the useful
    yv = np.transpose(yv)

    # ================================ TODO ================================
    # In this point calculate the cost of the neural network (feedforward)
    x = np.copy(X)

    for i in range(num_layers - 1):
        print("shape of x")
        print(np.shape(x))

        s = np.shape(Theta[i])
        theta = Theta[i][:, 0:s[1] - 1]
        x = np.dot(x, np.transpose(theta))
        x = x + Theta[i][:, s[1] - 1]
        x = sigmoid(x)

    cost = (yv * np.log(x) - (1 - yv) * np.log(1 - x)) / m
    cost = -np.sum(cost)

    somme = 0

    for i in range(num_layers - 1):
        somme += lambd * np.sum(Theta[i] ** 2) / (2 * m)

    cost += somme

    return cost

    

