import numpy as np
from sigmoid import sigmoid


def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    p = np.zeros((1, m))

    x = np.copy(X)

    for i in range(num_layers - 1):
        s = np.shape(Theta[i])
        theta = Theta[i][:, 1:s[1]]
        x = np.dot(x, np.transpose(theta))
        x = x + Theta[i][:, 0]
        x = sigmoid(x)

    print("x")
    print(x)
    print("np.argmax(x, axis=0)")
    prediction = np.argmax(x, axis=1)

    for i in range(m):
        p[0, i] = prediction[i]
    
    return p

