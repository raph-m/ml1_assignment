import numpy as np

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = np.zeros(np.shape(z))

    for a in range(len(g)):
        g[a] = 1/(1+np.exp(-z[a]))

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.

    return g
