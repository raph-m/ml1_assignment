import numpy as np


def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    size = np.shape(z)
    g = np.zeros(size)

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.

    if size[1]:
        for a in range(size[0]):
            for b in range(size[1]):
                g[a,b] = 1/(1+np.exp(-z[a,b]))
    else:
        for a in range(len(g)):
            g[a] = 1/(1+np.exp(-z[a]))

    return g