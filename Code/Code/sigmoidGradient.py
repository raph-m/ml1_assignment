import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):

    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z
    s = sigmoid(z)
    g = s*(1-s)
    # =========================== TODO ==================================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z.
    
    return g




