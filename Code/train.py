import numpy as np
from backwards import backwards
from roll_params import roll_params
from randInitializeWeights import randInitializeWeights
from unroll_params import unroll_params
from read_dataset import read_dataset
from predict import predict


def train_step(nn_weights, layers, X, y, num_labels, lambd, learning_rate):
    grad = backwards(nn_weights, layers, X, y, num_labels, lambd)
    nn_weights = nn_weights - learning_rate*grad


def train(nn_weights, layers, X, y, num_labels, lambd, learning_rate, epochs):

    for i in range(epochs):
        print("epoch number "+str(i))
        train_step(nn_weights, layers, X, y, num_labels, lambd, learning_rate)

size_training = 60000     # number of samples retained for training
size_test     = 10000     # number of samples retained for testing
images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)

input_layer_size   = 784        # 28x28 Input Images of Digits
num_labels         = 10         # 10 labels, from 0 to 9 (one label for each digit)


layers = [input_layer_size, 15, num_labels]
Theta = randInitializeWeights(layers)

# Unroll parameters
nn_weights = unroll_params(Theta)

epochs = 5
lambd = 3.0
learning_rate = 0.1

train(nn_weights, layers, images_test, labels_training, num_labels, lambd, learning_rate, epochs)

pred = predict(Theta, images_test)
print("accuracy 2 = "+str(np.mean(pred==labels_test)*100))

