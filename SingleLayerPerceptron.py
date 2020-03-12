import pandas as pd
import random
from matplotlib import pyplot as plt

def perceptron(inputs,weights,threshold):
    summ = 0
    for i in range(len(weights)):
        summ += inputs[i]*weights[i]
    if summ > threshold:
        return 1
    return 0

def updateWeights(inputs,weights,expected,output,learning_rate):
    new_weights = [0] * len(weights)
    for i in range(len(weights)):
        # new_weights[i] = weights[i] + learning_rate*expected*inputs[i] # Fausett
        new_weights[i] = weights[i] + (expected-output)*inputs[i] #another aproach
    return new_weights


def plot(data,weights):
    x1 = [i / 10 for i in range(-15, 15)]
    x2 = [(-1*weights[0] - weights[1]*x1_v) / weights[2] for x1_v in x1]
    plt.scatter(data['x1'], data['x2'])
    plt.plot(x1, x2)
    plt.show()

# input
data = pd.read_csv('Data/problemAND.csv', header=None, names=['x1','x2','expected'])
bias = [1] * len(data)
data.insert(0,'bias',bias,True)
print(data)

# weights
weights = [random.random() for input in range(len(data.columns)-1)]
# weights = [0.1 for input in range(len(data.columns)-1)]

# training
plot(data,weights)
threshold = 0
learning_rate = 0.5
for season in range(100):
    pre_weights = weights.copy()
    for inputs in data.values:
        expected = inputs[-1]
        output = perceptron(inputs,weights,threshold)
        if output != expected:
            weights = updateWeights(inputs,weights,expected,output,learning_rate)
            plot(data,weights)

    if weights == pre_weights:
        break

# test
print('\nTeste:\nObtido\tEsperado')
for inputs in data.values:
    expected = inputs[-1]
    output = perceptron(inputs,weights,threshold)
    print(str(output)+'\t\t'+str(expected))


