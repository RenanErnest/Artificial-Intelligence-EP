import pandas as pd

class SingleLayerPerceptron:

    def perceptron(self,inputs,weights,threshold):
        summ = 0
        for i in range(len(inputs)):
            summ += inputs[i]*weights[i]
        if summ > threshold:
            return 1
        return 0

    def updateWeights(self,inputs,weights,desired,learning_rate):
        new_weights = [0] * len(weights)
        for i in range(len(weights)):
            new_weights[i] = weights[i] + learning_rate*desired*inputs[i] # Fausett
            # new_weights[i] = weights[i] + (desired-output)*inputs[i] #another aproach
        return new_weights

    data = pd.read_csv('Data/problemAND.csv')
    print(data.head())

