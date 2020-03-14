import pandas as pd
import random
from matplotlib import pyplot as plt
import numpy as np


class TwoLayerMLP:
    
    def __init__(self, hidden_amount, inputs, epoch = 100, learning_rate = 0.2):
        # inputs treatment
        self.inputs = inputs
        bias = [1] * len(inputs) # initializng a list with bias 1 for every input(cases)
        self.inputs.insert(0, 'bias', bias, True) # inserting the bias into the dataframe to be an extra input for each case
        self.epoch = epoch # época
        self.learning_rate = learning_rate # ritmo de aprendizado
        
        # initializing weights with random values
        ''' 
        Entry amount minus one(because of the expected value in the last column) random weights.
        '''
        # matriz de pesos da camada de entrada para a camada escondida
        self.weights = [[random.random() for m in range(len(inputs.columns)-1)] for n in range(hidden_amount)] 
        print(self.weights) 
        self.hidden_amount = hidden_amount # número de perceptrons na camada escondida
        # matriz de pesos da camada escondida para a camada de saida
        self.hidden_weights = [random.random() for n in range(self.hidden_amount+1)]
        print(self.hidden_weights)


    '''
    A function that given the inputs and weights calculate the output of a perceptron k
    The step function adopted was the sigmoid function
    '''
    def perceptron(self, k, inputs):
        summ = 0
        for i in range(len(self.weights[k])):
            summ += inputs[i] * self.weights[k][i]
        return self.sigmoid(summ)


    def sigmoid(self, z):
        '''
        Sigmoid function
        z can be an numpy array or scalar
        '''
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def sigmoid_derivative(self, z):
        '''
        Derivative for Sigmoid function
        z can be an numpy array or scalar
        '''
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result


    def train(self):
        for epochx in range(self.epoch):
            print("Epoca: " +str(epochx))
            hidden_outputs = [0] * self.hidden_amount
            hidden_outputs.append(1)
            final_output = 0
            for inputs in self.inputs.values:
                print("for input")
                expected = inputs[-1]
                print(expected) #-1,-1, -> expected -1 (0)
                # Feedfoward
                for k in range(self.hidden_amount):
                    hidden_outputs[k] = self.perceptron(k,inputs)

                # adicionando o bias nas entradas para a camada de saida
                print(hidden_outputs)
                final_summ = 0
                print(self.hidden_weights)
                for i in range(len(self.hidden_weights)):
                    final_summ += hidden_outputs[i] * self.hidden_weights[i]

                final_output += final_summ #1 = bias
            output = self.sigmoid_derivative(final_output)
                    
    def predict(self):
        hidden_outputs = [0] * self.hidden_amount
        for inputs in self.inputs.values:
            # Feedfoward
            for k in range(self.hidden_amount):
                hidden_outputs[k] = self.perceptron(k,inputs)

            # adicionando o bias nas entradas para a camada de saida
            hidden_outputs.append(1)
            
            final_summ = 0
            for i in range(len(self.hidden_weights)):
                final_summ += hidden_outputs[i] * self.hidden_weights[i]
            
            final_output = self.sigmoid(final_summ)
            answer = 0
            if final_output > 0:
                answer = 1
            else:
                answer = -1

            print(inputs[-1],answer,'sigmoid return:',final_output)

                    

                
        






data = pd.read_csv('Data/problemXOR.csv', header=None, names=['x1','x2','expected'])
for i in range(len(data['expected'])):
    if data['expected'][i] == 0:
        data['expected'][i] = -1

model = TwoLayerMLP(2,data,1)
model.train()
model.predict()