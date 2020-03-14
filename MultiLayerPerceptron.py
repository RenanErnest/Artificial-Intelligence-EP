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
        self.epoch = epoch
        self.learning_rate = learning_rate
        
        # initializing weights with random values
        ''' 
        Entry amount minus one(because of the expected value in the last column) random weights.
        '''
        # matriz de pesos da camada de entrada para a camada escondida
        self.weights = [[random.random() for m in range(len(inputs.columns)-1)] for n in range(hidden_amount)]  

        self.hidden_amount = hidden_amount
        # matriz de pesos da camada escondida para a camada de saida
        self.weight_ho = [random.random() for n in range(self.hidden_amount+1)]


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
        for _ in range(self.epoch):
            hidden_outputs = [0] * self.hidden_amount
            hidden_outputs.append(1)
            for inputs in self.inputs.values: # for each case
                expected = inputs[-1]

                # Feedfoward
                # from input to hidden
                for k in range(self.hidden_amount):
                    hidden_outputs[k] = self.perceptron(k,inputs)
                
                # from hidden to output
                final_summ = 0
                for i in range(len(self.weight_ho)-1):
                    final_summ += hidden_outputs[i] * self.weight_ho[i]
                # final_summ += 1 * self.weight_ho[-1] # adding bias
                
                # step function
                final_output = 0
                if final_summ > 0:
                    final_output = 1
                else:
                    final_output = -1

                # Backpropagation
                final_error = expected - final_output
                if final_error != 0:
                    # calculating the errors of each perceptron in the hidden layer
                    hidden_errors = []
                
                    for i in range(len(hidden_outputs)):
                        hidden_errors.append(self.weight_ho[i] * final_error * self.sigmoid_derivative(hidden_outputs[i]))
                    
                    # calculating the errors of each input weights
                    errors = []
                    # for each neuron in the input layer
                    for k in range(self.hidden_amount):
                        error = 0
                        # for each neuron in the hidden layer
                        for j in range(len(hidden_outputs)):
                            error += hidden_errors[j] * self.weights[k][j]
                        errors.append(error * self.sigmoid_derivative(inputs[k]))
                
                    # Updating weights considering learning rate and error back propagation
                    # updating weights of the input layer
                    for i in range(self.hidden_amount):
                        for j in range(len(self.weight_ho)):
                            self.weights[i][j] += self.learning_rate*errors[i]*inputs[i]
                    # updating weights of the hidden layer
                    for j in range(len(self.weight_ho)):
                        self.weight_ho[j] += self.learning_rate*hidden_errors[j]*hidden_outputs[j]
                    
                    
    def predict(self):
        hidden_outputs = [0] * self.hidden_amount
        for inputs in self.inputs.values:
            # Feedfoward
            for k in range(self.hidden_amount):
                hidden_outputs[k] = self.perceptron(k,inputs)

            # adicionando o bias nas entradas para a camada de saida
            hidden_outputs.append(1)
            
            final_summ = 0
            for i in range(len(self.weight_ho)):
                final_summ += hidden_outputs[i] * self.weight_ho[i]
            
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

model = TwoLayerMLP(2,data,10,1)
model.train()
model.predict()