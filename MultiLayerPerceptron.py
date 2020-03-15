import pandas as pd
import random
from matplotlib import pyplot as plt
import numpy as np


class TwoLayerMLP:

    '''
        arguments:
        hidden_amount - represents the number of neurons in the hidden layer
        inputs - represent a dataframe containing the inputs and the target value on each last column
        epoch - represents the number of iterations in each training
        learning_rate - represents the amount of update on each error treatment
    '''
    def __init__(self, hidden_amount, inputs, epoch = 100, learning_rate = 0.2):
        # inserting the bias as an input for every case
        self.inputs = inputs
        bias = [1] * len(inputs) # initializng a list with bias 1 for every input(cases)
        self.inputs.insert(0, 'bias', bias, True) # inserting the bias into the dataframe to be an extra input for each case
        # after that our matrix has the columns as each input including the bias on the first column
        # and the rows representing each case

        # defining some class arguments, kind of global variables inside the class
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_amount = hidden_amount

        # initializing weights with random values
        # Entry amount minus one(because of the target value in the last column) random weights.
        # so we'll have all the weights including the bias's weight from every input node to every hidden node
        # the rows represent each destination node and the columns represent each input
        # matriz de pesos da camada de entrada para a camada escondida
        self.weightsInputToHidden = [[random.random() for m in range(len(inputs.columns)-1)] for n in range(hidden_amount)]

        # considering only one neuron in the output layer
        # it represents the weights for each output from every hidden layer perceptron including the weight for the bias
        # so we have the same amount of hidden layer perceptron outputs that is equal to the number of neurons in the hidden layer
        # we also already define a weight for the bias so our weights array contain the number of perceptrons
        # in the hidden layer plus one elements, the last one represents the weight for the bias
        # matriz de pesos da camada escondida para a camada de saida
        self.weightsHiddenToOutput = [random.random() for n in range(self.hidden_amount+1)]


    '''
    A function that given the inputs and weights calculate the output of a perceptron k
    The step function adopted was the sigmoid function
    '''
    def perceptron(self, k, inputs):
        summ = 0
        for i in range(len(self.weightsInputToHidden[k])):
            summ += inputs[i] * self.weightsInputToHidden[k][i]
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

    '''
        arguments:
        case_inputs - represents the inputs for each case, so this is only an array whose each element is an input
    '''
    def feedfoward(self, case_inputs):
        # generating the hidden outputs
        # the bias is already inserted in the hidden outputs on initializing
        hiddenOutputs = [1]

        # for each perceptron in the hidden layer we do a classical linear combination of each input times each weight
        # of the inputs for that perceptron
        for hidden_perceptron_weights in self.weightsInputToHidden:
            # linear combination of each input with each weight corresponding to that input for this perceptron
            summ = 0
            for input_index in range(len(case_inputs)):
                summ += case_inputs[input_index] * hidden_perceptron_weights[input_index]
            # appending the return of the sigmoid function with this summatory for our hidden outputs
            # at the end of this iteration we'll have an array containg the output of each perceptron of the hidden layer
            hiddenOutputs.append(self.sigmoid(summ))

        # at this point we'll do kind of the same thing but from the hidden layer to the output layer
        # we are assuming that the output layer contains only one node so we need only one input for that node
        summ = 0
        for hidden_output_index in range(len(hiddenOutputs)):
            summ += hiddenOutputs[hidden_output_index]*self.weightsHiddenToOutput[hidden_output_index]

        # applying the step function on our linear combination
        final_output = self.sigmoid(summ)

        return final_output, hiddenOutputs



    def train(self):
        for _ in range(self.epoch):
            for case_inputs in self.inputs.values: # for each case

                # the target value for each case is in the last column
                target = case_inputs[-1]
                # passing all the inputs without the target value to the feedfoward algorithm
                # and capturing the predicted value and the outputs of the hidden layer including already the bias
                predicted, hiddenOutputs = self.feedfoward(case_inputs[:-1])

                # turning the predicted value to binary only for tests with the XOR case
                predicted = 1 if predicted > 0 else -1

                # Backpropagation
                final_error = target - predicted
                if final_error != 0:
                    # calculating the errors of each perceptron in the hidden layer
                    hidden_errors = []
                
                    for i in range(len(hiddenOutputs)):
                        hidden_errors.append(self.weightsHiddenToOutput[i] * final_error * self.sigmoid_derivative(hiddenOutputs[i]))
                    
                    # calculating the errors of each input weights
                    errors = []
                    # for each neuron in the input layer
                    for k in range(self.hidden_amount):
                        error = 0
                        # for each neuron in the hidden layer
                        for j in range(len(hiddenOutputs)):
                            error += hidden_errors[j] * self.weightsInputToHidden[k][j]
                        errors.append(error * self.sigmoid_derivative(case_inputs[k]))
                
                    # Updating weights considering learning rate and error back propagation
                    # updating weights of the input layer
                    for i in range(self.hidden_amount):
                        for j in range(len(self.weightsHiddenToOutput)):
                            self.weightsInputToHidden[i][j] += self.learning_rate*errors[i]*case_inputs[i]
                    # updating weights of the hidden layer
                    for j in range(len(self.weightsHiddenToOutput)):
                        self.weightsHiddenToOutput[j] += self.learning_rate*hidden_errors[j]*hiddenOutputs[j]
                    
                    
    def predict(self):
        for case_inputs in self.inputs.values:
            # passing all the inputs without the target value to the feedfoward algorithm
            # and capturing the predicted value and the outputs of the hidden layer including already the bias
            predicted, hiddenOutputs = self.feedfoward(case_inputs[:-1])

            # turning the predicted value to binary only for tests with the XOR case
            binary_predicted = 1 if predicted > 0 else -1
            print('target:',case_inputs[-1],'predicted answer:',binary_predicted,'predicted value:',predicted)

                    

                
        






data = pd.read_csv('Data/problemXOR.csv', header=None, names=['x1','x2','target'])
for i in range(len(data['target'])):
    if data['target'][i] == 0:
        data['target'][i] = -1

model = TwoLayerMLP(2,data,10,1)
model.train()
model.predict()