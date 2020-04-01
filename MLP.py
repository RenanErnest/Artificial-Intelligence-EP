import numpy as np
import pandas as pd

''' A 2-layer MLP '''
class MLP:

    def __init__(self,inputNumber,hiddenNumber,outputNumber):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        ''' weights '''
        # a inputNumber+1(bias) x hiddenNumber matrix with random values. It already includes the bias weights
        # weightsInputToHidden[i][j] means the weight between the perceptron i of the hidden layer and the perceptron j of the input layer
        self.weightsInputToHidden = 2 * np.random.random((hiddenNumber,inputNumber+1)) - 1
        # a hiddenNumber+1(bias) x outputNumber matrix with random values. It already includes the bias weights
        # weightsHiddenToOutput[i][j] means the weight between the perceptron i of the output layer and the perceptron j of the hidden layer
        self.weightsHiddenToOutput = 2 * np.random.random((outputNumber,hiddenNumber+1)) - 1

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def feedfoward(self,inputValues):
        # inputValues: a single line of input containg the attributes x1, x2, ..., xn
        
        # adding the bias as an inputValue
        inputValues = np.insert(inputValues,0,1) # adding the value 1 in the index 0

        # input through hidden
        hiddenValues = np.zeros((self.hiddenNumber))
        for i in range(self.hiddenNumber): # for each hiddenNeuron
            summ = 0
            for j in range(len(inputValues)): # each inputNeuron
                summ += inputValues[j]*self.weightsInputToHidden[i][j]
            hiddenValues[i] = self.sigmoid(summ)

        # adding the bias as hiddenValue
        hiddenValues = np.insert(hiddenValues,0,1) # adding the value 1 in the index 0

        # hidden through output
        outputValues = np.zeros((self.outputNumber))
        for i in range(self.outputNumber):
            summ = 0
            for j in range(len(hiddenValues)):
                summ += hiddenValues[j]*self.weightsHiddenToOutput[i][j]
            outputValues[i] = self.sigmoid(summ)

        return outputValues,hiddenValues,inputValues
    
    def backpropagation(self,targetValues,inputValues,learningRate):
        
        (outputValues,hiddenValues,inputValues) = self.feedfoward(inputValues)

        outputErrors = np.zeros((self.outputNumber))
        for i in range(self.outputNumber):
            outputErrors[i] = (targetValues[i]-outputValues[i]) * self.sigmoid_derivative(outputValues[i])

        deltasHiddenToOutput = np.zeros((self.outputNumber,self.hiddenNumber+1))
        for i in range(self.outputNumber):
            for j in range(self.hiddenNumber+1):
                deltasHiddenToOutput[i][j] = learningRate*outputErrors[i]*hiddenValues[j]

        # hidden errors
        hiddenErrors = np.zeros((self.hiddenNumber))
        for i in range(1,self.hiddenNumber+1):
            summ = 0
            for j in range(self.outputNumber):
                summ += outputErrors[j]*self.weightsHiddenToOutput[j][i]
            hiddenErrors[i-1] = self.sigmoid_derivative(hiddenValues[i])*summ

        deltasInputToHidden = np.zeros((self.hiddenNumber,self.inputNumber+1))
        for i in range(self.hiddenNumber):
            for j in range(self.inputNumber+1):
                deltasInputToHidden[i][j] = learningRate*hiddenErrors[i]*inputValues[j]

        # updating the weights
        for i in range(len(self.weightsHiddenToOutput)):
            for j in range(len(self.weightsHiddenToOutput[i])):
                self.weightsHiddenToOutput[i][j] += deltasHiddenToOutput[i][j]
        
        for i in range(len(self.weightsInputToHidden)):
            for j in range(len(self.weightsInputToHidden[i])):
                self.weightsInputToHidden[i][j] += deltasInputToHidden[i][j]

    def train(self, trainSet, epochs=1000, learningRate=1, learningRateMultiplierPerEpoch=1):
        '''
            trainSet: a pandas dataframe with the values for training
        '''
        # data treatment
        inputs = data.drop(data.columns[-1],axis=1).values
        targets = data.drop(data.columns[:-1],axis=1).values
        
        for epoch in range(epochs):
            for inputValues,targetValues in zip(inputs,targets):
                if type(targetValues[0]) == type(np.array((2))):
                    targetValues = targetValues[0]
                self.backpropagation(targetValues,inputValues,learningRate)
                learningRate *= learningRateMultiplierPerEpoch
        
        return self.predict(inputs)
       

    def predict(self,inputs):
        output = []
        for inputValues in inputs:
            (outputValues,hiddenValues,inputValues) = self.feedfoward(inputValues)
            print(outputValues)
            output.append(outputValues)
        return output

    def openFile(self, filePath):
        '''
            filePath: an entire file path including the extension. For example: "Data/problemOR.csv"
        '''
        data = pd.read_csv(filePath, header = None)
        return data

'''letters'''
# mlp = MLP(63,20,7)
# data = mlp.openFile('Data/caracteres-limpo.csv')
# letter_codes = np.array([[1,0,0,0,0,0,0],
#                     [0,1,0,0,0,0,0],
#                     [0,0,1,0,0,0,0],
#                     [0,0,0,1,0,0,0],
#                     [0,0,0,0,1,0,0],
#                     [0,0,0,0,0,1,0],
#                     [0,0,0,0,0,0,1],
#                     [1,0,0,0,0,0,0],
#                     [0,1,0,0,0,0,0],
#                     [0,0,1,0,0,0,0],
#                     [0,0,0,1,0,0,0],
#                     [0,0,0,0,1,0,0],
#                     [0,0,0,0,0,1,0],
#                     [0,0,0,0,0,0,1],
#                     [1,0,0,0,0,0,0],
#                     [0,1,0,0,0,0,0],
#                     [0,0,1,0,0,0,0],
#                     [0,0,0,1,0,0,0],
#                     [0,0,0,0,1,0,0],
#                     [0,0,0,0,0,1,0],
#                     [0,0,0,0,0,0,1],
#                     ])

# data = data.drop(data.columns[-1],axis=1)
# data['new'] = list(letter_codes)
# output = mlp.train(data,100,0.5)
# letters = ['A','B','C','D','E','J','K']
# for outputValues in output:
#     for i in range(len(outputValues)):
#         if round(outputValues[i]) == 1:
#             print(letters[i])
#             break

# print()

# #ruido
# data = mlp.openFile('Data/caracteres-ruido.csv')
# data = data.drop(data.columns[-1],axis=1)
# data['new'] = list(letter_codes)
# # data treatment
# inputs = data.drop(data.columns[-1],axis=1).values
# output = mlp.predict(inputs)
# for outputValues in output:
#     for i in range(len(outputValues)):
#         if round(outputValues[i]) == 1:
#             print(letters[i])
#             break

'''XOR'''
mlp = MLP(2,4,1)
data = mlp.openFile('Data/problemXOR.csv')
output = mlp.train(data,1000,0.5)



