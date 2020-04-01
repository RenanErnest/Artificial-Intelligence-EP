import numpy as np
from numpy import exp, array, random, dot
import pandas as pd

''' A 2-layer MLP '''
class MLP:

    def __init__(self,inputNumber,hiddenNumber,outputNumber, filename, epochs, learningRate, targets=None):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber
        self.learningRate = learningRate
        self.biasWeightsToHidden = np.zeros((hiddenNumber))
        self.biasWeightsToOutput = np.zeros((outputNumber))
        ''' weights '''
        # a inputNumber x hiddenNumber matrix with random values.
        # weightsInputToHidden[i][j] means the weight between the perceptron i of the input layer and the perceptron j of the hidden layer
        self.weightsInputToHidden = 2 * np.random.random((inputNumber,hiddenNumber)) - 1
        #self.weightsInputToHidden = 2 * np.random.random((self.hiddenNumber,self.inputNumber)) - 1
        # a hiddenNumber x outputNumber matrix with random values
        # weightsHiddenToOutput[i][j] means the weight between the perceptron i of the hidden layer and the perceptron j of the output layer
        self.weightsHiddenToOutput = 2 * np.random.random((hiddenNumber,outputNumber)) - 1
        #self.weightsHiddenToOutput = 2 * np.random.random((self.outputNumber,self.hiddenNumber)) - 1

        # errors
        self.hiddenErrors = np.zeros((hiddenNumber))
        self.outputErrors = np.zeros((outputNumber))
        self.file = pd.read_csv("Data/" + filename + ".csv", header = None) 
        self.np_file = self.file.to_numpy()
        #só com o array unidimensional da saída
        # blz, se vc comentar essa parte funciona ?, só  cno tem que fazer isso aqui ó
        # to testando uma parada pra poder mandar os target se ja tiver, caso n tenha usa o do np_file
        if not targets:
           self.expectedOutput = self.np_file[:, -1].T # set the expectedOutput array to the last column of the dataframe
        else: 
           self.expectedOutput = targets
        
        # data treatment
        #o escalar esta sendo expectedOutput, n po, acho que é com o indice, na funcao que da o erro e sem o indice
        # new_targets = []
        # for i in range(len(self.expectedOutput)):
        #     if type(self.expectedOutput[i]) == type(np.zeros(1,dtype=np.int64)[0]):
        #         new_targets.append(list(self.expectedOutput[i])) #o print do numpy ta bugado tbm, tem vezes que printa 2x do nada
        # self.expectedOutput = np.array(new_targets)

        #print("Saídas esperadas:\n")
        #print(self.expectedOutput)
        self.inputs = np.delete(self.np_file, -1, axis=1) # set the input array
        self.train(epochs)

    def feedfoward_backpropagation(self,inputValues, expectedValues, learningRate):
        # inputValues: a single line of input containg the attributes x1, x2, ..., xn
        
        # input through hidden
        hiddenValues = np.zeros((self.hiddenNumber))
        for j in range(self.hiddenNumber):
            summ = 0
            for i in range(len(inputValues)):
                summ += inputValues[i]*self.weightsInputToHidden[i][j]
            # adding bias
            summ += self.biasWeightsToHidden[j]
            hiddenValues[j] = self.sigmoid(summ)

        # hidden through output
        outputValues = np.zeros((self.outputNumber))
        for j in range(self.outputNumber):
            summ = 0
            for i in range(len(hiddenValues)):
                summ += hiddenValues[i]*self.weightsHiddenToOutput[i][j]
            summ += self.biasWeightsToOutput[j]
            outputValues[j] = self.sigmoid(summ)
            #outputValues[j] = summ

        # calculating the errors
        # output errors
        for j in range(self.outputNumber):
            # print(outputValues.size > 1)
            # if outputValues.size > 1
            #     expectedValues - outputValues # é que esse array quando n tm 2 outputs só tem 1 elemento, ainda dava um erro pq era um escalar
            #     #acabei de arrumar isso ai, no construtor. pode comentar e testar? fiz agorinha... blz 
            # else:
            #só ver se é list
            if(type(expectedValues) == type([])):
                self.outputErrors[j] = expectedValues[j] - outputValues[j]
            else:
                self.outputErrors[j] =  expectedValues - outputValues
            
        #hidden errors
        for i in range(self.hiddenNumber):
            summ = 0
            for j in range(self.outputNumber):
                summ += self.outputErrors[j]*self.weightsHiddenToOutput[i][j]
            self.hiddenErrors[i] = summ * self.sigmoid_derivative(hiddenValues[i])

        # calculating delta of weights
        # hidden-output deltas
        hiddenToOutputDeltas = np.zeros((self.hiddenNumber,self.outputNumber))
        for i in range(self.hiddenNumber):
            for j in range(self.outputNumber):
                hiddenToOutputDeltas[i][j] = hiddenValues[i] * self.outputErrors[j]
        
        # input-hidden deltas
        inputToHiddenDeltas = np.zeros((self.inputNumber,self.hiddenNumber))
        for i in range(self.inputNumber):
            for j in range(self.hiddenNumber):
                inputToHiddenDeltas[i][j] = inputValues[i] * self.hiddenErrors[j]
        
        # updating weights
        # hidden through output weights
        for i in range(self.hiddenNumber):
            for j in range(self.outputNumber):
                self.weightsHiddenToOutput[i][j] += hiddenToOutputDeltas[i][j] * learningRate
        # hidden through output bias weights     
        for j in range(self.outputNumber):
            self.biasWeightsToOutput[j] += self.outputErrors[j] * learningRate

        # input through hidden weights
        for i in range(self.inputNumber):
            for j in range(self.hiddenNumber):
                self.weightsInputToHidden[i][j] += inputToHiddenDeltas[i][j] * learningRate
        # input through hidden bias weights
        for j in range(self.hiddenNumber):
            self.biasWeightsToHidden[j] += self.hiddenErrors[j] * learningRate


    def train(self,epochs):
       
        for epoch in range(epochs):
            for inputValues_index in range(len(self.inputs)):
                self.feedfoward_backpropagation(self.inputs[inputValues_index],self.expectedOutput[inputValues_index], self.learningRate)

        # predict
        for inputValues in self.inputs:
            print(self.predict(inputValues))
            '''
        for epoch in range(epochs):
            output_from_layer_1, output_from_layer_2 = self.think(self.inputs)

            #Calculate the error for layer 2
            layer2_error = self.expectedOutput - output_from_layer_2
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1
            layer1_error = layer2_delta.dot(self.weightsHiddenToOutput.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            # Calculate adjusts
            layer1_adjustment = self.inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            #Adjust the weights
            self.weightsInputToHidden += layer1_adjustment.T
            self.weightsHiddenToOutput = self.weightsHiddenToOutput + layer2_adjustment

        for inputValues in self.inputs:
            hidden_state, output = self.think(inputValues)
            print(output)
        '''
    def think(self, inputs):
        output_layer1 = self.sigmoid(dot(inputs, self.weightsInputToHidden.T))
        output_layer2 = self.sigmoid(dot(output_layer1, self.weightsHiddenToOutput.T))
        return output_layer1, output_layer2

    def predict(self,inputValues):
        # input through hidden
        hiddenValues = np.zeros((self.hiddenNumber))
        for j in range(self.hiddenNumber):
            summ = 0 
            for i in range(len(inputValues)):
                summ += inputValues[i]*self.weightsInputToHidden[i][j]
            hiddenValues[j] = self.sigmoid(summ)

        # hidden through output
        outputValues = np.zeros((self.outputNumber))
        for j in range(self.outputNumber):
            summ = 0
            for i in range(len(hiddenValues)):
                summ += hiddenValues[i]*self.weightsHiddenToOutput[i][j]
            outputValues[j] = self.sigmoid(summ)
        
        return outputValues
    
    def sigmoid(self, z):
        result = 1.0 / (1.0 + np.exp(-z))
        return result
        
    def sigmoid_derivative(self, z):
        result = z * (1 - z)
        return result
'''
filename = "problemOR"
data = pd.read_csv("Data/" + filename + ".csv", header = None) 
np_file = data.to_numpy()
inputs = np.delete(np_file, -1, axis=1)
print(inputs)'''
if __name__ == "__main__":
    random.seed(1)
    targets2OutputsXOR = np.array([[0,0],[0,1],[1,0],[1,1]])
    test = MLP(2,4,1,'problemXOR',10000, 1)
#test = MLP(2,2,1,'problemAND',1000, 1, targets=[-1,-1,-1,1])
# opa
# sad, ele usa as coordenadas do input como -1 e 1 msm ?
#[20:31, 30/03/2020] Renan Ernesto: tipo o seu no xor coloca certinho com 0s e 1s?
# [20:32, 30/03/2020] Joninho EACH: Sim
