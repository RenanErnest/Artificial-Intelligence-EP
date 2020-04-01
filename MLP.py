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
        
        # adding the bias as an inputValue to facilitate the calculus
        inputValues = np.insert(inputValues,0,1) # adding the value 1 in the index 0

        # feed values from input layer through hidden layer
        hiddenValues = np.zeros((self.hiddenNumber)) # an array for the values that will be calculated
        for i in range(self.hiddenNumber): # for each hiddenNeuron
            summ = 0
            for j in range(len(inputValues)): # each inputNeuron
                summ += inputValues[j]*self.weightsInputToHidden[i][j] # linear combination of all the inputValues with their respective weight to a hidden perceptron i
            hiddenValues[i] = self.sigmoid(summ) # applying the step function to the summatory

        # adding the bias as a hiddenValue to facilitate the calculus
        # this is a development preference, the weights already contain the bias's weights, then, the bias comes as an input
        # this have to be treated at the backpropagation because the bias node does not propagate an error because the previous layer is not connected to it
        hiddenValues = np.insert(hiddenValues,0,1) # adding the value 1 in the index 0

        # feed values from hidden layer through output layer
        outputValues = np.zeros((self.outputNumber))
        for i in range(self.outputNumber): # for each outputNeuron
            summ = 0
            for j in range(len(hiddenValues)): # each hiddenNeuron
                summ += hiddenValues[j]*self.weightsHiddenToOutput[i][j] # linear combination of all the hiddenValues with their respective weight to an output perceptron i
            outputValues[i] = self.sigmoid(summ) # applying the step function to the summatory

        return outputValues,hiddenValues,inputValues
    
    def backpropagation(self,targetValues,inputValues,learningRate):
        
        # executing the feedfoward and receiving all of the values of output from each neuron on each layer
        # so outputValues are the values of the output neurons, hidden values are the values of the hidden neurons, and so on
        (outputValues,hiddenValues,inputValues) = self.feedfoward(inputValues)

        # setup a matrix for calculate the output errors, and calculating it
        outputErrors = np.zeros((self.outputNumber))
        # for each neuron at the output layer we calculate the difference of the target value and the output neuron value
        # also we multiply that difference with the derivative of the step function applied to the respective output neuron value
        for i in range(self.outputNumber):
            outputErrors[i] = (targetValues[i]-outputValues[i]) * self.sigmoid_derivative(outputValues[i])

        # get the delta (change) from the hidden-layer to output-layer, considering the error of the output layer calculated above
        deltasHiddenToOutput = np.zeros((self.outputNumber,self.hiddenNumber+1))
        for i in range(self.outputNumber):
            for j in range(self.hiddenNumber+1):
                deltasHiddenToOutput[i][j] = learningRate*outputErrors[i]*hiddenValues[j]

        # setup a matrix for calculate the hidden errors, and calculate it using the errors get by the output layer
        hiddenErrors = np.zeros((self.hiddenNumber))
        for i in range(1,self.hiddenNumber+1):
            summ = 0
            for j in range(self.outputNumber):
                summ += outputErrors[j]*self.weightsHiddenToOutput[j][i] 
            hiddenErrors[i-1] = self.sigmoid_derivative(hiddenValues[i])*summ 

        # setup a matrix for calculate the input to hidden layer errors, using each hidden-layer neuron error 
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
        
        #for each epoch 
        for epoch in range(epochs):
            for inputValues,targetValues in zip(inputs,targets):
                if type(targetValues[0]) == type(np.array((2))): # data treatment
                    targetValues = targetValues[0]
                self.backpropagation(targetValues,inputValues,learningRate)
                learningRate *= learningRateMultiplierPerEpoch
        
        return self.predict(inputs)
       

    def predict(self,inputs):
        output = []
        for inputValues in inputs:
            #get the values of output neurons provided by feed foward method, as hidden and original input
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
mlp = MLP(63,20,7)
data = mlp.openFile('Data/caracteres-limpo.csv')
letter_codes = np.array([[1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1],
                    ])

data = data.drop(data.columns[-1],axis=1)
data['new'] = list(letter_codes)
output = mlp.train(data,100,0.5)
letters = ['A','B','C','D','E','J','K']
for outputValues in output:
    for i in range(len(outputValues)):
        if round(outputValues[i]) == 1:
            print(letters[i])
            break

print()

#ruido
data = mlp.openFile('Data/caracteres-ruido.csv')
data = data.drop(data.columns[-1],axis=1)
data['new'] = list(letter_codes)
# data treatment
inputs = data.drop(data.columns[-1],axis=1).values
output = mlp.predict(inputs)
for outputValues in output:
    for i in range(len(outputValues)):
        if round(outputValues[i]) == 1:
            print(letters[i])
            break

# '''XOR'''
# mlp = MLP(2,4,1)
# data = mlp.openFile('Data/problemXOR.csv')
# output = mlp.train(data,1000,0.5)



