import numpy as np

class MLP:

    def __init__(self, inputNumber, hiddenNumber, outputNumber):
        self.inputNumber = inputNumber
        self.hiddenNumber = hiddenNumber
        self.outputNumber = outputNumber

        ''' weights '''
        # a inputNumber+1(bias) x hiddenNumber matrix with random values. It already includes the bias weights
        # weightsInputToHidden[i][j] means the weight between the perceptron i of the hidden layer and the perceptron j of the input layer
        self.weightsInputToHidden = 2 * np.random.random((hiddenNumber, inputNumber + 1)) - 1
        # a hiddenNumber+1(bias) x outputNumber matrix with random values. It already includes the bias weights
        # weightsHiddenToOutput[i][j] means the weight between the perceptron i of the output layer and the perceptron j of the hidden layer
        self.weightsHiddenToOutput = 2 * np.random.random((outputNumber, hiddenNumber + 1)) - 1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedfoward(self, inputValues):
        # inputValues: a single line of input containg the attributes x1, x2, ..., xn

        # adding the bias as an inputValue to facilitate the calculus
        if len(inputValues) == self.inputNumber:
            inputValues = np.insert(inputValues, 0, 1)  # adding the value 1 in the index 0

        # feed values from input layer through hidden layer
        hiddenValues = np.zeros((self.hiddenNumber))  # an array for the values that will be calculated
        for i in range(self.hiddenNumber):  # for each hiddenNeuron
            summ = 0
            for j in range(len(inputValues)):  # each inputNeuron
                # linear combination of all the inputValues with their respective weight to a hidden perceptron i
                summ += inputValues[j] * self.weightsInputToHidden[i][j] 
            hiddenValues[i] = self.sigmoid(summ)  # applying the step function to the summation

        # adding the bias as a hiddenValue to facilitate the calculus
        # this is a development preference, the weights already contain the bias's weights, then, the bias comes as an input
        # this have to be treated at the backpropagation because the bias node does not propagate an error because the previous layer is not connected to it
        if len(hiddenValues) == self.hiddenNumber:
            # adding the value 1 in the index 0
            hiddenValues = np.insert(hiddenValues, 0, 1)

        # feed values from hidden layer through output layer
        outputValues = np.zeros((self.outputNumber))
        for i in range(self.outputNumber):  # for each outputNeuron
            summ = 0
            for j in range(len(hiddenValues)):  # each hiddenNeuron
                # linear combination of all the hiddenValues with their respective weight to an output perceptron i
                summ += hiddenValues[j] * self.weightsHiddenToOutput[i][j]
            outputValues[i] = self.sigmoid(summ)  # applying the step function to the summation

        return outputValues, hiddenValues, inputValues

    def backpropagation(self, targetValues, inputValues, learningRate):

        # executing the feedfoward and receiving all of the values of output from each neuron on each layer
        # so outputValues are the values of the output neurons, hidden values are the values of the hidden neurons, and so on
        (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)

        # setting a matrix for calculate the output errors, and calculating it
        outputErrors = np.zeros((self.outputNumber))
        # for each neuron at the output layer we calculate the difference of the target value and the output neuron value
        # also we multiply that difference with the derivative of the step function applied to the respective output neuron value
        for i in range(self.outputNumber):
            outputErrors[i] = (targetValues[i] - outputValues[i]) * self.sigmoid_derivative(outputValues[i])

        # getting the delta (change) of the weights from the hidden-layer through output-layer, considering the errors of the output layer calculated above
        deltasHiddenToOutput = np.zeros((self.outputNumber, self.hiddenNumber + 1))
        for i in range(self.outputNumber):
            for j in range(self.hiddenNumber + 1):
                # for each weight that are connected to a output neuron i we store the change for that weight in a deltas array
                # this change is calculated by the product of the learning rate, the error of the neuron i, and the value that following that weight "caused" the error
                deltasHiddenToOutput[i][j] = learningRate * outputErrors[i] * hiddenValues[j]

        # setting a matrix for calculate the hidden errors, and calculating it using the errors got by the output layer
        # here we have to be cautelous, the errors are only for the neurons, not the bias, but our weights matrix have the bias's weights
        # so we have to ignore the bias's weights in this step of backpropagating the error to previous nodes
        # this is why we iterate from index 1 through hiddenNumber+1
        hiddenErrors = np.zeros((self.hiddenNumber))
        for i in range(1, self.hiddenNumber + 1):
            summ = 0
            # calculating the linear combination of all the output layer neuron errors with the respective weight that leads to that error
            # for example an error of a hidden neuron i will be the summation of the product of all the errors of output neurons with the weight that connect the hidden neuron i with these neurons
            for j in range(self.outputNumber):
                summ += outputErrors[j] * self.weightsHiddenToOutput[j][i]
            # because of a neural network is a bunch of nested functions, the derivative in order to find the minimum error implies in several chain rules
            # so every error propagated has a value that is multiplied with the derivative of the step function applied to the value of the node that receives the error
            hiddenErrors[i - 1] = self.sigmoid_derivative(hiddenValues[i]) * summ

            # getting the delta (change) of the weights from the input-layer through the hidden-layer, considering the errors of the hidden layer calculated above
        deltasInputToHidden = np.zeros((self.hiddenNumber, self.inputNumber + 1))
        for i in range(self.hiddenNumber):
            for j in range(self.inputNumber + 1):
                # for each weight from the input layer that are connected to a hidden neuron i we store the change for that weight a the deltas array
                # this change is calculated by the product of the learning rate, the error of the neuron i, and the value that following that weight "caused" the error
                deltasInputToHidden[i][j] = learningRate * hiddenErrors[i] * inputValues[j]

        # updating the weights
        # only and finally, adding the deltas(changes) to the current weights
        for i in range(len(self.weightsHiddenToOutput)):
            for j in range(len(self.weightsHiddenToOutput[i])):
                self.weightsHiddenToOutput[i][j] += deltasHiddenToOutput[i][j]

        for i in range(len(self.weightsInputToHidden)):
            for j in range(len(self.weightsInputToHidden[i])):
                self.weightsInputToHidden[i][j] += deltasInputToHidden[i][j]

    def train(self, trainSet, trainLabel, testSet, epochs=1000, learningRate=1, learningRateMultiplierPerEpoch=1):
        '''
            trainSet: a pandas dataframe with the values for training
        '''
        # data treatment, creating a numpy array for only the inputs, and another for only the targets
        inputs = trainSet.values
        targets = trainLabel.values

        errorPerEpoch = ''
        outputPerEpoch = ''
        overfitTestPerEpoch = ''

        for epoch in range(epochs):

            ##################### FAZER PREDICT NO DATASET DE TESTE E CALCULAR O ERRO ##################### 
            testOutput = predict(testSet)
            erros = calcularErro(testOutput, outputLabel)
            for o, e in enumerate(erros):
                overfitTestPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nValue: ' + erros[o] + '\n'
            ###############################################################################################

            # for each epoch we iterate over all the input and targets of a specific case and send it to the backpropagation function
            for inputValues, targetValues in zip(inputs, targets):
                # this condition is a data treatment for targets with more than one value, for example targets that are arrays
                if type(targetValues[0]) == type(np.array((2))):
                    targetValues = targetValues[0]

                # making error and output per epoch log
                (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)
                errorPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nInput: ' + str(inputValues) + '\nError: ' + \
                                 str((targetValues - outputValues) * self.sigmoid_derivative(outputValues)) + '\n'
                outputPerEpoch += 'Epoch: ' + str(epoch + 1) + '\nInput: ' + str(inputValues) + '\nOutput: ' + str(
                    outputValues) + '\n'

                self.backpropagation(targetValues, inputValues, learningRate)
                # updating the learning rate according to the multiplier, if the multiplier is 1, we can assume that our learning rate is static
                learningRate *= learningRateMultiplierPerEpoch 

        # making error and output per epoch log
        # writetxt(problem + '_Errors_Per_Epoch', errorPerEpoch)
        # writetxt(problem + '_Outputs_Per_Epoch', outputPerEpoch)

        # at the end it returns the a prediction of the inputs that were used to train the model
        # what is expected is that the predictions match with the target values of each input case
        return self.predict(inputs)

    def calcularErro(self, outputs, expectedLabels):
        errorList = []
        for i,o in enumerate(outputs):
            errorList.append(expectedLabels[i] - outputs[i])
        return errorList

    def predict(self, inputs):
        output = []
        for inputValues in inputs:
            # calling feed foward for each input case and receiving the output for each case, that is stored in the output list
            (outputValues, hiddenValues, inputValues) = self.feedfoward(inputValues)
            output.append(outputValues)
        # at the end we return all the outputs of our model in a list
        return output