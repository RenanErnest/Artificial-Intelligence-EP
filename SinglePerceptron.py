import pandas as pd
import random
from matplotlib import pyplot as plt

class SinglePerceptron:

    '''
    A constructor.
    arguments:
    inputs - Pandas Dataframe: dataframe with all the inputs and the expected result in the last column
    '''
    def __init__(self, inputs):
        # inputs treatment
        self.inputs = inputs
        bias = [1] * len(inputs) # initializng a list with bias 1 for every input(cases)
        self.inputs.insert(0, 'bias', bias, True) # inserting the bias into the dataframe to be an extra input for each case

        # initializing weights with random values
        ''' 
        Entry amount minus one random weights.
        '''
        self.weights = [random.random() for n in range(len(inputs.columns)-1)]


    '''
    A function that given the inputs and weights calculate the output of a perceptron
    The threshold or limit adopted was: if the summatory is greater than 0 the output is 1. Otherwise the output is -1
    '''
    def perceptron(self, inputs):
        summ = 0
        for i in range(len(self.weights)):
            summ += inputs[i] * self.weights[i]
        if summ > 0:
            return 1
        return -1

    '''
    A function that given the current weights, an output and an expected output update the weights of a perceptron
    The new weight is obtained by the previous weight added by the expected output times the input times the learning rate
    '''
    def updateWeights(self, expected, output, inputs):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learning_rate*expected*inputs[i]
        # self.learning_rate /= 2

    '''
    A function to visualize the data and the line obtained from a perceptron and its weights.
    At the moment only works with 2D problems
    arguments:
    start - Integer: the x coordinate where the lines(perceptron functions) will start to be calculated at the x axis
    end - Integer: the x coordinate where the lines(perceptron functions) will end to be calculated at the x axis
    '''
    def plot2D(self, start, end):
        x1 = [i / 100 for i in range(start*100, end*100)]
        colors = []
        for y in self.inputs['expected']:
            colors.append('red') if y == -1 else colors.append('blue')
        plt.scatter(self.inputs['x1'], self.inputs['x2'],color=colors)
        x2 = [(-1 * self.weights[0] - self.weights[1] * x1_v) / self.weights[2] for x1_v in x1] # x1_v: value of x1
        plt.plot(x1, x2)
        plt.show()

    def train(self):
        self.plot2D(-2,2)
        self.learning_rate = 1
        for season in range(100):
            pre_weights = self.weights.copy()
            for inputs in self.inputs.values:
                expected = inputs[-1]
                output = self.perceptron(inputs)
                if output != expected:
                    self.updateWeights(expected,output,inputs)
                    self.plot2D(-2,2)

            if self.weights == pre_weights:
                break

    def predict(self):
        for inputs in self.inputs.values:
            expected = inputs[-1]
            output = self.perceptron(inputs)
            print(output,expected)

# input
data = pd.read_csv('Data/problemAND.csv', header=None, names=['x1','x2','expected'])
for i in range(len(data['expected'])):
    if data['expected'][i] == 0:
        data['expected'][i] = -1
myLayer = SinglePerceptron(data)
myLayer.train()
myLayer.predict()