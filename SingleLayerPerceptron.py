import pandas as pd
import random
from matplotlib import pyplot as plt
import copy

class SingleLayerPerceptron:

    '''
    arguments:
    perceptron_amount - Integer: number of perceptrons for this single layer
    inputs - Pandas Dataframe: dataframe with all the inputs and the expected result in the last column
    '''
    def __init__(self, perceptron_amount, inputs):
        # inputs treatment
        self.inputs = inputs
        bias = [1] * len(inputs) # initializng a list with bias 1 for every input(cases)
        self.inputs.insert(0, 'bias', bias, True) # inserting the bias into the dataframe to be an extra input for each case

        # initializing weights with random values
        ''' 
        Entry amount minus one random weights for each perceptron. 
        A n,m matrix where n is the number of perceptrons and m is the number of inputs. '''
        self.weights = [[random.random() for m in range(len(inputs.columns)-1)] for n in range(perceptron_amount)]

        self.perceptron_amout = perceptron_amount


    '''
    A function that given the inputs and weights calculate the output of a perceptron k
    The threshold or limit adopted was: if the summatory is greater than 0 the output is 1. Otherwise the output is 0
    '''
    def perceptron(self, k, inputs):
        summ = 0
        for i in range(len(self.weights[k])):
            summ += inputs[i] * self.weights[k][i]
        if summ > 0:
            return 1
        return -1

    '''
    A function that given the current weights, an output and an expected output update the weights of a perceptron k
    The new weight is obtained by the previous weight added by the error times the input
    '''
    def updateWeights(self, k, expected, output, inputs):
        for i in range(len(self.weights)):
            # new_weights[i] = weights[i] + learning_rate*expected*inputs[i] # Fausett
            self.weights[k][i] = self.weights[k][i] + self.learning_rate*expected*inputs[i]  # another aproach
        self.learning_rate /= 2

    '''
    A function to visualize the division of the data given the weights, at the moment only works with 2D problems
    arguments:
    start - Integer: the x coordinate where the lines(perceptron functions) will start to be calculated
    end - Integer: the x coordinate where the lines(perceptron functions) will end to be calculated
    '''
    def plot2D(self, start, end):
        x1 = [i / 100 for i in range(start*100, end*100)]
        colors = []
        for y in self.inputs['expected']:
            if y == -1:
                colors.append('red')
            else:
                colors.append('blue')
        plt.scatter(self.inputs['x1'], self.inputs['x2'],color=colors)
        for perceptron in self.weights:
            x2 = [(-1 * perceptron[0] - perceptron[1] * x1_v) / perceptron[2] for x1_v in x1] # x1_v: value of x1
            plt.plot(x1, x2)
        plt.show()

    def train(self):
        self.plot2D(-2,2)
        self.learning_rate = 1
        for season in range(100):
            pre_weights = copy.deepcopy(self.weights)
            for k in range(self.perceptron_amout):
                for inputs in self.inputs.values:
                    expected = inputs[-1]
                    output = self.perceptron(k,inputs)
                    if output != expected:
                        self.updateWeights(k,expected,output,inputs)
                        self.plot2D(-2,2)

            if self.weights == pre_weights:
                break

    def predict(self):
        for inputs in self.inputs.values:
            expected = inputs[-1]
            for k in range(self.perceptron_amout):
                output = self.perceptron(k,inputs)
                print(output,end=" ")
            print(expected)

# input
data = pd.read_csv('Data/problemAND.csv', header=None, names=['x1','x2','expected'])
for i in range(len(data['expected'])):
    if data['expected'][i] == 0:
        data['expected'][i] = -1
myLayer = SingleLayerPerceptron(1,data)
myLayer.train()
myLayer.predict()


