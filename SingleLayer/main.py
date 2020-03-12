import numpy as np
from Perceptron import Perceptron
import draw as dr #grafico

binary_inputs = []

#dataset das possibilidades
binary_inputs.append(np.array([-1, -1]))
binary_inputs.append(np.array([-1, 1]))
binary_inputs.append(np.array([1, -1]))
binary_inputs.append(np.array([1, 1]))

#saida esperada
AND_labels = np.array([0,0,0,1])
OR_labels = np.array([0,1,1,1])
XOR_labels = np.array([0,1,1,0])

#alocar objeto Perceptron com duas entradas
perceptron = Perceptron(2)
perceptron.train(binary_inputs, AND_labels)

inputs = np.array([1, -1])
result = perceptron.predict(inputs) 
print(result)
#=> 1

inputs = np.array([1, 1])
result = perceptron.predict(inputs) 
print(result)
#=> 0
dr.drawGraph(binary_inputs)