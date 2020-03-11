class SingleLayerPerceptron:

    def perceptron(self,inputs,weights,threshold):
        summ = 0
        for i in range(len(inputs)):
            summ += inputs[i]*weights[i]
        if summ > threshold:
            return 1
        return 0