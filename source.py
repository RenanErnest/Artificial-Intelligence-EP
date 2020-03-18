import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.array([
    [-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1],
    [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
    [-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1],
    [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
    [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
    [-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1],
    [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1],
    [1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
    [-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
    [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1],
    [-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1],
    [-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,1],
    [1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1],
    [-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1],
    [1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1],
    [-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1],
    [1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1]
])

y = np.array([
    [[1],[0],[0],[0],[0],[0],[0]],
    [[0],[1],[0],[0],[0],[0],[0]],
    [[0],[0],[1],[0],[0],[0],[0]],
    [[0],[0],[0],[1],[0],[0],[0]],
    [[0],[0],[0],[0],[1],[0],[0]],
    [[0],[0],[0],[0],[0],[1],[0]],
    [[0],[0],[0],[0],[0],[0],[1]],
    [[1],[0],[0],[0],[0],[0],[0]],
    [[0],[1],[0],[0],[0],[0],[0]],
    [[0],[0],[1],[0],[0],[0],[0]],
    [[0],[0],[0],[1],[0],[0],[0]],
    [[0],[0],[0],[0],[1],[0],[0]],
    [[0],[0],[0],[0],[0],[1],[0]],
    [[0],[0],[0],[0],[0],[0],[1]],
    [[1],[0],[0],[0],[0],[0],[0]],
    [[0],[1],[0],[0],[0],[0],[0]],
    [[0],[0],[1],[0],[0],[0],[0]],
    [[0],[0],[0],[1],[0],[0],[0]],
    [[0],[0],[0],[0],[1],[0],[0]],
    [[0],[0],[0],[0],[0],[1],[0]],
    [[0],[0],[0],[0],[0],[0],[1]]
])

#X_expected = np.array['A','B','C','D','E','J','K','A','B','']

num_i_units = 63
num_h_units = 63
num_o_units = 7

#tirei o codigo de pegar o dataset como arquivo
class MLP:
    '''
        arguments:
        inputs - represent a dataframe containing the inputs and the target value on each last column
        epoch - represents the number of iterations in each training
        learning_rate - represents the amount of update on each error treatment
        input_amt - represents the number of neurons in the input layer
        hidden_amt - represents the number of neurons in the hidden layer
        output_amt - represents the number of neurons in the output layer
    '''

    
    def __init__(self, epoch, inputs, input_amt, hidden_amt, output_amt, learning_rate):
        self.epoch = epoch
        self.input_amt = input_amt
        self.hidden_amt = hidden_amt
        self.output_amt = output_amt
        self.learning_rate = learning_rate

        np.random.seed(1)
        W1 = np.random.normal(0, 1, (hidden_amt, input_amt))
        W2 = np.random.normal(0, 1, (output_amt, hidden_amt))

        B1 = np.random.random((hidden_amt, 1))
        B2 = np.random.random((output_amt, 1))
        

learning_rate = 0.1 # 0.001, 0.01 <- Magic values
reg_param = 0 # 0.001, 0.01 <- Magic values
max_iter = 15000 # 5000 <- Magic value
m = 4 # Number of training examples

# The model needs to be over fit to make predictions. Which 
np.random.seed(1)
W1 = np.random.normal(0, 1, (num_h_units, num_i_units)) # 2x2
W2 = np.random.normal(0, 1, (num_o_units, num_h_units)) # 1x2

B1 = np.random.random((num_h_units, 1)) # 2x1
B2 = np.random.random((num_o_units, 1)) # 1x1

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))

def forward(x, predict=False):
    a1 = x.reshape(x.shape[0], 1) # Getting the training example as a column vector.

    z2 = W1.dot(a1) + B1 # 2x2 * 2x1 + 2x1 = 2x1
    a2 = sigmoid(z2) # 2x1

    z3 = W2.dot(a2) + B2 # 1x2 * 2x1 + 1x1 = 1x1
    a3 = sigmoid(z3)

    if predict: return a3
    return (a1, a2, a3)

dW1 = 0
dW2 = 0

dB1 = 0
dB2 = 0

cost = np.zeros((max_iter, 1))
for i in range(max_iter):
    c = 0

    dW1 = 0
    dW2 = 0

    dB1 = 0
    dB2 = 0
    for j in range(m):
        sys.stdout.write("\rIteration: {} and {}".format(i + 1, j + 1))

        # Forward Prop.
        a0 = X[j].reshape(X[j].shape[0], 1) # 63x1 # 2x1

        z1 = W1.dot(a0) + B1 #63x63 * 63x1 = 63x1 # 2x2 * 2x1 + 2x1 = 2x1
        a1 = sigmoid(z1) # 63x1 # 2x1

        z2 = W2.dot(a1) + B2 #7x63 * 63x1 = 7x1 # 1x2 * 2x1 + 1x1 = 1x1
        a2 = sigmoid(z2) # 7x1 # 1x1

        # Back prop.
        # o a2 é o chute, ai deu esse array aqui:
        '''
            [   [0.84964996]
                [0.43277716]
                [0.99149763]
                [0.89154581]
                [0.64775394]
                [0.74671131]
                [0.77987597]    ]
        '''
        ''' o target:
            [1 0 0 0 0 0 0]
        '''

        '''
            dz2 tinha que ser:
            [
                [0.84 - 1]
                [0.43 - 0]
                [...]
                ...
                o esquema é a gente deixar o target no mesmo formato que o a2, parece msm kkkkk, mas o target sempre é 1 lugar só que vamos subtrair o 1, então
                mas ai tem que englobar cada valor com [0] kkkkk, é trampo, mas vale tentar. ah bora confiar que o codigo dele ta certo e tentar arrumar as entradas mesmo
            ]
        '''
        dz2 = a2 - y[j] # 7x1 - ???? # 1x1  #aqui tinha que fazer - de cada elemento de a2 por cada elemento de y[j] sim,
        #print(y[j]) # ta fazendo errado essa parte, mas se pa eu saquei porque, tem que formatar nosso target, acho que era matriz antes
        # e agora ta meio que array, ele deu reshape ,ata, tendi, mas tipo, tem BO no de cima que ve o erro, pq ele tava acostumado
        # com valor unitario, e agora são varias outputs
        dW2 += dz2 * a1.T #7x1 .* 1x63 = 7x63 # 1x1 .* 1x2 = 1x2
        # deu erro uma linha depois kkkkk
        print(W2.T) # 63x7
        print(dz2) # 7x1
        print(a1) # 63x1
        # pera ae ja volto
         # nao ta dando pra multiplicar pq precisa deixar as matrizes compativeis, o numero de linhas precisa ser = de colunas, 
        #a multiplicação n é comutativa n*p != p*n, mas acho que o n de linha da primeira matriz precisa ser igual ao de colunas da 2 matriz era pra ter dado, que isso comutativa? ah sim, mas aparentemente ta no ordem
        # hm
        # e se criar uma matriz de dimensão igual com 1 onde nao tem valor ? Não faço ideia, to bolado com tudo já kkkk
        
        # #qkuerkkekkkkndo caçar outro cód
        # como transpoe essa matriz ?
        # vou caçar outro código enquanto isso
        #blz 
        w2_transpose = W2 # assim
        dz1 = np.multiply((W2 * dz2), sigmoid(a1, derv=True)) # (63x7 * 7x63 = 63x63) * 63x1 # (2x1 * 1x1) .* 2x1 = 2x1
        dW1 += dz1.dot(a0.T) # 2x1 * 1x2 = 2x2

        dB1 += dz1 # 2x1
        dB2 += dz2 # 1x1

        c = c + (-(y[j] * np.log(a2)) - ((1 - y[j]) * np.log(1 - a2)))
        sys.stdout.flush() # Updating the text.
    W1 = W1 - learning_rate * (dW1 / m) + ( (reg_param / m) * W1)
    W2 = W2 - learning_rate * (dW2 / m) + ( (reg_param / m) * W2)

    B1 = B1 - learning_rate * (dB1 / m)
    B2 = B2 - learning_rate * (dB2 / m)
    cost[i] = (c / m) + ( 
        (reg_param / (2 * m)) * 
        (
            np.sum(np.power(W1, 2)) + 
            np.sum(np.power(W2, 2))
        )
    )


for x in X:
    print("\n")
    print(x)
    print(forward(x, predict=True))

plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

#------------------------------------------------------------------------------------
#data = pd.read_csv('Data/problemXOR.csv', header=None, names=['x1','x2','target'])
#for i in range(len(data['target'])):
 #   if data['target'][i] == 0:
  #      data['target'][i] = -1

#model = TwoLayerMLP(2,data,100,0.5)
#model.train()
#model.predict()