from MLP import MLP
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from adjustText import adjust_text
import time

start = time.time()

# Reading data
votos = pd.read_csv('Data/house-votes-84.txt', header = None)
votos.columns = ['is-republican','handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution','physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback','education-spending','superfund-right-to-sue','crime','duty-free-exports','export-administration-act-south-africa']

# Data treatment
values_map = {'y': 1,'n': 0,'?': np.nan}
class_map = {'republican': 1,'democrat': 0}
votos = votos.replace({'is-republican': class_map})
votos = votos.replace(values_map)
# Deleting missing data lines
votos = votos.dropna()

# Houldout
train = votos.sample(frac=0.7, random_state=np.random.randint(1000)) #random state is a seed value
test = votos.drop(train.index)
validate = train.sample(frac=0.2, random_state=np.random.randint(1000))
train = train.drop(validate.index)

# TrainInputs and labels
trainInputs = train.drop(train.columns[0], axis=1)
trainLabel = train.drop(train.columns.difference(['is-republican']), 1)

# TestInputs and labels
testInputs = test.drop(test.columns[0], axis=1)
testLabel = test.drop(test.columns.difference(['is-republican']), 1)

# ValidateInputs and labels
validateInputs = validate.drop(validate.columns[0], axis=1)
validateLabel = validate.drop(validate.columns.difference(['is-republican']), 1)

# Grid Search
grid = {'hidden_neurons': [20, 40, 60, 80], 'learning_rate': [0,5, 0,3, 0,1], 'epochs': [300, 1000, 5000]}

grid = {'hidden_neurons': [80], 'learning_rate': [0.5], 'epochs': [5000]} # best combo

inputNumber = 16
outputNumber = 1
# nada cm um trab nas ferias kkk algo de errado n esta certo
# arrays de mse
mseTrainList = []
mseValidateList = []
minConfig = None
cmValues = []

models = {'params': set(), 'mses': set()}
for hn in grid['hidden_neurons']:
    for lr in grid['learning_rate']:
        for ep in grid['epochs']:
            mlp = MLP(inputNumber, hn, outputNumber)
            (mseTrainList, mseValidateList, minConfig) = mlp.train(trainInputs, trainLabel, validateInputs, validateLabel, ep, lr, earlyStop=True)
            for threshold in np.arange(0, 1.1, 0.1):
                cm = mlp.confusionMatrix(testInputs.values,testLabel.values, threshold)
                cmValues.append({'cm':cm, 'threshold': threshold})
            # models['params'].add(str((hn,lr,ep)))
            # models['mses'].add(mlp.mse(trainInputs.values, trainLabel.values))

# print(models) 

# Grid Search
# plt.plot(list(models['params']), list(models['mses']))
# plt.xlabel('Combinações', fontsize=18)
# plt.ylabel('MSE', fontsize=18)
# plt.title('Grid Search', fontsize=24)
# plt.xticks(rotation=45, ha="right")
# plt.show()

# # Early stop
# plt.plot(list(range(len(mseValidateList))), mseValidateList)
# plt.plot(list(range(len(mseTrainList))), mseTrainList)
# plt.xlabel('Epoch', fontsize=18)
# plt.ylabel('MSE', fontsize=18)
# plt.title('Early Stop', fontsize=24)
# plt.legend(['Validação','Treinamento'])
# plt.annotate(str('Mínimo ' + str((minConfig['epoch'], round(minConfig['mse'],6)))),
#             xy=(minConfig['epoch'],  minConfig['mse']), xycoords='data',
#             xytext=(-50, 30), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->"))
# plt.show()

# F-scores
results = []
for cm in cmValues:
    try:
        cmv = cm['cm']
        precision = cmv['truePositive']/(cmv['truePositive'] + cmv['falsePositive'])
        recall = cmv['truePositive']/(cmv['truePositive'] + cmv['falseNegative'])
        results.append({'matrix': cm,
                        'precision': precision, 
                        'recall': recall, 
                        'fscore': 2*precision*recall/(precision + recall)
                        })
    except:
        results.append({'invalid Threshold, division by zero: ',cm['threshold']})

for res in results:
    print(res)

# ROC
x = []
y = []
th = []
for cm in cmValues:
    cmv = cm['cm']
    try:
        xi = cmv['truePositive']/(cmv['truePositive'] + cmv['falseNegative']) # revocação
        yi = cmv['falsePositive']/(cmv['trueNegative'] + cmv['falsePositive'])
        x.append(xi)
        y.append(yi)
        th.append(cm['threshold'])
    except:
        pass

    
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(y, x)
plt.plot(y, x)
plt.xlabel('Especificidade', fontsize=18)
plt.ylabel('Sensibilidade', fontsize=18)
plt.title('ROC curve', fontsize=24)
texts = [plt.text(y, x, round(s,1)) for x,y,s in zip(x,y,th)]
adjust_text(texts, only_move={'points':'xy', 'text':'xy'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5), autoalign='xy')
# for i,coord in enumerate(zip(y,x)):
#     plt.annotate(str(round(th[i],1)),
#         xy=(coord[0],  coord[1]), xycoords='data',
#         xytext=(15*(1+(abs(i-len(y))/10)), -30*(1+(abs(i-len(y))/10))), textcoords='offset points',
#         arrowprops=dict(arrowstyle="->"))
plt.show()

print(time.time()-start)