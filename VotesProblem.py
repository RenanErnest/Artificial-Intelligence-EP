import MLP
import pandas as pd
import numpy as np

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

# Estratégia de grade (trocar nome sei la qual é)
grade = {'hidden_neurons': [6, 8, 12, 17, 34, 50], 'learning_rate': [0.5, 0.3, 0.1], 'epochs': [300, 1000, 5000]}