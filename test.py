
import pandas as pd
import numpy as np
import json
from utils import *
from copula_tools import *
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

gene_json_file = '../population_synthesis/million_genes_ddpm_category5_2_times_samples.json'

file = open(gene_json_file, 'r')
json_data = json.load(file)

# print(json_data.keys())
data = pd.DataFrame.from_dict(json_data, orient='index', columns=['age', 'sex', 'homeincome', 'hometype', 'carown',
                                                                  'driver', 'workdays', 'worktype', 'student', 'numHH', 'kidinHH', 'commode', 'comtime'])

# data = pd.DataFrame.from_dict(json_data, orient='index')
X = data.head(100)
# print(X['age'])

marginals = {}
for column in X.columns:
    print(column)
    count = X[column].value_counts()
    marginal = count.sort_index()
    marginal = marginal.to_dict()
    # print(marginal)
    for key in marginal.keys():
        marginal[key] = marginal[key] / 100.0
    # print(marginal)
    marginals[column] = marginal


# count = X['sex'].value_counts()
# print('X[age][0]: ', X['age'].iloc[0])
# print(type(X['sex']))
# # print(count.index)
# # print(count[0])
# print(count.sort_index())
# marginal = count.sort_index()
# marginal = marginal.to_dict()

# print(marginal)
# for key in marginal.keys():
#     marginal[key] = marginal[key] / 100.0
# print(marginal)

# x = {'sex': {'male': 0.51, 'female': 0.49}}
# x = {'age': marginal}

copular1 = CopulaScaler()
copular1.fit_from_marginals(marginals)

# print(copular1.ecdf_map)
# print(copular1.answer_mapping)
data = copular1.transform_from_marginals(X)
print(data)

est = HillClimbSearch(data)
dag = est.estimate(scoring_method='bdeuscore')

model = BayesianNetwork(dag)
model.fit(data)

inference = BayesianModelSampling(model)
gene_sample = inference.forward_sample(size=10000)
# print(gene_sample)
# print(gene_sample.columns)
reverse_data = copular1.inverse_transform(gene_sample)
print(reverse_data)
print(reverse_data['age'].value_counts())
