import json
# from utils import *
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

gene_json_file = '../population_synthesis/million_genes_ddpm_category5_2_times_samples.json'

file = open(gene_json_file, 'r')
json_data = json.load(file)

# print(json_data.keys())
data = pd.DataFrame.from_dict(json_data, orient='index', columns=['age', 'sex', 'homeincome', 'hometype', 'carown',
                                                                  'driver', 'workdays', 'worktype', 'student', 'numHH',
                                                                  'kidinHH', 'commode', 'comtime'])

# data = pd.DataFrame.from_dict(json_data, orient='index')
X = data.head(1000)
print(X.columns)
# print(X)
print(X.shape)

est = HillClimbSearch(X)
dag = est.estimate(scoring_method='bdeuscore')

print(dag.nodes())
print(dag.edges())
print(len(dag.edges()))

model = BayesianNetwork(dag)
model.fit(X)

inference = BayesianModelSampling(model)
gene_sample = inference.forward_sample(size=2)
print(gene_sample)

file.close()
