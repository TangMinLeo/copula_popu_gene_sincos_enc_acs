import json
import pandas as pd
from importance_resampling import importance_resamp
import numpy as np
import itertools


def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.

    Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis
    """
    columns = list(data1.columns.values)
    # Relative frequency
    data1_f = data1.value_counts(normalize=True, dropna=False)
    data2_f = data2.value_counts(normalize=True, dropna=False)
    # Total numbers of categories
    Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
    M = np.prod(Mi)
    # SRMSE
    SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
    return SRMSE


f = open('./gene_copula_ddpm3_decoded_55218.json', 'r')
data = json.load(f)
f.close()

print('len(gene_data): ', len(data))

f = open('./puma_mapped_montgomery.json', 'r')
target_data = json.load(f)
f.close()
print('len(target_data): ', len(target_data))
data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
# print(target_data[:10])
print(data_tar.head(10))

f = open('./puma_mapped_princegeorges.json', 'r')
training_data = json.load(f)
f.close()
print('len(training_data): ', len(training_data))

data_tra = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
print(data_tra.head(10))

resampled = importance_resamp(data_tra, data_tar, data)

resampled.to_csv('./resampled_from_gene_copula_ddpm3_decoded_55218_1.csv', index=False)
print('len(resampled): ', len(resampled))


srmse_dict = {}
columns = resampled.columns
for i in range(1, 6):
    tuples = list(itertools.combinations(columns, i))  # No repeated elements
    SRMSE = 0 
    for tuple in tuples:
        SRMSE += srmse(data_tar.drop(list(columns.difference(tuple)), axis=1), resampled.drop(list(columns.difference(tuple)), axis=1))
    SRMSE /= len(tuples)
    srmse_dict["SRMSE "+str(i)] = SRMSE
# srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
print(srmse_dict)
