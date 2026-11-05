import json
import pandas as pd
from importance_resampling import importance_resamp
import itertools
# from target_ca_resampling import srmse


f = open('gene_copula_ddpm6_decoded_1306799.json', 'r')
all_encoded = json.load(f)
f.close()
print('len(all_encoded): ', len(all_encoded))


f = open('./mapped_tn.json', 'r')
training_data = json.load(f)
f.close()
print('len(training_data): ', len(training_data))
# print(training_data[:10])

data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
print(data_train.head(10))


f = open('./mapped_tx.json', 'r')
target_data = json.load(f)
f.close()
print('len(target_data): ', len(target_data))

data_tar = pd.DataFrame(target_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
print(data_tar.head(10))


resampled = importance_resamp(data_train, data_tar, all_encoded)
print('len(resampled): ', len(resampled))

resampled.to_csv('./resampled_from_gene_copula_ddpm6_decoded_1306799_new.csv', index=False)

# data_gene = pd.DataFrame(all_encoded, columns=attr_list)

# srmse_dict = {}
# columns = data_train.columns

# for i in range(1, 6):
#     tuples = list(itertools.combinations(columns, i))  # No repeated elements
#     SRMSE = 0 
#     for tuple in tuples:
#         SRMSE += srmse(data_tar.drop(list(columns.difference(tuple)), axis=1), resampled.drop(list(columns.difference(tuple)), axis=1))
#     SRMSE /= len(tuples)
#     srmse_dict["SRMSE "+str(i)] = SRMSE
# # srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
# print('target and resampling ', srmse_dict)