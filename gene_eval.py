import json
import pandas as pd
from copula_tools import *

f = open('./mapped.json', 'r', encoding='utf')
training_data = json.load(f)
f.close()
print(len(training_data))

data_train = pd.DataFrame(training_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
copular = CopulaScaler()
copular.fit(data_train)
print(copular.ecdf_map)
u_dist = copular.u_dist

gene_data = []
f = open('./copula_popu_gene_acs_voc_enc_0_100000.json', 'r', encoding='utf')
gene1 = json.load(f)
f.close()
gene_data.extend(gene1)

f = open('./copula_popu_gene_acs_voc_enc_1_100000.json', 'r', encoding='utf')
gene2 = json.load(f)
f.close()
gene_data.extend(gene2)

f = open('./copula_popu_gene_acs_voc_enc_2_100000.json', 'r', encoding='utf')
gene3 = json.load(f)
f.close()
gene_data.extend(gene3)

print(len(gene_data))

data_gene = pd.DataFrame(gene_data, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
copular1 = CopulaScaler()
copular1.fit(data_gene)
print(copular1.ecdf_map)

