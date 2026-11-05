import json
import pandas as pd
# from importance_resampling import importance_resamp


f = open('./voca_map.json', 'r')
voca = json.load(f)
f.close()
voc_map = {}
for key, value in voca.items():
    voc_map[value] = key


# mapped.json是马里兰州的人口数据
f = open('./mapped_tn.json', 'r')
s = json.load(f)
f.close()
print(len(s))
print(s[0])

# transformed = []
# for item in s:
#     one_it = []
#     for it in item:
#         attr = int(it)
#         rel = voc_map[attr]
#         one_it.append(int(rel.split(':')[1]))
#     transformed.append(one_it)
# print(transformed[:10])
# print(len(transformed))

# source = pd.DataFrame(transformed, columns=['AGEP', 'SEX', 'RAC1P', 'ESR', 'HINCP', 'HHT', 'NP', 'WIF', 'HUPAC'])
# print(source.head())

# source.to_csv('maryland.csv', index=False)
