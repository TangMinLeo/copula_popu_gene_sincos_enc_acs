import matplotlib.pyplot as plt
a = [269.09, 683.97, 182.17, 99.43, 114.52, 107.03, 126.00, 146.70, 241.08, 129.19, 194.88, 113.10]
tick_labels = ['2024.11.30', '2024.10.31', '2024.09.30', '2024.08.31', '2024.07.31', '2024.06.30', '2024.05.31',
               '2024.04.30', '2024.03.31', '2024.02.29', '2024.01.31', '2023.12.31']

a.reverse()
tick_labels.reverse()

plt.plot(a)
plt.xticks(range(len(a)), tick_labels, rotation=25)

plt.show()
plt.tight_layout()
