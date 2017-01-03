import numpy as np
data = np.loadtxt("compare_4.txt")

sortedIndex = data[:, 2].argsort()
sortedData = data[sortedIndex]

mid = 0
for i in range(sortedData[:,0].size):
	if sortedData[i,2] >= 0.5 and mid == 0:
		mid = i

print mid
