import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
	f = open(filename)
	x = []
	ratio = []
	line = f.readline()
	while line:
		strs = line.split(',')
		x.append(int(strs[0]))
		ratio.append(float(strs[1]))
		line = f.readline()
	ratio = sorted(ratio)
	print ratio[0], ratio[len(ratio)-1]
	return ratio

def getCounts(ratio, interval):
	index = 0
	count = [0] * (100 / interval)
	for i in ratio:
		index = int(i*100) / interval - 1
		#print index
		count[index] += 1
	
	return count


filename = 'posRatio.txt'
interval = 5
ratio = loadData(filename)
count = getCounts(ratio, interval)

x = []
for i in range(len(count)):
	x.append(i*interval/100.0)
counts = []
val = 0
for i in count:
	val += i
	counts.append(val)
plt.plot(x, counts, 'p-')
print counts
plt.show()
