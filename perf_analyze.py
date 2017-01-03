import numpy as np
def loadData():
	data = np.loadtxt("perf_compare.txt")
	index = []
	for i in range(data[:,0].size):
		if(data[i,3] > data[i,2]):
			index.append(data[i,0])

	print calcRatio(index)
	print calcRatio(data[:,0])

def calcRatio(arr):
	light = 0
	mid = 0
	heavy = 0
	for i in arr:
		if i <= 10:
			light = light + 1
		elif i <= 50:
			mid = mid + 1
		else:
			heavy = heavy + 1

	ratio = []
	print len(arr)
	ratio.append(light*1.0/len(arr))
	ratio.append(mid*1.0/len(arr))
	ratio.append(heavy*1.0/len(arr))
	return ratio

if __name__ == '__main__':
	loadData()
