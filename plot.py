import numpy as np
import matplotlib.pyplot as plt

class _User:
	def __init__(self, index, rvwCount, posRatio):
		self.m_index = index
		self.m_posRatio = posRatio
		self.m_rvwCount = rvwCount

	def getIndex(self):
		return self.m_index

	def getPosRatio(self):
		return self.m_posRatio

	def getRvwCount(self):
		return self.m_rvwCount

def loadData(filename):
	f = open(filename)
	users = []
	index = 0
	rvwCount = 0
	posRatio = 0.0
	line = f.readline()
	while line:
		strs = line.split(',')
		index = int(strs[0])
		rvwCount = int(strs[1])
		posRatio = float(strs[2])
		users.append(_User(index, rvwCount, posRatio))
		line = f.readline()
	return users

def getCounts(users, interval):
	index = 0
	count = [0]*(100 / interval)
	print len(count)
	userGroups = [[] for i in xrange(100 / interval)]
	for u in users:
		print u.getPosRatio()
		index = int(u.getPosRatio()*100) / interval - 1
		print index
		count[index] += 1
		userGroups[index].append(users[u.getIndex()])
	return count, userGroups

# Calculate the light/medium/heavy ration in an interval.
def calcRatio(oneGroup):
	count = [0]*3
	ratio = []
	for u in oneGroup:
		if u.getRvwCount() <= 10:
			count[0] += 1
		elif u.getRvwCount() <= 50:
			count[1] += 1
		else:
			count[2] += 1
	for c in count:
		ratio.append(c*1.0/sum(count))
	return count, ratio

def merge(curGroup, oneGroup):
	for u in oneGroup:
		curGroup.append(u)
	return curGroup

filename = 'posRatio.txt'
interval = 10
users = loadData(filename)
count, userGroups = getCounts(users, interval)
for oneGroup in userGroups:
	print calcRatio(oneGroup)
print '-------------------'

curGroup = []
for oneGroup in userGroups:
	curGroup = merge(curGroup, oneGroup)
	print calcRatio(curGroup)
print '-------------------'

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
