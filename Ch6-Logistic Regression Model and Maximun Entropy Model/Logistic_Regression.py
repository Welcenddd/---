import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename): #载入数据集
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		dataMat.append([1, float(curLine[0]), float(curLine[1])]) # 考虑到计算方便，第一列添加常数项即w0=1
		labelMat.append(int(curLine[-1])) #标签

	return np.mat(dataMat), labelMat

def trainModel(dataSet,labels,tol=None,trainSteps=None):
	m,n = dataSet.shape
	stepSize = 0.001
	#这里采用定步长，理论上可以参考附录A（虽然是梯度下降，但是修改一下符号就得到梯度上升）的算法A.1中的第（3）步通过一维搜索得到每个迭代步的最优步长
	weights = np.zeros((n,1)) # 初始化权重为零
	if tol:
		iter = True
		while iter:
			for i in range(m):
				dL = labels[i] * dataSet[i] - np.exp(np.dot(dataSet[i],weights)) * dataSet[i] / (1 + np.exp(np.dot(dataSet[i],weights))) # 求出的是权重的每个分量的增量
				if np.linalg.norm(dL) < tol: iter = False
				weights += stepSize * dL.T
	elif trainSteps:
		for _ in range(trainSteps):
			for i in range(m):
				dL = labels[i] * dataSet[i] - np.exp(np.dot(dataSet[i], weights)) * dataSet[i] / (
							1 + np.exp(np.dot(dataSet[i], weights)))  # 求出的是权重的每个分量的增量
				weights += stepSize * dL.T

	return weights

def predict(weights,testDat):
	prob = np.exp(np.dot(testDat,weights)) / (1 + np.exp(np.dot(testDat,weights))) # Y=1的概率
	if prob >= 0.5:
		return 1
	return 0

def validModel(testDataSet,testLabels):
	# 在测试集上用准确率指标评价模型
	errorCount = 0
	for i in range(len(testDataSet)):
		if predict(weights,testDataSet[i]) != testLabels[i]:
			print('分类错误的样本序号为%s' % i)
			errorCount += 1
	print('模型的准确率为%.2f%%' % ((1-errorCount/len(testDataSet))*100))

def plotBestFit(weights): # 绘制数据分布图
	dataMat, labelMat = loadDataSet('testSet.txt')
	dataArr = np.array(dataMat)
	m = np.shape(dataMat)[0]
	xcord1 = [] # 正样本
	ycord1 = []
	xcord2 = [] # 负样本
	ycord2 = []
	# 根据数据集标签进行分类
	for i in range(m):
		if int(labelMat[i]) == 1: # 1为正样本

			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else: # 0为负样本
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5) # 绘制正样本
	ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5) # 绘制负样本
	x = np.arange(-3.0, 3.0, 0.1) # x轴坐标
	# w0*x0 + w1*x1 * w2*x2 = 0
	# x0 = 1, x1 = x, x2 = y
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.title('BestFit')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

if __name__ == '__main__':
	data,labels = loadDataSet('testSet.txt')
	weights = trainModel(data,labels,tol=0.01) # 根据tol参数和trainSteps参数选择迭代终止条件
	validModel(data,labels)
	plotBestFit(weights)

