# 参考《机器学习实战》 Peter
# 分类树模型

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] == value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] != value)[0], :]
    return mat0, mat1

def chooseBestSplit(dataSet,leafType,errType,ops):
    tolS = ops[0] # 迭代停止条件1——允许误差
    tolN = ops[1] # 迭代停止条件2——允许的划分后的数据集大小（个数）
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: # 如果每个样本都是相同的类型，返回它们的均值
        return None, leafType
    m,n = dataSet.shape
    G = errType(dataSet) # 原始数据的基尼指数
    bestG = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in  range(n-1): # 遍历每个特征和相应的特征值，求最好的划分参数
        for splitValue in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitValue)
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1) #划分之后的误差
            if newS < bestS: #如果小于最小的误差，更新参数
                bestS = newS
                bestIndex = featIndex
                bestValue = splitValue
    if S - bestS < tolS: # 如果误差减小不是很大，返回数据的均值
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue) # 划分数据
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN: #划分后如果数据太少，返回均值
        return None,leafType(dataSet)
    return bestIndex,bestValue

def regLeaf(dataSet): # 计算数据的均值
    return np.mean(dataSet[:,-1])

def regErr(dataSet): # 计算数据的总方差
    return np.var(dataSet[:,-1]) * dataSet.shape[0]

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: # 如果特征为空，说明已经划分到叶子结点
        return val
    retTree = {}
    retTree['spInd'] = feat # 分割特征
    retTree['spVal'] = val # 分割值
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops) # 递归创建子树
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def plotDataSet(filename):
    # 绘制数据分布散点图
    dataMat = loadDataSet(filename)
    m = len(dataMat)
    xcord = []
    ycord = []

    for i in range(m):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])

    fig = plt.figure()
    ax =  fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='b',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def isTree(object):
    return (type(object).__name__ == 'dict')

def getMean(tree): #递归求每个含有两个叶结点的最小子树的平均值
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree,testData):
    if testData.shape[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1]-tree['left'],2)) + np.sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    fileName = 'ex2.txt'
    plotDataSet(fileName) #绘制数据散点图
    myDat = np.mat(loadDataSet(fileName))
    myTree = createTree(myDat) # 创建树
    print('原始树结构为：\n', myTree)
    testFileName = 'ex2test.txt'
    testData = np.mat(loadDataSet(testFileName))
    print('剪枝后的树为：\n', prune(myTree, testData))



