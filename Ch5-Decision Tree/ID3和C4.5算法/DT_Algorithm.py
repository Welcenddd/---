# 参考《机器学习实战》 Peter
# 这里有两个数据集，一个是蓝皮书上的信贷问题，另一个是斧头书上的眼镜分类问题
# 可以根据需要修改特征选择依据，可以选择ID3算法和C4.5算法
# 本程序还给出了分类树的可视化实现
# 由于数据集过于简单，没有考虑剪枝，这个有条件的话以后再实现

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文

def createDataSet():
    # 参考p59表5.1的数据
    # 最右边一列表示标签类别，即是否批准贷款申请；前四列为特征
    dataSet = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好',    '否'],
               ['青年', '是', '否', '好',	'是'],
               ['青年', '是', '是', '一般',	'是'],
               ['青年', '否', '否', '一般',	'否'],
               ['中年', '否', '否', '一般',	'否'],
               ['中年', '否', '否', '好',	'否'],
               ['中年', '是', '是', '好',	'是'],
               ['中年', '否', '是', '非常好','是'],
               ['中年', '否', '是', '非常好','是'],
               ['老年', '否', '是', '非常好','是'],
               ['老年', '否', '是', '好',    '是'],
               ['老年', '是', '否', '好',	'是'],
               ['老年', '是', '否', '非常好','是'],
               ['老年', '否', '否', '一般',	'否']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']


    return dataSet, labels

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # fltLine = list(map(float,curLine))
        dataMat.append(curLine)
    labels = ['age', 'prescript','astigmatic','tearRate']
    return dataMat,labels

def calcEntropy(dataSet): #计算数据的熵，按照式（5.7）
    numSamples = len(dataSet) #样本数量
    featVal = {} #新建空字典，用于统计各个标签类别出现的次数
    for spl in dataSet: #遍历每个样本
        if spl[-1] not in featVal.keys(): #防止访问不存在的键时出现报错
            featVal[spl[-1]] = 0
        featVal[spl[-1]] += 1

    entropy = 0.0
    for feat in featVal.keys(): #根据式（5.7）计算经验熵
        prob_feat = featVal[feat] / numSamples
        entropy -= prob_feat * np.log2(prob_feat)

    return entropy

def calcGini(dataSet):
    numSamples = len(dataSet)  # 样本数量
    featVal = {}  # 新建空字典，用于统计各个标签类别出现的次数
    for spl in dataSet:  # 遍历每个样本
        if spl[-1] not in featVal.keys():  # 防止访问不存在的键时出现报错
            featVal[spl[-1]] = 0
        featVal[spl[-1]] += 1

    gini = 0.0
    for feat in featVal.keys():  # 根据式（5.7）计算经验熵
        prob_feat = featVal[feat] / numSamples
        gini += prob_feat * (1 - prob_feat)

    return gini

def splitDataSet(dataSet,axis,value): #将数据集按照指定的特征和特征值进行划分
    retDataSet = []
    for spl in dataSet:
        if spl[axis] == value:
            reducedFeatVec = spl[:axis]  #注意spl[:0]会返回一个空集
            reducedFeatVec.extend(spl[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet,labels,algorithmType='CART'):
    numFeat = len(dataSet[0]) - 1 # 减1是因为要求输入的数据集中最后一列元素为实例标签，也就是说只有前面几列才是特征
    if algorithmType == 'ID3':
        baseEntropy = calcEntropy(dataSet)
        bestInfoGain = 0.0
    elif algorithmType == 'C4.5':
        baseEntropy = calcEntropy(dataSet)
        bestInfoGain = 0.0
        bestInfoGainRate = 0.0
    elif algorithmType == 'CART':
        bestGini = np.inf
    else:
        return TypeError

    bestFeat = -1
    for feat in range(numFeat):
        newEntropy = 0.0
        DaEntropy = 0.0
        gini = 0.0
        featVec = [spl[feat] for spl in dataSet]
        uniqueFeatVec = set(featVec)

        # CART决策树
        if algorithmType == 'CART':
            for val in uniqueFeatVec:
                subDataSet = splitDataSet(dataSet,feat,val)
                prob = len(subDataSet) / len(dataSet)
                subProb = len(splitDataSet(subDataSet,-1,subDataSet[0][-1])) / float(len(subDataSet))
                gini += prob * (1 - subProb**2 - (1-subProb)**2)
            if gini < bestGini:
                bestGini = gini
                bestFeat = feat
            print('特征%s的最小基尼指数为%.2f' % (labels[feat],bestGini))
        elif algorithmType == 'ID3':
            ###########普通决策树##############
            for val in uniqueFeatVec:
                subDataSet = splitDataSet(dataSet,feat,val)
                prob = len(subDataSet) / len(dataSet)
                newEntropy += prob * calcEntropy(subDataSet)
                DaEntropy -= prob * np.log2(prob)
            infoGain = baseEntropy - newEntropy
            # ID3算法，按照信息增益进行特征选择
            if infoGain > bestInfoGain:
                bestFeat = feat
                bestInfoGain = infoGain
            print('特征%s的最大信息增益为%.2f' % (labels[feat],bestInfoGain))
        elif algorithmType == 'C4.5':
            # C4.5算法，按照信息增益比进行特征选择
            for val in uniqueFeatVec:
                subDataSet = splitDataSet(dataSet,feat,val)
                prob = len(subDataSet) / len(dataSet)
                newEntropy += prob * calcEntropy(subDataSet)
                DaEntropy -= prob * np.log2(prob)
                # print('训练数据关于特征A的值的熵: %.2f' % DaEntropy)
            infoGain = baseEntropy - newEntropy
            if DaEntropy == 0.0:
                continue
            infoGainRate = infoGain / DaEntropy
            if infoGainRate > bestInfoGainRate:
                bestFeat = feat
                bestInfoGainRate = infoGainRate
            print('特征%s的最大信息增益比为%.2f' % (labels[feat], bestInfoGainRate))
    return bestFeat

def majorVote(classList):
    classCount = {}
    for cls in classList:
        if cls not in classCount.keys():
            classCount[cls] = 0
        classCount[cls] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels,featlabels,algorithmType='CART'):
    classList = [spl[-1] for spl in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorVote(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet,labels,algorithmType)
    bestFeatLabel = labels[bestFeat]
    featlabels.append(bestFeatLabel)
    print(u"此时最优索引为：" + (bestFeatLabel))
    myTree = {bestFeatLabel:{}}
    del labels[bestFeat]
    bestFeatVals = [spl[bestFeat] for spl in dataSet]
    uniqueBestFeatVals = set(bestFeatVals)
    for val in uniqueBestFeatVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = createTree(splitDataSet(dataSet,bestFeat,val),subLabels,featlabels,algorithmType)

    return myTree

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=dict(arrowstyle='<-')) # va和ha分别表示注释文本的左端和低端对齐到指定位置

def plotMidText(cntrPt, parentPt, txtString):
    # 计算标注位置（箭头起始位置的中点处）
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    # 设置结点格式boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yoff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        # 如果是叶结点，绘制叶结点，并标注有向边属性值
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD

def createPlot(inTree):
    # 创建fig
    fig = plt.figure(1, facecolor="white")
    # 清空fig
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()

def getNumLeafs(myTree):
    # 递归获取树的叶结点总数
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #提取划分类别的标签作为关键字
    secondDict = myTree[firstStr] #得到该类别对应的子树
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    # 递归获取树的深度
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def classify(inputTree,featLabels,testVec):
    # 分类测试函数
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    # 蓝皮书训练例子
    # dataSet,labels = createDataSet()
    # myTree = createTree(dataSet,labels,[])
    # createPlot(myTree)

    # 斧头书的例子
    dataSet,labels = loadDataSet('lenses.txt')
    # print(dataSet)
    myTree = createTree(dataSet,labels,[],algorithmType='CART')
    createPlot(myTree)
