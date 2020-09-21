# 参考《机器学习实战》 Peter
# CART算法

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
from itertools import combinations

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

def featureSplit(features):
    count = len(features)#特征值的个数
    if count < 2:
        print("please check sample's features,only one feature value")
        return -1
    # 由于需要返回二分结果，所以每个分支至少需要一个特征值，所以要从所有的特征组合中选取1个以上的组合
    # itertools的combinations 函数可以返回一个列表选多少个元素的组合结果，例如combinations(list,2)返回的列表元素选2个的组合
    # 我们需要选择1-（count-1）的组合
    featureIndex = list(range(count))
    featureIndex.pop(0)
    combinationsList = []
    resList=[]
    # 遍历所有的组合
    for i in featureIndex:
        temp_combination = list(combinations(features, len(features[0:i])))
        combinationsList.extend(temp_combination)
        combiLen = len(combinationsList)
    # 每次组合的顺序都是一致的，并且也是对称的，所以我们取首尾组合集合
    # zip函数提供了两个列表对应位置组合的功能
    resList = zip(combinationsList[0:int(combiLen/2)], combinationsList[combiLen-1:int(combiLen/2)-1:-1])
    return resList

def splitDataSet(dataSet, axis, values):
    retDataSet = []
    if len(values) < 2:
        for featVec in dataSet:
            if featVec[axis] == values[0]:#如果特征值只有一个，不抽取当选特征
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    else:
        for featVec in dataSet:
            for value in values:
                if featVec[axis] == value:#如果特征值多于一个，选取当前特征
                    retDataSet.append(featVec)

    return retDataSet

# 返回最好的特征以及二分特征值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #
    bestGiniGain = 1.0
    bestFeature = -1
    bestBinarySplit=()
    for i in range(numFeatures):        #遍历特征
        featList = [example[i] for example in dataSet]#得到特征列
        uniqueVals = list(set(featList))       #从特征列获取该特征的特征值的set集合
        # 三个特征值的二分结果：
        # [(('young',), ('old', 'middle')), (('old',), ('young', 'middle')), (('middle',), ('young', 'old'))]
        for split in featureSplit(uniqueVals):
            GiniGain = 0.0
            if len(split)==1:
                continue
            (left,right)=split
            # 对于每一个可能的二分结果计算gini增益
            # 左增益
            left_subDataSet = splitDataSet(dataSet, i, left)
            left_prob = len(left_subDataSet)/float(len(dataSet))
            GiniGain += left_prob * calcGini(left_subDataSet)
            # 右增益
            right_subDataSet = splitDataSet(dataSet, i, right)
            right_prob = len(right_subDataSet)/float(len(dataSet))
            GiniGain += right_prob * calcGini(right_subDataSet)
            if (GiniGain <= bestGiniGain):       #比较是否是最好的结果
                bestGiniGain = GiniGain         #记录最好的结果和最好的特征
                bestFeature = i
                bestBinarySplit=(left,right)
    return bestFeature,bestBinarySplit

def majorVote(classList):
    classCount = {}
    for cls in classList:
        if cls not in classCount.keys():
            classCount[cls] = 0
        classCount[cls] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # print dataSet
    if classList.count(classList[0]) == len(classList):
        return classList[0]#所有的类别都一样，就不用再划分了
    if len(dataSet) == 1: #如果没有继续可以划分的特征，就多数表决决定分支的类别
        # print "here"
        return majorVote(classList)
    bestFeat,bestBinarySplit = chooseBestFeatureToSplit(dataSet)
    # print bestFeat,bestBinarySplit,labels
    bestFeatLabel = labels[bestFeat]
    if bestFeat==-1:
        return majorVote(classList)
    myTree = {bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    for value in bestBinarySplit:
        subLabels = labels[:]       # #拷贝防止其他地方修改
        if len(value)<2:
            del(subLabels[bestFeat])
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
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
    # myTree = createTree(dataSet,labels)
    # createPlot(myTree)

    # 斧头书的例子
    dataSet,labels = loadDataSet('lenses.txt')
    myTree = createTree(dataSet, labels)
    createPlot(myTree)
