# 参考《机器学习实战》Peter
# 数据集一共有六个样本，每个样本表示一条评论，
# 对应的标签为0表示不是侮辱性评论，为1表示是侮辱性评论，并且每条评论已经做好了分词

import numpy as np

def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0, 1, 0, 1, 0, 1]
    return postingList, labels

def createVocabList(dataSet):
    VocabList = [] # 创建空列表
    for data in dataSet:
        VocabList.extend(set(data)) #将所有的分词组成一个集合
    return VocabList

def setOfWords2Vec(vocabList,inputSet):
    ReturnVec = [0] * len(vocabList) #初始化词向量，0表示inputSet中的单词不在vocabList里，1表示存在
    for word in inputSet:
        if word in vocabList:
            ReturnVec[vocabList.index(word)] = 1 #vocabList.index(word)表示寻找word在vocabList中的索引

    return ReturnVec

def trainModel(trainMat,trainLabels):
    numSamples = len(trainMat)
    numWords = len(trainMat[0])
    P1 = sum(trainLabels) / numSamples #Y=1的先验概率

    P0Num = np.ones(numWords)#lambda=1，采用拉普拉斯平滑
    P1Num = np.ones(numWords)

    P0Denom = 2.0 # Sj=2
    P1Denom = 2.0
    for i in range(numSamples): #计算条件概率P(X|Y)
        if trainLabels[i] == 1: #计算条件概率P(X|Y=1)
            P1Num += trainMat[i] #统计侮辱类评论中的单词出现次数
            P1Denom += sum(trainMat[i]) #侮辱类评论的单词总数
        else:                    #计算条件概率P(X|Y=0)
            P0Num += trainMat[i] #统计非侮辱类评论中的单词出现次数
            P0Denom += sum(trainMat[i]) #非侮辱类评论的单词总数

    PY1_X = np.log(P1Num/P1Denom) #取对数防止数值下溢
    PY0_X = np.log(P0Num/P0Denom)

    return PY0_X,PY1_X,P1

def classify(PY0_X,PY1_X,P1,testVec):
    # 确定实例testVec的类
    P1 = sum(testVec * PY1_X) + np.log(P1)
    P0 = sum(testVec * PY0_X) + np.log(1-P1)
    if P1 > P0:
        return 1
    return 0


if __name__ == '__main__':
    postingList, labels = loadDataSet()
    vocabList = createVocabList(postingList)
    trainMat = []
    for comment in postingList:
        trainMat.append(setOfWords2Vec(vocabList,comment))
    PY0_X, PY1_X, P1 = trainModel(trainMat,labels)
    testComment = ['love', 'my', 'dalmation']
    testVec = setOfWords2Vec(vocabList,testComment)
    if classify(PY0_X,PY1_X,P1,testVec):
        print(testComment,'是侮辱性评论')
    else:
        print(testComment,'不是侮辱性评论')

