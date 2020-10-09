import numpy as np

def loadSimpleData():
    dataMat = [[1.0, 2.1],
               [1.5, 1.6],
               [1.3, 1.0],
               [1.0, 1.0],
               [2.0, 1.0]]
    classLabel = [1.0, 1.0, -1.0, -1.0, 1.0]
    return np.mat(dataMat),classLabel

def stumpClassify(dataMat,dim,threshVal,threshIneq):
    retArray = np.ones((dataMat.shape[0],1))
    if threshIneq == 'lt': #小于等于切分点的取-1
        retArray[dataMat[:,dim] <= threshVal] = -1.0
    else: #大于切分点的取1
        retArray[dataMat[:,dim] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabel,D): #算法8.1的第(2)(a)步，得到基本分类器
    dataMat = np.mat(dataArr) #为方便数据处理，转换为矩阵形式
    labelMat = np.mat(classLabel).T
    m,n = dataMat.shape
    numSteps = 10.0 #需要考虑的基本切分点的个数
    bestStump = {} #创建树桩对应的空字典
    bestClassEst = np.mat(np.zeros((m,1))) #初始化基本分类器
    minError = float('inf')
    for i in range(n): #遍历所有的特征
        rangeMin = dataMat[:,i].min() #特征列的最小值和最大值
        rangeMax = dataMat[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps #切分点的移动步长
        for j in range(-1,int(numSteps)+1): #遍历每个特征的所有切分点
            for inEqual in ['lt','gt']: #遍历大于切分点就赋值-1和小于切分点就赋值-1两种切分情况
                threshVal = rangeMin + float(j) * stepSize #切分点
                predictedVals = stumpClassify(dataMat,i,threshVal,inEqual)
                errArr = np.mat(np.ones((m,1))) #初始化误差矩阵/向量
                errArr[predictedVals == labelMat] = 0 #预测值与标签相同的误差为零
                weightedError = D.T * errArr #加权误差
                print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' % (i,threshVal,inEqual,weightedError))
                if weightedError < minError: #求最小的加权误差以及对应的切分方式、最好的基本分类器
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inEqual

    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabel,numIter=40): #模型训练
    weakClassifierArr = [] #创建空列表来存储弱分类器
    m = dataArr.shape[0]
    D = np.mat(np.ones((m,1))/m) #初始化权重向量
    aggClassEst = np.mat(np.zeros((m,1))) #初始化分类器
    for i in range(numIter):
        bestStump,error,classEst = buildStump(dataArr,classLabel,D) #通过训练数据得到本轮迭代的基本分类器
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) #本轮迭代中的基本分类器Gm(x)的系数alpha
        bestStump['alpha'] = alpha #添加到树桩字典中
        weakClassifierArr.append(bestStump) #添加到弱分类器列表中
        expon = np.multiply(-1*alpha*np.mat(classLabel).T,classEst) #式（8.4）的指数函数的指数部分
        D = np.multiply(D,np.exp(expon)) #式（8.4）的分子
        D = D / D.sum() #更新训练数据集的权值分布，即式（8.4）
        aggClassEst += alpha * classEst #式（8.6），基本分类器的线性组合
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabel).T, np.ones((m,1))) #计算分类错误的样本
        errorRate = aggErrors.sum() / m #计算分类错误率
        if errorRate == 0.0: #迭代终止条件2：误差率为0
            break
    return weakClassifierArr,aggClassEst

def adaBoostClassify(dataToClass,classifierArr): #分类函数
    dataMat = np.mat(dataToClass)
    m = dataMat.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)): #遍历所有的基本分类器
        classEst = stumpClassify(dataMat,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq']) #求每个分类器分类结果
        aggClassEst += classifierArr[i]['alpha'] * classEst #累加/线性组合得到所有分类器的最终表决结果
        print(aggClassEst) #输出每次的表决以便观察(有多少个基本分类器就是输出多少次)
    return np.sign(aggClassEst) #二分类表决

if __name__ == '__main__':
    data,label = loadSimpleData()
    D = np.mat(np.ones((5,1))/5)

    classifierArr,aggClassEst = adaBoostTrainDS(data,label,9)
    print(adaBoostClassify([[5.0,5.0],[0.0,0.0]],classifierArr))

