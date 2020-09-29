import numpy as np
import random
import matplotlib.pyplot as plt

class optStruct: #创建一个数据结构存储部分数据
    def __init__(self,dataMat,classLabel,C,toler):
        self.X = dataMat
        self.labelMat = classLabel
        self.C = C
        self.tol = toler
        self.m = dataMat.shape[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.E = np.mat(np.zeros((self.m,2)))

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append(list(map(float,[line[0], line[1]])))
        labelMat.append(float(line[-1]))

    return dataMat,labelMat

def calcEk(oS,k): #计算误差
    gxk = (np.multiply(oS.alphas,oS.labelMat)).T * (oS.X * oS.X[k,:].T) + oS.b #式（7.104），xi的预测值
    Ek = gxk - oS.labelMat[k]
    return Ek

def selectJrand(i,m): #随机选取变量
    j = i
    while j == i:
        j = random.uniform(0,m)
    return int(j)

def selectJ(i,oS,Ei): #按照p129的第2点中的原则选取第二个变量
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.E[i] = [1,Ei]
    validEList = np.nonzero(oS.E[:,0].A)[0]
    if len(validEList) > 1:
        for k in validEList: #遍历非零误差，寻找使|Ei-Ej|最大的j和Ej
            if k == i:
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return int(maxK),Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return int(j),Ej

def clipAlpha(aj,H,L): #对alpha进行修剪
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def updateEk(oS,k): #每更新一次alpha，就将对应的误差E更新
    Ek = calcEk(oS,k)
    oS.E[k] = [1,Ek]

def innerLoop(i,oS): #内层循环，内容与简化版的SMO算法基本相同，这里不再做注释
    Ei = calcEk(oS,i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > oS.C):
        j,Ej = selectJ(i,oS,Ei)  # 选取第二个变量
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0

        eta = oS.X[i,:] * oS.X[i,:].T + oS.X[j,:] * oS.X[j,:].T - 2.0 * oS.X[i,:] * oS.X[j,:].T
        if eta <= 0:
            print("eta <= 0")
            return 0
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[i, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0

def SMO_complete(dataMat,classLabel,C,toler,maxIter):
    oS = optStruct(np.mat(dataMat),np.mat(classLabel).transpose(),C,toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet): #在间隔边界遍历和整个训练集遍历上切换循环
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerLoop(i,oS)
                print('全样本遍历:第%d次迭代, 样本:%d, alpha优化次数:%d' % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0] #间隔边界上的支持向量点
            for i in nonBoundIs:
                alphaPairsChanged += innerLoop(i,oS)
                print('间隔边界上的遍历:第%d次迭代, 样本:%d, alpha优化次数:%d' % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('迭代次数:%d' % iter)
    return oS.b, oS.alphas

def calcWs(alphas, data, classLabel): #计算权重
    X = np.mat(data)
    labelMat = np.mat(classLabel).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def plotData(data,classLabel,w,b): #绘制数据散点图和决策边界
    dataPlus = []
    dataMinus = []
    # 寻找正样本点和负样本点
    for i in range(len(data)):
        if classLabel[i] > 0:
            dataPlus.append(data[i])
        else:
            dataMinus.append(data[i])
    dataPlus = np.array(dataPlus)
    dataMinus = np.array(dataMinus)
    plt.scatter(dataPlus.T[0],dataPlus.T[1],s=30,alpha=0.7)
    plt.scatter(dataMinus.T[0], dataMinus.T[1], s=30, alpha=0.7)

    x1 = max(data)[0]
    x2 = min(data)[0]

    a1,a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    #根据x.w + b = 0得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        # 支持向量机的点
        if(abs(alpha) > 0):
            x, y = data[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()



if __name__ == '__main__':
    data,label = loadDataSet('testSet.txt')
    b,alphas = SMO_complete(data,label,0.6,0.001,40)
    ws = calcWs(alphas,data,label)
    plotData(data,label,ws,b)

    errorCount = 0
    for i in range(len(data)):
        if np.sign(data[i]*np.mat(ws)+b) != np.sign(label[i]):
            errorCount += 1
    print('错误率为%.2f%%' % (errorCount/len(data)*100))



