import random
import numpy as np

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append(list(map(float,[line[0], line[1]])))
        labelMat.append(float(line[-1]))

    return dataMat,labelMat

def selectJrand(i,m): #随机选取第二个变量
    j = i
    while j == i:
        j = random.uniform(0,m)
    return int(j)

def clipAlpha(aj,H,L): #对alpha值进行修剪
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def SMO_simple(dataMat,label,C,toler,maxIter):
    dataMat = np.mat(dataMat)
    label = np.mat(label).transpose()
    b = 0
    m,n = dataMat.shape
    alphas = np.mat(np.zeros((m,1)))
    iterNum = 0
    while iterNum < maxIter: #只有在所有数据集上遍历maxIter次且alphas值不发生改变时，程序才会终止
        alphaPairsChanged = 0
        for i in range(m):
            gxi = (np.multiply(alphas,label)).T * (dataMat * dataMat[i,:].T) + b #式（7.104），xi的预测值
            Ei = gxi - label[i] # 式（7.105），对xi的预测值与真实值yi之差
            # 按照蓝书p129最上面那段话，先遍历所有满足0 < alpha[i] < C的点，如果误差太大（也就是不满足KKT条件第二条），这说明需要进行参数更新
            # 具体而言，从KKT条件进行判断：
            # yi*g(i) >= 1 and alpha[i] = 0 (正确分类的样本点)
            # yi*g(i) == 1 and 0 < alpha[i] < C (间隔边界上的样本点)
            # yi*g(i) <= 1 and alpha[i] = C (间隔边界和分离超平面之间的样本点)
            if (label[i] * Ei < -toler and alphas[i] < C) or (label[i] * Ei > toler and alphas[i] > C):
                j = selectJrand(i,m) #随机选取第二个变量
                gxj = (np.multiply(alphas,label)).T * (dataMat * dataMat[j,:].T) + b #xi的预测值
                Ej = gxj - label[j] #xi的误差
                alphaIold = alphas[i].copy() #记录更新前两个变量的值
                alphaJold = alphas[j].copy()

                if label[i] != label[j]: #按照p126求变量alphas[j]的取值范围
                    L = max(0,alphaJold-alphaIold)
                    H = min(C,C+alphaJold-alphaIold)
                else:
                    L = max(0,alphaJold+alphaIold-C)
                    H = min(C,alphaJold+alphaIold)
                if L == H: #如果两者相等，说明变量取值只能在边界上，就没有任何的优化空间，则跳过本次循环
                    print('L==H')
                    continue
                eta = dataMat[i,:] * dataMat[i,:].T + dataMat[j,:] * dataMat[j,:].T - 2.0 * dataMat[i,:] * dataMat[j,:].T #按照式（7.107）计算
                if eta <= 0: #如果eta<=0则跳出本次循环（这里只考虑简单的情况）
                    print('eta<=0')
                    continue
                alphas[j] += label[j] * (Ei - Ej) / eta #按照式（7.106）更新得到alpha[j]未修剪的值
                alphas[j] = clipAlpha(alphas[j],H,L) #对alpha[j]进行修剪
                if abs(alphas[j] - alphaJold) < 0.00001: #如果alpha[j]更新前后变化太小，跳过本次循环
                    print('alpha[j]变化太小')
                    continue
                alphas[i] += label[i] * label[j] * (alphaJold - alphas[j]) #按照式（7.109）更新alpha[i]
                # 按照式（7.115~7.116）计算b1和b2
                b1 = -Ei - label[i] * (dataMat[i,:]*dataMat[i,:].T) * (alphas[i]-alphaIold) - label[j] * (dataMat[j,:]*dataMat[i,:].T) * (alphas[j]-alphaJold) + b
                b2 = -Ej - label[i] * (dataMat[i,:]*dataMat[j,:].T) * (alphas[i]-alphaIold) - label[j] * (dataMat[j,:]*dataMat[j,:].T) * (alphas[j]-alphaJold) + b
                # 判断如何更新b的值
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                print('第%d次迭代, 样本%d, alpha优化次数为:%d' % (iterNum,i,alphaPairsChanged))
        if alphaPairsChanged == 0: #检查alphas的值是否有更新，如果有更新则将iter设为0之后继续运行程序。
            iterNum += 1
        else:
            iterNum = 0
        print('迭代次数:%d' % iterNum)

    return b,alphas

if __name__ == '__main__':
    data,label = loadDataSet('testSet.txt')
    b,alphas = SMO_simple(data,label,0.6,0.001,40)
    print(b,'\n')
    print(alphas[alphas>0])
    for i in range(len(data)): #输出支持向量
        if alphas[i] > 0:
            print(data[i], label[i])

