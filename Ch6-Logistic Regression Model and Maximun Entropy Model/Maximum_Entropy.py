from collections import defaultdict
import math

class maxEntropy(object):
    def __init__(self):
        self.feats = defaultdict(int) #用defaultdict防止访问不存在的键时出现报错
        self.trainSet = [] #训练集
        self.labels = set() #标签

    def loadDataSet(self,filename): #载入数据
        for line in open(filename):
            curLine = line.strip().split()
            if len(curLine) < 2: #至少需要两列
                continue
            label = curLine[0] #第一列是标签
            self.labels.add(label)
            for f in set(curLine[1:]): 
            #统计样本(x,y)出现的频数，y：label，x：采用unigram得到的分词，
            #也就是将每个短语划分为一个一个为的单词
                self.feats[(label,f)] += 1
            self.trainSet.append(curLine)

    def _initParams(self): #初始化参数
        self.size = len(self.trainSet) #数据集样本个数
        self.M = max([len(sample)-1 for sample in self.trainSet]) # GIS算法中的M参数
        self.Ep_ = [0.0] * len(self.feats) # 初始化特征函数关于经验分布的期望值，size为特征函数的个数
        for i,f in enumerate(self.feats): #i为序号（表示第i个特征函数），注意这里的特征函数是针对标签和某个分词共同出现与否这件事也就是f，它也是feats中的键
            self.Ep_[i] = float(self.feats[f]) / float(self.size) #按照p82最小面的式子求Ep_(fi)
            #联合分布P(X,Y)的经验分布(考虑到特征函数的取值，也是Ep_(f))
            self.feats[f] = i # 与i对应的特征函数
        self.w = [0.0] * len(self.feats) # 初始化权重向量为0
        self.lastW = self.w #将上一轮迭代得到的权重传递给变量lastW

    def train(self, max_iter=1000): #通过训练得到式（6.22）中的wi参数
        self._initParams() #初始化参数
        for i in range(max_iter): #迭代
            print('iter %d ...' % (i + 1))
            self.Ep = self.calcEp() # 按照式p83最上面的式子计算Ep(fi)
            self.lastW = self.w[:]
            for i, w in enumerate(self.w): # 更新wi
                delta = 1.0 / self.M * math.log(self.Ep_[i] / self.Ep[i]) #式（6.34）
                self.w[i] += delta
            # print(self.w)
            if self._convergence(self.lastW, self.w):# 判断是否收敛
                break

    def calcEp(self): #按照p83最上面的式子计算Ep(f)
        Ep = [0.0] * len(self.feats) #Ep的个数也是与特征函数个数相同，初始化Ep，存储与每个特征函数对应的Ep
        for data in self.trainSet: #遍历训练数据
            features = data[1:] #提取分词特征
            # calculate P(y|x)
            prob = self.calProb(features)
            for f in features:
                for w, l in prob:
                    # only focus on features from training data.
                    if (l, f) in self.feats:
                        # get feature id
                        idx = self.feats[(l, f)]
                        # sum(1/N * f(y,x)*P(y|x)), P(x) = 1/N
                        Ep[idx] += w * (1.0 / self.size)
        return Ep

    def calProb(self, features): #计算P(y|x) 式(6.22,6.23)
        weights = [(self.probWeights(features, l), l) for l in self.labels]
        # 对单个样本中所有的label（outdoor和indoor）与所有的特征（单个分词）的组合进行遍历，求得式(6.22)的分子部分
        Zw = sum([w for w, l in weights]) #规范化因子Zw(x)
        prob = [(w / Zw, l) for w, l in weights]
        return prob

    def probWeights(self,features,label):
        weight = 0.0
        for feat in features:
            if (label,feat) in self.feats:
                weight += self.w[self.feats[(label,feat)]]
        return math.exp(weight)

    def _convergence(self, lastW, w):
        for w1, w2 in zip(lastW, w):
            if abs(w1 - w2) >= 0.01:
                return False
        return True

    def predict(self, input): #预测
        features = input.strip().split() #特征分词
        prob = self.calProb(features) #计算
        prob.sort(reverse=True)
        return prob

if __name__ == '__main__':
    model = maxEntropy()
    model.loadDataSet(filename='data.txt')
    print(model.trainSet)
    print(model.feats)
    model.train()
    print(model.predict('Sunny'))





