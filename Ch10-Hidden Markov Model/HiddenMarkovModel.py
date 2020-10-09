#参考https://github.com/zhulei227/ML_Notes

import numpy as np

class HMM(object):
	"""docstring for HMM"""
	def __init__(self, hiddenStatusNum=None,visibleStatusNum=None):
		self.hiddenStatusNum = hiddenStatusNum
		self.visibleStatusNum = visibleStatusNum
		self.pi = None
		self.A = None
		self.B = None
	# 10.2 概率计算方法
	def predictJointVisibleProb(self,visibleList=None,forwardType='forward'): #计算式(10.12)
		if forwardType == 'forward': #前向算法
			alpha = self.pi * self.B[:,[visibleList[0]]] #计算初值, [visibleList[0]]表示观测序列的第一个值o1
			for step in range(1,len(visibleList)):
				alpha = self.A.T.dot(alpha) * self.B[:,[visibleList[step]]]
			return np.sum(alpha)
		else:
			beta = np.ones_like(self.pi)
			for step in range(len(visibleList)-2,-1,-1):
				beta = self.A.dot(self.B[:,[visibleList[step+1]]]*beta)
			return np.sum(self.pi * self.B[:,[visibleList[0]]]*beta)
	# 10.3 学习算法
	def fitWithHiddenStatus(self,visibleList,hiddenList): #监督学习方法
		#初始化模型参数
		self.pi = np.zeros(shape=(self.hiddenStatusNum,1))
		self.A = np.zeros(shape=(self.hiddenStatusNum,self.hiddenStatusNum))
		self.B = np.zeros(shape=(self.hiddenStatusNum,self.visibleStatusNum))

		for i in range(0,len(visibleList)): #遍历所有的观测序列(以文本分词为例,可能存在多个序列并且每个序列的长度都不一样)
			visibleStatus = visibleList[i] #第i个序列
			hiddenStatus = hiddenList[i] #第i个序列对应的隐状态
			self.pi[hiddenStatus[0]] += 1 #hiddenStatus[0]表示第i个序列对应的隐状态中的第一个隐状态,这里对其初始状态进行统计以便求出初始状态概率
			for j in range(0,len(hiddenStatus)-1): #遍历隐状态(以文本分词为例,则为遍历每个分词)
				self.A[hiddenStatus[j],hiddenStatus[j+1]] += 1 #统计t时刻(这里j对应t)处于状态hiddenStatus[j]转移到状态hiddenStatus[j+1]的频数
				self.B[hiddenStatus[j],visibleStatus[j]] += 1 #统计t时刻(这里j对应t)处于状态hiddenStatus[j]并且观测为visibleStatus[j]的频数
			self.B[hiddenStatus[j+1],visibleStatus[j+1]] += 1 #最后时刻的也要记得进行统计
		# 归一化
		self.pi = self.pi / np.sum(self.pi)
		self.A = self.A / np.sum(self.A, axis=0)
		self.B = self.B / np.sum(self.B, axis=0)

	# 无监督学习方法——Baum-Welch算法
	def fitWithoutHiddenStatus(self,visibleList=None,tol=1e-5,nIter=10):
		# 初始化, 算法10.4的第(1)步
		# 如果不指定模型参数,就先进行初始化操作
		if self.pi is None:
			self.pi = np.ones(shape=(self.hiddenStatusNum,1)) + np.random.random(size=(self.hiddenStatusNum,1))
			self.pi = self.pi / np.sum(self.pi)
		if self.A is None:
			self.A = np.ones(shape=(self.hiddenStatusNum,self.hiddenStatusNum)) + np.random.random(size=(self.hiddenStatusNum,self.hiddenStatusNum))
			self.A = self.A / np.sum(self.A, axis=0)
		if self.B == None:
			self.B = np.ones(shape=(self.hiddenStatusNum,self.visibleStatusNum)) + np.random.random(size=(self.hiddenStatusNum,self.visibleStatusNum))
			self.B = self.B / np.sum(self.B, axis=0)
		# 循环迭代计算/递推, 算法10.4的第(2)步
		for _ in range(0,nIter):
			# 在迭代计算之前，先用前向和后向算法计算alpha、beta以及gamma，因为参数的更新需要用到这三个参数
			alphas = [] #用来存储所有时刻的alpha
			alpha = self.pi * self.B[:,[visibleList[0]]] #按照式(10.15)计算初值alpha1(i)
			#注意这里取列表是为了保持切片之后还是一个二维数组,从而保证*运算正确(自己可以比较与self.B[:,visibleList[0]]的区别)
			alphas.append(alpha)
			for step in range(1,len(visibleList)): #前向算法,遍历时刻t=1到t=T(注意,根据观测序列O的定义,其长度len(visibleList)即为T)进行递推
				alpha = self.A.T.dot(alpha) * self.B[:,[visibleList[step]]] #运算规则同初值计算
				alphas.append(alpha)
			betas = [] #用来存储所有时刻的beta
			beta = np.ones_like(self.pi) #最后一时刻的beta, 式(10.19)
			betas.append(beta)
			for step in range(len(visibleList)-2,-1,-1): #从后往前递推, 式(10.20), 注意这里len(visibleList)-2对应的是T-1时刻, 因为0代表t=1时刻以及len(visibleList)-1对应的是T时刻
				beta = self.A.dot(self.B[:,[visibleList[step+1]]] * beta)
				betas.append(beta)
			betas.reverse() #反向切片使得betas中存储的beta是按照t=1一直到t=T时刻的顺序
			gammas = [] #用来存储单个状态概率计算公式(10.24)
			for i in range(0,len(alphas)): #这里的i就代表时刻t
				gammas.append((alphas[i] * betas[i])[:,0])   #计算式(10.24)的分子
			gammas = np.asarray(gammas) #转化为数组

			xi = np.zeros_like(self.A) #用来存储两个状态概率的计算公式(10.26)
			for i in range(0,self.hiddenStatusNum):
				for j in range(0,self.hiddenStatusNum):
					xi_i_j = 0.0
					for t in range(0,len(visibleList)-1):
						xi_i_j += alphas[t][i][0] * self.A[i,j] * self.B[j,visibleList[t+1]] * betas[t+1][j][0]
					xi[i][j] = xi_i_j #对式(10.15)在所有时刻上进行求和从而得到算法10.4中对参数a的更新公式的分子
			loss = 0.0
			for i in range(0,self.hiddenStatusNum): #计算参数pi的绝对误差/损失并更新
				new_pi_i = gammas[0][i] #t=0时刻的gamma值即为初始状态概率分布
				loss += np.abs(new_pi_i - self.pi[i][0])
				self.pi[i] = new_pi_i
			for i in range(0,self.hiddenStatusNum): #计算参数a的绝对误差/损失并更新
				for j in range(0,self.hiddenStatusNum):
					new_a_i_j = xi[i,j] / np.sum(gammas[:,i][:-1]) #对所有时刻的gamma值进行求和,注意这里的xi[i,j]已经对时刻t进行过求和
					loss += np.abs(new_a_i_j - self.A[i,j])
					self.A[i,j] = new_a_i_j
			for j in range(0,self.hiddenStatusNum): #计算参数b的绝对误差/损失并更新
				for k in range(0,self.visibleStatusNum):
					new_b_j_k = np.sum(gammas[:,j] * (np.asarray(visibleList)==k)) / np.sum(gammas[:,j]) # * (np.asarray(visibleList)==k)是表示取观测为k的时刻, 再与gammas[:,j]相乘就表示满足观测为指定值的所有gamma
					loss += np.abs(new_b_j_k - self.B[j,k])
					self.B[j,k] = new_b_j_k
			# 归一化
			self.pi = self.pi / np.sum(self.pi)
			self.A = self.A / np.sum(self.A, axis=0)
			self.B = self.B / np.sum(self.B, axis=0)
			if loss < tol: #迭代终止条件(所有模型参数的绝对误差之和是否小于容许值)
				break
	# 10.4 预测算法
	def predictHiddenStatus(self,visibleList):
		# 算法10.5的初始化
		delta = self.pi * self.B[:,[visibleList[0]]] 
		psi = [[0] * self.hiddenStatusNum]
		# 算法10.5的递推步骤
		for visibleIndex in range(1,len(visibleList)): #计算所有时刻所有可能的隐状态的所有可能的路径的概率
			newDelta = np.zeros_like(delta)
			newPsi = []
			for i in range(0,self.hiddenStatusNum): # 遍历t时刻的每个可能的隐状态也即1≤i≤N或隐状态集合Q
				bestPreIndex_i = -1 #初始化
				bestPreIndexValue_i = 0
				delta_i = 0
				# 遍历每个隐状态即1≤j≤N，求delta_t(i)和psi_t(i)
				for j in range(0,self.hiddenStatusNum):
					delta_i_j = delta[j][0] * self.A[j,i] * self.B[i,visibleList[visibleIndex]]
					if delta_i_j > delta_i:
						delta_i = delta_i_j
					bestPreIndexValue_i_j = delta[j][0] * self.A[j,i]
					if bestPreIndexValue_i_j > bestPreIndexValue_i:
						bestPreIndexValue_i = bestPreIndexValue_i_j
						bestPreIndex_i = j

				newDelta[i,0] = delta_i #将t时刻所有可能的隐状态中的最大路径的概率赋值给newDelta
				newPsi.append(bestPreIndex_i)
			delta = newDelta #更新delta的值
			psi.append(newPsi) #存储每个时刻的psi
		bestHiddenStatus = [np.argmax(delta)] #终止步骤，最优路径的终结点
		for psiIndex in range(len(visibleList)-1,0,-1): #根据psi和终结点进行最优路径回溯
			nextStatus = psi[psiIndex][bestHiddenStatus[-1]]
			bestHiddenStatus.append(nextStatus)
		bestHiddenStatus.reverse()
		return bestHiddenStatus

if __name__ == '__main__':
	pi = np.array([[0.2], [0.4], [0.4]]) #初始概率分布
	A = np.array([[0.5, 0.2, 0.3], #状态转移矩阵
				  [0.3, 0.5, 0.2],
				  [0.2, 0.3, 0.5]])
	B = np.array([[0.5, 0.5], #观测矩阵
				  [0.4, 0.6],
				  [0.7, 0.3]])

	hmm = HMM()
	hmm.pi = pi
	hmm.A = A
	hmm.B = B

	print(hmm.predictJointVisibleProb([0, 1, 0], forwardType='forward'))
	print(hmm.predictJointVisibleProb([0, 1, 0], forwardType='backward'))

	O = [  #6个观测序列
		[1, 2, 3, 0, 1, 3, 4],
		[1, 2, 3],
		[0, 2, 4, 2],
		[4, 3, 2, 1],
		[3, 1, 1, 1, 1],
		[2, 1, 3, 2, 1, 3, 4]
	]
	I = O #对应的隐状态
	# 有监督学习
	hmm = HMM(hiddenStatusNum=5, visibleStatusNum=5)
	hmm.fitWithHiddenStatus(visibleList=O, hiddenList=I)
	print(hmm.pi)
	print(hmm.A)
	print(hmm.B)

	# 无监督学习
	hmm = HMM(hiddenStatusNum=5, visibleStatusNum=5)
	hmm.fitWithoutHiddenStatus(O[0] + O[1] + O[2] + O[3] + O[4] + O[5])
	print(hmm.pi)
	print(hmm.A)
	print(hmm.B)

	#预测算法
	print(hmm.predictHiddenStatus([0,1,0]))