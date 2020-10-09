#将隐马尔可夫模型用于中文分词
import HiddenMarkovModel as HMM
import numpy as np

visibleSeqs = []
hiddenSeqs = []
char2idx = {}
idx2hidden = {0:'B', 1:'M', 2:'E', 3:'S'}
count = 0
for line in open('./data/people_daily_mini.txt', encoding='utf8'):
	visibleSeq = []
	hiddenSeq = []
	arrs = line.strip().split(' ')
	for item in arrs:
		if len(item)==1: #长度为1说明是单字
			hiddenSeq.append(3)
		elif len(item)==2: #长度为2说明是一个双字组成的词
			hiddenSeq.extend([0,2]) #注意这里不采用append
		else: #其他则可归纳由词首尾和若干个词中组成
			hiddenSeq.extend([0]+[1]*(len(item)-2)+[2])
		for c in item: #给每个分词进行序号标记/转
			if c in char2idx:
				visibleSeq.append(char2idx[c])
			else:
				char2idx[c] = count
				visibleSeq.append(count)
				count += 1
		visibleSeqs.append(visibleSeq)
		hiddenSeqs.append(hiddenSeq)

hmm = HMM.HMM(hiddenStatusNum=4,visibleStatusNum=len(char2idx))
hmm.fitWithHiddenStatus(visibleSeqs,hiddenSeqs) #已知隐状态的模型训练

def seg(vis,hid):
	rst = []
	for i in range(0,len(hid)):
		if hid[i] in [2,3]: #如果是2或者3表示隐状态为词尾或者单字，那么添加完对应的观测序列之后应该在后面添加空格表示分词
			rst.append(vis[i])
			rst.append('   ')
		else:
			rst.append(vis[i])
	return ''.join(rst)

# seq是给定的观测序列，然后调用predictHiddenStatus预测对应的隐状态
seq = '我和我的祖国，一刻也不能分离'
hid = hmm.predictHiddenStatus([char2idx[c] for c in seq])
print(seg(seq,hid))

seq = '小龙女说，我也想过过过过过过过的生活'
hid = hmm.predictHiddenStatus([char2idx[c] for c in seq])
print(seg(seq,hid))

seq = '我爱马云爸爸'
hid = hmm.predictHiddenStatus([char2idx[c] for c in seq])
print(seg(seq,hid))

print(np.log(hmm.predictJointVisibleProb([char2idx[c] for c in '我爱马云爸爸'])))
print(np.log(hmm.predictJointVisibleProb([char2idx[c] for c in '马云爸爸爱我'])))

