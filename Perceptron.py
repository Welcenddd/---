import numpy as np

def create_dataset():
    # 书中例2.1的数据
    data = np.mat([[3,3], [4,3], [1,1]])
    label = np.mat([1, 1, -1]).T
    # data的shape为(3,2); label的shape为(3,1)
    return data, label

def train_model(data,label,eta=1):
    #算法的原始形式--->算法2.1
    m,n = data.shape # m和n分别表示样本数量和特征维数
    w = np.zeros((1,n)) #shape是(1,n)
    b = 0
    continue_iter = True #设置迭代开关, True表示继续迭代, 否则表示没有误分类点也就停止迭代
    while continue_iter:
        continue_iter = False
        for i in range(m):
            if label[i] * (w * data[i].T + b) <= 0:
                w += eta * label[i] * data[i]
                b += eta * label[i]
                continue_iter = True
                break
    return w, b

def dual_train_model(data,label,eta=1):
    #算法的对偶形式--->算法2.2
    m, n = data.shape  # m和n分别表示样本数量和特征维数
    alpha = np.zeros((m,1))  # shape是(n,1)
    b = 0
    continue_iter = True  # 设置迭代开关, True表示继续迭代, 否则表示没有误分类点也就停止迭代
    while continue_iter:
        continue_iter = False
        for i in range(m):
            if label[i] * (data[i] * data.T * np.multiply(alpha,label)  + b) <= 0: #alpha:(m,1), label:(m,1), data:(m,n)
                alpha[i] += eta
                b += eta * label[i]
                continue_iter = True
                break
    return alpha, b

if __name__ == '__main__':
    data,label = create_dataset()
    # 原始形式的算法得到的结果
    w,b = train_model(data,label)
    print('由原始算法得到的分离超平面为%d * x1 + %d * x2 + (%d) = 0' % (int(w[0,0]),int(w[0,1]),int(b)))

    # 对偶形式算法得到的结果
    alpha, b = dual_train_model(data, label)
    w = data.T * np.multiply(alpha,label)
    print('由对偶算法得到的分离超平面为%d * x1 + %d * x2 + (%d) = 0' % (int(w[0, 0]), int(w[1, 0]), int(b)))
