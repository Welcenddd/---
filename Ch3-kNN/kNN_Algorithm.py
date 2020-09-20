import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 本来打算自己生产一个随机的数据集，但是发现效果不是很好，就不用下面这个函数了，改用鸢尾花数据集
def generate_data(num_samples=100,num_features=2):
    data_size = (num_samples,num_features)
    data = np.random.randint(0,100,data_size)

    label_size = (num_samples,1)
    labels = np.random.randint(0,2,label_size)
    return data, labels

def plot_data(data,labels,feat1=0,feat2=1): #feat1和feat2是想要画的特征序号,num_features为2的情况下默认为0和1
    markers = ['o','^','s','D','*','p','<','>'] #如果需要更多的marker和color，可参考https://www.cnblogs.com/qccc/p/12819459.html以及https://stackoverflow.com/questions/8409095/set-markers-for-individual-points-on-a-line-in-matplotlib
    colors = ['b','r','g','c','m','y','k','w']
    num_feature = data.shape[1]
    markers = markers[:num_feature]
    colors = colors[:num_feature]
    feat1 = range(num_feature)[feat1]
    feat2 = range(num_feature)[feat2]
    for num_feat in range(num_feature):
        data_feat = data[labels.ravel()==num_feat]
        plt.scatter(data_feat[:,feat1],data_feat[:,feat2],c=colors[num_feat],marker=markers[num_feat])
    plt.show()

# 载入鸢尾花数据集(总共150个样本，每类样本个数均为50，每个样本包含四类特征:花萼长度、花萼宽度、花瓣长度、花瓣宽度)
def load_data(filename,num_samples=150,convert_label=True,shuffle=True):
    df = pd.read_csv(filename, header=None) # 读取数据集
    random_index = np.arange(df.shape[0])
    if shuffle:
        np.random.shuffle(random_index)  # 打乱数据索引
    data = df.iloc[random_index[:num_samples],:4].values #前四列为特征
    labels = df.iloc[random_index[:num_samples],4].values #标签为第五列
    if convert_label:
        labels,word2num = text2num(labels) #将文本标签转化为数字标签
    else:
        word2num = None
    return data,labels,word2num

# 将文本标签转化为数字标签
def text2num(labels):
    num_samples = labels.shape[0]
    total_types = np.unique(labels) #提取标签总类别
    num_types = len(total_types)
    num_labels = np.arange(num_types)
    word2num = {word:num for word,num in zip(total_types,num_labels)} #创建从文本转化为数字的字典
    for i in range(num_samples):
        labels[i] = word2num[labels[i]] #将每个花的名字转化为数字
    return labels,word2num


# 样本之间的计算距离
def calc_distance(x1,x2,dist_type='Euclidean'):
    # p39式(3.2~3.5)的距离计算公式, 也可参考https://blog.csdn.net/qq_34562093/article/details/78234742
    if dist_type == 'Euclidean':
        return np.sqrt(np.sum(np.square(x2-x1)))
    elif dist_type == 'Manhattan':
        return np.sum(abs(x2-x1))
    elif dist_type == 'Chebyshev':
        return np.max(x2-x1)
    else:
        print('请输入正确的距离类型!')

# 计算离目标点最近的k个点
def get_nearest_k_points(data,label,x,k):
    dist_list = np.zeros(data.shape[0])
    for i in range(len(dist_list)):
        dist = calc_distance(data[i],x)
        dist_list[i] = dist

    sorted_index = np.argsort(dist_list) #按照距离从小到大进行排序，返回对应的索引
    sorted_index_topk = sorted_index[:k]
    vote_count = [0] * k
    for i in range(k):
        vote_count[label[sorted_index_topk[i]]] = +1 # 投票表决

    return vote_count.index(np.max(vote_count))

# 模型验证
def model_valid(filename_train='iris.data',filename_test='iris.data',k=5):
    data_train, labels_train, word2num_train = load_data(filename_train, convert_label=True, shuffle=True)
    data_test, labels_test, word2num_test = load_data(filename_test, convert_label=True, shuffle=False)
    num_test = data_test.shape[0]
    error_count = 0
    for i in range(num_test):
        pred = get_nearest_k_points(data_train,labels_train,data_test[i],k)
        if pred != labels_test[i]:
            error_count += 1
    print('准确率为:%.2f%%' % ((1-error_count/num_test)*100))

if __name__ == '__main__':

    data,labels,_ = load_data('iris.data', convert_label=True, shuffle=True)
    plot_data(data,labels,feat1=0,feat2=3) #根据feat1和feat2指定特征来绘制数据散点分布图

    model_valid(k=3) # 由于鸢尾花数据集没有验证集,所以直接采用原数据集

    # print('预测的类别为: %s' % num2word[tt])