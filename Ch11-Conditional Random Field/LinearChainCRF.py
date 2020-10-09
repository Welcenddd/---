from CRFFeatureFunction import CRFFeatureFunction
import numpy as np

class CRF(object):
    def __init__(self, epochs=10, lr=1e-3, tol=1e-5, output_status_num=None, input_status_num=None, unigram_rulers=None, bigram_rulers=None):
        """
        :param epochs: 迭代次数
        :param lr: 学习率
        :param tol:梯度更新的阈值
        :param output_status_num:标签状态数
        :param input_status_num:输入状态数
        :param unigram_rulers: 状态特征规则
        :param bigram_rulers: 状态转移规则
        """
        self.epochs = epochs
        self.lr = lr
        self.tol = tol
        # 为输入序列和标签状态序列添加一个头尾id(引进特殊的起点和终点状态标记y0=start,yn+1=stop)
        self.output_status_num = output_status_num + 2
        self.input_status_num = input_status_num + 2
        self.input_status_head_tail = [input_status_num, input_status_num + 1]
        self.output_status_head_tail = [output_status_num, output_status_num + 1]
        # 特征函数
        self.FF = CRFFeatureFunction(unigram_rulers, bigram_rulers)
        # 模型参数
        self.w = None

    def fit(self, x, y):
        """
        :param x: [[...],[...],...,[...]]
        :param y: [[...],[...],...,[...]]
        :return
        """
        # 为 x,y加头尾
        x = [[self.input_status_head_tail[0]] + xi + [self.input_status_head_tail[1]] for xi in x]
        y = [[self.output_status_head_tail[0]] + yi + [self.output_status_head_tail[1]] for yi in y]
        self.FF.fit(x, y) #创建特征函数
        self.w = np.ones(len(self.FF.feature_funcs)) * 1e-5 #初始化权重
        for _ in range(0, self.epochs):
            # 偷个懒，用随机梯度下降
            for i in range(0, len(x)):
                xi = x[i] #观测序列
                yi = y[i] #标记序列
                """
                1.求全局特征向量F(yi,xi)以及条件随机场P_w(yi|xi)
                """
                F_y_x = [] #全局特征向量, 式(11.18)
                Z_x = np.ones(shape=(self.output_status_num, 1)).T #规范化因子
                for j in range(1, len(xi)):
                    F_y_x.append(self.FF.map(yi[j - 1], yi[j], xi, j)) #F_y_x中的一行表示一个观测序列xi在位置j(书中为符号i)的特征匹配结果
                    # 构建M矩阵
                    M = np.zeros(shape=(self.output_status_num, self.output_status_num))
                    for k in range(0, self.output_status_num):
                        for t in range(0, self.output_status_num):
                            M[k, t] = np.exp(np.dot(self.w, self.FF.map(k, t, xi, j))) #式(11.21~11.23)即求位置j(书中为i)处的矩阵Mi(x)
                    # 前向算法求 Z(x)
                    Z_x = Z_x.dot(M) #求矩阵Mi(x)的连乘
                F_y_x = np.sum(F_y_x, axis=0) #沿第0维求和表示对所有的转移和状态特征在各个位置j(对应书中的符号i)进行求和得到全局特征向量,即式(11.13)
                Z_x = np.sum(Z_x) #规范化因子Zw(x)以start为起点stop为终点通过所有路径y1y2...yn的非规范化概率(矩阵Mi(x)的连乘)的和
                # 求P_w(yi|xi)
                P_w = np.exp(np.dot(self.w, F_y_x)) / Z_x #式(11.19)
                """
                2.求梯度,并更新
                """
                dw = (P_w - 1) * F_y_x
                self.w = self.w - self.lr * dw
                if (np.sqrt(np.dot(dw, dw) / len(dw))) < self.tol:
                    break

    def predict(self, x):
        """
        维特比求解最优的y
        :param x:[...]
        :return:
        """
        # 为x加头尾
        x = [self.input_status_head_tail[0]] + x + [self.input_status_head_tail[1]]
        # 初始化, 式(11.54), delta中的元素个数等于标记个数
        delta = np.asarray([np.dot(self.w, self.FF.map(self.output_status_head_tail[0], j, x, 1)) for j in range(0, self.output_status_num)])
        psi = [[0] * self.output_status_num] #初始化非规范化概率最大值的路径
        # 递推
        for visible_index in range(2, len(x) - 1): #递推求非规范化概率最大值以及相应的路径,visible_index表示书中delta_i(l)中的下标i
            new_delta = np.zeros_like(delta)
            new_psi = []
            for i in range(0, self.output_status_num): #遍历所有的标记l
                best_pre_index_i = -1
                best_pre_index_value_i = 0
                delta_i = 0
                for j in range(0, self.output_status_num): # j表示yi-1的所有可能标记,y表示yi的所有可能标记。
                    delta_i_j = delta[j] + np.dot(self.w, self.FF.map(j, i, x, visible_index))
                    if delta_i_j > delta_i:
                        delta_i = delta_i_j
                    best_pre_index_value_i_j = delta[j] + np.dot(self.w, self.FF.map(j, i, x, visible_index))
                    if best_pre_index_value_i_j > best_pre_index_value_i:
                        best_pre_index_value_i = best_pre_index_value_i_j #delta_i(l)
                        best_pre_index_i = j #delta_i(l)对应的j
                new_delta[i] = delta_i
                new_psi.append(best_pre_index_i)
            delta = new_delta #更新delta值(直接覆盖,不用缓存)
            psi.append(new_psi) #记录路径
        # 回溯
        best_hidden_status = [np.argmax(delta)]
        for psi_index in range(len(x) - 3, 0, -1):
            next_status = psi[psi_index][best_hidden_status[-1]]
            best_hidden_status.append(next_status)
        best_hidden_status.reverse()
        return best_hidden_status


if __name__ == '__main__':

    # 测试:参数学习
    x = [
        [1, 2, 3, 0, 1, 3, 4],
        [1, 2, 3],
        [0, 2, 4, 2],
        [4, 3, 2, 1],
        [3, 1, 1, 1, 1],
        [2, 1, 3, 2, 1, 3, 4]
                            ]
    y = x

    crf = CRF(epochs=20,lr=1e-4,tol=1e-6,output_status_num=5, input_status_num=5) #观测序列和标记序列可能取值范围的数量都为5
    crf.fit(x, y)
    print(crf.w)
    print(len(crf.w))

    # 测试:预测
    print(crf.predict(x[0]))