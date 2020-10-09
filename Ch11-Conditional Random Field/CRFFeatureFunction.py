import numpy as np
#提醒一点，一定要弄清楚特征函数中的x和y在这里指的是什么（x指观测到的数据/观测序列，例如分词“北”、“京”等，也可以是词性；
#而y指的是分词的标签/标记/标记序列，例如常见的标注集{B,M,E}或者其他的标注集。
#特征函数的创建要多从例11.1去理解每个符号的含义
#预测算法要把例11.3理解的很透彻才能看懂代码。

# 实现特征函数的功能
class CRFFeatureFunction(object):
    def __init__(self, unigram_rulers=None, bigram_rulers=None):
        """
        默认输入特征就一种类型
        :param unigram_rulers: 状态特征规则
        :param bigram_rulers: 状态转移规则
        """
        if unigram_rulers is None: #如果不指定状态特征规则,就进行初始化
            self.unigram_rulers = [
                [0],     # 当前特征x->标签y(表示当前x对当前y的影响)
                [1],     # 后一个特征x->标签y(表示后一个x对当前y的影响)
                [-1],    # 前一个特征x->标签y(表示前一个x对当前y的影响)
                [0, 1],  # 当前特征x和后一个特征x'->标签y(表示当前x和后一个x对当前y的影响)
                [-1, 0]  # 前一个特征x和当前特征x->标签y(表示前一个x和当前x对当前y的影响)
            ]
        else:
            self.unigram_rulers = unigram_rulers
        if bigram_rulers is None: #如果不指定状态转移规则,就进行初始化
            self.bigram_rulers = [
                None,  # 不考虑特征的影响，只考虑前一个标签和当前标签
                [0]    # 当前特征->前一个标签和当前标签
            ]
        else:
            self.bigram_rulers = bigram_rulers
        # 特征函数
        self.feature_funcs = []

    def fit(self, x, y):
        """
        构建特征函数，为了节省空间，训练集x,y中没有出现的特征和标签组合就不考虑了
        :param x: [[...],[...],...,[...]]
        :param y: [[...],[...],...,[...]]
        :return:
        """
        uni_cache = {} #建立所有的单字集合,格式为unigram_ruler:[...]的字典
        bi_cache = {} #建立所有的双字集合
        for i in range(0, len(x)):
            xi = x[i]
            yi = y[i]
            # 处理unigram_ruler
            for k, unigram_ruler in enumerate(self.unigram_rulers):
                if uni_cache.get(k) is None: #查找指定键的值,如果不存在则设为空列表
                    uni_cache[k] = []
                for j in range(max(0, 0 - np.min(unigram_ruler)), min(len(xi), len(xi) - np.max(unigram_ruler))):
                    # if :
                    #    unigram_ruler = [0], then j = 0,1,...,len(xi)-1    (按照python不取末尾的值,即只取到len(xi)-1)
                    #    unigram_ruler = [1], then j = 0,1,...,len(xi)-2
                    #    unigram_ruler = [-1], then j = 1,2,...,len(xi)-1
                    #    unigram_ruler = [0,1], then j = 0,1,...,len(xi)-2
                    #    unigram_ruler = [-1,0], then j = 1,2,...,len(xi)-1
                    #从以上分类可以分析出j表示序列中的位置也即式(11.11)的指标i
                    # if :
                    #    unigram_ruler = [0], 则表示只考虑当前位置的观测值的影响, 即特征函数sl(yi,x,i) i=i_cur
                    #    unigram_ruler = [1], 则表示只考虑后一位置的观测值的影响, 即特征函数sl(yi,x,i) i=i_cur+1
                    #    unigram_ruler = [-1], 则表示只考虑前一位置的观测值的影响, 即特征函数sl(yi,x,i) i=i_cur-1
                    #    unigram_ruler = [0,1], 则表示考虑当前位置和后一位置的观测值的影响, 即特征函数sl(yi,x,i), i=i_cur,i_cur+1
                    #    unigram_ruler = [-1,0], 则表示考虑当前位置和前一位置的观测值的影响, 即特征函数sl(yi,x,i) i=i_cur-1,i_cur
                    # 如果考虑前一个特征对当前标签的影响那么就要保证pos + j从0开始，一直到观测序列xi的长度-1；
                    # 而如果需要考虑后一个特征对当前特征的影响，就要保证pos + j从1开始，一直到观测序列xi的长度末尾
                    # 如果需要同时考虑前一个特征和后一个特征的影响，就要作综合考虑。
                    # 从这两个角度出发，就可以理解j的循环范围以及下面这行代码
                    key = "".join(str(item) for item in [xi[pos + j] for pos in unigram_ruler] + [yi[j]])
                    # if :
                    #    unigram_ruler = [0],
                    #    pos只能取0,那么xi[pos + j]=xi[j]-->item=xi[j]-->key=xi[j]和yi[j]组成的字符串,也就是观测xi[j]和标记yi[j]的组合
                    #
                    #    unigram_ruler = [1],
                    #    pos只能取1,那么xi[pos + j]=xi[j+1]-->item=xi[j+1]-->key=xi[j+1]和yi[j]组成的字符串,也就是观测xi[j+1]和标记yi[j]的组合
                    #
                    #    unigram_ruler = [-1],
                    #    pos只能取-1,那么xi[pos + j]=xi[j-1]-->item=xi[j-1]-->key=xi[j-1]和yi[j]组成的字符串,也就是观测xi[j-1]和标记yi[j]的组合
                    #
                    #    unigram_ruler = [0,1],
                    #    pos可取0和1,那么xi[pos + j]=[xi[j],xi[j+1]]-->item=xi[j]或xi[j+1]-->key=xi[j]、xi[j+1]以及yi[j]组成的字符串,也就是观测xi[j]、xi[j+1]和标记yi[j]的组合

                    #    unigram_ruler = [-1,0],
                    #    pos可取-1和0,那么xi[pos + j]=[xi[j-1],xi[j]]-->item=xi[j-1]或xi[j]-->key=xi[j-1]、xi[j]以及yi[j]组成的字符串,也就是观测xi[j-1]、xi[j]和标记yi[j]的组合
                    if key in uni_cache[k]:
                        continue
                    else:
                        self.feature_funcs.append([
                            'u',
                            unigram_ruler, #状态特征规则
                            [xi[j + pos] for pos in unigram_ruler], #根据规则得到的特征/观测x
                            yi[j] #相应的标签/输出y
                        ])
                        uni_cache[k].append(key)
            # 处理 bigram_ruler
            for k, bigram_ruler in enumerate(self.bigram_rulers):
                if bi_cache.get(k) is None:
                    bi_cache[k] = []
                # B的情况 tk(y_i-1,y_i,i)的形式i.e.不考虑观测x的影响
                if bigram_ruler is None: #None表示只考虑前一个标签状态和当前标签状态，而不考虑当前特征xi取值的情况，所以可以直接添加
                    for j in range(1, len(xi)):
                        key = "B" + "".join([str(yi[j - 1]), str(yi[j])])
                        if key in bi_cache[k]:
                            continue
                        else:
                            self.feature_funcs.append([
                                'B',
                                bigram_ruler, #状态转移规则
                                None, #不考虑观测的取值
                                [yi[j - 1], yi[j]]
                            ])
                            bi_cache[k].append(key)
                    continue #跳出本次关于k的循环,就不会执行非B的情况
                # 非B的情况 tk(y_i-1,y_i,x,i)的形式
                for j in range(max(1, 0 - np.min(bigram_ruler)), min(len(xi), len(xi) - np.max(bigram_ruler))):
                    #    bigram_ruler = [0], then j = 1,2,...,len(xi)-1
                    key = "".join(str(item) for item in [xi[pos + j] for pos in bigram_ruler] + [yi[j - 1], yi[j]])

                    #    bigram_ruler = [0], then j = 1,2,...,len(xi)-1
                    #    pos只能取0,那么xi[pos + j]=xi[j]-->item=xi[j]-->key=xi[j]和yi[j-1]、yi[j]组成的字符串,也就是观测xi[j]和标记yi[j-1]、yi[j]的组合
                    if key in bi_cache[k]:
                        continue
                    else:
                        self.feature_funcs.append([
                            'b',
                            bigram_ruler,
                            [xi[j + pos] for pos in bigram_ruler],
                            [yi[j - 1], yi[j]]
                        ])
                        bi_cache[k].append(key)
        del uni_cache
        del bi_cache

    def map(self, y_pre, y_cur, x_tol, i_cur):
        """
        返回是否match特征函数的list
        :param y_pre:
        :param y_cur:
        :param x_tol:
        :param i_cur:
        :return:
        """

        def map_func_(func):
            # y_pre='B', y_cur='E', i_cur=3
            try:
                # gram_type:unigram/bigram, ruler:0/1/-1或者他们之间的组合, xi:观测, yi:标记
                gram_type, ruler, xi, yi = func
                if gram_type == "u" and [x_tol[i + i_cur] for i in ruler] == xi and yi == y_cur:
                    # if:
                    #    ruler=[0], then x_tol[i + i_cur] = x_tol[i_cur] = x_tol[3] = "的"
                    #    ruler=[1], then x_tol[i + i_cur] = x_tol[i_cur+1] = x_tol[4]不存在返回0
                    #    ruler=[-1], then x_tol[i + i_cur] = x_tol[i_cur-1] = x_tol[2] = "我"
                    #    ruler=[0,1], then x_tol[i + i_cur] = x_tol[i_cur],x_tol[i_cur+1] = x_tol[3],x_tol[4]不存在返回0
                    #    ruler=[-1,0], then x_tol[i + i_cur] = x_tol[i_cur-1],x_tol[i_cur] = x_tol[2],x_tol[3]="我","的"

                    return 1
                elif gram_type == "b" and [x_tol[i + i_cur] for i in ruler] == xi and yi == [y_pre, y_cur]: #考虑当前特征取值
                    return 1
                elif gram_type == "B" and yi == [y_pre, y_cur]: #不考虑当前特征取值的情况
                    return 1
                else:
                    return 0
            except:
                # 越界的情况，默认不匹配
                return 0

        # 把map_func_函数一一作用于特征函数feature_funcs中的元素,去寻找匹配的元素, 匹配的返回1否则返回0
        return np.asarray(list(map(map_func_, self.feature_funcs)))
if __name__ == '__main__':

    # 测试
    x = [["我", "爱", "我", "的", "祖", "国"], ["我", "爱"]] #两组观测序列
    y = [["B", "E", "B", "E", "B", "E"], ["B", "E"]]       #两组标记序列
    ff = CRFFeatureFunction()
    ff.fit(x, y)  #训练数据 创建特征函数
    print(len(ff.feature_funcs))
    # 判断满足前一个标记y_pre='B',当前标记y_cur='E', 以及当前结点i_cur=3的特征函数,
    # ["我", "爱", "我", "的"]是给定观测序列，["B", "E"]是给定标记序列。
    # 下面这行代码的意思就是在训练得到所有特征函数的权值(也就是得到条件随机场模型)之后,给定某个观测序列即["我", "爱", "我", "的"],
    # 求P(y_pre="B", y_cur="E"|x=["我", "爱", "我", "的"])的概率 (理解这个很关键！！！)。
    # 由于i_cur=3,y_cur就对应y3,y_pre对应y2(如果标记序列按照[y1 y2 y3 ... yT]进行排序)
    print(ff.map("B", "E", ["我", "爱", "我", "的"], 3))