import numpy as np

def create_dataset(mu0,sig0,alpha0,mu1,sig1,alpha1,length=1000):
    # 生成观测数据
    data0 = np.random.normal(mu0,sig0,int(length*alpha0))
    data1 = np.random.normal(mu1,sig1,int(length*alpha1))

    data = []
    data.extend(data0)
    data.extend(data1)
    np.random.shuffle(data) # 打乱数据
    return np.array(data)

def calc_guass(mu,sig,y):
    # 计算高斯密度函数
    gauss_dens = 1 / (np.sqrt(2*np.pi)*sig) * np.exp(-(y-mu)**2 / (2*sig**2))

    return gauss_dens

def E_step(mu0,sig0,alpha0,mu1,sig1,alpha1,y):
    # 按照p165的第(2)步计算每个k对应的gamma
    gauss_dens0 = calc_guass(mu0,sig0,y)
    numerator0 = alpha0 * gauss_dens0
    gauss_dens1 = calc_guass(mu1,sig1,y)
    numerator1 = alpha1 * gauss_dens1

    denominator = numerator0 + numerator1
    gamma0 = numerator0 / denominator
    gamma1 = numerator1 / denominator
    return gamma0, gamma1

def M_step(mu0,mu1,gamma0,gamma1,y):
    # 按照p165的第(3)步计算新一轮的模型参数
    mu_0 = np.dot(gamma0,y) / np.sum(gamma0)
    mu_1 = np.dot(gamma1,y) / np.sum(gamma1)

    sig_0 = np.sqrt(np.dot(gamma0,(y-mu0)**2) / np.sum(gamma0)) # 注意p165给的是sigmma^2, 所以要记得开根号
    sig_1 = np.sqrt(np.dot(gamma1,(y-mu1)**2) / np.sum(gamma1))

    N = len(y)
    alpha_0 = np.sum(gamma0) / N
    alpha_1 = np.sum(gamma1) / N
    return  mu_0,mu_1,sig_0,sig_1,alpha_0,alpha_1

def train_EM(mu0,sig0,alpha0,mu1,sig1,alpha1,y,train_steps=500):
    # 训练阶段(2)-->(3)-->(2)-->(3)-->...
    step = 0
    while step < train_steps:
        gamma0, gamma1 = E_step(mu0,sig0,alpha0,mu1,sig1,alpha1,y)
        mu0, mu1, sig0, sig1, alpha0, alpha1 = M_step(mu0,mu1,gamma0,gamma1,y)
        step += 1

    return mu0,mu1,sig0,sig1,alpha0,alpha1

if __name__ == '__main__':
    # 迭代初值
    mu0 = 0.8; sig0 = 0.8; alpha0 = 0.9
    mu1 = 0.5; sig1 = 0.1; alpha1 = 0.1
    # 迭代步数
    train_steps = 100
    print('初始化参数为:mu0:%.2f, mu1:%.2f, sig0:%.2f, sig1:%.2f, alpha0:%.2f, alpha1:%.2f' %(mu0, mu1, sig0, sig1, alpha0, alpha1))
    # 生成观测数据
    y = create_dataset(mu0,sig0,alpha0,mu1,sig1,alpha1)
    # 开始迭代
    mu0,mu1,sig0,sig1,alpha0,alpha1 = train_EM(mu0,sig0,alpha0,mu1,sig1,alpha1,y,train_steps)
    print('估计参数为:mu0:%.2f, mu1:%.2f, sig0:%.2f, sig1:%.2f, alpha0:%.2f, alpha1:%.2f' %(mu0, mu1, sig0, sig1, alpha0, alpha1))

