import numpy as np
import os
import pickle

def get_lorenz_matrix(n, time=100, step=0.02, c=0.1, time_invariant=True, init_way='uniform', init_param=None):
    """

    :param n:
    :param time:
    :param step:
    :param c:
    :param time_invariant: 是否是时不变的
    :return: 返回一个[time // step, n*3]大小的矩阵，n*3是变量数目， time // step 是时间长度。
    """
    length = int(time / step)  #
    x = np.zeros((n * 3, length), dtype=np.float32)  # 初始化矩阵
    if init_way == 'uniform':
        x[:, 0] = np.random.rand(n * 3)  # 随机生成初始值
    elif init_way == 'norm':
        x[:, 0] = np.random.randn(n * 3) * init_param['std'] + init_param['mean']
    sigma = 10.0

    for i in range(1, length):

        if not time_invariant:
            sigma = 10.0 + 0.1 * i % 10

        x[0, i] = x[0, i - 1] + step * (sigma * (x[1, i - 1] - x[0, i - 1]) + c * x[(n - 1) * 3, i - 1])
        x[1, i] = x[1, i - 1] + step * (28 * x[0, i - 1] - x[1, i - 1] - x[0, i - 1] * x[2, i - 1])
        x[2, i] = x[2, i - 1] + step * (-8 / 3 * x[2, i - 1] + x[0, i - 1] * x[1, i - 1])

        for j in range(1, n):
            x[3 * j, i] = x[3 * j, i - 1] + step * (
                        10 * (x[3 * j + 1, i - 1] - x[3 * j, i - 1]) + c * x[3 * (j - 1), i - 1])
            x[3 * j + 1, i] = x[3 * j + 1, i - 1] + step * (
                        28 * x[3 * j, i - 1] - x[3 * j + 1, i - 1] - x[3 * j, i - 1] * x[3 * j + 2, i - 1])
            x[3 * j + 2, i] = x[3 * j + 2, i - 1] + step * (
                        -8 / 3 * x[3 * j + 2, i - 1] + x[3 * j, i - 1] * x[3 * j + 1, i - 1])

    return x.T



# if __name__ == '__main__':
#     temp = my_lorenz(30)
#     print(temp.shape)
#     print((temp[:, 0] == 0).sum())