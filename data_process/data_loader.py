import datetime
import h5py
import pandas as pd
import scipy.io as scio
import scipy.stats as stats

from data_process.data_generation import *


def load_lorenz_data(data_dir, n, skip_time_num, time=100, step=0.02, c=0.1, time_invariant=True,
                     init_way='uniform', init_param=None):
    # 数据文件名称
    data_file_name = 'lorenz_({})_n={}_time={}_step={}_c={}_init_way={}_init_param={}.pkl'.format(
        'time_invariant' if time_invariant else 'time-variant',
        n, time, step, c, init_way, init_param)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = None
    file_path = os.path.join(data_dir, data_file_name)

    if os.path.exists(file_path):
        # 数据文件存在直接加载
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        # 数据文件不存在，创建后保存
        data = get_lorenz_matrix(n, time, step, c, time_invariant, init_way, init_param)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    return data[skip_time_num:]


def get_select_idxs(total_time_len, train_len, embedding_len, nb_samples=None):
    """
    获取不重叠的数据索引，索引为开始预测的时间点
    :param total_time_len:
    :param train_len:
    :param embedding_len:
    :param nb_samples:
    :return:
    """
    max_nb_samples = (total_time_len - embedding_len - 1) // train_len
    assert 0 < nb_samples < max_nb_samples, 'nb_samples is too large!'
    if nb_samples is not None:
        return np.array([(i+1) * train_len for i in range(max_nb_samples)])[:nb_samples]
    else:
        return np.array([(i + 1) * train_len for i in range(max_nb_samples)])


def get_resampled_idx(data_dir, input_dim, input_drop_rate, target_idx, samples_num=30000, re_generate=False):
    """
    首先获取生成的关于数据的索引，数据量可以根据 C(dim, keep_dims) 来确定
    :param data_dir:
    :param input_dim:
    :param input_drop_rate:
    :param target_idx:
    :param samples_num:
    :param re_generate: 如果存在以前
    :return:
    """

    assert 0.0 <= input_drop_rate <= 1.0, 'invalid input_drop_rate'

    picked_dim = int(np.around(input_dim * (1 - input_drop_rate)))
    saved_file_name = 'resample_idxs_num={}_dim={}_picked_dim={}_drop_rate={}.pkl'.format(samples_num,
                                                                                          input_dim,
                                                                                          picked_dim,
                                                                                          input_drop_rate)
    rng = np.random.default_rng()
    all_idx = np.arange(input_dim)
    selected_idx = all_idx[all_idx != target_idx]

    if os.path.exists(os.path.join(data_dir, saved_file_name)) and not re_generate:
        print('loading resampled idxs from file: {}'.format(os.path.join(data_dir, saved_file_name)))
        with open(os.path.join(data_dir, saved_file_name), 'rb') as file:
            resampled_idxs = pickle.load(file)
    else:
        print('generating resampled idxs')
        resampled_idxs = np.zeros(shape=(samples_num, picked_dim), dtype=np.int32)
        for i in range(samples_num):
            # 打乱顺序后排序，一定包含 target_idx
            resampled_idxs[i] = np.sort(np.append(rng.permutation(selected_idx)[:picked_dim-1], target_idx))

        with open(os.path.join(data_dir, saved_file_name), 'wb') as file:
            pickle.dump(resampled_idxs, file)
            print('writing idxs to file: {}'.format(os.path.join(data_dir, saved_file_name)))

    return resampled_idxs


def resampled_data_generator(input_x, resampled_idxs, resampled_way, shuffle=True):
    """
    基于数据索引的一个数据生成器
    :param input_x:
    :param resampled_idxs:
    :param resampled_way:
    :param shuffle:
    :return:
    """
    idxs = np.arange(resampled_idxs.shape[0])
    if shuffle:
        rng = np.random.default_rng()
        idxs = rng.permutation(idxs)

    for idx in idxs:
        if resampled_way == 'dropout':
            resampled_data = np.copy(input_x)
            mask = np.zeros(shape=(input_x.shape[-1],))
            mask[resampled_idxs[idx]] = 1.0
            resampled_data = resampled_data * mask

            yield resampled_data
        elif resampled_way == 'pick':
            # print(resampled_idxs[idx])
            yield input_x[:, resampled_idxs[idx]]
