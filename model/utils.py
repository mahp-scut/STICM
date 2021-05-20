import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
from scipy.stats import pearsonr


def draw_line(values, file_path, fig_size=(6, 5), title=None):
    """
    简单的画曲线
    :param values:
    :param file_path:
    :return:
    """
    plt.rcParams['figure.figsize'] = fig_size  # 设置figure_size尺寸
    plt.title(title)
    plt.plot(np.arange(values.shape[0]), values, marker='.')
    plt.savefig(file_path)
    plt.clf()

class Node:
    def __init__(self, value, children:list=None):
        self.children = children
        self.value = value


def build_parameter_tree(*args):
    """
    构建参数tree
    :param args:
    :return:
    """
    last_children_nodes = None
    for arg in args:
        children_nodes = []
        for e in arg:
            children_nodes.append(Node(e, last_children_nodes))
        last_children_nodes = children_nodes
    root = Node('root', last_children_nodes)
    return root


def dfs_p_tree(tree:Node, params_group:list, params:list):
    """
    递归深度优先遍历获取所有的参数组
    :param tree:
    :param params_group: 某个参数组
    :param params: 保存所有参数
    :return:
    """

    # 先把本节点参数加入group中
    if tree.value != 'root':
        params_group.append(tree.value)

    if tree.children is None:  # 遍历到了叶子节点，就到底了
        params.append(params_group)
    else:
        for child in tree.children:
            dfs_p_tree(child, copy.deepcopy(params_group), params)  # 每次递归都深度复制当前参数组


def grid_search_parameter(*args):
    """
    建立参数tree之后，深度优先遍历获取所有组合
    :param args:
    :return:
    """
    result_params = []
    params_group = []
    param_tree = build_parameter_tree(*args)
    dfs_p_tree(param_tree, params_group, result_params)
    return result_params



def get_same_idxs(sample_time_len, embedding_len):
    """
    获取embedding矩阵中斜线相等的元素的index
    :param sample_time_len:
    :param embedding_len:
    :return: [斜线num, [斜线元素num ,2]]
    """
    same_idxs = []
    if embedding_len > sample_time_len:
        for start_col_idx in range(1, embedding_len):
            idxs = []
            row_num = min(start_col_idx + 1, sample_time_len)
            for row_idx in range(row_num):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 0] = row_idx
                idx[0, 1] =start_col_idx - row_idx
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

        # 下三角相等的
        for start_col_idx in range(embedding_len - sample_time_len + 1, embedding_len - 1):
            idx_count = embedding_len - start_col_idx
            idxs = []
            for i in range(idx_count):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 0] = sample_time_len - 1 - i
                idx[0, 1] = start_col_idx + i
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))
    else:
        for start_row_idx in range(1, sample_time_len):
            idxs = []
            col_num = min(start_row_idx+1, embedding_len)
            for col_idx in range(col_num):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 1] = col_idx
                idx[0, 0] = start_row_idx - col_idx
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

        for start_row_idx in range(sample_time_len - embedding_len + 1, sample_time_len - 1):
            idx_count = sample_time_len - start_row_idx
            idxs = []
            for i in range(idx_count):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 1] = embedding_len - 1 - i
                idx[0, 0] = start_row_idx + i
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

    return same_idxs


def embedding_2_predict_y(delay_embedding, config, batch_size, all_y=True):
    """

    :param delay_embedding:
    :param config:
    :param batch_size:
    :param all_y: 是否返回预测出来的embedding矩阵里面全部的均值y
    :return:
    """
    if all_y:
        same_idxs = get_same_idxs(config.TRAIN_LEN, config.EMBEDDING_LEN)
    else:
        same_idxs = get_same_idxs(config.TRAIN_LEN, config.EMBEDDING_LEN)[config.TRAIN_LEN - 1:]

    batch_predict_y = []
    for j in range(batch_size):
        predict_y = []
        if all_y:
            predict_y.append(delay_embedding[j, 0, 0])
        for i, same_idx in enumerate(same_idxs):
            same_y_s = tf.gather_nd(delay_embedding[j], same_idx)
            mean_y = tf.reduce_mean(same_y_s)
            predict_y.append(mean_y)
        predict_y.append(delay_embedding[j, -1, -1])
        predict_y = tf.stack(predict_y)
        # print(predict_y.shape)
        batch_predict_y.append(predict_y)
    batch_predict_y = tf.stack(batch_predict_y)
    return batch_predict_y


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b), axis=0))


def batch_rmses(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    return np.sqrt(np.mean(np.square(a - b), axis=1))

def batch_nrmses(pred, label, drop_zero_std=True):
    """

    :param pred: 预测值
    :param label: 标签
    :return:
    """
    # print(np.std(label, axis=1).shape)
    batch_std = np.std(label, axis=1)
    if drop_zero_std:
        valid_idx = np.nonzero(batch_std)[0]
        return batch_rmses(pred, label)[valid_idx] / batch_std[valid_idx]
    else:
        return batch_rmses(pred, label) / batch_std
    # return batch_rmses(pred, label) / np.mean(label, axis=1)




def pcc(a, b):
    """
    计算两个向量的PCC
    :param a:
    :param b:
    :return:
    """
    return pearsonr(a, b)[0]

def batch_pcc(a, b):
    """
    计算两个Batch的pcc
    :param a:
    :param b:
    :return:
    """
    pccs = []
    for x, y in zip(a, b):
        pcc = pearsonr(x, y)[0]
        if pcc is np.nan:
            pccs.append(0)
        else:
            pccs.append(pcc)
    return np.array(pccs)

def get_nearest_idxs(locs, target_idx):
    """
    返回距离目标变量最近的索引排序，从近到远
    :param locs: [n, 2]
    :return:
    """
    dis = np.sqrt(np.sum(np.square(locs - locs[target_idx]), axis=-1))
    return np.argsort(dis)


def draw_cmp_curves(file_path, predict_len, all_y, predict_results, func_names, colors,
                    fig_size=(6, 5), y_limit=None,
                    is_legned=True, title_rmse_idx=0, xlabel='Time', ylabel='Value',
                    markers=None, zorders=None, marker_sizes=None, line_widths=None,
                    dpi=600):
    """

    :param file_path:
    :param predict_len:
    :param all_y:
    :param predict_results:
    :param func_names:
    :param colors:
    :return:
    """
    total_len = len(all_y)
    train_len = total_len - predict_len

    if markers is None:
        markers = ['.' for _ in range(len(predict_results))]
    if zorders is None:
        zorders = [10 for _ in range(len(predict_results))]
    if marker_sizes is None:
        marker_sizes = [6 for _ in range(len(predict_results))]
    if line_widths is None:
        line_widths = [2 for _ in range(len(predict_results))]
    if y_limit is None:
        y_limit = [np.min(all_y) - 0.05 * (np.max(all_y) - np.min(all_y)),
                   np.max(all_y) + 0.05 * (np.max(all_y) - np.min(all_y))]

    plt.rcParams['figure.figsize'] = fig_size  # 设置figure_size尺寸
    if y_limit:
        plt.ylim(y_limit[0], y_limit[1])

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    x = np.arange(1, train_len+1)
    gt = all_y
    plt.plot(x, gt[:train_len], color='blue', marker='.', label='Known', zorder=10)  # 画真实的

    x = np.arange(train_len + 1, len(gt) + 1)
    gt = all_y
    plt.plot(x, gt[train_len:], color='cyan', marker='.', label='True label', zorder=10)  # 画真实的

    idx = 0
    for predict_y, func_name, color, marker, zorder, marker_size, line_width in zip(predict_results, func_names,
                                                                                    colors, markers, zorders,
                                                                                    marker_sizes, line_widths):
        x = np.arange(train_len, train_len + 2)
        connect = [all_y[train_len - 1], predict_y[0]]
        plt.plot(x, connect, color=color, zorder=zorder, lw=line_width)  # 画连接线

        if len(predict_y) == predict_len:
            x = np.arange(train_len + 1, total_len + 1)
            predict = predict_y
            rmse = np.sqrt(np.mean(np.square(all_y[-predict_len:] - predict)))
            if title_rmse_idx == idx:
                plt.title('RMSE: {:.3f}'.format(rmse))
            plt.plot(x, predict, color=color, marker=marker, label='{}'.format(func_name),
                     zorder=zorder, lw=line_width, ms=marker_size)
        else:
            x = np.arange(1, total_len + 1)
            predict = predict_y[-predict_len:]
            rmse = np.sqrt(np.mean(np.square(all_y[-predict_len:] - predict)))
            if title_rmse_idx:
                plt.title('RMSE: {:.3f}'.format(rmse))
            plt.plot(x, predict_y, color=color, marker='.', label='{}'.format(func_name))

        idx += 1

    if is_legned:
        # plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.1), ncol=1)
        plt.legend()

    if dpi:
        plt.savefig(file_path, dpi=dpi)
    else:
        plt.savefig(file_path)
    plt.clf()


def draw_long_term_curve(file_path, label, predict_values, fig_size, pcc, rmse):
    plt.rcParams['figure.figsize'] = fig_size  # 设置figure_size尺寸
    x = np.arange(predict_values.shape[0])
    plt.plot(x, label, color='cyan', marker='.', label='True label', zorder=10)  # 画真实的
    plt.plot(x, predict_values, color='red', marker='.', label='Predict value', zorder=15)  # 画真实的
    plt.title('PCC={:.3f}, RMSE={:.3f}'.format(pcc, rmse))
    plt.savefig(file_path, dpi=600)
    # plt.savefig(file_path)
    plt.clf()