import tensorflow as tf
from tensorflow.keras import backend as K
import inspect
import numpy as np
import model.utils as utils


class Embedding2Y(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(Embedding2Y, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, **kwargs):
        embedding = inputs
        target_y = utils.embedding_2_predict_y(embedding, self.config, self.config.BATCH_SIZE, all_y=False)
        return target_y

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.config.EMBEDDING_LEN - 1]

class ZeroLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return 0.0


class MaskEmbeddingLoss(tf.keras.losses.Loss):
    def __init__(self, config):
        """

        :param config:
        """
        self.known_mask = tf.convert_to_tensor(get_known_mask(config.TRAIN_LEN, config.EMBEDDING_LEN))
        self.config = config
        super(MaskEmbeddingLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)
        mask_embedding_loss = tf.reduce_mean(tf.multiply(
            tf.square(y_true - y_pred),
            self.known_mask)
        ) * self.config.LOSS_WEIGHTS.get('masked_embedding_loss', 1.0)
        return mask_embedding_loss


class ReconstructionLoss(tf.keras.losses.Loss):
    def __init__(self, config):
        self.config = config
        super(ReconstructionLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)

        reconstruction_loss = tf.reduce_mean(
            tf.square(y_true - y_pred)) * self.config.LOSS_WEIGHTS.get('reconstruction_loss', 1.0)

        return reconstruction_loss


def get_known_mask(time_len, embedding_len):
    """
    针对的矩阵是[sample_time_len, embedding_len]
    已知的赋值为1，未知的为0
    :param time_len:
    :param embedding_len:
    :return:
    """
    mask = np.ones(shape=[time_len, embedding_len], dtype=np.float32)

    # TODO：gene数据应该需要这个
    # mask = (mask * np.arange(embedding_len)[::-1]).astype(dtype=np.float32)

    for i in range(embedding_len - 1):
        mask[time_len - embedding_len + 1 + i, -(i+1):] = 0.0

    return mask

class TimeConsistencyLoss(tf.keras.layers.Layer):
    def __init__(self, loss_weight, train_len, embedding_len, batch_size, **kwargs):
        self.loss_weight = loss_weight
        self.train_len = train_len
        self.embedding_len = embedding_len
        self.same_idxs = utils.get_same_idxs(train_len, embedding_len)
        self.batch_size = batch_size
        super(TimeConsistencyLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):

        batch_consistent_losses = []
        for b in range(self.batch_size):
            embedding = inputs[b]

            consistent_loss = []
            # 计算对角线元素相等
            for i, same_idx in enumerate(self.same_idxs):
                same_y_s = tf.gather_nd(embedding, same_idx)
                mean_y = tf.reduce_mean(same_y_s)
                loss = tf.reduce_mean(tf.square(same_y_s - mean_y))
                consistent_loss.append(loss)

            consistent_loss = tf.stack(consistent_loss)
            consistent_loss = tf.reduce_mean(consistent_loss)
            batch_consistent_losses.append(consistent_loss)

        batch_consistent_losses = tf.stack(batch_consistent_losses)
        tc_loss = tf.reduce_mean(batch_consistent_losses)

        self.add_loss(tc_loss * self.loss_weight)
        self.add_metric(tc_loss, name='time_consistency_loss', aggregation='mean')
        return inputs


class Y_Predict_RMSE_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='predict_y_rmse', **kwargs):
        super(Y_Predict_RMSE_Metric, self).__init__(name=name, **kwargs)

        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.rmse = self.add_weight(name='rmse', initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)
        rmse = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)))

        self.count.assign_add(tf.shape(y_true)[0])
        self.rmse.assign_add(rmse)

    def result(self):
        return self.rmse / tf.cast(self.count, tf.float32)

    def reset_states(self):
        self.count.assign(0)
        self.rmse.assign(0)


class MaskEmbeddingLossMetric(tf.keras.metrics.Metric):
    def __init__(self, c, name='mask_embedding_loss', **kwargs):
        super(MaskEmbeddingLossMetric, self).__init__(name=name, **kwargs)
        self.known_mask = tf.convert_to_tensor(get_known_mask(c.TRAIN_LEN, c.EMBEDDING_LEN))

        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.me_loss = self.add_weight(name='me_loss', initializer=tf.zeros_initializer())
        self.loss_weight = c.LOSS_WEIGHTS.get('masked_embedding_loss', 1.0)

        self.c = c

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)
        mask_embedding_loss = tf.reduce_sum(
            tf.reduce_mean(tf.multiply(tf.square(y_true - y_pred),
                                       self.known_mask), axis=[-1, -2])) * self.loss_weight

        self.count.assign_add(tf.shape(y_true)[0])
        self.me_loss.assign_add(mask_embedding_loss)

    def result(self):
        return self.me_loss / tf.cast(self.count, tf.float32)

    def reset_states(self):
        self.count.assign(0)
        self.me_loss.assign(0)

    def get_config(self):
        config = super(MaskEmbeddingLossMetric, self).get_config()
        config['c'] = self.c
        return config


class ReconstructionLossMetric(tf.keras.metrics.Metric):
    def __init__(self, c, name='reconstruction_loss', **kwargs):
        super(ReconstructionLossMetric, self).__init__(name=name, **kwargs)

        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.r_loss = self.add_weight(name='r_loss', initializer=tf.zeros_initializer())
        self.loss_weight = c.LOSS_WEIGHTS.get('reconstruction_loss', 1.0)

        self.c = c

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)
        reconstruction_loss = tf.reduce_sum(
            tf.reduce_mean(tf.square(y_true - y_pred), axis=[-1, -2])) * self.loss_weight

        self.count.assign_add(tf.shape(y_true)[0])
        self.r_loss.assign_add(reconstruction_loss)

    def result(self):
        return self.r_loss / tf.cast(self.count, tf.float32)

    def reset_states(self):
        self.count.assign(0)
        self.r_loss.assign(0)

    def get_config(self):
        config = super(ReconstructionLossMetric, self).get_config()
        config['c'] = self.c
        return config