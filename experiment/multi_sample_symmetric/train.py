import sys
sys.path.append("../..")
from model.symmetric_tcn import TCN_AE
from model.tcn_ae_skip import MaskEmbeddingLoss, MaskEmbeddingLossMetric, ReconstructionLoss, ReconstructionLossMetric, Y_Predict_RMSE_Metric, ZeroLoss
from data_process import data_loader
from experiment.multi_sample_symmetric.multi_config import MultiConfig
import tensorflow as tf
import numpy as np
import os
import datetime


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, target_idx, train_len, embedding_len, select_idxs, data_matrix, batch_size, shuffle=True):
        """
        生成dropout的输入数据，以及embedding matrix、
        :param select_idxs: 时间点索引列表，多样本开始预测的时间点
        :param data_matrix: [time_len, input_dim]  所有数据
        :param batch_size: [train_len + ]
        """
        self.target_idx = target_idx
        self.select_idxs = select_idxs

        self.data_matrix = data_matrix
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_len = train_len
        self.embedding_len = embedding_len

    def __len__(self):
        return len(self.select_idxs) // self.batch_size

    def __getitem__(self, item):
        batched_input_data = []
        batched_target_embedding = []
        batched_target_y = []

        for i in range(self.batch_size):

            t_idx = self.select_idxs[item*self.batch_size + i]  # 预测开始的时间点
            input_data = self.data_matrix[t_idx - self.train_len: t_idx]
            total_y = self.data_matrix[t_idx - self.train_len: t_idx + self.embedding_len - 1, self.target_idx]

            target_embedding = []
            for i in range(self.train_len):  # train_len
                target_embedding.append(total_y[i:i + self.embedding_len])
            target_embedding = np.stack(target_embedding)

            batched_target_y.append(total_y[- self.embedding_len + 1:])
            batched_input_data.append(input_data)
            batched_target_embedding.append(target_embedding)

        batched_input_data = np.stack(batched_input_data)
        batched_target_embedding = np.stack(batched_target_embedding)
        batched_target_y = np.stack(batched_target_y)
        return batched_input_data, [batched_target_embedding, batched_target_y, batched_input_data]
        # return batched_input_data, [batched_target_embedding, batched_target_y]

    def get_item(self, item):
        return self.__getitem__(item)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.select_idxs)




if __name__ == '__main__':

    gpu_idx = 1
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)

    config = MultiConfig()

    data_dir = '../../data_files/lorenz'

    lorenz_data = data_loader.load_lorenz_data(data_dir, config.INPUT_DIM // 3,
                                               skip_time_num=2000, time_invariant=True, time=5000)

    epoch_samples = 1000
    select_idxs = data_loader.get_select_idxs(lorenz_data.shape[0],
                                              config.TRAIN_LEN,
                                              config.EMBEDDING_LEN,
                                              epoch_samples)

    time_stamp = datetime.datetime.now()
    log_dir = '({}) {}_tcn_symmetric_ae_d={}_m={}_L={}_resample_rate={:.2f}_dilation_rate={}_nodes={}'.format(
        config.DATA_NAME,
        time_stamp.strftime('%Y_%m_%d-%H_%M_%S'),
        config.INPUT_DIM,
        config.TRAIN_LEN, config.EMBEDDING_LEN,
        config.INPUT_DROP_RATE,
        config.ENCODER_DILATION_RATES,
        config.ENCODER_NODES
    )
    save_dir = os.path.join('../../logs', log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tf.keras.backend.clear_session()  # 清空GPU缓存

    data_generator = DataGenerator(target_idx=config.TARGET_IDX,
                                   train_len=config.TRAIN_LEN,
                                   embedding_len=config.EMBEDDING_LEN,
                                   select_idxs=select_idxs,
                                   data_matrix=lorenz_data,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=True)

    file_pattern = 'epoch:{epoch:04d}_loss:{loss:.3f}_e_loss:{output_1_mask_embedding_loss:.3f}' \
                   '_tc_loss:{time_consistency_loss:.3f}_r_loss:{output_3_reconstruction_loss:.3f}' \
                   '_predict_loss:{output_2_predict_y_rmse:.3f}.h5'

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                                      patience=3, min_lr=1e-5),
                 tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
                     save_dir, file_pattern),
                                                    save_weights_only=True,
                                                    save_freq='epoch')]

    optimizer = tf.keras.optimizers.Adam(config.LR)
    model = TCN_AE(config)
    model.compile(optimizer=optimizer,
                  loss=[MaskEmbeddingLoss(config),
                        ZeroLoss(),
                        ReconstructionLoss(config)],
                  metrics=[[MaskEmbeddingLossMetric(c=config)],
                           [Y_Predict_RMSE_Metric()],
                           [ReconstructionLossMetric(c=config)]])

    model.fit(x=data_generator,
              epochs=config.EPOCHES,
              max_queue_size=50,
              use_multiprocessing=False,
              callbacks=callbacks
              )


