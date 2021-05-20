import model.utils as utils
from model.symmetric_tcn import TCN_AE
from data_process import data_loader
from experiment.multi_sample_symmetric.multi_config import MultiConfig
from experiment.multi_sample_symmetric.train import DataGenerator
import tensorflow as tf
import numpy as np
import pickle
import os
import time

if __name__ == '__main__':
    gpu_idx = 1
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)

    config = MultiConfig()
    config.BATCH_SIZE = 1  # 预测的批大小为1

    data_dir = '../../data_files/lorenz'
    lorenz_data = data_loader.load_lorenz_data(data_dir, config.INPUT_DIM // 3,
                                               skip_time_num=2000, time_invariant=True, time=5000)

    # log_dir = os.path.join('../../logs',
    #                       'The log folder of training log files')
    # weight_file_name = 'The file containing saved parameters of STCN model'
   

    pic_results_dir = os.path.join(log_dir, 'all_dim_90_pic_results')
    if not os.path.exists(pic_results_dir):
        os.makedirs(pic_results_dir)

    epoch_samples = 1000
    select_idxs = data_loader.get_select_idxs(lorenz_data.shape[0],
                                              config.TRAIN_LEN,
                                              config.EMBEDDING_LEN,
                                              epoch_samples)

    data_generator = DataGenerator(target_idx=config.TARGET_IDX,
                                   train_len=config.TRAIN_LEN,
                                   embedding_len=config.EMBEDDING_LEN,
                                   select_idxs=select_idxs,
                                   data_matrix=lorenz_data,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=True)

    model = TCN_AE(config)
    model.build(input_shape=(None, config.TRAIN_LEN, config.INPUT_DIM))
    model.load_weights(os.path.join(log_dir, weight_file_name), by_name=True)

    rmse_errors = []
    train_ys = []
    label_ys = []
    predict_ys = []
    start_t = time.time()
    for i in range(len(select_idxs)):
        t_idx = select_idxs[i]

        data = data_generator.get_item(i)
        input_data = data[0]

        target_y = np.squeeze(data[1][1])
        predict_y = np.squeeze(model(input_data, training=False)[1][0])
        total_y = np.concatenate([input_data[0][:, config.TARGET_IDX], target_y])

        train_ys.append(input_data[0][:, config.TARGET_IDX])
        label_ys.append(target_y)
        predict_ys.append(predict_y)

        rmse_error = utils.rmse(predict_y, target_y)
        rmse_errors.append(rmse_error)

        # utils.draw_cmp_curves(file_path=os.path.join(pic_results_dir,
        #                                              't_idx:{}_rmse:{:.3f}.png'.format(t_idx, rmse_error)),
        #                       predict_len=len(target_y),
        #                       all_y=total_y,
        #                       predict_results=[predict_y],
        #                       func_names=[config.DATA_NAME],
        #                       colors=['red'],
        #                       fig_size=(8, 5))
        if (i + 1) % 100 == 0:
            print(i + 1)
    end_t = time.time()
    print('total time: {:.2f}ms'.format((end_t - start_t) * 1000))
    print(np.mean(rmse_errors))
    print(select_idxs[np.argsort(rmse_errors)[:50]])

    # 保存一下每个样本的损失
    rmse_errors = np.array(rmse_errors)
    np.save(os.path.join(pic_results_dir, 'samples_rmse_errors.npy'), rmse_errors)

    train_ys = np.array(train_ys, dtype=np.float32)
    label_ys = np.array(label_ys, np.float32)
    predict_ys = np.array(predict_ys, np.float32)

    with open(os.path.join(pic_results_dir, 'pred_results.pkl'), 'wb') as file:
        pickle.dump([train_ys, label_ys, predict_ys], file)
