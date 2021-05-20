import inspect
import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from model.tcn_ae_skip import TimeConsistencyLoss, Embedding2Y


class TemporalConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 repeat_times: int,
                 residual: bool = False,
                 last_norm: bool = True,
                 last_acti: bool = True,
                 last_drop: bool = True,
                 activation: str = 'relu',
                 norm_way: str = None,
                 kernel_initializer: str = 'he_normal',
                 kernel_regularizer: dict = 'l2',
                 weight_decay=0,
                 dropout_rate: float = 0,
                 **kwargs):
        """

        :param dilation_rate:
        :param nb_filters:
        :param kernel_size:
        :param repeat_times: 重复几次tcn层，一般为1，如果大于1的话一般和 residual=True 配合使用
        :param residual: 是否建立残差连接
        :param last_norm: (仅当 residual=False 的时候可以为 False) 最后一层是否normalization
        :param last_acti: (仅当 residual=False 的时候可以为 False) 最后一层是否使用激活函数
        :param last_drop: (仅当 residual=False 的时候可以为 False) 最后一层是否使用dropout
        :param activation:
        :param norm_way: 如果为None，就不normalization
        :param kernel_initializer:
        :param kernel_regularizer:
        :param weight_decay:
        :param dropout_rate:
        :param kwargs:
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.repeat_times = repeat_times
        self.residual = residual
        self.last_norm = last_norm
        self.last_acti = last_acti
        self.last_drop = last_drop
        self.activation = activation
        self.norm_way = norm_way
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self.build_output_shape = None
        self.layers = []
        self.shape_match_conv = None
        self.final_activation = None

        # 判断last_drop, last_acti, last_norm 和 residual是否冲突
        if self.residual:
            assert self.last_drop, 'residual block must has last drop'
            assert self.last_acti, 'residual block must has last acti'
            assert self.last_norm, 'residual block must has last norm'

        super(TemporalConvBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.layers.append(layer)
        self.layers[-1].build(self.build_output_shape)
        self.build_output_shape = self.layers[-1].compute_output_shape(self.build_output_shape)

    def build(self, input_shape):
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.build_output_shape = input_shape

            for i in range(self.repeat_times):
                # 因果卷积
                with K.name_scope('Conv1D_{}'.format(i)):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(tf.keras.layers.Conv1D(self.nb_filters,
                                                                        self.kernel_size,
                                                                        padding='causal',
                                                                        dilation_rate=self.dilation_rate,
                                                                        kernel_initializer=tf.keras.initializers.get(
                                                                            self.kernel_initializer),
                                                                        kernel_regularizer=tf.keras.regularizers.get(
                                                                            {'class_name': self.kernel_regularizer,
                                                                             'config': self.weight_decay})))

                # normalization layer
                if self.norm_way and (i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_norm)):
                    with K.name_scope('norm_{}_{}'.format(self.norm_way,
                                                          i)):  # name scope used to make sure weights get unique names
                        if self.norm_way == 'bn':
                            self._add_and_activate_layer(tf.keras.layers.BatchNormalization())
                        elif self.norm_way == 'ln':
                            self._add_and_activate_layer(tf.keras.layers.LayerNormalization())
                        else:
                            raise NotImplementedError()
                if i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_acti):
                    self._add_and_activate_layer(tf.keras.layers.Activation(self.activation))
                else:
                    self._add_and_activate_layer(tf.keras.layers.Activation('linear'))

                if i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_drop):
                    self._add_and_activate_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate))

            if self.residual:
                if self.nb_filters != input_shape[-1]:
                    # 1x1 conv to match the shapes (channel dimension).
                    name = 'matching_conv1D'
                    with K.name_scope(name):
                        # make and build this layer separately because it directly uses input_shape
                        self.shape_match_conv = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                                       kernel_size=1,
                                                                       padding='same',
                                                                       name=name,
                                                                       kernel_initializer=self.kernel_initializer)

                else:
                    name = 'matching_identity'
                    self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name=name)

                with K.name_scope(name):
                    self.shape_match_conv.build(self.build_output_shape)
                    self.build_output_shape = self.shape_match_conv.compute_output_shape(self.build_output_shape)

                self.final_activation = tf.keras.layers.Activation(self.activation)
                self.final_activation.build(self.res_output_shape)  # probably isn't necessary

                #############################################
                ##### ？？？？？？？ 不知道有啥用 ???????????
                #############################################
                # this is done to force Keras to add the layers in the list to self._layers
                self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
                self.__setattr__(self.final_activation.name, self.final_activation)

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

        super(TemporalConvBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=True, **kwargs):
        x = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)

        if self.residual:
            x2 = self.shape_match_conv(inputs)
            res_x = tf.keras.layers.add([x, x2])
            res_act_x = self.final_activation(res_x)
            return res_act_x
        return x


def get_dilation_add_idxs(batch_size, time_points, kernel_size, dilation_rate):
    """
    获取反卷积过程中空洞相加的索引
    :param batch_size:
    :param time_points:
    :param kernel_size:
    :param dilation_rate:
    :return:
    """
    final_time_len = time_points + (kernel_size - 1) * dilation_rate
    dilation_add_idxs = []
    for i in range(time_points):
        t_idxs = np.ones(shape=(final_time_len, 2), dtype=np.int32)
        t_idxs[:, 0] *= i
        t_idxs[:, 1] *= (kernel_size + 1)
        for j in range(kernel_size):
            t_idxs[i + j * dilation_rate, 1] = j
        dilation_add_idxs.append(t_idxs)
    dilation_add_idxs = np.stack(dilation_add_idxs)
    dilation_add_idxs = np.tile(dilation_add_idxs, reps=(batch_size, 1, 1, 1))
    return dilation_add_idxs


class MyDeConv1D(tf.keras.layers.Layer):
    def __init__(self, nb_filters, kernel_size, dilation_rate, kernel_initializer, config,
                 kernel_regularizer=None, tied_kernel=None, **kwargs):
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.tied_kernel = tied_kernel  # 是否有传入和encoder layer绑定的权重, shape:[kernel_size, input_dim, output_dim]
        self.config = config
        self.kernel = None
        self.bias = None
        self.gather_idxs = None

        super(MyDeConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.tied_kernel is not None:
            # 如果是提供了tied weights，那么应该检查shape
            kernel_size, encoder_input_dim, encoder_output_dim = self.tied_kernel.shape
            assert encoder_input_dim == self.nb_filters and encoder_output_dim == input_shape[-1] and kernel_size == self.kernel_size, 'wrong shape of tied kernel.'
            self.kernel = tf.transpose(self.tied_kernel, perm=[2, 0, 1])
        else:
            # 否则创建新的variable
            self.kernel = self.add_weight(name='deconv_kernel',
                                          shape=(input_shape[-1], self.kernel_size, self.nb_filters),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)

        # 初始化bias
        self.bias = self.add_weight(name='bias',
                                    shape=(self.nb_filters,),
                                    initializer='zeros',
                                    trainable=True)
        
        self.gather_idxs = get_dilation_add_idxs(self.config.BATCH_SIZE,
                                                 self.config.TRAIN_LEN,
                                                 self.kernel_size,
                                                 self.dilation_rate)
        super(MyDeConv1D, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        input = tf.expand_dims(tf.expand_dims(inputs, axis=-1), axis=-1)

        raw_output = tf.add(tf.multiply(input, self.kernel), self.bias)  # [b, m, input_dim, kernel_size, nb_filters]
        raw_output = tf.reduce_sum(raw_output, axis=-3)  # [b, m, kernel_size, nb_filters]


        # 这样不对！！！！
        # 构造新矩阵，并且赋值
        # final_output = tf.zeros(shape=(self.config.BATCH_SIZE,
        #                                self.config.TRAIN_LEN + (self.kernel_size - 1) * self.dilation_rate,
        #                                self.nb_filters))
        #
        # for i in range(-1, -self.config.TRAIN_LEN - 1, -1):  # 从后往前相加
        #     for j in range(-1, -self.kernel_size - 1, -1):
        #         final_output[:, i - self.dilation_rate * (np.abs(j) - 1), :] += raw_output[:, i, j, :]
        # TODO:先在最后添加0，然后使用tf.gather后相加
        raw_output = tf.concat([raw_output, tf.zeros(shape=(self.config.BATCH_SIZE,
                                                            self.config.TRAIN_LEN,
                                                            1,
                                                            self.nb_filters))], axis=2)
        final_output = tf.gather_nd(raw_output, self.gather_idxs, batch_dims=1)
        final_output = tf.reduce_sum(final_output, axis=1)  # [b, m, m+(kernel_size-1)*d, nb_filters] 在时间维度相加，axis=1
        # print(final_output.shape)
        # 截断后输出（从前面开始截断）
        return final_output[:, -self.config.TRAIN_LEN:, :]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.nb_filters]


class TemporalDeConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 repeat_times: int,
                 config,
                 residual: bool = False,
                 last_norm: bool = True,
                 last_acti: bool = True,
                 last_drop: bool = True,
                 activation: str = 'relu',
                 norm_way: str = None,
                 kernel_initializer: str = 'he_normal',
                 kernel_regularizer: str = 'l2',
                 weight_decay=0,
                 dropout_rate: float = 0,
                 tied_block: tf.keras.layers.Layer = None,
                 **kwargs):
        """

        :param dilation_rate:
        :param nb_filters:
        :param kernel_size:
        :param repeat_times: 重复几次tcn层，一般为1，如果大于1的话一般和 residual=True 配合使用
        :param residual: 是否建立残差连接
        :param last_norm: (仅当 residual=False 的时候可以为 False) 最后一层是否normalization
        :param last_acti: (仅当 residual=False 的时候可以为 False) 最后一层是否使用激活函数
        :param last_drop: (仅当 residual=False 的时候可以为 False) 最后一层是否使用dropout
        :param activation:
        :param norm_way: 如果为None，就不normalization
        :param kernel_initializer:
        :param kernel_regularizer:
        :param weight_decay:
        :param dropout_rate:
        :param tied_block: 绑定权重的模块层，为None就不绑定，重新初始化权重
        :param kwargs:
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.repeat_times = repeat_times
        self.residual = residual
        self.last_norm = last_norm
        self.last_acti = last_acti
        self.last_drop = last_drop
        self.activation = activation
        self.norm_way = norm_way
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.config = config

        self.tied_block = tied_block  # 绑定权重的模块层， 结构需要对称并且和参数对应

        self.build_output_shape = None
        self.layers = []
        self.shape_match_conv = None
        self.final_activation = None

        # 判断last_drop, last_acti, last_norm 和 residual是否冲突
        if self.residual:
            assert self.last_drop, 'residual block must has last drop'
            assert self.last_acti, 'residual block must has last acti'
            assert self.last_norm, 'residual block must has last norm'

        super(TemporalDeConvBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.layers.append(layer)
        self.layers[-1].build(self.build_output_shape)
        self.build_output_shape = self.layers[-1].compute_output_shape(self.build_output_shape)

    def get_tied_kernel(self, repeat_idx):
        if self.tied_block:
            tied_kernel = None
            for layer in self.tied_block.layers:
                # print("here here")
                # print(len(layer.weights))
                for weight in layer.weights:
                    if re.match('.*Conv1D_{}.*kernel.*'.format(repeat_idx), weight.name):
                        tied_kernel = weight
                        print(tied_kernel.name)
            if tied_kernel is None:
                raise Exception('can not find matched kernel!')
            return tied_kernel
        else:
            return None

    def build(self, input_shape):
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.build_output_shape = input_shape

            for i in range(self.repeat_times):
                # 因果卷积
                with K.name_scope('DeConv1D_{}'.format(i)):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(MyDeConv1D(self.nb_filters,
                                                            self.kernel_size,
                                                            dilation_rate=self.dilation_rate,
                                                            config=self.config,
                                                            kernel_initializer=tf.keras.initializers.get(
                                                                self.kernel_initializer),
                                                            kernel_regularizer=tf.keras.regularizers.get(
                                                                {'class_name': self.kernel_regularizer,
                                                                 'config': self.weight_decay}),
                                                            tied_kernel=self.get_tied_kernel(repeat_idx=i)))

                # normalization layer
                if self.norm_way and (i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_norm)):
                    with K.name_scope('norm_{}_{}'.format(self.norm_way,
                                                          i)):  # name scope used to make sure weights get unique names
                        if self.norm_way == 'bn':
                            self._add_and_activate_layer(tf.keras.layers.BatchNormalization())
                        elif self.norm_way == 'ln':
                            self._add_and_activate_layer(tf.keras.layers.LayerNormalization())
                        else:
                            raise NotImplementedError()
                if i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_acti):
                    self._add_and_activate_layer(tf.keras.layers.Activation(self.activation))
                else:
                    self._add_and_activate_layer(tf.keras.layers.Activation('linear'))

                if i < self.repeat_times - 1 or (i == self.repeat_times - 1 and self.last_drop):
                    self._add_and_activate_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate))

            if self.residual:
                if self.nb_filters != input_shape[-1]:
                    # 1x1 conv to match the shapes (channel dimension).
                    name = 'matching_conv1D'
                    with K.name_scope(name):
                        # make and build this layer separately because it directly uses input_shape
                        self.shape_match_conv = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                                       kernel_size=1,
                                                                       padding='same',
                                                                       name=name,
                                                                       kernel_initializer=self.kernel_initializer)

                else:
                    name = 'matching_identity'
                    self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name=name)

                with K.name_scope(name):
                    self.shape_match_conv.build(self.build_output_shape)
                    self.build_output_shape = self.shape_match_conv.compute_output_shape(self.build_output_shape)

                self.final_activation = tf.keras.layers.Activation(self.activation)
                self.final_activation.build(self.res_output_shape)  # probably isn't necessary

                #############################################
                ##### ？？？？？？？ 不知道有啥用 ???????????
                #############################################
                # this is done to force Keras to add the layers in the list to self._layers
                self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
                self.__setattr__(self.final_activation.name, self.final_activation)

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

        super(TemporalDeConvBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=True, **kwargs):
        x = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            # print(layer.name)
            x = layer(x, training=training) if training_flag else layer(x)

        if self.residual:
            x2 = self.shape_match_conv(inputs)
            res_x = tf.keras.layers.add([x, x2])
            res_act_x = self.final_activation(res_x)
            return res_act_x
        return x

class TCN_AE(tf.keras.Model):
    def __init__(self, config, embedding2y=True, *args, **kwargs):
        super(TCN_AE, self).__init__(*args, **kwargs)
        self.config = config
        self.embedding2y = embedding2y

        self.encoder_layers = None  # 编码器层
        self.decoder_layers = None  # 解码器层

        self.tc_loss = None  # 时间一致性损失层
        self.embedding2y_layer = None  # 从embedding到预测Y的层

        self.build_output_shape = None

    def build(self, input_shape):
        print('model build is called')
        self.build_output_shape = input_shape

        # 构建编码器和解码器
        encoder_ouput_shapes = []  # 记录每一层编码器的输出，在skip connection的时候可以用
        self.encoder_layers = self._build_encoder()
        for layer in self.encoder_layers:
            layer.build(self.build_output_shape)
            self.build_output_shape = layer.build_output_shape
            encoder_ouput_shapes.append(self.build_output_shape)

        self.tc_loss = TimeConsistencyLoss(self.config.LOSS_WEIGHTS.get('tc_loss', 1.0),
                                           self.config.TRAIN_LEN,
                                           self.config.EMBEDDING_LEN,
                                           self.config.BATCH_SIZE)
        self.tc_loss.build(self.build_output_shape)

        if self.embedding2y:
            self.embedding2y_layer = Embedding2Y(self.config)
            self.embedding2y_layer.build(self.build_output_shape)

        self.decoder_layers = self._build_decoder(self.encoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            last_output_shape = self.build_output_shape
            if 0 <= i - 1 < len(self.config.DECODER_ENCODER_SKIP_MAP):  # 如果有跳跃连接，那么上一层的输出应该拼接上跳跃链接的维度
                last_output_shape[-1] += encoder_ouput_shapes[self.config.DECODER_ENCODER_SKIP_MAP[i - 1]][-1]

            layer.build(last_output_shape)
            self.build_output_shape = layer.build_output_shape

        super(TCN_AE, self).build(input_shape)  # make sure self.built is True

    def _build_encoder(self):
        layers = []
        if self.config.TCN_BLOCK_RESIDUAL:
            for i, nodes in enumerate(self.config.ENCODER_NODES):
                if self.config.ENCODER_DILATION_RATES is not None:
                    dilation_rate = self.config.ENCODER_DILATION_RATES[i]
                else:
                    dilation_rate = 2 ** i  # 空洞卷积系数
                layers.append(TemporalConvBlock(dilation_rate=dilation_rate,
                                                nb_filters=nodes,
                                                kernel_size=self.config.KERNEL_SIZE,
                                                repeat_times=self.config.TCN_BLOCK_REPEAT_TIMES,
                                                norm_way=self.config.NORMALIZATION,
                                                kernel_initializer=self.config.KERNEL_INITIALIZER,
                                                kernel_regularizer=self.config.KERNEL_REGULARIZER,
                                                weight_decay=self.config.WEIGHT_DECAY,
                                                dropout_rate=self.config.DROP_RATE,
                                                residual=True))
            print('build encoder ........')
            layers.append(TemporalConvBlock(dilation_rate=1,
                                            nb_filters=self.config.ENCODER_NODES[-1],
                                            kernel_size=1,
                                            repeat_times=1,
                                            norm_way=self.config.NORMALIZATION,
                                            kernel_initializer=self.config.KERNEL_INITIALIZER,
                                            kernel_regularizer=self.config.KERNEL_REGULARIZER,
                                            weight_decay=self.config.WEIGHT_DECAY,
                                            dropout_rate=self.config.DROP_RATE,
                                            last_acti=False,
                                            last_norm=False,
                                            last_drop=False))

            return tf.keras.Sequential(layers=layers, name='encoder')
        else:
            for i, nodes in enumerate(self.config.ENCODER_NODES):
                if self.config.ENCODER_DILATION_RATES is not None:
                    dilation_rate = self.config.ENCODER_DILATION_RATES[i]
                else:
                    dilation_rate = 2 ** i  # 空洞卷积系数
                if i == len(self.config.ENCODER_NODES) - 1:
                    last_norm = False
                    last_drop = False
                    last_acti = False
                else:
                    last_norm = True
                    last_drop = True
                    last_acti = True

                layers.append(TemporalConvBlock(dilation_rate=dilation_rate,
                                                nb_filters=nodes,
                                                kernel_size=self.config.KERNEL_SIZE,
                                                repeat_times=self.config.TCN_BLOCK_REPEAT_TIMES,
                                                norm_way=self.config.NORMALIZATION,
                                                kernel_initializer=self.config.KERNEL_INITIALIZER,
                                                kernel_regularizer=self.config.KERNEL_REGULARIZER,
                                                weight_decay=self.config.WEIGHT_DECAY,
                                                dropout_rate=self.config.DROP_RATE,
                                                last_acti=last_acti,
                                                last_norm=last_norm,
                                                last_drop=last_drop,
                                                name='tcn_encoder_block_{}'.format(i),
                                                ))

            return layers

    def _build_decoder(self, encoder_blocks):
        if self.config.DECODER_NODES is not None and len(self.config.DECODER_NODES) > 0:
            decoder_nodes = self.config.DECODER_NODES
        else:
            decoder_nodes = self.config.ENCODER_NODES[:-1][::-1]
            decoder_nodes.append(self.config.INPUT_DIM)

        layers = []

        if self.config.TCN_BLOCK_RESIDUAL:
            for i, nodes in enumerate(decoder_nodes):
                if self.config.DECODER_DILATION_RATES is not None:
                    dilation_rate = self.config.DECODER_DILATION_RATES[i]
                else:
                    dilation_rate = 2 ** i  # 空洞卷积系数
                layers.append(TemporalConvBlock(dilation_rate=dilation_rate,
                                                nb_filters=nodes,
                                                kernel_size=self.config.KERNEL_SIZE,
                                                repeat_times=self.config.TCN_BLOCK_REPEAT_TIMES,
                                                norm_way=self.config.NORMALIZATION,
                                                kernel_initializer=self.config.KERNEL_INITIALIZER,
                                                kernel_regularizer=self.config.KERNEL_REGULARIZER,
                                                weight_decay=self.config.WEIGHT_DECAY,
                                                dropout_rate=self.config.DROP_RATE,
                                                residual=True))
            layers.append(TemporalConvBlock(dilation_rate=1,
                                            nb_filters=decoder_nodes[-1],
                                            kernel_size=1,
                                            repeat_times=1,
                                            norm_way=self.config.NORMALIZATION,
                                            kernel_initializer=self.config.KERNEL_INITIALIZER,
                                            kernel_regularizer=self.config.KERNEL_REGULARIZER,
                                            weight_decay=self.config.WEIGHT_DECAY,
                                            dropout_rate=self.config.DROP_RATE,
                                            last_acti=False,
                                            last_norm=False,
                                            last_drop=False))
            return tf.keras.Sequential(layers=layers, name='decoder')
        else:
            for i, nodes in enumerate(decoder_nodes):
                if self.config.DECODER_DILATION_RATES is not None:
                    dilation_rate = self.config.DECODER_DILATION_RATES[i]
                else:
                    dilation_rate = 2 ** i  # 空洞卷积系数
                if i == len(decoder_nodes) - 1:
                    last_norm = False
                    last_drop = False
                    last_acti = False
                else:
                    last_norm = True
                    last_drop = True
                    last_acti = True
                tied_block = encoder_blocks[self.config.DECODER_ENCODER_TIED_IDX[i]] \
                    if len(self.config.DECODER_ENCODER_TIED_IDX) > 0 else None
                layers.append(TemporalDeConvBlock(dilation_rate=dilation_rate,
                                                  nb_filters=nodes,
                                                  kernel_size=self.config.KERNEL_SIZE,
                                                  repeat_times=self.config.TCN_BLOCK_REPEAT_TIMES,
                                                  config=self.config,
                                                  norm_way=self.config.NORMALIZATION,
                                                  kernel_initializer=self.config.KERNEL_INITIALIZER,
                                                  kernel_regularizer=self.config.KERNEL_REGULARIZER,
                                                  weight_decay=self.config.WEIGHT_DECAY,
                                                  dropout_rate=self.config.DROP_RATE,
                                                  last_acti=last_acti,
                                                  last_norm=last_norm,
                                                  last_drop=last_drop,
                                                  name='tcn_decoder_block_{}'.format(i),
                                                  tied_block=tied_block
                                                  ))

            return layers

    def call(self, inputs, training=None, mask=None):
        x = inputs

        # 记录每一层编码器的输出
        encoder_outputs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
            encoder_outputs.append(x)

        # 解码器包含skip connection
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x, training=training)
            # 建立skip connection
            if i < len(self.config.DECODER_ENCODER_SKIP_MAP):
                skip_encoder_idx = self.config.DECODER_ENCODER_SKIP_MAP[i]
                if skip_encoder_idx >= 0:
                    x = tf.concat([x, encoder_outputs[skip_encoder_idx]], axis=-1)

        embedding = encoder_outputs[-1]
        embedding = self.tc_loss(embedding)
        reconstruct_x = x

        if self.embedding2y:
            target_y = self.embedding2y_layer(embedding)
            return embedding, target_y, reconstruct_x

        else:
            return embedding, reconstruct_x

        # if self.embedding2y:
        #     target_y = self.embedding2y_layer(embedding)
        #     return embedding, target_y
        #
        # else:
        #     return embedding



# tf.keras.layers.Lambda()
# conv = tf.keras.layers.Conv1D(filters=10, kernel_size=3,
#                               padding='causal', dilation_rate=2)
# conv.build(input_shape=[None, 50, 90])
#
# for weight in conv.weights:
#     print(weight.name)
#     print(weight.shape)
# print(conv.weights)

