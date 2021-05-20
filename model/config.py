
class BaseConfig(object):
    def __init__(self):
        ############################
        # 数据相关参数
        ##########################
        self.DATA_NAME = None
        # 输入的数据维度
        self.INPUT_DIM = None
        # 编码器的节点数量
        self.ENCODER_NODES = None
        # 解码器的节点数量
        self.DECODER_NODES = None
        # 训练长度
        self.TRAIN_LEN = None
        # 嵌入长度
        self.EMBEDDING_LEN = None
        # 输入丢弃率，类似于resample
        self.INPUT_DROP_RATE = None
        # 目标变量索引
        self.TARGET_IDX = None


        ############################
        # 模型相关参数
        ############################

        # 损失权重
        self.LOSS_WEIGHTS = {'tc_loss': 1.0,
                             'masked_embedding_loss': 1.0,
                             'reconstruction_loss': 1.0}

        # 正则化方式
        self.KERNEL_REGULARIZER = 'l1'
        # 正则化权重
        self.WEIGHT_DECAY = {'l': 0}
        # self.WEIGHT_DECAY = {'l1': 1e-3, 'l2': 1e-3}  # 'l1_l2'
        # 学习速率
        self.LR = 1e-3
        # 批大小
        self.BATCH_SIZE = None
        # drop
        self.DROP_RATE = 0.0
        # 激活函数
        self.ACTIVATION = 'elu'
        # 层间归一化方法
        self.NORMALIZATION = 'bn'
        # self.NORMALIZATION = 'ln'
        # self.NORMALIZATION = None
        # 权重初始化方式
        self.KERNEL_INITIALIZER = 'he_normal'
