from model.config import BaseConfig

class MultiConfig(BaseConfig):
    def __init__(self):
        super(MultiConfig, self).__init__()

        self.DATA_NAME = 'lorenz_multi_sample'

        self.INPUT_DIM = 90

        self.TRAIN_LEN = 50

        self.EMBEDDING_LEN = 16

        self.ENCODER_DILATION_RATES = [1, 2, 4, 8]

        self.ENCODER_NODES = [128, 64, 32, self.EMBEDDING_LEN]

        self.DECODER_DILATION_RATES = [8, 4, 2, 1]

        self.DECODER_NODES = [32, 64, 128, self.INPUT_DIM]

        self.DECODER_ENCODER_TIED_IDX = []

        self.DECODER_ENCODER_SKIP_MAP = [2, 1, 0]

        self.TCN_BLOCK_REPEAT_TIMES = 1
        
        self.TCN_BLOCK_RESIDUAL = False
        
        self.KERNEL_SIZE = 3
        
        self.INPUT_DROP_RATE = 0.0
        
        self.TARGET_IDX = 0

        self.LOSS_WEIGHTS = {'masked_embedding_loss': 3.0,
                             'reconstruction_loss': 1.0,
                             'tc_loss': 3.0}

        self.KERNEL_REGULARIZER = 'l2'
        
        self.WEIGHT_DECAY = {'l': 0.0}

        self.LR = 1e-3
        
        self.BATCH_SIZE = 4
        
        self.EPOCHES = 50
        
        self.DROP_RATE = 0.0
        
        self.ACTIVATION = 'relu'

        self.NORMALIZATION = 'ln'

        self.KERNEL_INITIALIZER = 'he_normal'