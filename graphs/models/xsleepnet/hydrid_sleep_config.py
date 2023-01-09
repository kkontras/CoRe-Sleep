class Config(object):
    def __init__(self):

        # common settings
        self.epoch_seq_len = 20 # seq_len
        self.nchannel = 3 # number of channels
        self.nclass = 5

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 10*self.epoch_seq_len
        self.batch_size = 32
        self.evaluate_every = 100
        self.checkpoint_every = 100

        # seqsleepnet settings
        self.seq_ndim = 129 # freq
        self.seq_frame_seq_len = 29 # time

        #self.seq_frame_step = self.seq_frame_seq_len
        #self.seq_epoch_step = self.epoch_seq_len
        self.seq_nhidden1 = 64
        self.seq_nlayer1 = 1
        self.seq_attention_size1 = 32
        self.seq_nhidden2 = 64
        self.seq_nlayer2 = 1

        self.seq_nfilter = 32
        self.seq_nfft = 256
        self.seq_samplerate = 100
        self.seq_lowfreq = 0
        self.seq_highfreq = 50

        # deepsleepnet settings
        self.deep_nlayer = 2
        #self.deep_ndim = 1  # frequency dimension
        self.deep_ntime = 3000  # time dimension
        self.deep_nhidden = 256  # nb of neurons in the hidden layer of the GRU cell
        #self.deep_nstep = 20  # the number of time steps per series

        self.dropout_rnn = 0.75
        self.dropout_cnn = 0.5

        self.early_stop_count = 200

        self.num_fold_training_data = 37 # the number of folds to parition the training subjects. To circumvent the memory
                                        # problem when the data is large, only one fold of the data is alternatively loaded at a time.
        self.num_fold_testing_data = 10

        self.warmup_evaluate_step = 20