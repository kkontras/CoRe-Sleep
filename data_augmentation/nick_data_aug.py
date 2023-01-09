import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pandas as pd
from sklearn import model_selection

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv1D, Activation, GlobalAveragePooling1D
from tensorflow.keras.layers import MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Add, Concatenate
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from numpy.random import default_rng

 

utils.setup_tf()
rng = default_rng()

 

training_path = '/volume1/scratch/nseeuws/neonatal/data_conv_training.mat'
testing_path = '/volume1/scratch/nseeuws/neonatal/data_conv_testing.mat'

n_splits = 5

 

(x, y), (x_test, y_test) = utils.prepare_eeg(
    training_path, testing_path, clip_value=300, normalize=True)

 

(N, T, CH) = x.shape

 

skf = model_selection.StratifiedKFold(n_splits=5)

# Model definition

 

def ResBlock(input_layer, n_filters, kernel_size):
    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)

    x = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(x)

    return x


 

def get_resnet_clf():
    input_eeg = Input(shape=(T, CH))

    n_filters = 16
    x0 = Conv1D(filters=n_filters, kernel_size=5, strides=1, padding='same', activation=None)(input_eeg)

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    n_filters = 32
    x0 = Conv1D(filters=n_filters, kernel_size=5, strides=5, padding='same', activation=None)(x0)

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    n_filters = 64
    x0 = Conv1D(filters=n_filters, kernel_size=5, strides=5, padding='same', activation=None)(x0)

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = ResBlock(x0, n_filters=n_filters, kernel_size=5)
    x0 = Add()([x, x0])

    x = GlobalAveragePooling1D()(x0)

    output = Dense(units=1, activation='sigmoid')(x)

    clf = Model(input_eeg, output)
    return clf


 

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=64, shuffle=True):
        super().__init__()
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key_array = np.arange(self.images.shape[0], dtype=np.uint32)
        self.on_epoch_end()
        self.rng = np.random.default_rng()

        (N, T, CH) = images.shape
        self.T = T
        self.CH = CH

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.asarray(self.images[keys], dtype=np.float32)
        y = np.asarray(self.labels[keys], dtype=np.float32)

        x = self.perturb(x)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def perturb(self, x):
        T = self.T
        CH = self.CH
        N = self.batch_size
        # Frequency masking
        p = 0.75
        n_trials = 1
        n_sec = 20
        mask_bool = self.rng.binomial(p=p, n=n_trials, size=(N, 1, CH))
        start = self.rng.integers(low=0, high=(T / 2) + 1 - 60, size=(N, 1, CH))
        duration = self.rng.integers(low=5, high=200, size=(N, 1, CH))

        x_fft = np.fft.rfft(a=x, axis=1)

        for i_batch in range(N):
            for i_ch in range(CH):
                if mask_bool[i_batch, 0, i_ch]:
                    start_ = start[i_batch, 0, i_ch]
                    stop_ = start_ + duration[i_batch, 0, i_ch]
                    x_fft[i_batch, start_:stop_, i_ch] = 0.
        x = np.fft.irfft(a=x_fft, n=T, axis=1)

        # Time masking
        p = 0.75
        n_trials = 1
        n_sec = 20
        mask_bool = self.rng.binomial(p=p, n=n_trials, size=(N, 1, CH))
        start = self.rng.integers(low=0, high=T - 5 * n_sec, size=(N, 1, CH))
        duration = self.rng.integers(low=0, high=30 * n_sec, size=(N, 1, CH))

        for i_batch in range(N):
            for i_ch in range(CH):
                if mask_bool[i_batch, 0, i_ch]:
                    start_ = start[i_batch, 0, i_ch]
                    stop_ = start_ + duration[i_batch, 0, i_ch]
                    x[i_batch, start_:stop_, i_ch] = 0.

        # Scaling
        scaling = self.rng.uniform(low=0.5, high=2., size=(N, 1, CH))
        x = scaling * x

        # Flipping
        flip = np.asarray(self.rng.integers(low=0, high=2, size=(N, 1, 1)), dtype=np.float32)
        flip = 2 * (flip - .5)
        x = flip * x

        return x


# Training

 

batch_size = 256
epochs = 250

 

n_nqs = np.sum(y == 0)
n_qs = np.sum(y == 1)
class_weight = {0: 1, 1: n_nqs / n_qs}

 

scheduler = utils.make_scheduler(factor=0.5, period=10)
schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

callbacks = [schedule, stopper]
# callbacks = []

 

loss = []
loss_val = []
acc_train = []
acc_val = []

kappa = []
auc = []
acc = []
sens = []
spec = []

 

for idx_train, idx_val in skf.split(X=x, y=y):
    print('========  EPOCH  =====')
    x_train = x[idx_train]
    y_train = y[idx_train]

    x_val = x[idx_val]
    y_val = y[idx_val]

    clf = get_resnet_clf()
    optimizer = Adam(lr=1e-3)
    clf.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    generator = DataGenerator(images=x_train, labels=y_train, batch_size=batch_size, shuffle=True)

    history = clf.fit(x=generator,
                      batch_size=batch_size, epochs=250, validation_data=(x_val, y_val),
                      shuffle=True, class_weight=class_weight, callbacks=callbacks, verbose=0)

    loss.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    acc_train.append(history.history['accuracy'])
    acc_val.append(history.history['val_accuracy'])

    y_proba = clf.predict(x_test)[:, 0]
    y_pred = y_proba > 0.5

    auc_ = metrics.roc_auc_score(y_true=y_test, y_score=y_proba)
    kappa_ = metrics.cohen_kappa_score(y1=y_test, y2=y_pred)
    acc_ = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    sens_ = metrics.recall_score(y_true=y_test, y_pred=y_pred)
    spec_ = metrics.recall_score(y_true=y_test == 0, y_pred=y_pred == 0)

    auc.append(auc_)
    kappa.append(kappa_)
    acc.append(acc_)
    sens.append(sens_)
    spec.append(spec_)

    print('Accuracy     -  {0:.3f}'.format(acc_))
    print('Kappa        -  {0:.3f}'.format(kappa_))
    print('AUC          -  {0:.3f}'.format(auc_))
    print('Sensitivity  -  {0:.3f}'.format(sens_))
    print('Specificity  -  {0:.3f}'.format(spec_))

# Storing run data

 

df = pd.DataFrame({'Accuracy': acc, 'Kappa': kappa, 'AUC': auc, 'Sensitivity': sens, 'Specificity': spec})

 

df.describe()

 

# df.to_csv('Data/aug_full.csv')

 
