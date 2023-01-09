import tensorflow as tf
import scipy.io as sio
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input


def setup_tf():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)

            
def load_data(trainingPath, validationPath,clip_value=300):

    # Loading the training data
    matTR = sio.loadmat(trainingPath)
    eeg = np.transpose(matTR['data']['x'][0][0],(2,1,0))
    Y = matTR['data']['y'][0][0]
    Y = np.reshape(Y,newshape=(Y.shape[0],))
    del matTR

    eeg = np.clip(eeg, a_min=-clip_value, a_max=clip_value)

    # Loading validation data
    matTR = sio.loadmat(validationPath)
    eeg_val = np.transpose(matTR['data']['x'][0][0],(2,1,0))
    Y_val = matTR['data']['y'][0][0]
    Y_val = np.reshape(Y_val,newshape=(Y_val.shape[0]))
    del matTR

    eeg_val = np.clip(eeg_val, a_min=-clip_value, a_max=clip_value)

    return eeg,Y,eeg_val,Y_val


def load_data_noval(path,clip_value=300):
    matTR = sio.loadmat(path)
    eeg = np.asarray(matTR['EEG'])
    del matTR

    eeg = np.clip(eeg,a_min=-clip_value,a_max=clip_value)

    return eeg


def prepare_eeg(trainingPath,validationPath,clip_value=300,normalize=False):
    X,Y,X_val,Y_val = load_data(trainingPath=trainingPath,validationPath=validationPath,clip_value=clip_value)

    X_test = np.asarray(X_val,dtype=np.float32)
    Y_test = np.copy(Y_val)
    del X_val,Y_val

    (N,T,_) = X.shape

    #patientTR = sio.loadmat(patientPath)
    #patientID = np.reshape(np.asarray(patientTR['patient'],dtype=np.uint),newshape=(N,))
    #del patientTR
    #IDs = np.unique(patientID)
#
    #for i,ID in enumerate(IDs[0:3]):
    #    if i==0:
    #        X_val = X[patientID==ID]
    #        Y_val = Y[patientID==ID]
    #    else:
    #        X_val = np.append(X_val,X[patientID==ID],axis=0)
    #        Y_val = np.append(Y_val,Y[patientID==ID],axis=0)
    #for i,ID in enumerate(IDs[3:]):
    #    if i==0:
    #        X_train = X[patientID==ID]
    #        Y_train = Y[patientID==ID]
    #    else:
    #        X_train = np.append(X_train,X[patientID==ID],axis=0)
    #        Y_train = np.append(Y_train,Y[patientID==ID],axis=0)
#
    if normalize:
        eeg_mean = np.mean(X, axis=(0, 1))
        eeg_std = np.std(X, axis=(0, 1))

        X = X/eeg_std
        X_test = X_test/eeg_std
    return (X, Y), (X_test,Y_test)


def get_amir_base(T=900, CH=8):
    input_eeg = Input(shape=(T, CH, 1))
    
    x = Conv2D(filters=3, kernel_size=(10, 1),
               padding='same', activation='relu')(input_eeg)
    x = Conv2D(filters=3, kernel_size=(3, 1),
               padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = Conv2D(filters=5, kernel_size=(5, 1),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=5, kernel_size=(1, 3),
               padding='valid', activation='relu')(x)
    x = MaxPool2D(pool_size=(5, 1), strides=(3, 1), padding='same')(x)
    
    x = Conv2D(filters=7, kernel_size=(5, 1),
               padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(6, 1), strides=(4, 1))(x)
    
    x = Conv2D(filters=10, kernel_size=(37, 1),
               padding='valid', activation='relu')(x)
    
    x = Flatten()(x)
    x = Dense(units=1, activation='sigmoid')(x)
    
    return Model(input_eeg, x)



def make_scheduler(factor=0.5, period=100):
    def scheduler(epoch, lr):
        if ((epoch+1) % period == 0):
            return lr*factor
        else:
            return lr
    return scheduler