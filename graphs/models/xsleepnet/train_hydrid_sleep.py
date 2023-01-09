import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from hydrid_sleep_net import Hydrid_Sleep_Net
from hydrid_sleep_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

#from datagenerator_from_list_v2 import DataGenerator
#from equaldatagenerator_from_list_v2 import EqualDataGenerator

from datagenerator_wrapper import DataGeneratorWrapper

from gradient_policy import GradientPolicy

from scipy.io import loadmat

import copy

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_train_data_check", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data_check", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data_check", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

# seqsleepnet settings
tf.app.flags.DEFINE_float("dropout_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_attention_size1", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_nhidden2", 64, "Sequence length (default: 20)")

# deepsleepnet settings
tf.app.flags.DEFINE_float("dropout_cnn", 0.5, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("deep_nhidden", 256, "Sequence length (default: 20)")

# common settings
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 32)")

# flag for early stopping
tf.app.flags.DEFINE_boolean("early_stopping", False, "whether to apply early stopping (default: False)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout_rnn = FLAGS.dropout_rnn
config.epoch_seq_len = FLAGS.seq_len
config.seq_nfilter = FLAGS.seq_nfilter
config.seq_nhidden1 = FLAGS.seq_nhidden1
config.seq_nhidden2 = FLAGS.seq_nhidden2
config.seq_attention_size1 = FLAGS.seq_attention_size1

config.dropout_cnn = FLAGS.dropout_cnn
#config.epoch_seq_len = FLAGS.seq_len
#config.epoch_step = FLAGS.seq_len
config.deep_nhidden = FLAGS.deep_nhidden

eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

# 1 channel case
if (not eog_active and not emg_active):
    print("eeg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    train_gen_check_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data_check),
                                             num_fold=1,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             num_fold=1,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    #test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
    #                                         num_fold=1,
    #                                         data_shape_1=[config.deep_ntime],
    #                                         data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
    #                                         seq_len = config.epoch_seq_len,
    #                                         shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_check_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    valid_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    #test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    nchannel = 1

elif(eog_active and not emg_active):
    print("eeg and eog active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    train_gen_check_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data_check),
                                                   eog_filelist=os.path.abspath(FLAGS.eog_train_data_check),
                                             num_fold=1,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                                   eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
                                             num_fold=1,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    #test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
    #                                              eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
    #                                         num_fold=1,
    #                                         data_shape_1=[config.deep_ntime],
    #                                         data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
    #                                         seq_len = config.epoch_seq_len,
    #                                         shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    train_gen_check_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    train_gen_check_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    valid_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    valid_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    #test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    #test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    nchannel = 2
elif(eog_active and emg_active):
    print("eeg, eog, and emg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    train_gen_check_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data_check),
                                                   eog_filelist=os.path.abspath(FLAGS.eog_train_data_check),
                                                   emg_filelist=os.path.abspath(FLAGS.emg_train_data_check),
                                             num_fold=1,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                                   eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
                                                   emg_filelist=os.path.abspath(FLAGS.emg_eval_data),
                                             num_fold=1,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    #test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
    #                                              eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
    #                                              emg_filelist=os.path.abspath(FLAGS.emg_eval_data),
    #                                         num_fold=1,
    #                                         data_shape_1=[config.deep_ntime],
    #                                         data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
    #                                         seq_len = config.epoch_seq_len,
    #                                         shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    train_gen_wrapper.compute_emg_normalization_params()
    train_gen_check_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    train_gen_check_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    train_gen_check_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    valid_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    valid_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    valid_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    #test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    #test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    #test_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    nchannel = 3

# as there is only one fold, there is only one partition consisting all subjects,
# and next_fold should be called only once
train_gen_check_wrapper.new_subject_partition() # next data fold
train_gen_check_wrapper.next_fold()
valid_gen_wrapper.new_subject_partition() # next data fold
valid_gen_wrapper.next_fold()
#test_gen_wrapper.new_subject_partition()
#test_gen_wrapper.next_fold()

config.nchannel = nchannel


# variable to keep track of best fscore
#best_fscore = 0.0
#best_acc = 0.0
#best_kappa = 0.0
#min_loss = float("inf")
# variable to keep track of performance for model selection
best_acc_joint = 0.0
best_acc_all = 0.0
best_fscore_joint = 0.0
best_fscore_all = 0.0
best_kappa_joint = 0.0
best_kappa_all = 0.0
# Training
# ==================================================

# Initialize equal weights for all three branches of the network
w1 = 1.0/3
w2 = 1.0/3
w3 = 1.0/3

# to saved the weights of the selected model
best_w1_acc = w1
best_w2_acc = w2
best_w3_acc = w3

best_w1_fscore = w1
best_w2_fscore = w2
best_w3_fscore = w3

best_w1_kappa = w1
best_w2_kappa = w2
best_w3_kappa = w3

train_size = len(train_gen_check_wrapper.gen.reduce_data_index)
valid_size = len(valid_gen_wrapper.gen.reduce_data_index)
gp_tf = GradientPolicy(train_size=train_size, valid_size=valid_size, average_win=20)
gp_raw = GradientPolicy(train_size=train_size, valid_size=valid_size, average_win=20)
gp_joint = GradientPolicy(train_size=train_size, valid_size=valid_size, average_win=20)

early_stop_count_all_acc = 0
early_stop_count_joint_acc = 0
early_stop_count_all_fscore = 0
early_stop_count_joint_fscore = 0
early_stop_count_all_kappa = 0
early_stop_count_joint_kappa = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    #session_conf.gpu_options.allow_growth = False
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = Hydrid_Sleep_Net(config=config)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            #optimizer = tnf.train.AdamOptimizer(1e-4)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # gradient clipping
            # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow/36501922
            #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
            #train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

        '''
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(han.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        '''

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())

        def train_step(x1_batch, x2_batch, y_batch):
            """
            A single training step
            """
            print("w1 {}, w2 {}, w3 {}".format(w1, w2, w3))

            seq_frame_seq_len = np.ones(len(x1_batch)*config.epoch_seq_len,dtype=int) * config.seq_frame_seq_len
            epoch_seq_len = np.ones(len(x1_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
              net.input_x1: x1_batch,
              net.input_x2: x2_batch,
              net.input_y: y_batch,
              net.dropout_rnn: config.dropout_rnn,
              net.epoch_seq_len: epoch_seq_len,
              net.seq_frame_seq_len: seq_frame_seq_len,
              net.dropout_cnn: config.dropout_cnn,
              net.w1 : w1,
              net.w2 : w2,
              net.w3 : w3,
              net.istraining: 1
            }
            _, step, output_loss, total_loss, accuracy = sess.run(
               [train_op, global_step, net.output_loss, net.loss, net.accuracy],
               feed_dict)
            return step, output_loss, total_loss, accuracy

        def dev_step(x1_batch, x2_batch, y_batch):
            seq_frame_seq_len = np.ones(len(x1_batch)*config.epoch_seq_len,dtype=int) * config.seq_frame_seq_len
            epoch_seq_len = np.ones(len(x1_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x1: x1_batch,
                net.input_x2: x2_batch,
                net.input_y: y_batch,
                net.dropout_rnn: 1.0,
                net.epoch_seq_len: epoch_seq_len,
                net.seq_frame_seq_len: seq_frame_seq_len,
                net.dropout_cnn: 1.0,
                net.w1 : w1,
                net.w2 : w2,
                net.w3 : w3,
                net.istraining: 0
            }
            deep_loss, seq_loss, joint_loss, output_loss, total_loss, \
            deep_yhat, seq_yhat, joint_yhat, yhat = sess.run(
                   [net.deep_loss, net.seq_loss, net.joint_loss, net.output_loss, net.loss,
                    net.deep_predictions, net.seq_predictions, net.joint_predictions, net.predictions], feed_dict)
            return deep_loss, seq_loss, joint_loss, output_loss, total_loss, deep_yhat, seq_yhat, joint_yhat, yhat

        def _evaluate(gen, log_filename):
            # Validate the model on the entire evaluation test set after each epoch
            output_loss1 =0
            output_loss2 =0
            output_loss3 =0
            output_loss =0
            total_loss = 0
            deep_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            seq_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            joint_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])

            factor = 60

            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x1_batch, x2_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = joint_yhat_[n]
                    yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x1_batch, x2_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = joint_yhat_[n]
                    yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = yhat_[n]
            deep_yhat = deep_yhat + 1
            seq_yhat = seq_yhat + 1
            joint_yhat = joint_yhat + 1
            yhat = yhat + 1

            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} {:g} {:g} {:g} ".format(output_loss1, output_loss2, output_loss3, output_loss, total_loss))
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(deep_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    text_file.write("{:g} ".format(acc_n))
                    acc1 += acc_n
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(seq_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    text_file.write("{:g} ".format(acc_n))
                    acc2 += acc_n
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(joint_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    text_file.write("{:g} ".format(acc_n))
                    acc3 += acc_n
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    if n == config.epoch_seq_len - 1:
                        text_file.write("{:g} \n".format(acc_n))
                    else:
                        text_file.write("{:g} ".format(acc_n))
                    acc += acc_n
            acc1 /= config.epoch_seq_len
            acc2 /= config.epoch_seq_len
            acc3 /= config.epoch_seq_len
            acc /= config.epoch_seq_len

            return acc1, acc2, acc3, acc, \
                   deep_yhat, seq_yhat, joint_yhat, yhat, \
                   output_loss1, output_loss2, output_loss3, \
                   output_loss, total_loss

        def _evaluate_reduce(gen, log_filename):
            # Validate the model on the entire evaluation test set after each epoch
            output_loss1 =0
            output_loss2 =0
            output_loss3 =0
            output_loss =0
            total_loss = 0
            deep_yhat = np.zeros([config.epoch_seq_len, len(gen.reduce_data_index)])
            seq_yhat = np.zeros([config.epoch_seq_len, len(gen.reduce_data_index)])
            joint_yhat = np.zeros([config.epoch_seq_len, len(gen.reduce_data_index)])
            yhat = np.zeros([config.epoch_seq_len, len(gen.reduce_data_index)])

            factor = 60

            num_batch_per_epoch = np.floor(len(gen.reduce_data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x1_batch, x2_batch, y_batch, label_batch_ = gen.next_batch_reduce(factor*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = joint_yhat_[n]
                    yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.reduce_pointer < len(gen.reduce_data_index)):
                actual_len, x1_batch, x2_batch, y_batch, label_batch_ = gen.rest_batch_reduce(factor*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.reduce_data_index)] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.reduce_data_index)] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.reduce_data_index)] = joint_yhat_[n]
                    yhat[n, (test_step-1)*factor*config.batch_size : len(gen.reduce_data_index)] = yhat_[n]
            deep_yhat = deep_yhat + 1
            seq_yhat = seq_yhat + 1
            joint_yhat = joint_yhat + 1
            yhat = yhat + 1

            # groundtruth
            y = np.zeros([config.epoch_seq_len, len(gen.reduce_data_index)])
            for n in range(config.epoch_seq_len):
                y[n,:] = gen.label[gen.reduce_data_index - (config.epoch_seq_len - 1) + n]

            deep_yhat = np.reshape(deep_yhat, len(gen.reduce_data_index)*config.epoch_seq_len)
            seq_yhat = np.reshape(seq_yhat, len(gen.reduce_data_index)*config.epoch_seq_len)
            joint_yhat = np.reshape(joint_yhat, len(gen.reduce_data_index)*config.epoch_seq_len)
            yhat = np.reshape(yhat, len(gen.reduce_data_index)*config.epoch_seq_len)
            y = np.reshape(y, len(gen.reduce_data_index)*config.epoch_seq_len)

            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} {:g} {:g} {:g} ".format(output_loss1, output_loss2, output_loss3, output_loss, total_loss))
                acc1 = accuracy_score(y, deep_yhat) # due to zero-indexing
                text_file.write("{:g} ".format(acc1))
                acc2 = accuracy_score(y, seq_yhat) # due to zero-indexing
                text_file.write("{:g} ".format(acc2))
                acc3 = accuracy_score(y, joint_yhat) # due to zero-indexing
                text_file.write("{:g} ".format(acc3))
                acc = accuracy_score(y, yhat) # due to zero-indexing

                fscore1 = f1_score(y, deep_yhat, average='macro')  # due to zero-indexing
                text_file.write("{:g} ".format(fscore1))
                fscore2 = f1_score(y, seq_yhat, average='macro')  # due to zero-indexing
                text_file.write("{:g} ".format(fscore2))
                fscore3 = f1_score(y, joint_yhat, average='macro')  # due to zero-indexing
                text_file.write("{:g} ".format(fscore3))
                fscore = f1_score(y, yhat, average='macro')  # due to zero-indexing

                kappa1 = cohen_kappa_score(deep_yhat, y)  # due to zero-indexing
                text_file.write("{:g} ".format(kappa1))
                kappa2 = cohen_kappa_score(seq_yhat, y)  # due to zero-indexing
                text_file.write("{:g} ".format(kappa2))
                kappa3 = cohen_kappa_score(joint_yhat, y)  # due to zero-indexing
                text_file.write("{:g} ".format(kappa3))
                kappa = cohen_kappa_score(yhat, y)  # due to zero-indexing
                text_file.write("{:g} \n".format(kappa))


            return acc1, acc2, acc3, acc, \
                   fscore1, fscore2, fscore3, fscore, \
                   kappa1, kappa2, kappa3, kappa, \
                   deep_yhat, seq_yhat, joint_yhat, yhat, \
                   output_loss1, output_loss2, output_loss3, \
                   output_loss, total_loss


        print("{} Start validation 0".format(datetime.now()))
        eval_acc1_0, eval_acc2_0, eval_acc3_0, eval_acc_0, \
        eval_fscore1_0, eval_fscore2_0, eval_fscore3_0, eval_fscore_0, \
        eval_kappa1_0, eval_kappa2_0, eval_kappa3_0, eval_kappa_0, \
        eval_yhat1_0, eval_yhat2_0, eval_yhat3_0, eval_yhat_0, \
        eval_output_loss1_0, eval_output_loss2_0, eval_output_loss3_0, eval_output_loss_0, eval_total_loss_0 = \
            _evaluate_reduce(gen=valid_gen_wrapper.gen, log_filename="eval_result_log.txt")
        train_acc1_0, train_acc2_0, train_acc3_0, train_acc_0, \
        train_fscore1_0, train_fscore2_0, train_fscore3_0, train_fscore_0, \
        train_kappa1_0, train_kappa2_0, train_kappa3_0, train_kappa_0, \
        train_yhat1_0, train_yhat2_0, train_yhat3_0, train_yhat_0, \
        train_output_loss1_0, train_output_loss2_0, train_output_loss3_0, train_output_loss_0, train_total_loss_0 = \
            _evaluate_reduce(gen=train_gen_check_wrapper.gen, log_filename="train_result_log.txt")
        #test_acc1_0, test_acc2_0, test_acc3_0, test_acc_0, \
        #test_yhat1_0, test_yhat2_0, test_yhat3_0, test_yhat_0, \
        #test_output_loss1_0, test_output_loss2_0, test_output_loss3_0, test_output_loss_0, test_total_loss_0 = \
        #    evaluate(gen=test_gen_wrapper.gen, log_filename="test_result_log.txt")
        train_gen_check_wrapper.gen.reset_reduce_pointer()
        valid_gen_wrapper.gen.reset_reduce_pointer()
        #test_gen_wrapper.gen.reset_reduce_pointer()

        gp_raw.add_point(train_output_loss1_0, eval_output_loss1_0)
        gp_tf.add_point(train_output_loss2_0, eval_output_loss2_0)
        gp_joint.add_point(train_output_loss3_0, eval_output_loss3_0)

        start_time = time.time()
        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            train_gen_wrapper.new_subject_partition()

            for data_fold in range(config.num_fold_training_data):
                # load data of the current fold
                train_gen_wrapper.next_fold()
                # IMPORTANT HERE: the number of epoch is reduced by a factor of config.seq_len to encourage scanning through the subjects quicker
                # then the number of training epoch need to increased by a factor of config.seq_len accordingly (in the config file)
                train_batches_per_epoch = np.floor(len(train_gen_wrapper.gen.data_index) / config.batch_size / config.epoch_seq_len).astype(np.uint32)
                step = 1
                while step < train_batches_per_epoch:
                    # Get a batch
                    x1_batch, x2_batch, y_batch, label_batch = train_gen_wrapper.gen.next_batch(config.batch_size)
                    train_step_, train_output_loss_, train_total_loss_, train_acc_ = train_step(x1_batch, x2_batch, y_batch)
                    time_str = datetime.now().isoformat()

                    # average acc
                    acc_ = 0
                    for n in range(config.epoch_seq_len):
                        acc_ += train_acc_[n]
                    acc_ /= config.epoch_seq_len

                    print("{}: step {}, output_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_))
                    step += 1

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % config.evaluate_every == 0:
                        # Validate the model on the entire evaluation test set after each epoch
                        print("{} Start validation".format(datetime.now()))

                        eval_acc1, eval_acc2, eval_acc3, eval_acc, \
                        eval_fscore1, eval_fscore2, eval_fscore3, eval_fscore, \
                        eval_kappa1, eval_kappa2, eval_kappa3, eval_kappa, \
                        eval_yhat1, eval_yhat2, eval_yhat3, eval_yhat, \
                        eval_output_loss1, eval_output_loss2, eval_output_loss3, eval_output_loss, eval_total_loss = \
                            _evaluate_reduce(gen=valid_gen_wrapper.gen, log_filename="eval_result_log.txt")
                        train_acc1, train_acc2, train_acc3, train_acc, \
                        train_fscore1, train_fscore2, train_fscore3, train_fscore, \
                        train_kappa1, train_kappa2, train_kappa3, train_kappa, \
                        train_yhat1, train_yhat2, train_yhat3, train_yhat, \
                        train_output_loss1, train_output_loss2, train_output_loss3, train_output_loss, train_total_loss = \
                            _evaluate_reduce(gen=train_gen_check_wrapper.gen, log_filename="train_result_log.txt")
                        #test_acc1, test_acc2, test_acc3, test_acc, \
                        #test_yhat1, test_yhat2, test_yhat3, test_yhat, \
                        #test_output_loss1, test_output_loss2, test_output_loss3, test_output_loss, test_total_loss = \
                        #    _evaluate_reduce(gen=test_gen_wrapper.gen, log_filename="test_result_log.txt")


                        saved_file = 0
                        #if(eval_output_loss <= min_loss ):
                        #    min_loss = eval_output_loss

                        early_stop_count_all_acc += 1
                        if(eval_acc >= best_acc_all):
                            early_stop_count_all_acc = 0 # reset

                            best_acc_all = eval_acc
                            best_w1_acc = w1
                            best_w2_acc = w2
                            best_w3_acc = w3
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                            save_path = saver.save(sess, checkpoint_name)
                            saved_file = 1

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_acc_all')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best_all_acc.txt"), "a") as text_file:
                                #text_file.write("{:g} {:g} {:g}\n".format(train_acc, eval_acc, test_acc))
                                text_file.write("{:g} {:g} {:g} {:g} {:g} {:g}\n".format(train_acc, eval_acc, train_fscore, eval_fscore, train_kappa, eval_kappa))

                        early_stop_count_all_fscore += 1
                        if (eval_fscore >= best_fscore_all):
                            early_stop_count_all_fscore = 0  # reset

                            best_fscore_all = eval_fscore
                            best_w1_fscore = w1
                            best_w2_fscore = w2
                            best_w3_fscore = w3
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) + '.ckpt')
                            if (saved_file == 0):
                                save_path = saver.save(sess, checkpoint_name)
                                saved_file = 1

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_fscore_all')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best_all_fscore.txt"), "a") as text_file:
                                # text_file.write("{:g} {:g} {:g}\n".format(train_acc, eval_acc, test_acc))
                                text_file.write(
                                    "{:g} {:g} {:g} {:g} {:g} {:g}\n".format(train_acc, eval_acc, train_fscore,
                                                                             eval_fscore, train_kappa, eval_kappa))

                        early_stop_count_all_kappa += 1
                        if (eval_kappa >= best_kappa_all):
                            early_stop_count_all_kappa = 0  # reset

                            best_kappa_all = eval_kappa
                            best_w1_kappa = w1
                            best_w2_kappa = w2
                            best_w3_kappa = w3
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) + '.ckpt')
                            if (saved_file == 0):
                                save_path = saver.save(sess, checkpoint_name)
                                saved_file = 1

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_kappa_all')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best_all_kappa.txt"), "a") as text_file:
                                text_file.write(
                                    "{:g} {:g} {:g} {:g} {:g} {:g}\n".format(train_acc, eval_acc, train_fscore,
                                                                             eval_fscore, train_kappa, eval_kappa))

                        early_stop_count_joint_acc += 1
                        if(eval_acc3 >= best_acc_joint):
                            early_stop_count_joint_acc = 0 # reset

                            best_acc_joint = eval_acc3
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                            if(saved_file == 0):
                                save_path = saver.save(sess, checkpoint_name)
                                saved_file = 1

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_acc_joint')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best_joint_acc.txt"), "a") as text_file:
                                text_file.write("{:g} {:g} {:g} {:g} {:g} {:g}\n".format(train_acc3, eval_acc3, train_fscore3, eval_fscore3, train_kappa3, eval_kappa3))

                        early_stop_count_joint_fscore += 1
                        if (eval_fscore3 >= best_fscore_joint):
                            early_stop_count_joint_fscore = 0  # reset

                            best_fscore_joint = eval_fscore3
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) + '.ckpt')
                            if (saved_file == 0):
                                save_path = saver.save(sess, checkpoint_name)
                                saved_file = 1

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_fscore_joint')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best_joint_fscore.txt"), "a") as text_file:
                                text_file.write(
                                    "{:g} {:g} {:g} {:g} {:g} {:g}\n".format(train_acc3, eval_acc3, train_fscore3,
                                                                             eval_fscore3, train_kappa3, eval_kappa3))

                        early_stop_count_joint_kappa += 1
                        if (eval_kappa3 >= best_kappa_joint):
                            early_stop_count_joint_kappa = 0  # reset

                            best_kappa_joint = eval_kappa3
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) + '.ckpt')
                            if (saved_file == 0):
                                save_path = saver.save(sess, checkpoint_name)
                                saved_file = 1

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_kappa_joint')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best_joint_kappa.txt"), "a") as text_file:
                                text_file.write(
                                    "{:g} {:g} {:g} {:g} {:g} {:g}\n".format(train_acc3, eval_acc3, train_fscore3,
                                                                             eval_fscore3, train_kappa3, eval_kappa3))

                        gp_raw.add_point(train_output_loss1, eval_output_loss1)
                        gp_tf.add_point(train_output_loss2, eval_output_loss2)
                        gp_joint.add_point(train_output_loss3, eval_output_loss3)

                        if (current_step / config.evaluate_every > config.warmup_evaluate_step):  # 100 warm-up steps
                            w1_, g1, o1 = gp_raw.compute_weight()
                            w2_, g2, o2 = gp_tf.compute_weight()
                            w3_, g3, o3 = gp_joint.compute_weight()
                            if (w1_ + w2_ + w3_ == 0.):
                                w1 = 1. / 3
                                w2 = 1. / 3
                                w3 = 1. / 3
                            else:
                                w_sum = w1_ + w2_ + w3_
                                w1 = w1_ / w_sum
                                if (w1 > 0.):
                                    w1 = np.max([w1, 1e-5])  # clipping to avoid numberical issue
                                w2 = w2_ / w_sum
                                if (w2 > 0.):
                                    w2 = np.max([w2, 1e-5])  # clipping to avoid numberical issue
                                w3 = w3_ / w_sum
                                if (w3 > 0.):
                                    w3 = np.max([w3, 1e-5])  # clipping to avoid numberical issue
                            with open(os.path.join(out_dir, "w.txt"), "a") as text_file:
                                text_file.write(
                                    "{:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} \n".format(o1, g1, w1_, w1,
                                                                                                            o2, g2, w2_, w2,
                                                                                                            o3, g3, w3_,
                                                                                                            w3))
                        print("w1 {}, w2 {}, w3 {}".format(w1, w2, w3))


                        train_gen_check_wrapper.gen.reset_reduce_pointer()
                        valid_gen_wrapper.gen.reset_reduce_pointer()
                        #test_gen_wrapper.gen.reset_reduce_pointer()

                        if(FLAGS.early_stopping == True):
                            print('EARLY STOPPING enabled!')
                            # early stop after 10 training steps without improvement.
                            if((early_stop_count_all_acc >= config.early_stop_count) and
                                    (early_stop_count_joint_acc >= config.early_stop_count) and
                                    (early_stop_count_joint_fscore >= config.early_stop_count) and
                                    (early_stop_count_joint_fscore >= config.early_stop_count) and
                                    (early_stop_count_joint_kappa >= config.early_stop_count) and
                                    (early_stop_count_joint_kappa >= config.early_stop_count)):
                                end_time = time.time()
                                with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                                    text_file.write("{:g}\n".format((end_time - start_time)))
                                with open(os.path.join(out_dir, "final_w_acc.txt"), "a") as text_file:
                                    text_file.write("{:g} {:g} {:g}\n".format(best_w1_acc, best_w2_acc, best_w3_acc))
                                with open(os.path.join(out_dir, "final_w_fscore.txt"), "a") as text_file:
                                    text_file.write("{:g} {:g} {:g}\n".format(best_w1_fscore, best_w2_fscore, best_w3_fscore))
                                with open(os.path.join(out_dir, "final_w_kappa.txt"), "a") as text_file:
                                    text_file.write("{:g} {:g} {:g}\n".format(best_w1_kappa, best_w2_kappa, best_w3_kappa))
                                quit()
                        else:
                            print('EARLY STOPPING disabled!')

                #train_generator.reset_pointer()

        end_time = time.time()
        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
        with open(os.path.join(out_dir, "final_w_acc.txt"), "a") as text_file:
            text_file.write("{:g} {:g} {:g}\n".format(best_w1_acc, best_w2_acc, best_w3_acc))
        with open(os.path.join(out_dir, "final_w_fscore.txt"), "a") as text_file:
            text_file.write("{:g} {:g} {:g}\n".format(best_w1_fscore, best_w2_fscore, best_w3_fscore))
        with open(os.path.join(out_dir, "final_w_kappa.txt"), "a") as text_file:
            text_file.write("{:g} {:g} {:g}\n".format(best_w1_kappa, best_w2_kappa, best_w3_kappa))
