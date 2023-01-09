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

from scipy.io import loadmat, savemat


# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
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
tf.app.flags.DEFINE_integer("deep_nhidden", 512, "Sequence length (default: 20)")

# common settings
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 32)")

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
#config.seq_epoch_step = FLAGS.seq_len
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

if (not eog_active and not emg_active):
    print("eeg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             num_fold=config.num_fold_training_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=True)
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
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
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                                  eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
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
    test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
                                                  eog_filelist=os.path.abspath(FLAGS.eog_test_data),
                                                  emg_filelist=os.path.abspath(FLAGS.emg_test_data),
                                             num_fold=config.num_fold_testing_data,
                                             data_shape_1=[config.deep_ntime],
                                             data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                             seq_len = config.epoch_seq_len,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params()
    train_gen_wrapper.compute_eog_normalization_params()
    train_gen_wrapper.compute_emg_normalization_params()
    test_gen_wrapper.set_eeg_normalization_params(train_gen_wrapper.eeg_meanX, train_gen_wrapper.eeg_stdX)
    test_gen_wrapper.set_eog_normalization_params(train_gen_wrapper.eog_meanX, train_gen_wrapper.eog_stdX)
    test_gen_wrapper.set_emg_normalization_params(train_gen_wrapper.emg_meanX, train_gen_wrapper.emg_stdX)
    nchannel = 3

# as there is only one fold, there is only one partition consisting all subjects, and next_fold should be called only once
#test_gen_wrapper.new_subject_partition()
#test_gen_wrapper.next_fold()

config.nchannel = nchannel

# shuffle training data here
#del train_generator
#test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)
#print("Test set: {:d}".format(test_generator.data_size))
#print("/Test batches per epoch: {:d}".format(test_batches_per_epoch))
del train_gen_wrapper


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
                net.w1 : best_w1,
                net.w2 : best_w2,
                net.w3 : best_w3,
                net.istraining: 0
            }
            output_loss1, output_loss2, output_loss3, output_loss, total_loss, \
            deep_yhat, seq_yhat, joint_yhat, yhat, deep_score, seq_score, joint_score, score = sess.run(
                   [net.deep_loss, net.seq_loss, net.joint_loss, net.output_loss, net.loss,
                    net.deep_predictions, net.seq_predictions, net.joint_predictions, net.predictions,
                    net.deep_scores, net.seq_scores, net.joint_scores, net.score], feed_dict)
            return output_loss1, output_loss2, output_loss3, output_loss, total_loss, \
                   deep_yhat, seq_yhat, joint_yhat, yhat, \
                   deep_score, seq_score, joint_score, score

        def evaluate(gen_wrapper):
            output_loss1 =0
            output_loss2 =0
            output_loss3 =0
            output_loss =0
            total_loss = 0

            N = int(np.sum(gen_wrapper.file_sizes) - (config.epoch_seq_len - 1)*len(gen_wrapper.file_sizes))
            deep_yhat = np.zeros([config.epoch_seq_len, N])
            seq_yhat = np.zeros([config.epoch_seq_len, N])
            joint_yhat = np.zeros([config.epoch_seq_len, N])
            yhat = np.zeros([config.epoch_seq_len, N])
            y = np.zeros([config.epoch_seq_len, N])

            deep_score = np.zeros([config.epoch_seq_len, N, config.nclass])
            seq_score = np.zeros([config.epoch_seq_len, N, config.nclass])
            joint_score = np.zeros([config.epoch_seq_len, N, config.nclass])
            score = np.zeros([config.epoch_seq_len, N, config.nclass])

            count = 0
            gen_wrapper.new_subject_partition()
            for data_fold in range(config.num_fold_testing_data):
                # load data of the current fold
                gen_wrapper.next_fold()
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_, \
                deep_score_, seq_score_, joint_score_, score_, \
                output_loss1_, output_loss2_, output_loss3_, \
                output_loss_, total_loss_ = _evaluate(gen_wrapper.gen)

                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_

                deep_yhat[:, count : count + len(gen_wrapper.gen.data_index)] = deep_yhat_
                seq_yhat[:, count : count + len(gen_wrapper.gen.data_index)] = seq_yhat_
                joint_yhat[:, count : count + len(gen_wrapper.gen.data_index)] = joint_yhat_
                yhat[:, count : count + len(gen_wrapper.gen.data_index)] = yhat_

                deep_score[:, count : count + len(gen_wrapper.gen.data_index),:] = deep_score_
                seq_score[:, count : count + len(gen_wrapper.gen.data_index), :] = seq_score_
                joint_score[:, count : count + len(gen_wrapper.gen.data_index), :] = joint_score_
                score[:, count : count + len(gen_wrapper.gen.data_index), :] = score_

                # groundtruth
                for n in range(config.epoch_seq_len):
                    y[n,count : count + len(gen_wrapper.gen.data_index)] =\
                        gen_wrapper.gen.label[gen_wrapper.gen.data_index - (config.epoch_seq_len - 1) + n]
                count += len(gen_wrapper.gen.data_index)


            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc = 0
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(deep_yhat[n,:], y[n,:]) # due to zero-indexing
                acc1 += acc_n
                acc_n = accuracy_score(seq_yhat[n,:], y[n,:]) # due to zero-indexing
                acc2 += acc_n
                acc_n = accuracy_score(joint_yhat[n,:], y[n,:]) # due to zero-indexing
                acc3 += acc_n
                acc_n = accuracy_score(yhat[n,:], y[n,:]) # due to zero-indexing
                acc += acc_n
            acc1 /= config.epoch_seq_len
            acc2 /= config.epoch_seq_len
            acc3 /= config.epoch_seq_len
            acc /= config.epoch_seq_len
            return acc1, acc2, acc3, acc, \
                   deep_yhat, seq_yhat, joint_yhat, yhat, \
                   deep_score, seq_score, joint_score, score, \
                   output_loss1, output_loss2, output_loss3, \
                   output_loss, total_loss

        def _evaluate(gen):
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

            deep_score = np.zeros([config.epoch_seq_len, len(gen.data_index), config.nclass])
            seq_score = np.zeros([config.epoch_seq_len, len(gen.data_index), config.nclass])
            joint_score = np.zeros([config.epoch_seq_len, len(gen.data_index), config.nclass])
            score = np.zeros([config.epoch_seq_len, len(gen.data_index), config.nclass])

            num_batch_per_epoch = np.floor(len(gen.data_index) / (20*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x1_batch, x2_batch, y_batch, label_batch_ = gen.next_batch(20*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_, \
                deep_score_, seq_score_, joint_score_, score_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size] = joint_yhat_[n]
                    yhat[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size] = yhat_[n]

                    deep_score[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size,:] = deep_score_[n]
                    seq_score[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size,:] = seq_score_[n]
                    joint_score[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size,:] = joint_score_[n]
                    score[n, (test_step-1)*20*config.batch_size : test_step*20*config.batch_size,:] = score_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x1_batch, x2_batch, y_batch, label_batch_ = gen.rest_batch(20*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_, \
                deep_score_, seq_score_, joint_score_, score_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*20*config.batch_size : len(gen.data_index)] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*20*config.batch_size : len(gen.data_index)] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*20*config.batch_size : len(gen.data_index)] = joint_yhat_[n]
                    yhat[n, (test_step-1)*20*config.batch_size : len(gen.data_index)] = yhat_[n]

                    deep_score[n, (test_step-1)*20*config.batch_size : len(gen.data_index),:] = deep_score_[n]
                    seq_score[n, (test_step-1)*20*config.batch_size : len(gen.data_index),:] = seq_score_[n]
                    joint_score[n, (test_step-1)*20*config.batch_size : len(gen.data_index),:] = joint_score_[n]
                    score[n, (test_step-1)*20*config.batch_size : len(gen.data_index),:] = score_[n]
            deep_yhat = deep_yhat + 1
            seq_yhat = seq_yhat + 1
            joint_yhat = joint_yhat + 1
            yhat = yhat + 1
            '''
            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc = 0
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(deep_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc1 += acc_n
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(seq_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc2 += acc_n
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(joint_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc3 += acc_n
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc += acc_n
            acc1 /= config.epoch_seq_len
            acc2 /= config.epoch_seq_len
            acc3 /= config.epoch_seq_len
            acc /= config.epoch_seq_len
            '''
            return deep_yhat, seq_yhat, joint_yhat, yhat, \
                   deep_score, seq_score, joint_score, score, \
                   output_loss1, output_loss2, output_loss3, output_loss, total_loss


        # read optimal weight
        with open(os.path.join(out_dir, "final_w_acc.txt"), "r") as w_file:
            line = w_file.readline()
            ws = line.split(" ")
            best_w1 = float(ws[0].strip())
            best_w2 = float(ws[1].strip())
            best_w3 = float(ws[2].strip())
        print("w1 {}, w2 {}, w3 {}".format(best_w1, best_w2, best_w3))

        saver = tf.train.Saver(tf.all_variables())
        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model_acc_all")
        saver.restore(sess, best_dir)
        print("Model all loaded")

        deep_acc, seq_acc, joint_acc, test_acc, \
        deep_yhat, seq_yhat, joint_yhat, test_yhat,\
        deep_score, seq_score, joint_score, test_score,\
        deep_loss, seq_loss, joint_loss, test_loss, test_total_loss = \
            evaluate(gen_wrapper=test_gen_wrapper)

        #savemat(os.path.join(out_path, FLAGS.output_file), dict(
        savemat(os.path.join(out_path, "test_ret_all.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score, output_loss = test_loss,
                                                             deep_yhat = deep_yhat, deep_acc = deep_acc, deep_score = deep_score, deep_loss = deep_loss,
                                                             seq_yhat = seq_yhat, seq_acc = seq_acc, seq_score = seq_score, seq_loss = seq_loss,
                                                             joint_yhat = joint_yhat, joint_acc = joint_acc, joint_score = joint_score, joint_loss = joint_loss,
                                                             total_loss = test_total_loss))
        test_gen_wrapper.gen.reset_pointer()


        saver = tf.train.Saver(tf.all_variables())
        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model_acc_joint")
        saver.restore(sess, best_dir)
        print("Model joint loaded")

        deep_acc, seq_acc, joint_acc, test_acc, \
        deep_yhat, seq_yhat, joint_yhat, test_yhat,\
        deep_score, seq_score, joint_score, test_score,\
        deep_loss, seq_loss, joint_loss, test_loss, test_total_loss = \
            evaluate(gen_wrapper=test_gen_wrapper)

        #savemat(os.path.join(out_path, FLAGS.output_file), dict(
        savemat(os.path.join(out_path, "test_ret_joint.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score, output_loss = test_loss,
                                                             deep_yhat = deep_yhat, deep_acc = deep_acc, deep_score = deep_score, deep_loss = deep_loss,
                                                             seq_yhat = seq_yhat, seq_acc = seq_acc, seq_score = seq_score, seq_loss = seq_loss,
                                                             joint_yhat = joint_yhat, joint_acc = joint_acc, joint_score = joint_score, joint_loss = joint_loss,
                                                             total_loss = test_total_loss))
        test_gen_wrapper.gen.reset_pointer()

