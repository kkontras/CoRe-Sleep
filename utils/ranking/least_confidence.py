import numpy as np

def least_conf(data,num_classes):
    raise Exception("Error: Evaluation should get [eval_batch,num_classes] or [eval_batch,num_classes,iterations], none of them has given!")

#     # num_labels = float(num_classes)
#     least_conf_ranks = []
#     for i in data :
#         prob_dist = calculate_probs(i)
#         simple_least_conf = np.nanmax(prob_dist)  # most confident prediction, ignoring NaNs
#         normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
#         least_conf_ranks.append(normalized_least_conf)
#     return np.array(least_conf_ranks)
#
def margin_conf(samples,num_classes):
    raise Exception("Error: Evaluation should get [eval_batch,num_classes] or [eval_batch,num_classes,iterations], none of them has given!")

#     if (samples.shape.__len__() < 3):
#         Ratio_acq = 1 - np.max(samples, axis=1)
#         print(Ratio_acq.shape)
#         return np.array(Ratio_acq)
#     elif (samples.shape.__len__() == 3):
#         # #samples-batchsize-#classes
#         samples = samples.swapaxes(1,2)
#         samples = samples.swapaxes(0,1)
#         print(samples.shape)
#         expected_p = np.mean(samples, axis=0)
#         print(expected_p.shape)
#         expected_p[::-1].sort()
#         Ratio_acq =   # [batch size]
#         print(Ratio_acq.shape)
#         return np.array(Ratio_acq)
#     else:
#         raise Exception("Error: Evaluation should get [eval_batch,num_classes] or [eval_batch,num_classes,iterations], none of them has given!")
#
#     # num_labels = float(num_classes)
#     margin_conf_ranks = []
#     for i in data:
#         prob_dist = calculate_probs(i)
#         prob_dist[::-1].sort()  # sort probs so that largest is at prob_dist[0]
#         difference = (prob_dist[0] - prob_dist[1])
#         margin_conf = 1 - difference
#         margin_conf_ranks.append(margin_conf)
#     return np.array(margin_conf_ranks)

def ratio_conf(samples,num_classes):
    if (samples.shape.__len__() < 3):
        Ratio_acq = 1 - np.max(samples, axis=1)
        return np.array(Ratio_acq)
    elif (samples.shape.__len__() == 3):
        # #samples-batchsize-#classes
        expected_p = np.mean(samples, axis=0)
        Ratio_acq = 1-np.max(expected_p,axis=1)  # [batch size]
        return np.array(Ratio_acq)
    else:
        raise Exception("Error: Evaluation should get [eval_batch,num_classes] or [eval_batch,num_classes,iterations], none of them has given!")

def entropy_conf(samples,num_classes):

    if (samples.shape.__len__() < 3):
        Entropy_acq = - np.sum(samples * np.log(samples + 1e-10), axis=-1)  # [batch size]
        return np.array(Entropy_acq)
    elif (samples.shape.__len__() == 3):
        # #samples-batchsize-#classes
        expected_p = np.mean(samples, axis=0)
        Entropy_acq = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
        return np.array(Entropy_acq)
    else:
        raise Exception("Error: Evaluation should get [eval_batch,num_classes] or [eval_batch,num_classes,iterations], none of them has given!")

def bald_conf(samples,num_classes):

    if (samples.shape.__len__() < 3):
        raise Exception("Error: Evaluation should get [eval_batch,num_classes,iterations], bald doesnt work with single prob predictions")
    elif (samples.shape.__len__() == 3):
        # #samples-batchsize-#classes
        expected_entropy = - np.mean(np.sum(samples * np.log(samples + 1e-10), axis=-1), axis=0)  # [batch size]
        expected_p = np.mean(samples, axis=0)
        entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
        BALD_acq = entropy_expected_p - expected_entropy
        return np.array(BALD_acq)
    else:
        raise Exception("Error: Evaluation should get [eval_batch,num_classes] or [eval_batch,num_classes,iterations], none of them has given!")
