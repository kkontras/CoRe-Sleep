import numpy as np

def least_conf(data,num_classes):
    num_labels = float(num_classes)
    least_conf_ranks = []
    prob_dist = calculate_probs(data,num_labels)
    simple_least_conf = np.nanmax(prob_dist)  # most confident prediction, ignoring NaNs
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    least_conf_ranks.append(normalized_least_conf)
    return np.array(least_conf_ranks)

def margin_conf(data,num_classes):
    num_labels = float(num_classes)
    margin_conf_ranks = []
    prob_dist = calculate_probs(data, num_labels)
    prob_dist[::-1].sort()  # sort probs so that largest is at prob_dist[0]
    difference = (prob_dist[0] - prob_dist[1])
    margin_conf = 1 - difference
    margin_conf_ranks.append(margin_conf)
    return np.array(margin_conf_ranks)


def ratio_conf(data,num_classes):
    num_labels = float(num_classes)
    ratio_conf_ranks = []
    prob_dist = calculate_probs(data, num_labels)
    prob_dist[::-1].sort()  # sort probs so that largest is at prob_dist[0]
    ratio_conf = prob_dist[1] / prob_dist[0]
    ratio_conf_ranks.append(ratio_conf)
    return np.array(ratio_conf_ranks)

def entropy_conf(data,num_classes):
    num_labels = float(num_classes)
    entropy_conf_ranks = []
    prob_dist = calculate_probs(data, num_labels)
    log_probs = prob_dist * np.log2(prob_dist+0.00001)  # multiply each probability by its base 2 log
    raw_entropy = 0 - np.sum(log_probs)
    normalized_entropy = raw_entropy / np.log2(prob_dist.size)
    entropy_conf_ranks.append(normalized_entropy)
    return np.array(entropy_conf_ranks)

def bald_conf(data,num_classes):
    # num_labels = float(num_classes)
    bald_conf_ranks = []
    expected_entropy = - np.mean(np.sum(data * np.log(data + 1e-10), axis=-1), axis=0)  # [batch size]
    expected_p = data
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    BALD_acq = entropy_expected_p - expected_entropy
    bald_conf_ranks.append(BALD_acq)
    return np.array(bald_conf_ranks)

def calculate_probs(predicted_classes, num_classes):
    '''
    This function is to calculate the probabilities for each class given the softmax output
    :param predicted_classes: matrix num_datapoints X num_ensembles (or dropout_iterations)
    :param num_classes:
    :return: For each datapoint it returns a vector with 10 elements, corresponding to the prob of each class
    '''
    probs = np.mean(predicted_classes,axis = 1)
    return probs