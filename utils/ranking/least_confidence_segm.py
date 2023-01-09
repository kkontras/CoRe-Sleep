import numpy as np

def least_conf(data,num_classes):
    raise Exception("Error: Least Confidence is not supported yet.")

def margin_conf(data,num_classes):
    raise Exception("Error: Margin Confidence is not supported yet.")

def ratio_conf(samples,num_classes):
    if (samples.shape.__len__() != 5):
        raise  Exception("In acquisition score functions there should have been given the format #batch-h-w-#samples-#class")
    else:
        # #batch-h-w-#samples-#class
        expected_p = np.mean(samples, axis=3)
        Ratio_acq = 1-np.max(expected_p,axis=3)  # [batch size]
        return np.array(Ratio_acq)

def entropy_conf(samples,num_classes):

    if (samples.shape.__len__() != 5):
        raise  Exception("In acquisition score functions there should have been given the format #batch-h-w-#samples-#class")
    else:
        # #batch-h-w-#samples-#class
        expected_p = np.mean(samples, axis=3)
        Entropy_acq = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
        return np.array(Entropy_acq)

def bald_conf(samples,num_classes):

    if (samples.shape.__len__() != 5):
        raise  Exception("In acquisition score functions there should have been given the format #batch-h-w-#samples-#class")
    else:
        # #batch-h-w-#samples-#class
        expected_entropy = - np.mean(np.sum(samples * np.log(samples + 1e-10), axis=-1), axis=3)
        expected_p = np.mean(samples, axis=3)
        entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # batch size
        BALD_acq = entropy_expected_p - expected_entropy
        return np.array(BALD_acq)
