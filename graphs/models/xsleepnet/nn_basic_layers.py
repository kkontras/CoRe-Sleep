import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.layers as layers
from graphs.models.xsleepnet.bnlstm import BNLSTMCell

"""
Predefine all necessary layer for the R-CNN
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        #weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        #biases = tf.get_variable('biases', shape=[num_filters])
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        #print(conv.get_shape())

        # Add biases
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        bias = tf.nn.bias_add(conv, biases)
        #print(bias.get_shape())
        #bias = tf.reshape(bias, conv.get_shape().as_list())
        bias = tf.reshape(bias, tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu

def conv_bn(x, filter_height, filter_width, num_filters, stride_y, stride_x, is_training, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        #weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        #biases = tf.get_variable('biases', shape=[num_filters])
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        #print(conv.get_shape())

        # Add biases
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        bias = tf.nn.bias_add(conv, biases)
        #print(bias.get_shape())
        #bias = tf.reshape(bias, conv.get_shape().as_list())
        bias = tf.reshape(bias, tf.shape(conv))

        conv_bn = tf.contrib.layers.batch_norm(bias, center=True, scale=True,
                                                is_training=is_training, scope=scope)

        # Apply relu function
        relu = tf.nn.relu(conv_bn, name=scope.name)

        return relu

def fc(x, num_in, num_out, name, relu=True):
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    #with tf.variable_scope(name) as scope:
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out])
        biases = tf.get_variable('biases', [num_out])

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)  # Apply ReLu non linearity
            return relu
        else:
            return act

def fc_bn(x, num_in, num_out, is_training, name, relu=True):
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out])
        biases = tf.get_variable('biases', [num_out])

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        act_bn = tf.contrib.layers.batch_norm(act,
                                      center=True, scale=True,
                                      is_training=is_training,
                                      scope='bn')

        if relu == True:
            relu = tf.nn.relu(act_bn)  # Apply ReLu non linearity
            return relu
        else:
            return act_bn

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def recurrent_layer(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    recurrent_cell = tf.contrib.rnn.GRUCell(nhidden)
    if(nlayer > 1):
        recurrent_cell = tf.contrib.rnn.MultiRNNCell([recurrent_cell] * nlayer,
                                                     state_is_tuple=True)  # due to built with LN LSTM
    # input dropout
    recurrent_cell = tf.contrib.rnn.DropoutWrapper(recurrent_cell, input_keep_prob=input_keep_prob)
    #output dropout
    recurrent_cell = tf.contrib.rnn.DropoutWrapper(recurrent_cell, output_keep_prob=output_keep_prob)
    return recurrent_cell

def bidirectional_recurrent_layer(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    fw_cell = tf.contrib.rnn.GRUCell(nhidden)
    bw_cell = tf.contrib.rnn.GRUCell(nhidden)
    #fw_cell = BNLSTMCell(num_units=nhidden,
    #                     is_training_tensor=self.phase,
    #                     max_bn_steps = self.config.n_step)
    #bw_cell = BNLSTMCell(num_units=nhidden,
    #                     is_training_tensor=self.phase,
    #                     max_bn_steps = self.config.n_step)
    if (nlayer > 1):
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
    return fw_cell,bw_cell

def bidirectional_recurrent_layer_output(fw_cell, bw_cell, input_layer, sequence_len, scope=None):
    ((fw_outputs,
      bw_outputs),
     (fw_state,
      bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                    cell_bw=bw_cell,
                                                    inputs=input_layer,
                                                    sequence_length=sequence_len,
                                                    dtype=tf.float32,
                                                    swap_memory=True,
                                                    scope=scope))
    outputs = tf.concat((fw_outputs, bw_outputs), 2)

    def concatenate_state(fw_state, bw_state):
        if isinstance(fw_state, LSTMStateTuple):
            state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
            state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
            state = LSTMStateTuple(c=state_c, h=state_h)
            return state
        elif isinstance(fw_state, tf.Tensor):
            state = tf.concat((fw_state, bw_state), 1,
                              name='bidirectional_concat')
            return state
        elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                      len(fw_state) == len(bw_state)):
            # multilayer
            state = tuple(concatenate_state(fw, bw)
                          for fw, bw in zip(fw_state, bw_state))
            return state

        else:
            raise ValueError(
                'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state

# new version of tensorflow has the new way to create multiplayer RNN
def bidirectional_recurrent_layer_bn_new(nhidden, nlayer, seq_len=1, is_training=False, input_keep_prob=1.0, output_keep_prob=1.0):
    if (nlayer == 1):
        #fw_cell = tf.contrib.rnn.GRUCell(num_units=nhidden)
        #bw_cell = tf.contrib.rnn.GRUCell(num_units=nhidden)
        fw_cell = BNLSTMCell(num_units=nhidden,
                             is_training_tensor=is_training,
                             max_bn_steps = seq_len)
        bw_cell = BNLSTMCell(num_units=nhidden,
                             is_training_tensor=is_training,
                             max_bn_steps = seq_len)
    else:
        # problem with the new version described here
        # https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k
        '''
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        '''
        fw_cell_ = []
        bw_cell_ = []
        for i in range(nlayer):
            #fw_cell_.append(tf.contrib.rnn.GRUCell(num_units=nhidden))
            #bw_cell_.append(tf.contrib.rnn.GRUCell(num_units=nhidden))
            fw_cell_.append(BNLSTMCell(num_units=nhidden,
                             is_training_tensor=is_training,
                             max_bn_steps = seq_len))
            bw_cell_.append(BNLSTMCell(num_units=nhidden,
                                 is_training_tensor=is_training,
                                 max_bn_steps = seq_len))
        fw_cell = tf.contrib.rnn.MultiRNNCell(cells=fw_cell_, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell(cells=bw_cell_, state_is_tuple=True)


        # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
    return fw_cell,bw_cell

# new version of tensorflow has the new way to create multiplayer RNN
def bidirectional_recurrent_layer_gru_new(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    if (nlayer == 1):
        fw_cell = tf.contrib.rnn.GRUCell(num_units=nhidden)
        bw_cell = tf.contrib.rnn.GRUCell(num_units=nhidden)
    else:
        # problem with the new version described here
        # https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k
        '''
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        '''
        fw_cell_ = []
        bw_cell_ = []
        for i in range(nlayer):
            fw_cell_.append(tf.contrib.rnn.GRUCell(num_units=nhidden))
            bw_cell_.append(tf.contrib.rnn.GRUCell(num_units=nhidden))
        fw_cell = tf.contrib.rnn.MultiRNNCell(cells=fw_cell_, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell(cells=bw_cell_, state_is_tuple=True)


        # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
    return fw_cell,bw_cell

def bidirectional_recurrent_layer_lstm_new(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    if (nlayer == 1):
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True)
    else:
        # problem with the new version described here
        # https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k
        '''
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * nlayer,
                                              state_is_tuple=True)  # due to built with LN LSTM
        '''
        fw_cell_ = []
        bw_cell_ = []
        for i in range(nlayer):
            fw_cell_.append(tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True))
            bw_cell_.append(tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True))
        fw_cell = tf.contrib.rnn.MultiRNNCell(cells=fw_cell_, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell(cells=bw_cell_, state_is_tuple=True)


        # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
    return fw_cell,bw_cell

# new version of tensorflow has the new way to create multiplayer RNN
def bidirectional_recurrent_layer_output_new(fw_cell, bw_cell, input_layer, sequence_len, scope=None):
    ((fw_outputs,
      bw_outputs),
     (fw_state,
      bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                    cell_bw=bw_cell,
                                                    inputs=input_layer,
                                                    sequence_length=sequence_len,
                                                    dtype=tf.float32,
                                                    swap_memory=True,
                                                    scope=scope))
    outputs = tf.concat((fw_outputs, bw_outputs), 2)

    def concatenate_state(fw_state, bw_state):
        if isinstance(fw_state, LSTMStateTuple):
            state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
            state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
            state = LSTMStateTuple(c=state_c, h=state_h)
            return state
        elif isinstance(fw_state, tf.Tensor):
            state = tf.concat((fw_state, bw_state), 1,
                              name='bidirectional_concat')
            return state
        elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                      len(fw_state) == len(bw_state)):
            # multilayer
            state = tuple(concatenate_state(fw, bw)
                          for fw, bw in zip(fw_state, bw_state))
            return state

        else:
            raise ValueError(
                'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state




def attention(inputs, attention_size, time_major=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    return output, alphas
