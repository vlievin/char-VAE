#!/usr/bin/env python
"""
Implementation of a Variational Recurrent Autoencoder (https://arxiv.org/abs/1412.6581), a Variational Autoencoder (https://arxiv.org/abs/1606.05908) with recurrent Neural Networks as encoder and decoder.
The aim of this project is to obtain similar results to https://arxiv.org/abs/1511.06349 (Generating sentences from a continous space).

__author__ = "Valentin Lievin, DTU, Denmark"
__copyright__ = "Copyright 2017, Valentin Lievin"
__credits__ = ["Valentin Lievin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Valentin Lievin"
__email__ = "valentin.lievin@gmail.com"
__status__ = "Development"
"""    
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops


class Vrae:
    def __init__(self, char2word_state_size, 
                 char2word_num_layers, 
                 encoder_state_size, 
                 encoder_num_layers, 
                 decoder_state_size, 
                 decoder_num_layers, 
                 latent_dim, 
                 batch_size, 
                 num_symbols, 
                 input_keep_prob,
                 output_keep_prob, 
                 latent_loss_weight, 
                 dtype_precision, 
                 cell_type, 
                 peephole, 
                 sentiment_feature = False,
                 teacher_forcing=True):
        """
        Initi Variational Recurrent Autoencoder (VRAE) for sequences. The model clears the current tf graph and implements this model as the new graph. 
        Args:
            char2word_state_size (Natural Integer): size of the state of the RNN cells in the char2word RNN
            char2word_num_layers (Natural Integer): size of the state of the RNN cells in the char2word RNN
            encoder_state_size (Natural Integer): size of the states of the RNN cells in the encoder (word level)
            encoder_num_layers (Natural Integer): number of layers in the RNN cells of the encoder (word level)
            decoder_state_size (Natural Integer): size of the states of the RNN cells in the decoder (char level)
            decoder_num_layers (Natural Integer): number of layers in the RNN cells of the decoder (char level)
            latent_dim (Natural Integer): dimension of the latent space
            batch_size (Natural Integer): batch size
            num_symbols (Natural Integer): number of symbols in the data (number of unique characters if used with characters or vocabulary size if used with words)
            input_keep_prob (float): dropout keep probability for the inputs
            output_keep_prob (float): dropout keep probability for the outputs
            latent_loss_weight (float): weight used to weaken the regularization/latent loss
            dtype_precision (Integer): dtype precision
            cell_type (string): type of cell: LSTM,GRU,LNLSTM
            peephole (boolean): use peepholes or not for LSTM
            sentiment_feature (boolean): input sentiment_feature
            teacher_forcing (bool): use teacher forcing during training
        Returns 
        """
        if dtype_precision==16:
            dtype = tf.float16
        else:
            dtype = tf.float32
        # clear the default graph
        tf.reset_default_graph()
        self.batch_size_value = batch_size
        # placeholders
        self.use_sentiment_feature = sentiment_feature
        self.sentiment_feature = tf.placeholder( dtype , [None,3] , name="sentiment_feature")
        self.batch_size = tf.placeholder( tf.int32 , name='batch_size')
        self.input_keep_prob_value = input_keep_prob
        self.output_keep_prob_value = output_keep_prob
        self.x_input = tf.placeholder( tf.int32, [None, None], name='input_placeholder')
        self.x_input_lenghts = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.weights_input = tf.placeholder( tf.int32, [None, None], name='weights_placeholder')
        self.end_of_words = tf.placeholder( tf.int32, [None, None, 2], name='end_of_words_placeholder')
        self.batch_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_word_length')
        self.input_keep_prob = tf.placeholder(dtype,name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(dtype,name="output_keep_prob")
        self.max_sentence_size = tf.reduce_max(self.x_input_lenghts )
        self.training = tf.placeholder( tf.bool, name="training_state")
        self.teacher_forcing = teacher_forcing
        with tf.name_scope("training_parameters"):
            self.B = tf.placeholder(dtype, name='Beta_deterministic_warmup')
            self.learning_rate = tf.placeholder(dtype, shape=[], name='learning_rate')
            self.epoch = tf.placeholder(dtype, shape=[], name='epoch')
        # summaries
        tf.summary.scalar("Beta", self.B)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("epoch", self.epoch)
        tf.summary.scalar("sentences_max_length", self.max_sentence_size)
        # prepare the input
        with tf.name_scope("input_transformations"):
            inputs_onehot = tf.one_hot(self.x_input, num_symbols, axis= -1, dtype=dtype)   # one hot encoding
            data_dim = int(inputs_onehot.shape[2])
            #rnn_inputs = tf.reverse(inputs_onehot, [1])   # reverse input
            rnn_inputs = inputs_onehot
        
        # encoder
        encoder_output = char2word_encoder(char2word_state_size,
                                           char2word_num_layers, 
                                           encoder_state_size, 
                                           encoder_num_layers,
                                           rnn_inputs, 
                                           self.x_input_lenghts, 
                                           self.end_of_words, 
                                           self.batch_word_lengths, 
                                           self.batch_size, 
                                           dtype,
                                           cell_type, 
                                           peephole, 
                                           self.input_keep_prob, 
                                           self.output_keep_prob) 
        # sentiment feature
        if self.use_sentiment_feature:
            stochastic_layer_input = tf.concat( [self.sentiment_feature , encoder_output] , 1)
        else:
            stochastic_layer_input = encoder_output
        # stochastic layer
        self.z, self.z_mu, self.z_ls2 = stochasticLayer(stochastic_layer_input, latent_dim, self.batch_size,
                                                        dtype, scope="stochastic_layer")
        # decoder
        self.decoder_output = decoder(self.z, self.batch_size, decoder_state_size, decoder_num_layers, 
                                      data_dim, self.x_input_lenghts, cell_type, peephole,
                                      self.input_keep_prob, self.output_keep_prob, inputs_onehot, self.training,dtype, scope="decoder") 
        # loss
        self.loss, self.reconstruction_loss, self.latent_loss = loss_function(self.decoder_output, self.x_input, 
                                  self.weights_input,self.z_ls2, self.z_mu, 
                                  self.B, latent_loss_weight, dtype, scope="loss") 
        # optimizer
        self.optimizer = optimizationOperation(self.loss, self.learning_rate, scope="optimizer")   # optimizer
        # merge summaries: summarize variables
        self.merged_summary = tf.summary.merge_all()
    
    def step(self, sess, padded_batch_xs, beta, learning_rate, batch_lengths, batch_weights, end_of_words_value, batch_word_lengths_value, epoch,sentiment_feature) :
        """ 
        train the model for one step
        Args:
            sess: current Tensorflow session
            padded_batch_xs: padded input batch
            beta: beta parameter for deterministic warmup
            learning_rate: learning rate (potentially controled during training)
            batch_lengths: sentences lengths 
            batch_weights: sentences weights
            end_of_words_value: indexes of end of words (spaces)
            batch_word_lengths_value: number of words
            epoch: current epoch
            sentiment_feature: sentiment feature
        Returns:
            a tuple of values:
                optimizer op
                current loss
                summary op
        """
        return sess.run([self.optimizer, self.loss, self.reconstruction_loss, self.latent_loss, self.merged_summary, self.max_sentence_size ], feed_dict={self.x_input: padded_batch_xs, 
                                                           self.B:beta, 
                                                           self.learning_rate: learning_rate,
                                                           self.x_input_lenghts:batch_lengths,
                                                           self.weights_input: batch_weights,
                                                           self.input_keep_prob:self.input_keep_prob_value, 
                                                           self.output_keep_prob:self.output_keep_prob_value,
                                                           self.epoch: epoch,
                                                           self.batch_size:self.batch_size_value,
                                                           self.end_of_words: end_of_words_value,
                                                           self.batch_word_lengths:batch_word_lengths_value,
                                                           self.training: self.teacher_forcing,
                                                           self.sentiment_feature:sentiment_feature
                                                            })
    
    def reconstruct(self, sess, padded_batch_xs, batch_lengths, batch_weights,end_of_words_value,batch_word_lengths_value,sentiment_feature):
        """
        Feed a batch of inputs and reconstruct it
        Args:
            sess: current Tensorflow session
            padded_batch_xs: padded input batch
            batch_lengths: sentences lengths 
            end_of_words_value: indexes of corresponding to the end of words
            batch_word_lengths_value: word lengths
            sentiment_feature : sentiment features
        Returns:
            tuple x_reconstruct,z_vals,z_mean_val,z_log_sigma_sq_val, sequence_loss
                x_reconstruct: reconstruction of the input
                z_vals: sampled values of z (prior)
                z_mean_val: mean_z values of the prior
                z_log_sigma_sq_val: log of sigma^2 of the prior
                sequence_loss: average cross entropy
        """
        return sess.run((self.decoder_output,self.z, self.z_mu, self.z_ls2, self.loss), 
                        feed_dict={self.x_input: padded_batch_xs,
                                   self.x_input_lenghts:batch_lengths,
                                   self.weights_input: batch_weights, 
                                   self.B: 1,
                                   self.input_keep_prob:1, 
                                   self.output_keep_prob:1,
                                   self.batch_size:self.batch_size_value,
                                   self.end_of_words: end_of_words_value,
                                   self.batch_word_lengths:batch_word_lengths_value,
                                   self.sentiment_feature: sentiment_feature,
                                   self.training: False})
    
    def zToX(self,sess,z_sample,s_length):
        """
        Reconstruct X from a latent variable z.
        Args:
            sess: current Tensorflow session
            z_sample (numpy array): z sample, array of dimension (latent_dim x1)
            s_length: sentence_length
        Returns:
            x generated from z 
        """
        s_lengths = [s_length]
        z_samples = [z_sample]
        none_input = [[0]]
        return sess.run((self.decoder_output), feed_dict={self.z: z_samples,
                                                          self.x_input_lenghts:s_lengths,
                                                          self.input_keep_prob:1, 
                                                          self.output_keep_prob:1,
                                                          self.batch_size:1,
                                                          self.training: False,
                                                          self.x_input:none_input
                                                         })
    
    def XToz(self,sess,seq_ids,seq_len,words_endings,seq_words_len, sentiment):
        """
        Project X to the latent space Z
        Args:
            sess: current Tensorflow session
            seq_ids:
            seq_len:
            words_endings:
            seq_words_len:
            sentiment:
        Returns:
            x generated from z 
        """
        return sess.run((self.z_mu), feed_dict={self.x_input: [seq_ids],
                                                   self.x_input_lenghts:[seq_len],
                                                   self.end_of_words: [words_endings],
                                                   self.batch_word_lengths:[seq_words_len],
                                                self.input_keep_prob:1, 
                                                self.output_keep_prob:1,
                                                self.batch_size:1,
                                                self.sentiment_feature:[sentiment],
                                                self.training: False})
    
def char2word_encoder( char2word_state_size, 
                      char2word_num_layers, 
                      encoder_state_size, 
                      encoder_num_layers,
                      rnn_inputs,
                      batch_char_lengths, 
                      end_of_words, 
                      batch_word_lengths, 
                      batch_size, 
                      dtype, 
                      cell_type, 
                      peephole, 
                      input_keep_prob, 
                      output_keep_prob,
                      scope = "hierarchical_encoder"):
    # cell type
    if cell_type == 'GRU':
        cell_fn = tf.contrib.rnn.GRUCell
    elif cell_type == 'LSTM':
        cell_fn = tf.contrib.rnn.LSTMCell
    elif cell_type == 'LNLSTM':
        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
    elif cell_type == "UGRNN":
        cell_fn = tf.contrib.rnn.UGRNNCell
    elif cell_type == "GLSTM":
        cell_fn = tf.contrib.rnn.GLSTMCell
    elif cell_type == "LSTMBlockFusedCell":
        cell_fn = tf.contrib.rnn.LSTMBlockFusedCell
                
    with tf.name_scope(scope):
        with tf.name_scope("char2word_encoder"):
            # char2RNN cell
            cell_fn = tf.contrib.rnn.LSTMCell
            cells = []
            for _ in range(2 * char2word_num_layers):
                cell = cell_fn(char2word_state_size)
                cell = tf.contrib.rnn.DropoutWrapper( cell, output_keep_prob=output_keep_prob, input_keep_prob=input_keep_prob)
                cells.append(cell)
            cell_fw = tf.contrib.rnn.MultiRNNCell( cells[:char2word_num_layers] )
            cell_bw = tf.contrib.rnn.MultiRNNCell( cells[char2word_num_layers:] )
            # char2word RNN
            char_rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length = batch_char_lengths, dtype=dtype, scope="char2word_encoder_rnn")   
            char_rnn_outputs = tf.concat([ char_rnn_outputs[0][:, :1 , :]  , char_rnn_outputs[1][:, :-1 , :]   ] , 1) # bw outputs are already reversed
            # gather
            rnn_words_outputs = tf.gather_nd(char_rnn_outputs, end_of_words)
        with tf.name_scope("char2word_encoder"):
            # encoder cell
            cell_fn = tf.contrib.rnn.LSTMCell
            cells = []
            for _ in range(2*encoder_num_layers):
                cell = cell_fn(encoder_state_size)
                cell = tf.contrib.rnn.DropoutWrapper( cell, output_keep_prob=output_keep_prob, input_keep_prob=input_keep_prob)
                cells.append(cell)
            word_cell_fw = tf.contrib.rnn.MultiRNNCell( cells[:encoder_num_layers] )
            word_cell_bw = tf.contrib.rnn.MultiRNNCell( cells[encoder_num_layers:] )
            # encoder RNN
            _, sentence_encoder_final_state = tf.nn.bidirectional_dynamic_rnn(word_cell_fw, word_cell_bw, rnn_words_outputs, sequence_length=batch_word_lengths, dtype=dtype, scope="sentence_encoder_rnn")
            if cell_type == 'LSTM' or cell_type == "UGRNN" or cell_type == "LNLSTM" or cell_type == "GRU":
                sentence_encoder_final_state = tf.concat([ state[encoder_num_layers-1][0] for state in sentence_encoder_final_state] , 1)
            else:
                sentence_encoder_final_state = tf.concat([ state[encoder_num_layers-1] for state in sentence_encoder_final_state] , 1)
        return sentence_encoder_final_state
    
    
    
def encoder(state_size, num_layers, rnn_inputs, dtype, cell_type, peephole, input_keep_prob, output_keep_prob, scope="encoder"):
    """
    Encoder of the VRAE model. It corresponds to the approximation of p(z|x), thus encodes the inputs x into a higher level representation z. The encoder is Dynamic Recurrent Neural Network which takes a batch of sequence of arbitray lengths as inputs. The output is the last state of the last cell and corresponds to a representation of the whole input.
    This is the character-level encoder.
    Args:
        state_size (Natural Integer): state size for the RNN cell
        num_layers (Natural Integer): number of layers for the the RNN cell
        rnn_inputs (Tensor): input tensor (batch_size x None x input_dimension)
        cell_type (String): type of cell to use
        peephole (Boolean): use peephole for lstm
        dtype (string): dtype
        input_keep_prob (float): dropout keep probability for the inputs
        output_keep_prob (float): dropout keep probability for the outputs
        scope (string): scope name
    Returns:
            (Tensor) the last state of the RNN, dimension (batch_size x state_size)
    """
    with tf.name_scope(scope):
        with tf.variable_scope('encoder_cell'):
            if cell_type == 'GRU':
                cell_fn = tf.contrib.rnn.GRUCell
            elif cell_type == 'LSTM':
                cell_fn = tf.contrib.rnn.LSTMCell
            elif cell_type == 'LNLSTM':
                cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
            elif cell_type == "UGRNN":
                cell_fn = tf.contrib.rnn.UGRNNCell
            elif cell_type == "GLSTM":
                cell_fn = tf.contrib.rnn.GLSTMCell
            elif cell_type == "LSTMBlockFusedCell":
                cell_fn = tf.contrib.rnn.LSTMBlockFusedCell
            
            cells = []
            for _ in range(2 * num_layers):
                cell = cell_fn(state_size)
                cell = tf.contrib.rnn.DropoutWrapper( cell, output_keep_prob=output_keep_prob, input_keep_prob=input_keep_prob)
                cells.append(cell)
            cell_fw = tf.contrib.rnn.MultiRNNCell( cells[:num_layers] )
            cell_bw = tf.contrib.rnn.MultiRNNCell( cells[num_layers:] )
            rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, dtype=dtype, scope="Encoder_rnn")
        if cell_type == 'LSTM' or cell_type == 'UGRNN':
            final_state = tf.concat([ state[num_layers-1][0] for state in final_state] , 1)
        else:
            final_state = tf.concat([ state[num_layers-1] for state in final_state] , 1)
        return final_state


def stochasticLayer(encoder_output, latent_dim, batch_size,dtype, scope="stochastic_layer"):
    """
    The stochastic layer represents the prior distribution Z. We choose to model the prior as a Gaussian distribution with parameters mu and sigma. The distribution is represented by these two parameters only (mu and sigma) as introduced by https://arxiv.org/abs/1312.6114. Then we can draw samples epsilon from a normal distribution N(0,1) and obtain the samples z = mu + epsilon * sigma from the prior. This is what we call the "reparametrization trick" and allows us to train the model using SGD.
    Args:
        encoder_output (Tensor): input tensor (batch_size x encoder_state_size)
        latent_dim (Natural Integer): dimension of the latent space
        batch_size (Natural Integer): batch length
        scope (string): scope name
    Returns:
        A tuple z,z_mu,z_ls2:
            z: samples drawn from the prior
            z_mu: tensor representing 
    """
    with tf.name_scope(scope):
        # reparametrization trick
        with tf.name_scope("Z"):
            z_mu = tf.contrib.layers.fully_connected( inputs=encoder_output,num_outputs=latent_dim, activation_fn=None, scope="z_mu" ) 
            z_ls2 = tf.contrib.layers.fully_connected( inputs=encoder_output,num_outputs=latent_dim, activation_fn=None, scope="z_ls2" ) 
            
        # sample z from the latent distribution
        with tf.name_scope("z_samples"):
            with tf.name_scope('random_normal_sample'):
                eps = tf.random_normal((batch_size, latent_dim), 0, 1, dtype=dtype) # draw a random number
            with tf.name_scope('z_sample'):
                z = tf.add(z_mu, tf.multiply(tf.sqrt(tf.exp(z_ls2)), eps))  # a sample it from Z -> z
        # summaries
        tf.summary.histogram("z_mu", z_mu)
        tf.summary.histogram("z_ls2", z_ls2)
        tf.summary.histogram("z", z)
        
        return z,z_mu,z_ls2


def dynamic_rnn_with_projection_layer( cell_dec, z_input, x_input_lenghts, W_proj, b_proj, batch_size, state_size, data_dim, x_inputs,training,dtype, scope="dynamic_rnn_with_projection_layer"):
    """
    A custom dynamic rnn implemented using the raw_rnn class from Tensorflow. The difference with the dynamic_rnn is the use of a projection layer to feed the true output value to the next step. Indeed, for each cell, the output is a tensor of size (batch_size x state_size). Here we project this output into the expected output value, thus we obtain a Tensor (batch_size x data_dim). Then we output this expected output to the next cell. This makes the model more robust.
    Args:
        cell_dec (tf.nn.rnn_cell): RNN cell
        z_input (Tensor): input Tensor of size (batch_size x state_size) Typically the samples z projected to the dimension of the decoder
        x_input_lengths (Tensor): a Tensor of integers of size (batch_size, ). Lenght of the input sequences.
        W_proj (tf.Variable): weights of the projection layer.
        b_proj (tf.Variable): biases of the projection layer.
        batch_size (Natural Integer): batch size.
        state_size (Natural Integer): RNN cell state size.
        data_dim (Natural Integer): dimension of the data.
        x_inputs (Tensor): inputs
        training (bool): training phase or not
        dtype (string): dtype to be used   
        scope (string): scope name
    """
    # following dynamic_rnn implementation https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
    
    x_inputs = tf.transpose( x_inputs , [1,0,2]) # set time first
    flat_input = nest.flatten(x_inputs)
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]
    
    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)
    
    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]
    
    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
              "Input size (depth of inputs) must be accessible via shape inference,"
              " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
        raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
        raise ValueError(
          "Batch_size is not the same for all the elements in the input.")
    
    base_name = "encoder_input_ta"
    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        tensor_array_name=base_name + name)

    input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                   for i in range(len(flat_input)))

    input_ta = tuple(ta.unstack(input_)
                   for ta, input_ in zip(input_ta, flat_input))
    
    input_ta = input_ta[0]
    
    with tf.name_scope(scope):
        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            #prev_out = cell_output
            elements_finished = (time >= x_input_lenghts) # array or bool
            finished = tf.reduce_all(elements_finished) # check if all elements finished and get a single boolean
            if cell_output is None:  # time == 0
                next_cell_state = cell_dec.zero_state(batch_size, dtype)
                next_input_value = tf.concat([z_input, tf.zeros([batch_size,data_dim], dtype=dtype)], 1) 
            else:
                #emit_output = tf.add(tf.matmul(W_proj,prev_out), b_proj)
                next_cell_state = cell_state
                predicted_previous_output = tf.cond(training, 
                                                    lambda: input_ta.read(time-1), 
                                                    lambda: tf.nn.softmax(tf.add(tf.matmul(cell_output, W_proj), b_proj) ))
                next_input_value = tf.cond( # removing this condition leads to the read TensorArray problem: used for dynamic rray
                    finished,
                    lambda:tf.concat([ tf.zeros([batch_size,state_size], dtype=dtype), predicted_previous_output], 1) ,
                    lambda:tf.concat([z_input, predicted_previous_output ], 1) )
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, data_dim + state_size], dtype=dtype),
                lambda: next_input_value )
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)
        return tf.nn.raw_rnn(cell_dec, loop_fn)#, parallel_iterations = 1)


def decoder(z, batch_size, state_size, num_layers, data_dim, x_input_lenghts, cell_type, peephole, input_keep_prob, output_keep_prob, x_inputs, training, dtype, scope="decoder"):
    """"
    Decoder of the VRAE model. This neural network approximates the posterior distribution p(x|z). The decoder transforms samples z from the prior distribution to a reconstruction of x.
    Args:
        z (Tensor): samples z from the prior distribution (batch_size x latent_dim)
        batch_size (Natural Integer): batch size
        state_size (Natural Integer): size of the RNN cell
        num_layers (Natural Integer): number of layers in the RNN cell
        data_dim (Natural Integer): dimension of the data
        x_input_lenghts (Tensor): lengths of the inputs (batch_len, )
        dtype (string): dtype to be used
        cell_type (string): type of RNN cell
        peephole (Boolean): use peephole for lstm
        input_keep_prob (float): dropout keep probability for the inputs
        output_keep_prob (float): dropout keep probability for the outputs
        x_inputs (Tensor): inputs
        training (bool): training phase or not
        scope (string): scope name
    Returns:
        A tensor of size (batch_size x None x data_dim) which is a reconstruction of x
    """
    with tf.name_scope(scope):
        # projection layer
        with tf.name_scope("projection_layer"):
            W_proj = tf.Variable(tf.random_uniform([state_size, data_dim], 0, 1, dtype=dtype), dtype=dtype)
            b_proj = tf.Variable(tf.zeros([data_dim], dtype=dtype), dtype=dtype)
        # connect z to the RNN
        h_z2dec = tf.contrib.layers.fully_connected(z, state_size, scope="z2initial_decoder_state", activation_fn=None)
        # RNN Cell
        if cell_type == 'GRU':
            cell_fn = tf.contrib.rnn.GRUCell
        elif cell_type == 'LSTM':
            cell_fn = tf.contrib.rnn.LSTMCell
        elif cell_type == 'LNLSTM':
            cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
        elif cell_type == "UGRNN":
            cell_fn = tf.contrib.rnn.UGRNNCell
        elif cell_type == "GLSTM":
            cell_fn = tf.contrib.rnn.GLSTMCell
        cells = []
        for _ in range(num_layers):
            cell = cell_fn(state_size)
            cell = tf.contrib.rnn.DropoutWrapper( cell, output_keep_prob=output_keep_prob, input_keep_prob=input_keep_prob)
            cells.append(cell)
        dec_cell = tf.contrib.rnn.MultiRNNCell(cells)                                 
        # RNN decoder
        outputs_ta, final_state, _ = dynamic_rnn_with_projection_layer( dec_cell, h_z2dec, x_input_lenghts, W_proj, b_proj, batch_size, state_size, data_dim, x_inputs, training, dtype, scope="dynamic_rnn_with_projection_layer")
         # project the output
        rnn_outputs_decoder = outputs_ta.stack()
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(rnn_outputs_decoder))
        decoder_outputs_flat = tf.reshape(rnn_outputs_decoder, (-1, state_size))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W_proj), b_proj)
        rnn_outputs_decoder = tf.transpose( tf.reshape(decoder_logits_flat, (decoder_max_steps, batch_size, data_dim)) , [1,0,2])
        # no softmax here: softmax is applied in the loss function 
        return rnn_outputs_decoder
                    

def sentence_loss(x_reconstr_mean, x_input, weights_input, dtype, scope="sentence_loss"):
    """
    Sentence loss based on tf.contrib.seq2seq.sequence_loss. This is an reduced element-wise cross entropy.
    Args:
        x_reconstr_mean (Tensor): reconstruction of the input (batch_len x None x data_dim)
        x_input (Tensor): model input (batch_len x None x data_dim)
        weights_input (Tensor): model input weights (batch_len x None). A list of integer to indicate if 
            the current element is a real element (1) or a element added for padding (0).
        dtype (string): dtype
        scope (string): scope name
    Returns:
        Reconstruction loss (Variable)
    """
    with tf.name_scope(scope):
        return tf.contrib.seq2seq.sequence_loss(x_reconstr_mean, x_input, tf.cast( weights_input, dtype) )
                    
def latent_loss_function(z_ls2, z_mu, scope="latent_loss"):
    """
    Latent loss. Acts as a regularization and shape the prior distribution as normal distribution N(0,1). This is used to limit the capacity of the latent distribution and push the model to optimize its content by placing similar items close to another.
    Args:
        z_ls2 (Tensor): log of the squarred value of sigma, a parameter which controls the prior distribution
        z_mu (Tensor): value of mu, a parameter which controls the prior distribution
        scope (string): scope name
    Returns: 
        Latent loss (Variable)
    """
    with tf.name_scope(scope):
        return -0.5 * tf.reduce_sum(1 + z_ls2 - tf.square(z_mu) - tf.exp(z_ls2), 1)
        
def loss_function(x_reconstr_mean, x_input, weights_input,z_ls2, z_mu, B, latent_loss_weight, dtype, scope="loss"):
    """
    Loss function of the VRAE model: reconstruction loss + Beta * latent_loss_weight * latent_loss.
    Args:
        x_reconstr_mean (Tensor): reconstruction of the input (batch_len x None x data_dim)
        x_input (Tensor): model input (batch_len x None x data_dim)
        weights_input (Tensor): model input weights (batch_len x None). A list of integer to indicate if 
            the current element is a real element (1) or a element added for padding (0).
        z_ls2 (Tensor): log of the squarred value of sigma, a parameter which controls the prior distribution
        z_mu (Tensor): value of mu, a parameter which controls the prior distribution
        B (Placeholder): value of Beta used for the deterministic warm-up
        latent_loss_weight (float): weight used to weaken the latent_loss and help the model to optimize the reconstruction
        dtype (string): dtype
        scope (string): scope name
    Returns:
        loss of the VRAE model 
    """
    with tf.name_scope(scope):
        reconstruction_loss = sentence_loss(x_reconstr_mean, x_input, weights_input, dtype)
        latent_loss = latent_loss_function(z_ls2, z_mu) # L2 regularization
        #l2 = 0.00001 * sum(
        #    tf.nn.l2_loss(tf_var)
        #        for tf_var in tf.trainable_variables()
        #        if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        #)
        loss = tf.reduce_mean(reconstruction_loss + B * latent_loss_weight * latent_loss )
        # summaries
        tf.summary.scalar("reconstruction_loss", reconstruction_loss)
        tf.summary.scalar("latent_loss", tf.reduce_mean(latent_loss) )
        tf.summary.scalar("loss", loss)
        return loss, reconstruction_loss, latent_loss
                    
def optimizationOperation(cost, learning_rate, scope="training_step"):
    """
    optimizationStep
    Args:
        cost: loss function
        learning_rate (float or placeholder): learning rate
    Returns:
        Tensorflow optimizer
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        with tf.name_scope('train_step'):
            return tf.train.AdamOptimizer(learning_rate).minimize(cost)
        