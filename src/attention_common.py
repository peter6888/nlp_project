"""attention_common.py"""
# Copyright 2018 Copyright 2018 Stelios Serghiou, Peter Li, Apurva Pancholi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

def intra_temporal_context(decoder_states, encoder_states, eti, enc_padding_mask):
    '''
    Caculate the intra_temporal context
    :param decoder_states:
    :param encoder_states:
    :param eti:
    :param enc_padding_mask:
    :return:
    '''
    # Calculate encoder distribution
    # Mask padded sequences
    temporal_attention = intra_temporal_attention(decoder_states,
                                                      encoder_states, eti)
    attn_dist = masked_attention(temporal_attention, enc_padding_mask)

    # Equation (5)
    # encoder_states: batch_size x attn_length x 1 x attn_size
    # attn_dist: batch_size x attn_length
    # Result has shape (batch_size, 1, encoder_hidden_size)
    # --> After squeeze (batch_size, encoder_hidden_size)
    temporal_context = tf.squeeze(
        tf.einsum('btkh,bt->bkh', encoder_states, attn_dist))
    # print(temporal_context.get_shape())--> (16, 512)
    context_vector = temporal_context

    return context_vector, attn_dist


def intra_temporal_attention(decoder_states, encoder_states, eti_list):
    '''
    Get Intra-Temporal Attention Score. Refs to original paper section 2.1 https://arxiv.org/abs/1705.04304
    :param decoder_state:
    :param coverage: None
    :return:attention score
    '''
    # Extract hidden state from list and tuple of decoder states
    decoder_state = decoder_states[-1][1]
    # decoder_state[1].get_shape() (batch_size, hidden_vec_size)

    decoder_hidden_vec_size = decoder_state.get_shape()[1].value
    encoder_hidden_vec_size = encoder_states.get_shape()[3].value

    # Intra-Temporal Attention
    with variable_scope.variable_scope("IT_Attention"):

        # Equation (2) W_e_attn for h_d (hidden decoder vectors) and
        # h_e (hidden encoder vectors)
        W_e_attn = tf.get_variable('W_e_attn',
                                   shape=(1, 1,
                                          encoder_hidden_vec_size,
                                          decoder_hidden_vec_size),
                initializer=tf.contrib.layers.xavier_initializer())

        decoder_T = len(decoder_states)

        encoder_states_dot_W = nn_ops.conv2d(encoder_states, W_e_attn,
                                            [1, 1, 1, 1],
                                            "SAME")
        # shape (batch_size,?,1,decoder_hidden_vec_size)

        # tf.logging.info("encoder_states_dot_W.shape {}".format(encoder_states_dot_W.get_shape()))
        # encoder_states_dot_W.shape (16, len_attn, 1, 256)

        decoder_state = tf.expand_dims(tf.expand_dims(decoder_state, 1), 1)
        # reshape to (batch_size, 1, 1, decoder_hidden_vec_size)

        e = math_ops.reduce_sum(decoder_state * encoder_states_dot_W, [2, 3])
        # shape: (batch_size x attn_length)

        # Equation (3)
        if decoder_T == 1:
            e_prime = tf.exp(e)
        else:
            denominator = tf.reduce_sum(tf.exp(eti_list), axis=0)
            e_prime = tf.divide(tf.exp(e), denominator)
        # tf.logging.info("e_prime.shape:{}".format(e_prime.get_shape())) # (batch_size, attn_length)

        # append to eti list after e_prime been calculated
        eti_list.append(e)
        # tf.logging.info("e.shape:{}".format(e.get_shape()))
        # e.shape:(batch_size, ?)

        # Equation (4)
        attn_score = tf.nn.softmax(e_prime)
        # tf.logging.info("attn_score.shape:{}".format(attn_score.get_shape())) # attn_score.shape:(16, attn_length)

        return attn_score

def intra_decoder_attention(decoder_states_stack):
    '''
    Get Intra-Decoder Attention Score. Refs to original paper section 2.2
    https://arxiv.org/abs/1705.04304.
    Args:
        decoder_states_stack: tensor of decoder hidden states
            size = [T, batch_size, decoder_hidden_size]
    Returns:
        attn_score: tensor of attnetion scores alpha_d_tt (Equation 7),
            size = [batch_size, T - 1]
    '''
    batch_size = decoder_states_stack[-1].get_shape()[0].value
    decoder_T = decoder_states_stack.get_shape()[0]
    decoder_state = decoder_states_stack[-1]
    # decoder_state[1].get_shape() (batch_size, hidden_vec_size)

    decoder_hidden_vec_size = decoder_state.get_shape().as_list()[-1]

    # Intra-Decoder Attention
    with variable_scope.variable_scope("ID_Attention"):

        # W_d_attn of Equation 6
        W_d_attn = tf.get_variable('W_d_attn',
            shape=(decoder_hidden_vec_size, decoder_hidden_vec_size),
            initializer=tf.contrib.layers.xavier_initializer())

        if decoder_T > 1:
            # Equation (6)
            # return shape [T-1, batch_size, hidden_state_size]
            decoder_states_dot_W = tf.einsum(
                "ij,tbi->tbj", W_d_attn, decoder_states_stack[:-1])

            e = tf.einsum("tbi,bi->bt", decoder_states_dot_W, decoder_state)
            # return shape [batch_size, T-1]

            # Equation (7)
            attn_score = tf.nn.softmax(e)
            # shape (batch_size, decoder_T-1)
        else:
            attn_score = tf.zeros([batch_size, 1])

        return attn_score

def intra_decoder_context(decoder_states_stack):
    '''
    :param decoder_states_stack: shape T x batch_size x decoder_hidden_size
    :return: intra_decoder_context shape batch_size x decoder_hidden_size
    '''
    T, batch_size, decoder_hidden_size = decoder_states_stack.get_shape().as_list()
    decoder_attention = intra_decoder_attention(decoder_states_stack)
    # Equation (8)
    # decoder_states_stack: T x batch_size x decoder_hidden_size
    # decoder_attention: batch_size x T - 1
    # Result has shape (batch_size, decoder_hidden_size)
    if T > 1:
        decoder_context = tf.einsum('tbh,bt->bh',
                                    decoder_states_stack[:-1, :, :],
                                    decoder_attention)
                                    # ignore the last e
    else:
        decoder_context = tf.zeros(shape=[batch_size, decoder_hidden_size])

    return decoder_context

def masked_attention_with_softmax(e, masks):
    """Take softmax of e then apply enc_padding_mask and re-normalize"""
    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
    attn_dist *= masks  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

def masked_attention(e, enc_padding_mask):
    '''
    Apply enc_padding_mask on encoder attention, and re-normalized it
    Args:
        e: tensor of original encoder attention scores
            size = batch_size x attn_length
        enc_padding_mask: tensor of masks
            size = batch_size x attn_length
    Returns:
        attn_dist: masked attention score,
            size = batch_size x attn_length
    '''
    attn_dist = e
    attn_dist *= enc_padding_mask  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # tf.logging.info("Linear Matrix [total_arg_size:{},output_size:{}]".format(total_arg_size, output_size))
    '''
    INFO:tensorflow:Linear Matrix [total_arg_size:640,output_size:128]
    INFO:tensorflow:Linear Matrix [total_arg_size:512,output_size:512]
    '''
    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term

def insert_zeros_at_end(old_tensor):
    '''
    Experiement how to insert zero for a Tensor with shape (15, 256) to (16, 256)
    Returns:
    '''
    zero_tensor = tf.zeros(shape=[1, old_tensor.get_shape().as_list()[1]])
    new_tensor = tf.concat([old_tensor, zero_tensor], axis=0)
    return new_tensor

############ The unit tests ###############
def test_intra_temporal_attention(args):
    batch_size = 5
    max_total_time = 4
    input_vector_size = 3
    hidden_vector_size = 2
    lstm_encode_cell = tf.nn.rnn_cell.LSTMCell(hidden_vector_size)
    lstm_decode_cell = tf.nn.rnn_cell.LSTMCell(hidden_vector_size)
    initial_state = lstm_decode_cell.zero_state(batch_size, tf.float32)

    eti_list = []

    #construct encoder_states
    inputs = tf.random_normal(shape=(batch_size, max_total_time, input_vector_size))
    encoder_states, state  = tf.nn.dynamic_rnn(lstm_encode_cell, inputs, dtype=tf.float32, swap_memory=True)
    encoder_states = tf.expand_dims(encoder_states, axis=2)

    decoder_states = []
    for i in range(max_total_time):
        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
        inputs = tf.random_normal(shape=(batch_size, input_vector_size))
        output, state = lstm_decode_cell(inputs, initial_state)
        decoder_states.append(state)
        attn_score = intra_temporal_attention(decoder_states, encoder_states, eti_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _attn_score, _eti_list = sess.run([attn_score, eti_list])
        print(_attn_score, _eti_list)

''' run output
[[0.24959728 0.250187   0.25239825 0.24781743]
 [0.25023326 0.24915825 0.25067964 0.24992885]
 [0.2526203  0.25018734 0.25049213 0.24670024]
 [0.24964409 0.25249588 0.25024527 0.2476147 ]
 [0.25222957 0.24989128 0.24894954 0.2489296 ]] [array([[ 0.00243569,  0.00110005, -0.00362626,  0.0062934 ],
       [ 0.01002057, -0.01420728,  0.00234492, -0.00138108],
       [-0.00786934,  0.00460867,  0.00512778,  0.02276652],
       [-0.00048143, -0.00333946, -0.00568223, -0.00368661],
       [-0.02969718,  0.00526314,  0.01831158,  0.03807773]],
      dtype=float32), array([[ 4.3471083e-03,  2.1533279e-03, -5.5117412e-03,  1.0610560e-02],
       [-1.8117115e-02,  2.3800917e-02, -2.7742297e-03,  2.0854485e-03],
       [ 3.1152950e-03, -7.1943197e-03, -8.9176204e-03, -2.2364363e-02],
       [-1.7991727e-03, -1.5096837e-03, -1.0802690e-02, -1.2376806e-02],
       [-2.4162211e-02, -5.2276533e-05,  8.1756189e-03,  3.6634713e-02]],
      dtype=float32), array([[ 3.55663244e-04,  3.51795170e-05, -1.16347615e-03,
         1.32937881e-03],
       [ 5.47397137e-03, -7.99820386e-03,  1.46524783e-03,
        -8.06200551e-04],
       [-1.19531397e-02, -6.22492004e-03, -9.17457324e-03,
         1.69785414e-03],
       [ 7.78692338e-05, -2.95359176e-03, -2.40339548e-03,
         1.50260632e-04],
       [-1.24974195e-02, -2.50677811e-03,  3.82442726e-04,
         2.21831650e-02]], dtype=float32), array([[-0.00031409,  0.00076142,  0.00503228, -0.00376655],
       [-0.00027401, -0.00310992,  0.00265443, -0.00072571],
       [ 0.00479821, -0.00225311, -0.0024122 , -0.0124967 ],
       [-0.00236128,  0.00713076, -0.00551229, -0.01508032],
       [-0.01548023, -0.00179417,  0.00250703,  0.02576774]],
      dtype=float32)]
'''

def get_decoder_states_stack():
    batch_size = 5
    max_total_time = 4
    input_vector_size = 3
    hidden_vector_size = 2
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_vector_size)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    inputs = tf.random_normal(shape=(batch_size, input_vector_size))
    decoder_states = []
    for _ in range(max_total_time):
        _, hidden_state = lstm_cell(inputs, initial_state)
        decoder_states.append(hidden_state)

    _, decoder_states_list = map(list, zip(*decoder_states))
    decoder_states_stack = tf.stack(decoder_states_list)

    return decoder_states_stack

def test_intra_decoder_attention(args):
    #(decoder_states, decoder_states_stack):
    '''
    decoder_states - list(Tensor)
    :param args:
    :return:
    '''

    dec_stack = get_decoder_states_stack()
    attn = intra_decoder_attention(dec_stack)

    with variable_scope.variable_scope("ID_Attention"):
        variable_scope.get_variable_scope().reuse_variables()
        # Read W_d_attn
        W_d_attn = tf.get_variable('W_d_attn')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _attn, _stack, _W = sess.run([attn, dec_stack, W_d_attn])
        print(_attn)
        print("stack values")
        print(_stack)
        print("W values:")
        print(_W)

        h_T = _stack[-1] # shape (batch_size, vector_size)

        #Equation (6)
        w_dot_h_T = np.einsum("bi, ij->bj", h_T, _W)
        print("w_dot_h_T")
        print(w_dot_h_T)
        e_d_t_t_prime = np.einsum("bi,tbi->bt", w_dot_h_T, _stack[:-1])
        print("Equation (6) result:")
        print(e_d_t_t_prime)

        #Equation (7)
        exp_e_s = np.exp(e_d_t_t_prime)
        print("After np.exp()")
        print(exp_e_s)
        print("sum over time:")
        print(np.sum(exp_e_s, axis=1, keepdims=True))
        print("After divide and sum over")
        result = exp_e_s / np.sum(exp_e_s, axis=1, keepdims=True)
        print(result)

'''
--first run result---
[[0.3333333  0.3333333  0.3333333  0.3333333 ]
 [0.33333334 0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334 0.33333334]
 [0.3333333  0.3333333  0.3333333  0.3333333 ]
 [0.33333334 0.33333334 0.33333334 0.33333334]]
--new run result--
[[0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334]
 [0.33333334 0.33333334 0.33333334]]
'''

def test_intra_decoder_context(args):
    context = intra_decoder_context(get_decoder_states_stack())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _context = sess.run(context)
        print("context.shape {}".format(_context.shape))
        print(_context)
    return

def test_insert_zeros_at_end(args):
    batch_size = 4
    vector_size = 5
    old_tensor = tf.random_normal(shape=(batch_size,vector_size))
    new_tensor = insert_zeros_at_end(old_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _old_tensor, _new_tensor = sess.run([old_tensor, new_tensor])
        print("Tensor before insert zero shape {}".format(_old_tensor.shape))
        print(_old_tensor)
        print("Tensor after insert zero {}".format(_new_tensor.shape))
        print(_new_tensor)

def test_attention_mask(args):
    '''
    Unit test for attention mask
    '''
    batch_size = 4
    vector_size = 3
    old_attn = tf.random_normal(shape=[batch_size, vector_size])
    mask = tf.constant([1.0, 1.0, 0])
    new_attn = masked_attention(old_attn, mask)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _new_attn, _mask, _old_attn = sess.run([new_attn, mask, old_attn])
        print("old_attn---")
        print(_old_attn)
        print("mask----")
        print(_mask)
        print("new_attn---")
        print(_new_attn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test tokenization for matching parameter dimensions')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser(
        'test1', help='Test attention mask')
    command_parser.set_defaults(func=test_attention_mask)

    command_parser = subparsers.add_parser(
        'test2', help='test intra decoder attention')
    command_parser.set_defaults(func=test_intra_decoder_attention)

    command_parser = subparsers.add_parser(
        'test3', help='test intra temporal attention')
    command_parser.set_defaults(func=test_intra_temporal_attention)

    command_parser = subparsers.add_parser(
        'test4', help='test insert zero at the end of Tensor')
    command_parser.set_defaults(func=test_insert_zeros_at_end)

    command_parser = subparsers.add_parser(
        'test5', help='test intra decoder context')
    command_parser.set_defaults(func=test_intra_decoder_context)

    ARGS = parser.parse_args()
    if not hasattr(ARGS, 'func'):
        parser.print_help()
    else:
        ARGS.func(ARGS)
