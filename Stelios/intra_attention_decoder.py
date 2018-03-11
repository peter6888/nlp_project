"""intra_attention_decoder.py"""
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications Copyright 2018 Stelios Serghiou, Peter Li, Apurva Pancholi

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

"""This file defines the decoder"""

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


# Note: this function is based attention_decoder
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def intra_attention_decoder(decoder_inputs, initial_state, encoder_states,
                            enc_padding_mask, cell,
                            initial_state_attention=False, pointer_gen=True,
                            use_coverage=False, prev_coverage=None):
    """
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
      enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
      cell: rnn_cell.RNNCell defining the cell function and size.
      initial_state_attention:
        Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
      pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
      use_coverage: boolean. If True, use coverage mechanism.
      prev_coverage:
        If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

    Returns:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x cell.output_size]. The output vectors.
      state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
      attn_dists: A list containing tensors of shape (batch_size,attn_length).
        The attention distributions for each decoder step.
      p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
      coverage: Coverage vector on the last step computed. None if use_coverage=False.
    """
    with variable_scope.variable_scope("attention_decoder") as scope:

        # ABI'S ATTENTION MECHANISM:

        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value
        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value

        # Reshape encoder_states (need to insert a dim)
        # actual shape of encoder_states.get_shape (16, ?, 1, 512)
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        # now is shape (batch_size, attn_len, 1, attn_size)

        # To calculate attention, we calculate
        # v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and
        # (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        # Get the weight matrix W_h and apply it to each encoder state to get
        # (W_h h_i), the encoder features
        W_h = variable_scope.get_variable("W_h",
                                          size=[1, 1, attn_size,
                                                attention_vec_size])
        encoder_features = nn_ops.conv2d(encoder_states, W_h,
                                         [1, 1, 1, 1], "SAME")
        # shape (batch_size, attn_length, 1, attention_vec_size)

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable("v", [attention_vec_size])
        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = variable_scope.get_variable(
                    "w_c", [1, 1, 1, attention_vec_size])

        # for beam search mode with coverage
        if prev_coverage is not None:
            # reshape from (batch_size, attn_length) to
            # (batch_size, attn_len, 1, 1)
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

        # OUR ATTENTION MECHANISM:

        # decoder_states is defined when we call hybrid_states below
        def intra_temporal_attention(decoder_states):
            '''
            Get Intra-Temporal Attention Score. Refs to original paper section 2.1 https://arxiv.org/abs/1705.04304
            :param decoder_state:
            :param coverage: None
            :return:attention score
            '''
            decoder_state = decoder_states[-1][1]
            # decoder_state[1].get_shape() (batch_size, hidden_vec_size)
            decoder_hidden_vec_size = decoder_state.get_shape()[1].value
            encoder_hidden_vec_size = encoder_states.get_shape()[3].value
            # tf.logging.info("hidden vector size - encoder:{}, decoder:{}".format(encoder_hidden_vec_size, decoder_hidden_vec_size)) # encoder:512, decoder:256

            with variable_scope.variable_scope("IT_Attention"):
                # Intra-Temporal Attention
                # Equation (2) W_e_attn for h_d (hidden decoder vectors) and
                # h_e (hidden encoder vectors)
                W_e_attn = tf.get_variable('W_e_attn',
                                           shape=(1, 1,
                                                  encoder_hidden_vec_size,
                                                  decoder_hidden_vec_size),
                                           initializer=tf.contrib.layers.xavier_initializer())
                decoder_T = len(decoder_states)
                encoder_states_dot_W = nn_ops.conv2d(encoder_states,
                                                     W_e_attn, [1, 1, 1, 1],
                                                     "SAME")
                # shape (batch_size, attn_length, 1, decoder_hidden_vec_size)

                # caculate eti[decoder_T - 1], which is a list have length
                # len(encoder_states)
                # tf.logging.info("encoder_states_dot_W.shape {}".format(encoder_states_dot_W.get_shape()))
                # encoder_states_dot_W.shape (16, ?, 1, 256)

                decoder_state = tf.expand_dims(
                    tf.expand_dims(decoder_state, 1), 1)
                # reshape to (batch_size, 1, 1, decoder_hidden_vec_size)

                e = math_ops.reduce_sum(
                    decoder_state * encoder_states_dot_W, axis=[2, 3])
                # shape (batch_size, attn_length)

                # Equation (3)
                if decoder_T == 1:
                    e_prime = tf.exp(e)
                else:
                    # Stelios
                    denominator = tf.reduce_sum(eti, axis=0)
                    e_prime = tf.divide(tf.exp(e), denominator)
                # tf.logging.info("e_prime.shape:{}".format(e_prime.get_shape()))
                # (batch_size, attn_length)

                # append to eti list after e_prime has been calculated
                # Stelios
                eti.append(e_prime)
                # tf.logging.info("e.shape:{}".format(e.get_shape()))
                # e.shape:(batch_size, attn_length)

                # Equation (4)
                # Stelios
                attn_score = tf.nn.softmax(e_prime)
                # tf.logging.info("attn_score.shape:{}".format(attn_score.get_shape()))
                # attn_score.shape:(16, attn_length)

                return attn_score

        def intra_decoder_attention(decoder_states):
            '''
            Get Intra-Decoder Attention Score. Refs to original paper section 2.2 https://arxiv.org/abs/1705.04304
            :param decoder_state:
            :param coverage: None
            :return:attention score with shape [batch_size, T]
            '''
            decoder_state = decoder_states[-1][1]
            # decoder_state[1].get_shape()
            # (batch_size, hidden_vec_size)

            decoder_hidden_vec_size = decoder_state.get_shape()[1].value

            with variable_scope.variable_scope("ID_Attention"):
                # Intra-Decoder Attention
                # W_d_attn for h_d (hidden decoder vectors) and h_d (hidden
                # decoder vectors)
                W_d_attn = tf.get_variable('W_d_attn',
                                           shape=(1, 1,
                                                  decoder_hidden_vec_size,
                                                  decoder_hidden_vec_size),
                                           initializer=tf.contrib.layers.xavier_initializer())
                decoder_T = len(decoder_states)

                if decoder_T > 1:
                    # Stelios
                    # Change from (T, batch_size, dec_hidden_size) to
                    # (batch_size, T, dec_hidden_size)
                    decoder_states_stack = tf.transpose(decoder_states_stack,
                                                        perm=[1, 0, 2])
                    # Equation (6)
                    decoder_states_ex = tf.expand_dims(
                        decoder_states_stack, axis=2)
                    # tf.logging.info("decoder_states_ex:{}".format(
                    #     decoder_states_ex.get_shape()))
                    decoder_states_dot_W = nn_ops.conv2d(decoder_states_ex,
                                                         W_d_attn,
                                                         [1, 1, 1, 1],
                                                         "SAME")
                    # shape (batch_size, decoder_T, 1, decoder_hidden_vec_size)

                    # Stelios
                    e = tf.einsum("ijkl,il->ij",
                                  decoder_states_dot_W, decoder_state)
                    # Stelios: (batch_size, decoder_T)

                    # Equation (7)
                    # Stelios
                    attn_score = tf.nn.softmax(e[:, :-1])  # ignore last e
                    # Stelios ?shape (batch_size, decoder_T - 1)
                else:
                    # Stelios - we do not need this
                    # (I have left it in in case it is being used elsewhere..)
                    attn_score = tf.zeros([batch_size, 1])

                return attn_score

        def hybrid_attention(decoder_states, coverage=None):
            '''
            The hybrid attention model which concat Intra Temporal Attention and Intra-Decoder Attention to get context and distrubution
            :param decoder_states: decoder hidden states shape list([batch_size, ])
            :param coverage:
            :return: context vector, attention distribution
            '''

            def masked_attention(e):
                """Take softmax of e then apply enc_padding_mask and re-normalize"""
                # Stelios
                attn_dist = e
                attn_dist *= enc_padding_mask  # apply mask
                masked_sums = tf.reduce_sum(attn_dist, axis=1)
                # shape (batch_size)

                # re-normalize
                return attn_dist / tf.reshape(masked_sums, [-1, 1])

            temporal_attention = intra_temporal_attention(decoder_states)
            decoder_attention = intra_decoder_attention(decoder_states)

            # encoder_states: [batch_size x attn_length x 1 x attn_size]
            # Equation (5) - result has shape (batch_size, 1, enc_hidden_size)
            # --> After squeeze (batch_size, encoder_hidden_size)
            temporal_context = tf.squeeze(
                tf.einsum('ijkl,ij->ikl', encoder_states, temporal_attention))
            # print(temporal_context.get_shape())--> (16, 512)

            # Equation (8) - result has shape (batch_size, decoder_hidden_size)
            # Stelios: I have changed this on the assumption that size of
            # decoder_states_stack = [decoder_T x batch_size x dec_hidden_size]
            if len(decoder_states) > 1:
                decoder_context = tf.einsum('ijk,ji->jk',
                                            decoder_states_stack[:-1, :, :],
                                            decoder_attention)
            else:
                decoder_context = tf.zeros(
                    shape=[decoder_attention.get_shape().as_list()[0] - 1,
                           decoder_states[-1][1].get_shape().as_list()[1]])

            # Calculate attention distribution
            # To-do: 2.3 a different way to caculate the distribution
            attn_dist = masked_attention(temporal_attention)
            context_vector = temporal_context

            return context_vector, attn_dist, decoder_context, coverage

        # Run process
        tf.logging.info("Using Intra Temporal + Decoder Attention Model")

        # The eti in equation (1), eti is a list length of decoder_steps_length, and each eti[t] is a list of length encoder_steps_length
        eti = []
        # eti and ett do NOT share same weight

        outputs = []
        attn_dists = []
        p_gens = []
        decoder_states = []
        temoral_attention_scores = []
        input_contexts = []
        decoder_contexts = []
        state = initial_state
        # don't need initial_state for caculation
        # decoder_states.append(state)

        # initialize coverage to None or whatever was passed in
        coverage = prev_coverage

        context_vector = array_ops.zeros([batch_size, attn_size])
        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_size])

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we
            # can pass it through a linear layer with this step's input to get
            # a modified version of the input
            context_vector, _, decoder_context, coverage = hybrid_attention(
                [initial_state], coverage)
            # in decode mode, this is what updates the coverage vector

        for i, inp in enumerate(decoder_inputs):
            tf.logging.info(
                "Adding attention_decoder timestep %i of %i", i,
                len(decoder_inputs))
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the
            # same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError(
                    "Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Keep the decoder states
            decoder_states.append(state)
            _, decoder_states_list = map(list, zip(*decoder_states))
            decoder_states_stack = tf.stack(decoder_states_list)
            # print(decoder_states_stack.get_shape())
            # (T, batch_size, decoder_hidden_size)

            # Run the attention mechanism
            # always true in decode mode
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    # you need this because you've already run the initial attention(...) call
                    context_vector, attn_dist, decoder_context, _ = hybrid_attention(
                        decoder_states, coverage)
                    # don't allow coverage to update
            else:
                context_vector, attn_dist, decoder_context, coverage = hybrid_attention(
                    decoder_states, coverage)
            decoder_contexts.append(decoder_context)
            attn_dists.append(attn_dist)
            temoral_attention_scores.append(attn_dist)
            input_contexts.append(context_vector)

            # Calculate p_gen
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear(
                        [context_vector, state.c, state.h, x], 1, True)
                    # a scalar

                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector],
                                cell.output_size, True)
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = array_ops.reshape(coverage, [batch_size, -1])

        # common part of return
        decoder_rets = {"outputs": outputs, "state": state,
                        "attn_dists": attn_dists, "p_gens": p_gens, "coverage": coverage}
        # extra returns for Socher model
        decoder_rets["temoral_attention_scores"] = temoral_attention_scores
        decoder_rets["input_contexts"] = input_contexts

        # Stelios - not sure what this is doing, but with my changes in the
        # attention mechanism it may not be needed and may cause trouble
        if len(decoder_contexts) > 0:
            decoder_contexts.insert(0, tf.zeros_like(
                decoder_contexts[-1]))  # append left with zeros
        decoder_rets["decoder_contexts"] = decoder_contexts

        return decoder_rets


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
