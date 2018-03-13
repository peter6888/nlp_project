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

"""This file defines the intra decoder"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from attention_common import linear, masked_attention
from attention_common import intra_decoder_context, intra_temporal_attention

# Note: this function is based attention_decoder
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def intra_attention_decoder(decoder_inputs, initial_state, encoder_states,
                            enc_padding_mask, cell,
                            initial_state_attention=False, pointer_gen=True,
                            use_coverage=False, prev_coverage=None, input_attention=1, use_intra_decoder_attention=False):
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
    tf.logging.info("input_attention is {}, ('0-Pointer-generator-attention, 1-Intra-Temporal Attention.')".format(input_attention))
    with variable_scope.variable_scope("attention_decoder") as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value
        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value

        # Reshape encoder_states (need to insert a dim)
        # actual shape of encoder_states.get_shape (16, ?, 1, 512)
        # now is shape (batch_size, attn_len, 1, attn_size)
        encoder_states = tf.expand_dims(encoder_states, axis=2)

        def hybrid_attention(decoder_states, coverage=None):
            '''
            The hybrid attention model which concat Intra Temporal Attention and Intra-Decoder Attention to get context and distrubution
            Args:
                decoder_states: list of decoder hidden states shape,
                    size = list([batch_size, hidden_dim])
                coverage: initialized to None or a previous coverage tensor
            Returns:
                context vector: tensor of weighed encoder hidden states,
                    size = [batch_size x hidden_dims]
                attn_dist: context vector reweighted to account for masking
                    size = [batch_size x hidden_dims]
                decoder context: tensor of weighted decoder hidden states,
                    size = [batch_size x hidden_dims]
                coverage: as per Abi's code to prevent repetition
            '''

            temporal_attention = intra_temporal_attention(decoder_states,
                                                          encoder_states, eti)


            # Calculate encoder distribution
            # Mask padded sequences
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

            # Equation (8)
            # decoder_states_stack: T x batch_size x decoder_hidden_size
            # decoder_attention: batch_size x T - 1
            # Result has shape (batch_size, decoder_hidden_size)
            decoder_context = intra_decoder_context(decoder_states_stack)

            return context_vector, attn_dist, decoder_context, coverage

        # USING ATTENTION
        tf.logging.info("Using Intra Temporal + Decoder Attention Model")

        # The eti in equation (1), eti is a list length of decoder_steps_length, and each eti[t] is a list of length encoder_steps_length
        eti = []  # eti and ett does NOT share same weight
        outputs = []  # stores decoder hidden state outputs
        attn_dists = []
        p_gens = []  # probabilities for pointer generator model of Abi
        decoder_states = []  # hidden states from each decoder step
        temporal_attention_scores = []
        input_contexts = []  # encoder weighted hidden states by attention
        decoder_contexts = []  # decoder weighted hidden states by attention
        state = initial_state  # state to be fed into the first decoder step

        old_dists = []
        old_contexts = []

        # don't need initial_state for caculation
        # decoder_states.append(state)
        coverage = prev_coverage  # initialize to None or specific value
        context_vector = array_ops.zeros([batch_size, attn_size])
        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_size])

        if initial_state_attention:  # true in decode mode
            decoder_states_stack = tf.zeros(shape=[1, batch_size, initial_state[1].get_shape().as_list()[1]])
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _, decoder_context, coverage = hybrid_attention(
                [initial_state], coverage)
            old_context_vector, _ = attention(encoder_states, initial_state, enc_padding_mask)
            # in decode mode, this is what updates the coverage vector

        for i, inp in enumerate(decoder_inputs):
            tf.logging.info("Adding attention_decoder timestep %i of %i", i,
                            len(decoder_inputs))

            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same
            # size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError(
                    "Could not infer input size from input: %s" % inp.name)
            print("inp shape:{}".format(inp.get_shape().as_list()))
            if i==0:
                print("initial_state[1].shape {}".format(initial_state[1].get_shape()))
                intra_context_vector = tf.zeros(shape=[batch_size, initial_state[1].get_shape().as_list()[1]])
                print("inp.shape {}, context_vector.shape {}, intra_context_vector.shape {}".format(inp.get_shape().as_list(), context_vector.get_shape().as_list(), intra_context_vector.get_shape()))
                x = linear([inp] + [context_vector] + [context_vector] + [intra_context_vector], input_size, True)
            else:
                x = linear([inp] + [context_vector] + [context_vector] + [decoder_context], input_size, True)
            print("x shape:{}".format(x.get_shape().as_list()))

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Keep the decoder states
            decoder_states.append(state)
            _, decoder_states_list = map(list, zip(*decoder_states))
            decoder_states_stack = tf.stack(decoder_states_list)
            # print(decoder_states_stack.get_shape()) #(T,batch_size, decoder_hidden_size)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    # you need this because you've already run the initial attention(...) call

                    context_vector, attn_dist, decoder_context, _ = hybrid_attention(decoder_states, coverage)
                    old_context_vector, old_attn_dist = attention(encoder_states, state, enc_padding_mask)
                    # don't allow coverage to update
            else:
                context_vector, attn_dist, decoder_context, coverage = hybrid_attention(decoder_states, coverage)
                old_context_vector, old_attn_dist = attention(encoder_states, state, enc_padding_mask)

            attn_dists.append(attn_dist)
            temporal_attention_scores.append(attn_dist)
            input_contexts.append(context_vector)
            decoder_contexts.append(decoder_context)

            old_dists.append(old_attn_dist)
            old_contexts.append(old_context_vector)

            # Calculate p_gen
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear(
                        [context_vector, state.c, state.h, x], 1, True)  # a scalar
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Append hidden states
            outputs.append(cell_output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = array_ops.reshape(coverage, [batch_size, -1])

        # Common part of return
        decoder_rets = {"outputs": outputs, "state": state,
                        "attn_dists": attn_dists, "p_gens": p_gens,
                        "coverage": coverage}
        # Extra returns for Socher model
        decoder_rets["temporal_attention_scores"] = temporal_attention_scores
        decoder_rets["input_contexts"] = input_contexts if input_attention==1 else old_contexts
        decoder_rets["decoder_contexts"] = decoder_contexts

        return decoder_rets

def attention(encoder_states, decoder_state, enc_padding_mask):
        """Calculate the context vector and attention distribution from the decoder state.

        Args:
          decoder_state: state of the decoder
          coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

        Returns:
          context_vector: weighted sum of encoder_states
          attn_dist: attention distribution
          coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
        """
        print("attention.encoder_states.get_shape() {}".format(encoder_states.get_shape().as_list()))
        batch_size = encoder_states.get_shape()[
            0].value  # if this line fails, it's because the batch size isn't defined
        attn_size = encoder_states.get_shape()[
            -1].value  # if this line fails, it's because the attention length isn't defined
        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        with variable_scope.variable_scope("Attention"):
            # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
            W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1],
                                             "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

            # Get the weight vector v
            v = variable_scope.get_variable("v", [attention_vec_size])
            # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
            decoder_features = linear(decoder_state, attention_vec_size,
                                      True)  # shape (batch_size, attention_vec_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                              1)  # reshape to (batch_size, 1, 1, attention_vec_size)

            def masked_attention(e):
                """Take softmax of e then apply enc_padding_mask and re-normalize"""
                attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                attn_dist *= enc_padding_mask  # apply mask
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                    [2, 3])  # calculate e

            # Calculate attention distribution
            attn_dist = masked_attention(e)

            # Calculate the context vector from attn_dist and encoder_states
            context_vector = math_ops.reduce_sum(
                array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                [1, 2])  # shape (batch_size, attn_size).
            context_vector = array_ops.reshape(context_vector, [-1, attn_size])

        return context_vector, attn_dist

