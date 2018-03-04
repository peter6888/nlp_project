# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications Copyright 2018 Stylianos Serghiou
#
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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# TODO: Make this code similar to homework assignments


class SummarizationModel(object):
    """A class to represent a sequence-to-sequence model for text summarization.
    Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph, i.e. entry points for input data."""
        hps = self._hps

        # TODO: Add dropout

        # Encoder
        self._enc_batch = tf.placeholder(
            tf.int32, shape=[hps.batch_size, None], name='enc_batch')
        # These are the max text lengths for each batch
        # We need these for dynamic_rnn unrolling
        self._enc_lens = tf.placeholder(
            tf.int32, shape=[hps.batch_size], name='enc_lens')
        # These are marks for the padded timesteps
        # We need these to avoid bias in total loss
        self._enc_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, None], name='enc_padding_mask')

        # if FLAGS.pointer_gen:
        #     self._enc_batch_extend_vocab = tf.placeholder(
        #         tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
        #     self._max_art_oovs = tf.placeholder(
        #         tf.int32, [], name='max_art_oovs')

        # Decoder
        self._dec_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

        # if hps.mode == "decode" and hps.coverage:
        #     self.prev_coverage = tf.placeholder(
        #         tf.float32, [hps.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate
        placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for encoder.
        """

        # SS: defined in batcher.py
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        # This is the length of sentences in each batch
        # Each batch can have a different length b/c we are using dynamic_rnn
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

        # if FLAGS.pointer_gen:
        #    feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
        #    feed_dict[self._max_art_oovs] = batch.max_art_oovs
        # if not just_enc:
        feed_dict[self._dec_batch] = batch.dec_batch
        feed_dict[self._target_batch] = batch.target_batch
        feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps,
          emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of
          shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim     because it's the concatenation of the forwards and
            backwards states.
          fw_state, bw_state:
                Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[    batch_size,hidden_dim])
        """

        # SS: rand_unif_init is defined within model.py
        with tf.variable_scope('encoder'):
            # Define the forward and backward cells
            cell_fw = tf.contrib.rnn.LSTMCell(
                self._hps.hidden_dim, initializer=self.rand_unif_init)
            cell_bw = tf.contrib.rnn.LSTMCell(
                self._hps.hidden_dim, initializer=self.rand_unif_init)

            # Run the forward and backward cells
            # encoder_inputs are the embeddings for the words in the batch
            (outputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw,
                encoder_inputs, dtype=tf.float32, sequence_length=seq_len,
                swap_memory=True
            )

            # concatenate the forwards and backwards outputs
            encoder_outputs = tf.concat(axis=2, values=outputs)

        return encoder_outputs, fw_state, bw_state

    def _reduce_states(self, fw_state, bw_state):
        """Add to the graph a linear layer to reduce the encoder's final FW and
        BW state into a single initial state for the decoder. This is needed
        because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        # TODO: Change initialization to Xavier
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell output and state
            w_reduce_c = tf.get_variable('w_reduce_c',
                                         shape=[hidden_dim * 2, hidden_dim],
                                         dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h',
                                         shape=[hidden_dim * 2, hidden_dim],
                                         dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c',
                                            shape=[hidden_dim],
                                            dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h',
                                            shape=[hidden_dim],
                                            dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            # Concatenation of fw and bw cell output
            old_c = tf.concat(axis=1, values=[fw_state.c, bw_state.c])
            z_c = tf.nn.xw_plus_b(old_c, w_reduce_c, bias_reduce_c)
            new_c = tf.nn.relu(z_c)

            # Concatenation of fw and bw state
            old_h = tf.concat(axis=1, values=[fw_state.h, bw_state.h])
            z_h = tf.nn.xw_plus_b(old_h, w_reduce_h, bias_reduce_h)
            new_h = tf.nn.relu(z_h)

        # Return new cell and state
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, decoder_inputs, padding_mask):
        """Add attention decoder to the graph. In train or eval mode, you call
        this once to get output on ALL steps. In decode (beam search) mode, you
        call this once for EACH decoder step.

        Args:
          inputs: inputs to the decoder (word embeddings). A list of tensors shape (    batch_size, emb_dim)

        Returns:
          outputs: List of tensors; the outputs of the decoder
          out_state: The final state of the decoder
          attn_dists: A list of tensors; the attention distributions
          p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
          coverage: A tensor, the current coverage vector
        """
        hps = self._hps

        # Build RNN cell
        decoder_cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim,
                                               initializer=self.rand_unif_init)

        # Helper (identify ids of argmax from RNN output logits)
        # self._dec_padding_mask = padding_mask
        decoder_lengths = tf.reduce_sum(padding_mask, axis=1)
        # decoder_inputs = emb_dec_inputs
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_inputs, decoder_lengths, time_major=True)

        vsize = self._vocab.size()
        projection_layer = layers_core.Dense(vsize, use_bias=False)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state, output_layer=projection_layer)

        return decoder

    # decoder_outputs = self._target_batch
    # target_weights = self._dec_padding_mask
    def _add_seq2seq(self, decoder, decoder_outputs, target_weights):
        """ Dynamic decoding step.
        """
        batch_size = self._hps.batch_size

        # Dynamic decoding
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, output_time_major=True)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)

        loss = (tf.reduce_sum(crossent * target_weights) / batch_size)

        return loss

    def _add_train_op(self):
        max_gradient_norm = self._hps.max_grad_norm
        learning_rate = self._hps.lr

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      max_gradient_norm)
        # Optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

        return train_op
