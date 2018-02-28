# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:40:45 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

def encode(self, hidden_dims, initializer, encoder_inputs, sequence_length):
    with tf.variable_scope('encoder_scope'):
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_dims, initializer=self.initializer, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_dims, initializer=self.initializer, state_is_tuple=True)
        encoder_outputs, encoder_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=sequence_length, swap_memory=True)
        encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
    return encoder_outputs, encoder_output_states
        