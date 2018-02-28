# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:29:50 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

def intra_decoder_attention(input, decoder_state, previous_word, time_step, previous_edtj, previous_decoder_state):
    with tf.variable_scoe('intra_temporal_attention_scope'):
        
        W_d_attn = tf.get_variable('W_d_attn', shape=[decoder_state.shape()[1],previous_decoder_state.shape()[0]], intializer=tf.contrib.layers.xavier_initializer())
        
        e_d_tt_dash = tf.matmul(tf.matmul(tf.transpose(decoder_state), W_d_attn), previous_decoder_state)
        
        #eti_dash = tf.exp(eti)/tf.reduce_sum(tf.exp(previous_edtj))
        
        previous_edtj.append(e_d_tt_dash)    
        
        attention_score_e_d_tt_dash = tf.exp(e_d_tt_dash)/tf.reduce_sum(tf.exp(previous_edtj))
        
        #context vector
        c_d_t = tf.reduce_sum(tf.matmul(attention_score_e_d_tt_dash, previous_decoder_state))