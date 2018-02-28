# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:51:46 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

def intra_temporal_attention(input, encoder_state, decoder_state, previous_word, time_step, previous_eti, previous_eti_dash):
    with tf.variable_scoe('intra_temporal_attention_scope'):
        
        W_e_attn = tf.get_variable('W_e_attn', shape=[decoder_state.shape()[1],encoder_state.shape()[0]], intializer=tf.contrib.layers.xavier_initializer())
        
        eti = tf.matmul(tf.matmul(tf.transpose(decoder_state), W_e_attn), encoder_state)
        
        if(time_step==1):
            eti_dash = tf.exp(eti)
        else:
            eti_dash = tf.exp(eti)/tf.reduce_sum(tf.exp(previous_eti))
        
        previous_eti.append(eti)    
        
        attention_score_eti = tf.sigmoid(eti_dash)
        
        #context vector
        c_e_t = tf.reduce_sum(tf.matmul(attention_score_eti, encoder_state))
        
    