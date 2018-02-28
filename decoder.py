# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:41:06 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

def decode(hidden_dims, initializer, previous_word, previous_attention_score):
        cell = tf.contrib.rnn.LSTMCell(hidden_dims, state_is_tuple=True, initializer=initializer)
        #prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
        outputs, out_state, attn_dists, p_gens, coverage = intra_temporal_attention(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, previous_word=previous_word)
        return outputs, out_state, attn_dists, p_gens, coverage
    
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
        

def intra_decoder_attention(input, decoder_state, previous_word, time_step, previous_edtj, previous_decoder_state):
    with tf.variable_scoe('intra_decoder_attention_scope'):
        
        W_d_attn = tf.get_variable('W_d_attn', shape=[decoder_state.shape()[1],previous_decoder_state.shape()[0]], intializer=tf.contrib.layers.xavier_initializer())
        
        e_d_tt_dash = tf.matmul(tf.matmul(tf.transpose(decoder_state), W_d_attn), previous_decoder_state)
        
        #eti_dash = tf.exp(eti)/tf.reduce_sum(tf.exp(previous_edtj))
        
        previous_edtj.append(e_d_tt_dash)    
        
        attention_score_e_d_tt_dash = tf.exp(e_d_tt_dash)/tf.reduce_sum(tf.exp(previous_edtj))
        
        #context vector
        c_d_t = tf.reduce_sum(tf.matmul(attention_score_e_d_tt_dash, previous_decoder_state))
        
def token_generator_and_pointer(user_pointer=False):
    with tf.variable_scoe('token_generator_and_pointer_scope'):
        
        W_out = tf.get_variable('W_d_attn', shape=[decoder_state.shape()[1],previous_decoder_state.shape()[0]], intializer=tf.contrib.layers.xavier_initializer())
        b_out = tf.get_variable('b_out',shape=[], initializer=tf.zeros_initializer(tf.float32))
        
        W_u = tf.get_variable('W_u', shape=[decoder_state.shape()[1],previous_decoder_state.shape()[0]], intializer=tf.contrib.layers.xavier_initializer())
        b_u = tf.get_variable('b_u',shape=[], initializer=tf.zeros_initializer(tf.float32))
        
        concat_decoder_state_context_vector = tf.contact([decoder_state, c_e_t, c_d_t], axis=1)
        
        ut = tf.sigmoid(tf.matmul(W_u, concat_decoder_state_context_vector) + bu)
        
        p_yt = tf.multiply(ut, attention_score_eti) + tf.multiply(ut, tf.nn.softmax(tf.matmul(W_out, concat_decoder_state_context_vector) + b_out))