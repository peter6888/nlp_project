# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:41:30 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

class abstractiveSummarizationModel():
    def __init__(self):
        #TODO
        
    def add_placeholders(self):
        #TODO
        self.previous_eti = tf.placeholder(tf.float32, [batch_size, None], name='previous_attention_score')
        
        #self.previous_decoder_state = tf.placeholder(tf.float32, [batch_size, None], name='previous_attention_score')
        
        
    def create_feed_dict() 
        
        