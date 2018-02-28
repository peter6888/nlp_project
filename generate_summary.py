# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:33:25 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('deep_rl_abstractive_summarization', False, 'If True, deep rl model for abstractive summarization. If False, use baseline model.')