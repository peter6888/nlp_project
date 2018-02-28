# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:37:35 2018

@author: Apurva Pancholi
"""

import tensorflow as tf

matA = tf.constant([[7, 8, 9, 10],[1,2,3,4]])

print (tf.shape(matA))

print (matA.get_shape()[2])