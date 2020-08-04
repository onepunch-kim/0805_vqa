import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.ops import conv2d, fc


class Model(object):
    def __init__(self, q_dim, a_dim, is_train=True):
        self.is_training = tf.placeholder_with_default(is_train, [], name='is_training') # for batch normaization
        self.q_dim = q_dim # question dimension
        self.a_dim = a_dim # answer dimension

    def build(self, img, q, scope='Classifier'):
        # Normalize input images into 0 ~ 1
        img = img / 255.

        with tf.variable_scope(scope):

            ###################### MODIFY HERE ###################### 
            ## Compute film(q) for gamma, beta

            conv_1 = conv2d(img, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_1')
            ## Affine transform of conv_1
            conv_2 = conv2d(conv_1, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_2')
            ## Affine transform of conv_2
            conv_3 = conv2d(conv_2, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_3')  
            ## Affine transform of conv_3
            
            #########################################################

            features = tf.reshape(conv_3, [tf.shape(conv_3)[0], -1])
            fc_1 = fc(features, 256, activation_fn=tf.nn.relu, name='fc_1')
            fc_2 = fc(fc_1, 256, activation_fn=tf.nn.relu, name='fc_2')
            fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=self.is_training, scope='fc_3/')
            logits = fc(fc_2, self.a_dim, activation_fn=None, name='fc_3')

        return logits