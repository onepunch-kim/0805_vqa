import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.ops import conv2d, fc


class Model(object):
    def __init__(self, q_dim, a_dim, is_train=True):
        self.is_training = tf.placeholder_with_default(is_train, [], name='is_training')
        self.q_dim = q_dim
        self.a_dim = a_dim

    def build(self, img, q, scope='Classifier'):
        # Normalize input images into 0 ~ 1
        img = img / 255.

        with tf.variable_scope(scope):

            ###################### MODIFY HERE ######################

            def film(q, scope='film'):
                with tf.variable_scope(scope):
                    cond = fc(q, 3 * 2 * 24, activation_fn=None, name='cond')
                    cond = tf.reshape(cond, [-1, 3, 2, 24])
                    return cond

            def modulate(conv, gamma, beta):
                gamma = tf.reshape(gamma, [-1, 1, 1, 24])
                beta = tf.reshape(beta, [-1, 1, 1, 24])
                return (1 + gamma) * conv + beta

            q_embed = fc(q, 256, name='fc_q')
            cond = film(q_embed)

            conv_1 = conv2d(img, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_1')
            conv_1 = modulate(conv_1, cond[:, 0, 0, :], cond[:, 0, 1, :])
            conv_2 = conv2d(conv_1, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_2')
            conv_2 = modulate(conv_2, cond[:, 1, 0, :], cond[:, 1, 1, :])
            conv_3 = conv2d(conv_2, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_3')
            conv_3 = modulate(conv_3, cond[:, 2, 0, :], cond[:, 2, 1, :])

            #########################################################

            features = tf.reshape(conv_3, [tf.shape(conv_3)[0], -1])
            fc_1 = fc(features, 256, activation_fn=tf.nn.relu, name='fc_1')
            fc_2 = fc(fc_1, 256, activation_fn=tf.nn.relu, name='fc_2')
            fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=self.is_training, scope='fc_3/')
            logits = fc(fc_2, self.a_dim, activation_fn=None, name='fc_3')

        return logits