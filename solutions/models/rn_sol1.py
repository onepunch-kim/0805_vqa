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

        def _positional_encoding(features):
            # Append two features of positional encoding to the given feature maps
            d = features.get_shape().as_list()[1]
            indices = tf.range(d)
            x = tf.tile(tf.reshape(indices, [d, 1]), [1, d])
            y = tf.tile(tf.reshape(indices, [1, d]), [d, 1])
            pos = tf.cast(tf.stack([x, y], axis=2)[None] / d, tf.float32)
            pos = tf.tile(pos, [tf.shape(img)[0], 1, 1, 1])
            return tf.concat([features, pos], axis=3)

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope):
                fc_1 = fc(g, 256, activation_fn=tf.nn.relu, name='fc_1')
                fc_1 = slim.dropout(fc_1, keep_prob=0.5, is_training=self.is_training, scope='fc_3/')
                logits = fc(fc_1, self.a_dim, activation_fn=None, name='fc_3')
                return logits

        with tf.variable_scope(scope):
            conv_1 = conv2d(img, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_1')
            conv_2 = conv2d(conv_1, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_2')
            conv_3 = conv2d(conv_2, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_3')

            conv_pos = _positional_encoding(conv_3)

            ###################### MODIFY HERE ######################

            def g_theta(o_pair, q, d, scope='g_theta'):
                with tf.variable_scope(scope):
                    q = q[:, None, None, :]
                    q = tf.tile(q, [1, d * d, d * d, 1])
                    o = tf.concat([o_pair, q], axis=3)
                    g_1 = conv2d(o, 256, self.is_training,
                                 k_h=1, k_w=1, s_h=1, s_w=1,
                                 activation_fn=tf.nn.relu, name='g_1')
                    g_2 = conv2d(g_1, 256, self.is_training,
                                 k_h=1, k_w=1, s_h=1, s_w=1,
                                 activation_fn=tf.nn.relu, name='g_2')
                    return g_2

            d = conv_3.get_shape().as_list()[1]
            o_1 = tf.tile(tf.reshape(conv_pos, [-1, d*d, 1, 26]), [1, 1, d*d, 26])
            o_2 = tf.tile(tf.reshape(conv_pos, [-1, 1, d*d, 26]), [1, d*d, 1, 26])
            o_pair = tf.concat([o_1, o_2], axis=3)
            g = g_theta(o_pair, q, d)

            all_g = tf.reduce_sum(g, [1, 2])

            #########################################################

            logits = f_phi(all_g)

        return logits