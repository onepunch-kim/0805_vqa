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
            conv_3 = conv2d(conv_2, 24, self.is_training, activation_fn=tf.nn.relu, name='conv_3') # (b,d,d,c)

            # Adding positional information (x,y) into features: size (b,d,d,c+2)
            conv_pos = _positional_encoding(conv_3) 

            ###################### MODIFY HERE ###################### 
            
            conv_q = tf.concat([tf.reshape(conv_pos, [tf.shape(conv_pos)[0], -1]), q], axis=1) 
            
            fc_1 = fc(conv_q, 256, activation_fn=tf.nn.relu, name='fc_1') 
            all_g = fc_1

            #########################################################

            logits = f_phi(all_g)

        return logits
