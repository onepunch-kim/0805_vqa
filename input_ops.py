import tensorflow as tf


def create_input_ops(data, batch_size, scope='inputs', shuffle=True):

    class generator(object):
        def __init__(self, data):
            self.img, self.q, self.a = data.query_dataset()

        def __call__(self):
            for triple in zip(self.img, self.q, self.a):
                yield triple

    with tf.name_scope(scope):
        dataset = tf.data.Dataset.from_generator(generator(data),
                                                 (tf.float32, tf.float32, tf.float32))
        dataset = dataset.repeat(None)
        if shuffle:
            capacity = 8 * batch_size
            dataset = dataset.shuffle(capacity)
        dataset = dataset.batch(batch_size)

    img, q, a = dataset.make_one_shot_iterator().get_next()
    return (img, q, a)
