import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from input_ops import create_input_ops
from sort_of_clevr.dataset import SortOfCLEVR
from sort_of_clevr import NUM_COLOR
from sort_of_clevr import Q_DIM, NUM_ANS
from utils import log


class EvalManager(object):
    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, prediction, groundtruth):
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def report(self):
        log.info("Computing scores...")
        correct_prediction_nr = 0
        count_nr = 0
        correct_prediction_r = 0
        count_r = 0

        for pred, gt in zip(self._predictions, self._groundtruths):
            for i in range(pred.shape[0]):
                # relational
                if np.argmax(gt[i, :]) < NUM_COLOR:
                    count_r += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_r += 1
                # non-relational
                else:
                    count_nr += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_nr += 1

        avg_nr = float(correct_prediction_nr) / count_nr
        log.infov("Average accuracy of non-relational questions: {}%".format(avg_nr * 100))
        avg_r = float(correct_prediction_r) / count_r
        log.infov("Average accuracy of relational questions: {}%".format(avg_r * 100))
        avg = float(correct_prediction_r + correct_prediction_nr) / (count_r + count_nr)
        log.infov("Average accuracy: {}%".format(avg * 100))


class Evaler(object):
    def __init__(self, cfg, dataset):
        self.config = cfg
        self.train_dir = cfg.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = cfg.batch_size
        self.dataset = dataset
        self.batch = create_input_ops(dataset, self.batch_size, shuffle=False)

        # --- create model ---
        if cfg.model == 'baseline':
            from models.baseline import Model
        elif cfg.model == 'rn':
            from models.rn import Model
        elif cfg.model == 'film':
            from models.film import Model
        else:
            raise ValueError(cfg.model)
        log.infov("Using Model class : %s", Model)
        self.model = Model(Q_DIM, NUM_ANS, is_train=False)

        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, cfg.image_size, cfg.image_size, 3],
        )
        self.q = tf.placeholder(name='q', dtype=tf.float32, shape=[cfg.batch_size, Q_DIM])
        self.a = tf.placeholder(name='a', dtype=tf.float32, shape=[cfg.batch_size, NUM_ANS])

        logits = self.model.build(self.img, self.q)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.a, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.all_preds = tf.nn.softmax(logits)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver()

        self.checkpoint_path = cfg.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = int(length_dataset / self.batch_size) + 1
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager()
        try:
            for s in range(max_steps):
                step, loss, step_time, prediction_pred, prediction_gt = \
                    self.run_single_step(self.batch)
                self.log_step_message(s, loss, step_time)
                evaler.add_batch(prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        evaler.report()
        log.infov("Evaluation complete.")

    def run_single_step(self, batch):
        _start_time = time.time()
        img, q, a = self.session.run(batch)
        [step, accuracy, all_preds, all_targets, _] = self.session.run(
            [self.global_step, self.accuracy, self.all_preds, self.a, self.step_op],
            feed_dict={
                self.img: img,  # [B, h, w, c]
                self.q: q,  # [B, n]
                self.a: a,  # [B, m]
                self.model.is_training: False
            }
        )
        _end_time = time.time()
        return step, accuracy, (_end_time - _start_time), all_preds, all_targets

    def log_step_message(self, step, accuracy, step_time):
        step_time = max(step_time, 0.001)
        instance_per_sec = self.batch_size / step_time
        log.infov((f" [test {step:4d}] "
                   f"Acc.: {accuracy * 100:.2f}% "
                   f"({step_time:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'rn', 'film'])
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/SortOfCLEVR_4_200000_32')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--image_size', type=int, default=32)

    config = parser.parse_args()
    test_dataset = SortOfCLEVR(config.dataset_path, split='test')
    evaler = Evaler(config, test_dataset)

    log.warning("dataset: %s", config.dataset_path)
    evaler.eval_run()


if __name__ == '__main__':
    main()
