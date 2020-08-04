import os
import time
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from sort_of_clevr.dataset import SortOfCLEVR
from sort_of_clevr import Q_DIM, NUM_ANS
from input_ops import create_input_ops
from utils import log


class Trainer(object):
    def __init__(self, cfg, train_dataset, val_dataset):
        self.config = cfg
        dataset_base = os.path.basename(os.path.normpath(cfg.dataset_path))
        hyper_parameter_str = dataset_base + '_lr_' + str(cfg.learning_rate)
        self.train_dir = './train_dir/%s-%s/%s/%s' % (
            cfg.model,
            cfg.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = cfg.batch_size
        self.train_batch = create_input_ops(train_dataset, self.batch_size)
        self.val_batch = create_input_ops(val_dataset, self.batch_size)

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
        self.model = Model(Q_DIM, NUM_ANS)

        # define placeholders: (image, question, answer)
        self.img = tf.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, cfg.image_size, cfg.image_size, 3],
        )
        self.q = tf.placeholder(name='q', dtype=tf.float32, shape=[cfg.batch_size, Q_DIM])
        self.a = tf.placeholder(name='a', dtype=tf.float32, shape=[cfg.batch_size, NUM_ANS])

        # compute logits and cross-entropy loss
        logits = self.model.build(self.img, self.q)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.a)
        loss = tf.reduce_mean(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.a, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = loss, accuracy

        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("loss", self.loss)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate = cfg.learning_rate
        if cfg.lr_weight_decay:
            # learning rate scheduling (optional)
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.95,
                name='decay_lr'
            )
        self.check_op = tf.no_op()
        # Adam optimizer
        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='optimizer_loss'
        )

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(self.train_dir + '/train')
        self.val_writer = tf.summary.FileWriter(self.train_dir + '/val')

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir, is_chief=True,
            saver=None, summary_op=None,
            summary_writer=self.train_writer,
            save_summaries_secs=100,
            global_step=self.global_step,
        )
        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = cfg.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")

        max_steps = 50000 # total iterations
        output_save_step = 5000
        for s in range(max_steps):
            step, accuracy, summary, loss, data_time, foward_time = self.run_single_step() # each training step

            # periodic inference
            if s % 100 == 0:
                self.train_writer.add_summary(summary, global_step=step)
                accuracy_val, summary_val = self.run_test() # evaluation
                self.val_writer.add_summary(summary_val, global_step=step)
                self.log_step_message(step, accuracy, accuracy_val, loss, data_time, foward_time)

            if s % output_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)

    def run_single_step(self):
        _start = time.time()
        img, q, a = self.session.run(self.train_batch)
        _data = time.time() - _start
        _start = time.time()

        fetch = [self.global_step, self.accuracy, self.summary_op,
                 self.loss, self.check_op, self.optimizer]

        fetch_values = self.session.run(
            fetch, feed_dict={
                self.img: img,  # [B, h, w, c]
                self.q: q,  # [B, n]
                self.a: a  # [B, m]
            })
        [step, accuracy, summary, loss] = fetch_values[:4]

        _forward = time.time() - _start
        return step, accuracy, summary, loss, _data, _forward

    def run_test(self):
        img, q, a = self.session.run(self.val_batch)
        accuracy_val, summary_val = self.session.run(
            [self.accuracy, self.summary_op], feed_dict={
                self.img: img,  # [B, h, w, c]
                self.q: q,  # [B, n]
                self.a: a,  # [B, m]
                self.model.is_training: False
            }
        )
        return accuracy_val, summary_val

    def log_step_message(self, step, accuracy, accuracy_test, loss, data_time, foward_time, is_train=True):
        mode = (is_train and 'train' or 'val')
        step_time = max(data_time + foward_time, 0.001)
        instance_per_sec = self.batch_size / step_time
        log_fn = (is_train and log.info or log.infov)
        log_fn((f" [{mode:5s} {step:4d}] "
                f"Loss: {loss:.5f} "
                f"Acc_TR: {accuracy * 100:.2f} "
                f"Acc_TE: {accuracy_test * 100:.2f} "
                f"({data_time:.3f} + {foward_time:.3f} sec/batch, {instance_per_sec:.3f} /sec)"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'rn', 'film']) # model architecture

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False) # if True, learning rate is decayed with exponential scheduling

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/SortOfCLEVR_4_200000_32')
    parser.add_argument('--image_size', type=int, default=32)

    config = parser.parse_args()
    train_dataset = SortOfCLEVR(config.dataset_path, split='train')
    val_dataset = SortOfCLEVR(config.dataset_path, split='val')
    trainer = Trainer(config, train_dataset, val_dataset)

    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()

if __name__ == '__main__':
    main()
