#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
poem_generator.py

Created on 2018/2/3.
Copyright (c) 2018 linpingta@163.com. All rights reserved.
"""

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import time
import logging
import argparse
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class PoemGenerator(object):
    def __init__(self, args, vocab_size):
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.batch_input = tf.placeholder(tf.int32,  [args.batch_size, None])
        self.batch_output = tf.placeholder(tf.int32,  [args.batch_size, None])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

        embedding = tf.get_variable("embedding", [vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.batch_input)
    
        outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        labels = tf.one_hot(tf.reshape(self.batch_output, [-1]), depth=vocab_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)

        #target_reshape = tf.reshape(self.batch_output, [-1])
        #loss = legacy_seq2seq.sequence_loss_by_example(
        #        [self.logits],
        #        [target_reshape],
        #        [tf.ones([tf.to_int32(self.max_length)])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(loss)

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def inference(self, sess, word_list, vocab, input_word='', word_len=7, sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))

        if not input_word:
            return ""
        if input_word not in vocab:
            return input_word
        x = np.zeros((1, 1))
        x[0, 0] = vocab[input_word]
        feed = {self.batch_input: x, self.initial_state: state}
        [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = input_word
        for n in range(word_len - 1):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[input_word]
            feed = {self.batch_input: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if input_word == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = word_list[sample]
            ret += pred
            input_word = pred
        return ret
