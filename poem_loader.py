#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
poem_loader.py

Created on 2018/2/3.
Copyright (c) 2018 linpingta@163.com. All rights reserved.
"""

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import time
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import Counter, defaultdict
import re

from title_rhythm import TitleRhythmDict


class PoemLoader(object):
    """
    处理输入诗词数据
    output: 
    1.vocab_size, vocab, 每个字对应encoding
    2.batch input/output
    """
    def __init__(self, args):
        self.batch_input_ = []
        self.batch_output_ = []
        self.length_list = []
        self.int_word_dict_ = {}
        self.word_int_dict_ = {}

        self.data_dir_ = args.data_dir
        self.ci_words_file_ = os.path.join(self.data_dir_, 'qsc.txt')
        self.vocab_file_ = os.path.join(self.data_dir_, "ci_vocab.pkl")
        self.tensor_file_ = os.path.join(self.data_dir_, "ci_data.npy")

        self.batch_size_ = args.batch_size
        self.title_set_ = set([])
        self.chars = []
        self.tensor = []
        self.vocab_size = 0
        self.pointer = 0
        self.num_batches = 0

    @property
    def batch_input(self):
        return self.batch_input_

    @batch_input.setter
    def batch_input(self, value):
        self.bathc_input_ = value

    @property
    def batch_output(self):
        return self.batch_output_

    @batch_output.setter
    def batch_output(self, value):
        self.bathc_output_ = value

    def _load_title_set(self):
		with open(self.ci_words_file_, 'r') as fp_r:
			count = 1
			while 1:
				line = fp_r.readline()
				line = line.strip().decode("utf-8")
				if not line:
					continue
				if line == "END":
					break
#				if (u"，" not in line) and (u"。" not in line): # title part
#                    title = line.split(' ')[0]
#                    self.title_set_.add(title)

    def _title_in_line(self, line):
        for ci_pai in self.title_set_:
            if ci_pai in line:
                return ci_pai
        return None

    def _build_ci_word_dict(self):
        ci_title_content_dict = {}
#        with open(self.ci_words_file_, 'r') as fp_r:
#			count = 1
#            title = ""
#            content = ""
#			while 1:
#				line = fp_r.readline()
#				line = line.strip().decode("utf-8")
#				if not line:
#					continue
#				if line == "END":
#					break
#				if (u"，" not in line) and (u"。" not in line): # title part
#                    new_title = self._title_in_line(line):
#                    if new_title:
#                        if title and content:
#                            ci_title_content_dict.setdefault(title, []).append(content)
#                            title = ""
#                            content = ""
#                        new_title = title
#                else:
#                    # content part
#                    content += line
#                    content.replace("\n", "")
#                count += 1
#                if count > 10:
#                    break
        return ci_title_content_dict

    def _generate_batch(self, ci_title_content_dict):
        ci_words = []
#        for title, ci_contents in ci_title_content_dict.iteritems():
#            ci_words += [ ci_content for ci_content in ci_contents]
#        counter = Counter(ci_words)
#        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
#        word_counter_set, _ = zip(*count_pairs)
#        self.word_int_dict_ = dict(zip(word_counter_set, range(len(word_counter_set))))
#        self.int_word_dict_ = { v:k for k, v in word_int_dict.iteritems() }
#
#        # promise same ci title in same batch
#        ci_title_vec_dict = {}
#        for title, ci_contents in ci_title_content_dict.iteritems():
#            for ci_content in ci_contents:
#                ci_vec = []
#                [ ci_vec.append(self.word_int_dict_[word] for word in ci_content ]
#                ci_title_vec_dict.setdefault(title, []).append(ci_vec)
#
#        # combine ci in a total list
#        ci_vec_list = []
#        for title, ci_vec in ci_title_vec_dict.iteritems():
#            [ ci_vec_list.append(ci_vec_elem) for ci_vec_elem in ci_vec ]

    def _is_title(self, line, logger):
        return (u"，" not in line) and (u"。" not in line)

    def preprocess(self, logger):
        contents = []
        word_count_dict = defaultdict(int)
        with open(self.ci_words_file_, 'r') as fp:
            title = ""
            count = 0
            while 1:
                line = fp.readline()
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                if line == "END":
                    break
                if self._is_title(line, logger):
                    continue
                sentence_pattern = re.compile(u"，|。")
                sub_contents = sentence_pattern.split(line)

                for sub_content in sub_contents:
                    #print "sub_content ", sub_content
                    for word in sub_content:
                        word_count_dict[word] += 1
                        #print "word ", word
                    if not sub_content:
                        continue
                    contents.append(sub_content)
                count += 1
        #        if count > 10:
        #            break
        counter = Counter(word_count_dict)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        #print "chars ", self.chars
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        # add unk here
        self.vocab_size += 1
        self.vocab['unk'] = len(self.vocab)

        #print "vocab ", self.vocab
        with open(self.vocab_file_, 'wb') as f:
            pickle.dump(self.chars, f)

        for content in contents:
            index_content = map(self.vocab.get, content)
            self.tensor.append(index_content)
            #print "index_content ", index_content
        np.save(self.tensor_file_, self.tensor)

        # tensor size表示总共句数，batch_size表示每次处理句数
        self.num_batches = int(len(self.tensor) / self.batch_size_)

    def load_preprocessed(self, logger):
        # 加载词表
        with open(self.vocab_file_, 'r') as fp:
            self.chars = pickle.load(fp)
        self.vocab_size = len(self.chars)
        # mapping char to index
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(self.tensor_file_)

        # add unk here
        self.vocab_size += 1
        self.vocab['unk'] = len(self.vocab)

        # tensor size表示总共句数，batch_size表示每次处理句数
        self.num_batches = int(self.tensor.size / self.batch_size_)

    def process_data(self, logger):
        try:
            if not (os.path.exists(self.vocab_file_) and os.path.exists(self.tensor_file_)):
                print("reading text file")
                self.preprocess(logger)
            else:
                print("loading preprocessed files")
                self.load_preprocessed(logger)
            self.create_batches(logger)
            return 1
        except Exception as e:
            logger.exception(e)
            return -1

    def create_batches(self, logger):
        print "tensor len ", len(self.tensor)
        print "vocab size ", self.vocab_size
        for i in range(self.num_batches):
            max_range = (i + 1) * self.batch_size_ if (i + 1) * self.batch_size_ < len(self.tensor) else len(self.tensor)
            partial_tensor_data = self.tensor[i * self.batch_size_ : (i + 1) * self.batch_size_]
            #print partial_tensor_data
            partial_tensor_data = list(partial_tensor_data)
            max_length = max(map(len, partial_tensor_data))
            #print "max_length ", max_length
            x_data = np.full((self.batch_size_, max_length), self.vocab['unk'], np.int32)
            y_data = np.copy(x_data)

            for j, sentence in enumerate(partial_tensor_data):
                # reverse sentence
                reverse_sentence = sentence
                reverse_sentence.reverse()
                #print "reverse_sentence len ", len(reverse_sentence)
                x_data[j, 0 : len(reverse_sentence)] = reverse_sentence
                y_data[j, 0 : len(reverse_sentence) - 1] = x_data[j, 1 : len(reverse_sentence)]
                y_data[j, len(reverse_sentence) - 1] = x_data[j, 0]
            #print "x_data ", x_data
            #print "y_data ", y_data
            self.batch_input_.append(x_data)
            self.batch_output_.append(y_data)
            self.length_list.append(max_length)

    def next_batch(self, logger):
        x, y = self.batch_input_[self.pointer], self.batch_output_[self.pointer]
        max_length = self.length_list[self.pointer]
        self.pointer += 1
        return x, y, max_length

    def reset_batch_pointer(self, logger):
        self.pointer = 0

    def check_data_valid(self, logger):
        if len(self.batch_input_) != len(self.batch_output_):
            logger.error("poem_loader batch_input_len[%s] != batch_output_len[%s]" % (len(self.batch_input_), len(self.batch_output_)))
            return -1
        return 1
 
def main():
    import argparse
    import logging
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    data_loader = PoemLoader(args) 
    data_loader.process_data(logger)
    data_loader.check_data_valid(logger)
 

if __name__ == '__main__':
    main()
