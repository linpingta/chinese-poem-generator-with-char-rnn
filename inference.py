#-*-coding: utf-8 -*-
import tensorflow as tf
import random

import argparse
import os
import re
try:
    import cPickle as pickle
except ImportError:
    import pickle

from poem_generator import PoemGenerator
from title_rhythm import TitleRhythmDict

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--word_len', type=int, default=5,
                        help='number of characters to sample')
    parser.add_argument('--generate_type', type=str, default='one',
                        help='generate one line or one ci')
    parser.add_argument('--title', type=str, default='',
                        help='ci pai')
    parser.add_argument('--input_word', type=text_type, default=u' ',
                        help='rhythm used in ci')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    inference(args)

def inference_one(model, sess, chars, vocab, input_word, word_len, sample):
    #reverse_words = model.inference(sess, chars, vocab, input_word, word_len, sample).encode('utf-8')
    reverse_words = model.inference(sess, chars, vocab, input_word, word_len, sample)
    return reverse_words

def build_pingze_rhythm_words_dict():
    pingze_rhythm_dict = []
    with open('data/psy.txt', 'r') as fp_r:
        while True:
            line = fp_r.readline()
            line = line.strip().decode("utf-8")
            if not line:
				continue
            if line == "END":
				break
            if u"：" in line: # Chinese title part
                rhythm_word = line[-2]
                next_line = fp_r.readline().strip().decode("utf-8")
                rhythm_word_list = []
                [ rhythm_word_list.append(word) for word in next_line ]
                pingze_rhythm_dict.append(rhythm_word_list)
    return pingze_rhythm_dict

def load_ci_pai_info():
    ci_pai_info_dict = {}
    sentence_pattern = re.compile(u",|.")
    for title, title_info in TitleRhythmDict.iteritems():
        end_tag_index = 0
        rhythm_tags = []
        for word in title_info:
            if word == ',':
                end_tag_index += 1
            elif word == '.':
                rhythm_tags.append(end_tag_index)
                end_tag_index += 1
        sentences = re.findall(r"[0-9]+", title_info)
        word_lens = []
        for i, sentence in enumerate(sentences):
            use_rhythm = False if i not in rhythm_tags else True
            word_lens.append((len(sentence), use_rhythm))
        
        ci_pai_info_dict[title] = word_lens
    return ci_pai_info_dict

def inference(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)

    # change args for inference
    saved_args.batch_size = 1
    vocab_size = len(vocab)
    model = PoemGenerator(saved_args, vocab_size)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            if args.generate_type == 'one': # generate one word
                ret_word = inference_one(model, sess, chars, vocab, args.input_word, args.word_len, args.sample)
                print "ret_words ", ret_word
                words = ret_word[::-1]
                print "words ", words
            else:
                ci_pai_info_dict = load_ci_pai_info()
                pingze_word_list = build_pingze_rhythm_words_dict()

                if args.title not in ci_pai_info_dict:
                    print "title[%s] not in ci_pai_list " % args.title
                    return

                random_idx = random.randint(0, len(pingze_word_list) - 1)
                rhythm_words = pingze_word_list[random_idx]
                #rhythm_words = [u'明', u'行']
                #print "rhythm_words ", rhythm_words
                already_used_rhythm_set = set([])
                ci_words_list = []
                for i, (word_len, use_rhythm) in enumerate(ci_pai_info_dict[args.title]):
                    if use_rhythm:
                        random_idx = random.randint(0, len(rhythm_words) - 1)
                        random_count = 0
                        while random_idx in already_used_rhythm_set:
                            random_idx = random.randint(0, len(rhythm_words) - 1)
                            random_count += 1
                            if random_count > 3:
                                random_idx = 0
                                break
                        already_used_rhythm_set.add(random_idx)
                        rhythm_word = rhythm_words[random_idx] 
                        #print "my rhythm test ", rhythm_word
                        #print "my rhythm test ", type(rhythm_word)
                    else:
                        random_idx = random.randint(0, 2000 - 1 - 1)
                        #random_idx = random.randint(0, len(chars) - 1 - 1)
                        rhythm_word = chars[random_idx]
                        #print "my test ", rhythm_word
                        #print "my test ", type(rhythm_word)
                    print "generate ", i, use_rhythm, rhythm_word, word_len
                    ret_word = inference_one(model, sess, chars, vocab, rhythm_word, word_len, args.sample)
                    print "word ", ret_word[::-1]
                    ci_words_list.append(ret_word[::-1])
                print "final result ", '，'.join(ci_words_list)


if __name__ == '__main__':
    main()
