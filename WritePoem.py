# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 11/03/2017 9:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import tensorflow as tf
from model import rnn_model
from poems import process_poems,process_file
import numpy as np
import random
import time
name = 'AllTang'
start_token = 'B'
end_token = 'E'
model_dir = './model/' + name + '/'
corpus_file = './data/' + name + '.txt'

lr = 0.0002


def to_word(predict, vocabs, poem_):
    predict = predict[0]
    predict /= np.sum(predict)
    # print(vocabs[-1])
    sample1,sample2,sample3 = np.random.choice(np.arange(len(predict)), size=3, replace=False,p=predict)
    sample = sample1
    if(vocabs[sample] == ' ' or vocabs[sample] == '\n' or vocabs[sample] == 'B'):
        sample = sample2
    # if(poem_[-1] == '，' or poem_[-1] == '。') and (vocabs[sample] == '。' or vocabs[sample] == '。'):
    #     sample = sample2
    # print(predict[sample], end=' ')
    if sample > len(vocabs):
        # 是空格
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_poem(begin_word):
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        # print(checkpoint)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        poem_ = ''
        word = begin_word or to_word(predict, vocabularies, poem=poem_)


        i = 0
        # while not poem_ or ' ' in poem_:
        if 1:
            print("开始作诗")
            poem_ = ''
            while word != end_token and word != start_token:
            # while 1:
                # if word == ' ':
                #     poem_ =  gen_poem(begin_word)
                #     break
                poem_ += word
                print(poem_)
                i += 1
                # if i > 24:
                #     break
                x = np.array([[word_int_map[word]]])
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = ' '
                # while word == ' ':
                if 1:
                    time.sleep(1)
                    random.seed(time.time())
                    word = to_word(predict, vocabularies, poem_)
        print('\n')
        return poem_
def gen_poem_cangtou(begin_word_all):

    begin_word = begin_word_all[0]
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=10, learning_rate=lr)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        # print(checkpoint)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        poem_ = ''
        # word = begin_word or to_word(predict, vocabularies, poem=poem_)


        i = 0
        poem_all = ''
        # while not poem_ or ' ' in poem_:
        for one in begin_word_all:
            print("开始作诗")
            poem_ = ''
            word = one
            while word != end_token and word != start_token:
                # if word == ' ':
                #     poem_ =  gen_poem(begin_word)
                #     break
                poem_ += word
                print(poem_)
                i += 1
                if i > 7:
                    break
                x = np.array([[word_int_map[one]]])
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = ' '
                # while word == ' ':
                if 1:
                    time.sleep(1)
                    random.seed(time.time())
                    word = to_word(predict, vocabularies, poem_)
            poem_all += poem_ + '\n'
        print('\n')
        return poem_all

def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 3:
            print(s + '。')

if __name__ == '__main__':
    begin_char = input('## please input the first character:')
    poem = gen_poem(begin_char)
    print(poem + '\n' + '\n')
    pretty_print_poem(poem_=poem)