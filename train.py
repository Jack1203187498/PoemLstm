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
import os
import tensorflow as tf
from model import rnn_model
from poems import process_poems, generate_batch, process_file
import random
import time
start_time = time.time()
name = 'AllTang'
tf.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.flags.DEFINE_string('model_dir', os.path.abspath('./model/' + name + '/'), 'model save path.')
tf.flags.DEFINE_string('file_path', os.path.abspath('./data/' + name + '.txt'), 'file name of poems.')
tf.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.flags.FLAGS



def run_training():
    # 模型保存路径不存在则创建
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # process_poems对古诗进行预处理
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    # batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    # 占位向量
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    # 使用lstm模型进行训练
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        # 如果之前训练过就找回之前的训练结果
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        try:
            n_chunk = len(poems_vector) // FLAGS.batch_size
            for epoch in range(start_epoch, FLAGS.epochs):
                #每次对其中的数据shuffle一次
                # print(type(poems_vector))
                random.shuffle(poems_vector)
                # print(type(poems_vector))
                # 这里有每一次输入的训练数据的列表
                batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
                n = 0
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d/%d, batch: %d/%d, training loss: %.6f' % (epoch, FLAGS.epochs - 1, batch, n_chunk - 1, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
                alltime = time.time() - start_time
                print("Time: %d h %d min % ds" % (alltime // 3600, (alltime - alltime // 3600 * 3600) // 60, alltime % 60))
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()