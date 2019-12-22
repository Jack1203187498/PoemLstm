
# -*- coding: utf-8 -*-
# file: poems.py
# author: JinTian
# time: 08/03/2017 7:39 PM
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
import collections
import numpy as np

start_token = 'B'
end_token = 'E'

def process_file(file_name):
    # poems -> list of numbers
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        all = f.read()
        print(all)
        temp = ""
        length = 50
        for items in all:
            temp += items
            if(len(temp)) == length:
                content = start_token + temp + end_token
                poems.append(content)
                temp = ""
    # poems = sorted(poems, key=len)
    print(poems)
    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)

    # words是排序过的字
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)

    words.append(' ')
    L = len(words)
    # word_int_map是所有古诗的映射
    word_int_map = dict(zip(words, range(L)))
    poems_vector = [list(map(lambda word: word_int_map.get(word, L), poem)) for poem in poems]

    return poems_vector, word_int_map, words

def process_poems(file_name):
    # poems -> list of numbers
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                content = line.strip()
                content = content.replace(' ', '')
                # 可能是去除有特殊符号的古诗叭
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                # 如果长度太短或者太长
                if len(content) < 5 or len(content) > 79:
                    continue
                # poems有所有标记过开始结束的古诗
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # poems = sorted(poems, key=len)

    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)

    # words是排序过的字
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)

    words.append(' ')
    L = len(words)
    # word_int_map是所有古诗的映射
    word_int_map = dict(zip(words, range(L)))
    poems_vector = [list(map(lambda word: word_int_map.get(word, L), poem)) for poem in poems]

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    # 需要进行的送数据批次
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        # 取出一段数据，batches为取出这段数据
        start_index = i * batch_size
        end_index = start_index + batch_size
        batches = poems_vec[start_index:end_index]
        # 找到这个batch所有里面最长的诗句
        length = max(map(len, batches))

        # 生成batch_size * length的空向量，填充为' '所对应数字
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)

        # row是序号，代表batch_size中的第几行诗，batch是数据
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch
        # 将x的数据复制给y并向后移动一位
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # 这里生成所有测试数据
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches