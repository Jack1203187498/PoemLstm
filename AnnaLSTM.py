# coding: utf-8
#
# 主要参考自https://github.com/NELSONZHAO/zhihu/blob/master/anna_lstm/anna_lstm.py
# Debug代码参考包括https://blog.csdn.net/weixin_40759186/article/details/82893626等
# 数据集中:
#     Anna.txt来自https://github.com/NELSONZHAO/zhihu/blob/master/anna_lstm/anna_lstm.py
#     USA.txt来自同学
# 将原来的功能函数封装在class里面
# 增加了模型和测试分离的功能
# 测试环境为python3.5, TensorFlow 1.10.0
#
# 实验中模型储存在model文件夹下相应名称文件夹中
# 实验中训练数据储存在data文件夹中,为txt格式,utf-8编码
# 实验中的具体实现可先参考AnneLSTM中的注释,其中大部分为原作者注释
# 已经支持断点继续训练,但是不会删除上一次训练最后保存的模型,只要手动删掉就好
# 支持中文路径(应该以前也支持就是我没用),支持自己生成不存在路径
# 实验所使用数据文件要求编码为utf-8,目前尚不支持其他文件编码形式
#
# 有待增加或改进的内容：
# 输出的规范化，以及更长的输出预测
# 对中文的支持效果还没有给出确定结论,目前可以确定支持中文
# 增加断点检测功能,便于训练中断时,下次继续训练(已经解决)
# 对于训练误差的统计以及可视化数据
# 可能能有更多的拓展性
# 实验中一些参数调整,包括但不限于:
#   learning rate的函数下降式调整调整
#   更多的epoch
#   更少的训练时间
#   不知所云的对抗网络
#   生成中文时选择的随机参数可能需要调整,英文中5效果良好
#
# 实验进程:
# 训练73219词的USA,时间1h28min47s,epoch = 20, 训练步数 = 940 效果一般,可以生成,有些单词为自造
# 示例中示范Anna.txt,词数353909, epoch = 20, 训练步数 = 3920, 最终效果很好,目测如果跑更多时间的话有优化空间
# 实验结果支持中文字符,目前没有进行大规模数据测试,还不能给出确定结论,目测实验数据要远大于1万字
#
# 尝试斗罗大陆.txt,一共2854525个字,100*100模式,epoch = 20
import time
import numpy as np
import tensorflow as tf
import os
class LstmAnna():
    def __init__(self, name='深渊', prime="我", numWords=1000):
        self.time0 = time.time()
        self.name = name
        # 对文件的处理
        self.process_file()
        # 关于LSTM的参数

        self.batch_size = 100  # Sequences per batch    一个batch中包含的序列数量
        self.num_steps = 100  # Number of sequence steps per batch  一个序列中包含的数据数量
        self.lstm_size = 512  # Size of hidden layers in LSTMs
        self.num_layers = 2  # Number of LSTM layers
        self.learning_rate = 0.001  # Learning rate
        self.keep_prob = 0.3  # Dropout keep probability
        self.epochs = 20
        self.save_every_n = 1
        # 关于文章的参数
        self.prime = prime
        self.wordnum =numWords
        self.top_n = 30
        self.checkpoints = 'model/' + name + '/'
    def process_file(self):
        with open('data/' + self.name + '.txt', 'r',encoding='utf-8') as f:
            text = f.read()
        # 返回所有不重复
        # 此处不排序的话每一次结果都不一样，输出会乱
        self.vocab = sorted(set(text))
        # print(self.vocab)
        self.vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_vocab = dict(enumerate(self.vocab))
        self.encoded = np.array([self.vocab_to_int[c] for c in text], dtype=np.int32)
    def get_batches(self, arr, n_seqs, n_steps):
        # 对已有的数组进行mini-batch分割
        # arr: 待分割的数组
        # n_seqs: 一个batch中序列个数
        # n_steps: 单个序列包含的字符数
        batch_size = n_seqs * n_steps
        self.n_batches = int(len(arr) / batch_size)
        # 这里我们仅保留完整的batch，对于不能整出的部分进行舍弃
        arr = arr[:batch_size * self.n_batches]
        # 重塑
        arr = arr.reshape((n_seqs, -1))
        for n in range(0, arr.shape[1], n_steps):
            # inputs
            x = arr[:, n:n + n_steps]
            # targets
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y
    # 上面的代码定义了一个generator，调用函数会返回一个generator对象，我们可以获取一个batch。
    def get_generator(self):
        self.batches = self.get_batches(self.encoded, 10, 50)
        x, y = next(self.batches)
    def build_LSTM(self):
        self.model = CharRNN(len(self.vocab), batch_size=self.batch_size, num_steps=self.num_steps,
                    lstm_size=self.lstm_size, num_layers=self.num_layers,
                    learning_rate=self.learning_rate)
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)
        self.runLSTM()
    def runLSTM(self):
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.latest_checkpoint(self.checkpoints)
            start_epoch = 0
            counter = 0
            if checkpoint:
                saver.restore(sess, checkpoint)
                start_epoch = int(checkpoint.split('-')[-2])
                counter = int(checkpoint.split('-')[-1].split('.')[0])
                print("## restore from the checkpoint epoch:{}  step:{}".format(start_epoch, counter))
                start_epoch += 1
            for e in range(start_epoch, self.epochs):
                # Train network
                new_state = sess.run(self.model.initial_state)
                loss = 0
                for x, y in self.get_batches(self.encoded, self.batch_size, self.num_steps):
                    counter += 1
                    start = time.time()
                    feed = {self.model.inputs: x,
                            self.model.targets: y,
                            self.model.keep_prob: self.keep_prob,
                            self.model.initial_state: new_state}
                    batch_loss, new_state, _ = sess.run([self.model.loss,
                                                         self.model.final_state,
                                                         self.model.optimizer],
                                                        feed_dict=feed)
                    end = time.time()
                    # control the print lines
                    if counter % 1 == 0:
                        print('轮数: {}/{}... '.format(e + 1, self.epochs),
                              '训练步数: {}... '.format(counter),
                              '训练误差: {:.4f}... '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end - start)))
                # 在每一层训练结束的时候存储模型
                saver.save(sess, self.checkpoints + self.name + "-{}-{}.ckpt".format(e, counter))
                time01 = time.time() - self.time0
                print("Time: %d h %d min % ds" % (time01 // 3600, (time01 - time01 // 3600 * 3600) // 60, time01 % 60))
            # saver.save(sess, "model/" + self.name + "/i{}_l{}.ckpt".format(counter, self.lstm_size))
        print("训练完成")
    def pick_top_n(self, preds):
        """
        从预测结果中选取前top_n个最可能的字符
        preds: 预测结果
        vocab_size
        top_n
        """
        p = np.squeeze(preds)
        # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-self.top_n]] = 0
        # 归一化概率
        p = p / np.sum(p)
        # 随机选取一个字符
        c = np.random.choice(len(self.vocab), 1, p=p)[0]
        return c
    def sample(self):
        samples = [c for c in self.prime]
        # sampling=True意味着batch的size=1 x 1
        model = CharRNN(len(self.vocab), lstm_size=self.lstm_size, sampling=True)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 加载模型参数，恢复训练
            checkpoint = tf.train.latest_checkpoint(self.checkpoints)
            try:
                saver.restore(sess, checkpoint)
            except:
                return "模型还未建立!"
            new_state = sess.run(model.initial_state)
            for c in self.prime:
                x = np.zeros((1, 1))
                # 输入单个字符
                x[0, 0] = self.vocab_to_int[c]
                feed = {model.inputs: x,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                preds, new_state = sess.run([model.prediction, model.final_state],
                                            feed_dict=feed)

            c = self.pick_top_n(preds)
            # 添加字符到samples中
            samples.append(self.int_to_vocab[c])

            # 不断生成字符，直到达到指定数目
            for i in range(self.wordnum):
                x[0, 0] = c
                feed = {model.inputs: x,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                preds, new_state = sess.run([model.prediction, model.final_state],
                                            feed_dict=feed)

                c = self.pick_top_n(preds)
                samples.append(self.int_to_vocab[c])
        return ''.join(samples)
    def writeAll(self):
        print("开始在" + self.checkpoints + "目录下 写" + str(self.wordnum) + "个词")
        self.samp = self.sample()
        print(self.samp)
# 使用tf.nn.dynamic_run来运行RNN序列
class CharRNN:
    def __init__(self, num_classes, batch_size=100, num_steps=100,lstm_size=512, num_layers=2, learning_rate=0.001,grad_clip=5, sampling=False):
        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets, self.keep_prob = self.build_inputs(batch_size, num_steps)
        # LSTM层
        cell, self.initial_state = self.build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        # 预测结果
        self.prediction, self.logits = self.build_output(outputs, lstm_size, num_classes)
        # Loss 和 optimizer (with gradient clipping)
        self.loss = self.build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = self.build_optimizer(self.loss, learning_rate, grad_clip)
    # 构建输入层
    def build_inputs(self, num_seqs, num_steps):
        inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
        targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
        # 加入keep_prob
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs, targets, keep_prob
    # 返回drop，原代码此处有bug
    def get_a_cell(self, lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    def build_lstm(self, lstm_size, num_layers, batch_size, keep_prob):
        # 构建lstm层
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        # 添加dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        # 堆叠
        # cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * num_layers)
        cell = tf.contrib.rnn.MultiRNNCell([self.get_a_cell(lstm_size, keep_prob) for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        return cell, initial_state

    def build_output(self, lstm_output, in_size, out_size):
        # 构造输出层
        seq_output = tf.concat(lstm_output, axis=1)  # tf.concat(concat_dim, values)
        # reshape
        x = tf.reshape(seq_output, [-1, in_size])
        # 将lstm层与softmax层全连接
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(out_size))
        # 计算logits
        logits = tf.matmul(x, softmax_w) + softmax_b
        # softmax层返回概率分布
        out = tf.nn.softmax(logits, name='predictions')
        return out, logits

    def build_loss(self, logits, targets, lstm_size, num_classes):
        # 根据logits和targets计算损失
        # One-hot编码
        y_one_hot = tf.one_hot(targets, num_classes)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        loss = tf.reduce_mean(loss)
        return loss

    def build_optimizer(self, loss, learning_rate, grad_clip):
        # 构造Optimizer
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

if __name__ == '__main__':
    #  name='USA', prime="The ", numWords=1000
    lstm = LstmAnna(name='USA', numWords=100, prime='The ')
    # 构建模型,在有模型的时候请注释掉
    # lstm.build_LSTM()
    # 写文本,在无模型时候返回模型还未建立
    lstm.writeAll()
