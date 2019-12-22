# coding: utf-8
#
# 主要参考自https://github.com/NELSONZHAO/zhihu/blob/master/anna_lstm/anna_lstm.py
# Debug代码参考包括：
#   https://blog.csdn.net/weixin_40759186/article/details/82893626
#   https://github.com/hzy46/Char-RNN-TensorFlow
# 数据集中:
#     Anna.txt来自https://github.com/NELSONZHAO/zhihu/blob/master/anna_lstm/anna_lstm.py
#     USA.txt来自同学
#     各个电子书来自某些盗版网站
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
# 对中文的支持效果还没有给出确定结论,目前可以确定支持中文(已经解决)
# 增加断点检测功能,便于训练中断时,下次继续训练(已经解决)
# 对于训练误差的统计以及可视化数据
# 可能能有更多的拓展性
# 实验中一些参数调整,包括但不限于:
#   learning rate的函数下降式调整调整
#   更多的epoch
#   更少的训练时间
#   不知所云的对抗网络
#   生成中文时选择的随机参数可能需要调整,英文中5效果良好
#   生成文本时候keep_prob
# 七律输出繁体高频字的问题（实际不是繁体字（balabala一大堆），但是看着很怪）

# 实验进程:
# 示例中示范Anna.txt,词数35万, epoch = 20, 训练步数 = 3920, 最终效果很好,目测如果跑更多时间的话有优化空间
# 实验结果支持中文字符,目前没有进行大规模数据测试,还不能给出确定结论,目测实验数据要远大于1万字
# 实验中样本数据如果是中文的话需要把embedding打开
# 训练73219词的USA,时间1h28min47s,epoch = 20, 训练步数 = 940 效果一般,可以生成,有些单词为自造
# 斗罗大陆.txt,一共285万字,100*100模式,epoch = 20，训练时间4 h 50 min，训练步数5860，训练误差在3.3左右
# 红楼梦.txt，一共79万字，100*100模式，epoch = 20，训练时间4 h 9 min，训练步数1640，训练误差4.05左右
# 修改了文本生成的函数，现在可以支持一次生成多个不同文本了
# 红楼梦，epoch = 20，训练误差降到3.15左右
# 七律，训练误差4.5左右（忘了），实验结果很好，可以写出正常七律，训练时间20h+,65*65,目前epoch为14
# 继续训练，训练时间7h59min2s，训练从16到20共5个epoch，训练误差4.3左右，训练步数20860
# 五律
# 轮数: 20/20...  训练步数: 24560...  训练误差: 4.0080...  3.1452 sec/batch
# Time: 7 h 37 min  3s
# 轮数: 60/60...  训练步数: 6179...  训练误差: 3.5502...  4.6254 sec/batch
# 轮数: 60/60...  训练步数: 6180...  训练误差: 3.5241...  4.5299 sec/batch
# Time: 2 h 52 min  29s
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import os
import random
class LstmAnna():
    def __init__(self, name='深渊', isChinese = False, num_steps = 100, epochs = 20, batch_size = 100):
        self.time0 = time.time()
        self.name = name

        # 对文件的处理
        self.isChinese = isChinese
        self.process_file()


        # 关于LSTM的参数
        self.batch_size = batch_size       # Sequences per batch    一个batch中包含的序列数量
        self.num_steps = num_steps        # Number of sequence steps per batch  一个序列中包含的数据数量
        self.lstm_size = 512        # Size of hidden layers in LSTMs
        self.num_layers = 2         # Number of LSTM layers LSTM层数
        self.learning_rate = 0.001  # Learning rate
        self.keep_prob = 0.5        # Dropout keep probability 每个元素被保留的概率，那么 keep_prob:1就是所有元素全部保留的意思。
                                    # 一般在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数，
        self.epochs = epochs        # 训练的轮数
        # self.save_every_n = 1     # 之前涉及保存的步数，新版本已经弃用
        #
        # 关于文章的参数
        self.prime = ""
        self.wordnum = 0
        self.top_n = 5
        self.checkpoints = 'model/' + name + '/'
    def process_file(self):
        with open('data/' + self.name + '.txt', 'r',encoding='utf-8') as f:
            text = f.read()
        # 返回所有不重复
        # 此处不排序的话每一次结果都不一样，输出会乱
        self.vocab = sorted(set(text))
        self.vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_vocab = dict(enumerate(self.vocab))

        # 统计文本中出现字符出现次数
        self.vocab_times = dict.fromkeys(self.vocab, 0)
        for i in text:
            self.vocab_times[i] += 1
        if(self.isChinese):
            with open('data/' + self.name + '-Word.txt', 'w', encoding='utf-8') as f:
                vocab_times = sorted(self.vocab_times.items(),key = lambda d:d[1],reverse= True)
                for i in range(150):
                    if vocab_times[i][0] in ['，','。','\n']:
                        continue
                    f.write(vocab_times[i][0])
                # f.write('\n')
        # print(self.vocab_times['羣'])
        self.encoded = np.array([self.vocab_to_int[c] for c in text], dtype=np.int32)
    def get_batches(self, arr, n_seqs, n_steps):
        # 对已有的数组进行mini-batch分割
        # arr: 待分割的数组
        # n_seqs: 一个batch中序列个数
        # n_steps: 单个序列包含的字符数
        batch_size = n_seqs * n_steps
        self.n_batches = int(len(arr) / batch_size)
        print(self.n_batches)
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
    # 上面的代码定义了一个generator（生成器），调用函数会返回一个generator对象，我们可以获取一个batch。
    # def get_generator(self):
    #     self.batches = self.get_batches(self.encoded, 10, 50)
    #     x, y = next(self.batches)
    def build_LSTM(self):
        self.model = CharRNN(len(self.vocab), batch_size=self.batch_size, num_steps=self.num_steps,
                    lstm_size=self.lstm_size, num_layers=self.num_layers,
                    learning_rate=self.learning_rate,use_embedding=self.isChinese)
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)
        self.runLSTM()
    def runLSTM(self):
        # 只保留最新的模型
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.latest_checkpoint(self.checkpoints)
            start_epoch = 0
            counter = 0
            # 代表之前跑过模型
            if checkpoint:
                saver.restore(sess, checkpoint)
                start_epoch = int(checkpoint.split('-')[-2])
                counter = int(checkpoint.split('-')[-1].split('.')[0])
                print("## restore from the checkpoint epoch:{}  step:{}".format(start_epoch, counter))
                start_epoch += 1
            # 从之前跑过的轮数/0开始进行训练
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
                    batch_loss, new_state, _ = sess.run([self.model.loss, self.model.final_state, self.model.optimizer],
                                                        feed_dict=feed)

                    end = time.time()
                    # control the print lines
                    if counter % 1 == 0:
                        print('轮数: {}/{}... '.format(e + 1, self.epochs),
                              '训练步数: {}... '.format(counter),
                              '训练误差: {:.4f}... '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end - start)))
                saver.save(sess, self.checkpoints + self.name + "-{}-{}.ckpt".format(e, counter))
                time01 = time.time() - self.time0
                print("Time: %d h %d min % ds" % (time01 // 3600, (time01 - time01 // 3600 * 3600) // 60, time01 % 60))
            # saver.save(sess, "model/" + self.name + "/i{}_l{}.ckpt".format(counter, self.lstm_size))
        print("训练完成")
    def pick_top_n(self, preds):    # 从预测结果中选取前top_n个最可能的字符
        p = np.squeeze(preds)
        # 将除了top_n个预测值的位置都置为0
        p[np.argsort(p)[:-self.top_n]] = 0
        # 归一化概率
        p = p / np.sum(p)
        # 随机选取一个字符
        c = np.random.choice(len(self.vocab), 1, p=p)[0]
        # 删除出现次数太少的情况
        times = 0
        while self.vocab_times[self.int_to_vocab[c]] < 250 or (self.int_to_vocab[c] in self.samples and self.int_to_vocab[c] not in ['，','。','\n']):
            # print(self.int_to_vocab[c] + "太少了或者出现过")
            c = np.random.choice(len(self.vocab), 1, p=p)[0]
            times += 1
            if times > 100:
                break
        return c
    def sample(self):
        self.samples = [c for c in self.prime]
        # sampling=True意味着batch的size=1 x 1
        model = CharRNN(len(self.vocab), lstm_size=self.lstm_size, sampling=True, use_embedding=self.isChinese)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 加载模型参数，恢复训练
            checkpoint = tf.train.latest_checkpoint(self.checkpoints)
            try:
                saver.restore(sess, checkpoint)
            except:
                return "模型还未建立!"
            new_state = sess.run(model.initial_state)
            # print(self.prime)
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
            self.samples.append(self.int_to_vocab[c])
            # 不断生成字符，直到达到指定数目，此处还未优化，如句号结束，换行结束
            for i in range(self.wordnum):
                x[0, 0] = c
                feed = {model.inputs: x,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                preds, new_state = sess.run([model.prediction, model.final_state],
                                            feed_dict=feed)
                c = self.pick_top_n(preds)
                c1 = self.int_to_vocab[c]
                # print(c1)
                if c1 == '閒':
                    c1 = '闲'
                if c1 == '鴈':
                    c1 = '雁'
                if c1 == '歎':
                    c1 = '叹'
                if c1 == '髪':
                    c1 = '发'
                if c1 == '羣':
                    c1 = '群'
                if c1 == '巖':
                    c1 = '岁'
                if c1 == '隠':
                    c1 = '隐'
                if c1 == '臯':
                    c1 = '皋'
                if c1 == '疎':
                    c1 = '疏'
                if c1 == '癡':
                    c1 = '痴'

                self.samples.append(c1)
        return ''.join(self.samples)
    def writeAll(self, prime, numWords):
        self.prime = prime
        self.wordnum = numWords
        print("开始在" + self.checkpoints + "目录模型下\n写" + str(self.wordnum) + "个词" + "\n以" + self.prime + "开始")
        self.samp = self.sample()
        print(self.samp)
        return self.samp
    def writePoemQilv(self, prime='', numWords=64):
        if prime:
            self.prime = prime
        else:
            with open('data/' + self.name + '-Word.txt', 'r', encoding='utf-8') as f:
                all = f.read()
                # all = all.split('\n')
                self.prime = random.choice(all)
        self.wordnum = numWords + 20
        # print("开始在" + self.checkpoints + "目录模型下\n写" + str(numWords) + "个词" + "\n以" + self.prime + "开始")
        while 1:
            self.samp = self.sample()
            # print(self.samp)
            poem = self.samp.split('\n')
            for item in poem:
                if(len(item) == numWords):
                    self.poem = item
                    # print(str(len(item)) + ':', end='')
                    # print(poem)
                    print(item)
                    return item
    def writePoemWulv(self, prime='', numWords=48):
        if prime:
            self.prime = prime
        else:
            with open('data/' + self.name + '-Word.txt', 'r', encoding='utf-8') as f:
                all = f.read()
                # all = all.split('\n')
                self.prime = random.choice(all)
        self.wordnum = numWords + 50
        # print("开始在" + self.checkpoints + "目录模型下\n写" + str(numWords) + "个词" + "\n以" + self.prime + "开始")
        if 1:
            self.samp = self.sample()
            # print(self.samp)
            poem = self.samp.split('\n')
            for item in poem:
                # print(item)
                if(len(item) == numWords):
                    self.poem = item
                    # print(str(len(item)) + ':', end='')
                    # print(poem)
                    print(item)
                    return item
    def writePoemWuJue(self, prime='', numWords=24):
        if prime:
            self.prime = prime
        else:
            with open('data/' + self.name + '-Word.txt', 'r', encoding='utf-8') as f:
                all = f.read()
                # all = all.split('\n')
                self.prime = random.choice(all)
        self.wordnum = numWords + 20
        # print("开始在" + self.checkpoints + "目录模型下\n写" + str(numWords) + "个词" + "\n以" + self.prime + "开始")
        while 1:
            self.samp = self.sample()
            # print(self.samp)
            poem = self.samp.split('\n')
            for item in poem:
                if(len(item) == numWords):
                    self.poem = item
                    # print(str(len(item)) + ':', end='')
                    # print(poem)
                    print(item)
                    return item
    def writePoemQiJue(self, prime='', numWords=32):
        if prime:
            self.prime = prime
        else:
            with open('data/' + self.name + '-Word.txt', 'r', encoding='utf-8') as f:
                all = f.read()
                # all = all.split('\n')
                self.prime = random.choice(all)
        self.wordnum = numWords + 20
        # print("开始在" + self.checkpoints + "目录模型下\n写" + str(numWords) + "个词" + "\n以" + self.prime + "开始")
        while 1:
            self.samp = self.sample()
            # print(self.samp)
            poem = self.samp.split('\n')
            for item in poem:
                if(len(item) == numWords):
                    self.poem = item
                    # print(str(len(item)) + ':', end='')
                    # print(poem)
                    print(item)
                    return item
# 使用tf.nn.dynamic_run来运行RNN序列
class CharRNN:
    def __init__(self, num_classes, batch_size=100, num_steps=100,lstm_size=512, num_layers=2,
                 learning_rate=0.001,grad_clip=5, sampling=False,
                 use_embedding=False, embedding_size=256):
        # 如果sampling是True，则采用SGD
        if sampling == True:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps
        # 是否使用embedding层
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        # 实际上为len(vocab)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # 梯度裁剪
        self.grad_clip = grad_clip

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets, self.keep_prob = self.build_inputs()
        # LSTM层
        cell, self.initial_state = self.build_lstm()
        # 对输入进行one-hot编码(并没有)
        # 根据输入情况,中文不用one-hot编码
        # 运行RNN
        self.outputs, state = dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)
        self.final_state = state

        # 预测结果
        self.prediction, self.logits = self.build_output()

        # Loss 和 optimizer (with gradient clipping)
        self.loss = self.build_loss()
        self.optimizer = self.build_optimizer()
    # 构建输入层
    def build_inputs(self):
        inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='inputs')
        targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')
        # 加入keep_prob
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 对于中文，需要使用embedding层
        # 英文字母，没有必要用embedding层
        # 通常embedding大小从100到300不等，取决于词汇库的大小。超过300维度会导致效益递减
        # 减少维度优化了时间，大概缩短到原先的2/3的时间
        if self.use_embedding is False:
            self.lstm_inputs = tf.one_hot(inputs, self.num_classes)
        else:
            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                self.lstm_inputs = tf.nn.embedding_lookup(embedding, inputs)
        return inputs, targets, keep_prob
    # 返回drop，原代码此处有bug
    def get_a_cell(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        return drop
    def build_lstm(self):
        # 构建lstm层
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        # 添加dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        # 堆叠
        # cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * num_layers)
        cell = tf.contrib.rnn.MultiRNNCell([self.get_a_cell() for _ in range(self.num_layers)])
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        return cell, initial_state
    def build_output(self):
        in_size, out_size = self.lstm_size, self.num_classes
        # 构造输出层
        seq_output = tf.concat(self.outputs, axis=1)  # tf.concat(concat_dim, values)
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
    def build_loss(self):
        # 根据logits和targets计算损失
        # One-hot编码
        # num_classes使用的是len(self.vocab)
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        # tf.reshape将tensor变换为参数shape形式,原数量不变
        # logits.get_shape返回一个元组,代表logits的张量大小,如2*3返回(2,3)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
        # 计算logits和labels之间的softmax交叉熵。其中类是互斥的，适用于一个图像对应一个标签的损失值计算
        # 1.将labels变为one_hot编码；2.计算logits的Softmax; 3.计算交叉熵Cross-Entropy
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
        # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
        loss = tf.reduce_mean(loss)
        return loss
    def build_optimizer(self):
        # 构造Optimizer，使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer
if __name__ == '__main__':
    #  name='USA', prime="The ", numWords=1000, isChinese
    #  模型初始化
    lstm = LstmAnna(name='Anna', isChinese=False,
                    # num_steps中：正常训练文本取100，七绝33，七律65，五绝64*64，五律49
                    num_steps=64,
                    epochs=60,
                    batch_size=64)
    # 构建模型
    # lstm.build_LSTM()
    # 写文本
    # for i in range(10):
    #     lstm.writePoemWuJue()
    lstm.writeAll(prime='The ', numWords=1000)
