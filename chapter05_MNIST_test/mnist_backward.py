"""
反向传播优化网络参数

def backward(mnist):
    x =
    y_ =
    y =
    global_step =
    loss =
    <正则化，指数衰减学习率、滑动平均>

    train_step =










"""
import tensorflow as tf
import input_data
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 32                 # 一轮喂入神经网络的图片数
LEARNING_RATE_BASE = 0.1        # 最开始的学习率
LEARNING_RATE_DECAY = 0.99      # 学习率衰减率
REGULARIZER = 0.0001            # 正则化系数
STEPS = 10000                   # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"    # 模型保存路径
MODEL_NAME = 'mnist_model'      # 模型保存名称


def backward(mnist):
    """"""
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])        # 使用placeholder给x，y标占位
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)       # 调用前向传播程序计算出张量y
    global_step = tf.Variable(0, trainable=False)   # 给轮数计数器赋初值，

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection("losses"))  # 调用包含正则化的损失函数 loss

    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    # 实例化saver
    saver = tf.train.Saver()

    # 在with结构中初始化所有变量
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        exclude = ['']

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 用for循环迭代STEPS轮(i从0到STEPS-1)
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)     # 每次读入BATCH_SIZE张图片
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})    # 喂入神经网络，执行训练过程

            # 每1,000轮打印出当前的loss值。要想得到loss值，需要在前面运行sess.run才有结果
            if i % 1000 == 0:
                print("当前轮数为 %d" %i)
                print("After %d training step(s), losses on trianing batch is %f. " % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)    # 保存模型到当前会话中


def main():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    print("训练集大小为: ", mnist.train.num_examples)
    print("验证集大小为：", mnist.validation.num_examples)
    backward(mnist)


# 模块化测试代码
if __name__ == "__main__":
    main()