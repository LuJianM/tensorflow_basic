"""
前向传播确定网络结构

需要定义参数w，参数b，推算y的网络结构
def forward（）：确定网络结构

def get_weight():实现参数w的初始化工作

def get_bias():实现参数b的初始化工作

"""

import tensorflow as tf


INPUT_NODE = 784    # 输入节点 784（一张图片28*28，共有784个像素点）个
OUTPUT_NODE = 10    # 输出10数，这个等于要预测类别的数目。在这里表示的是该图片为0~9之间的概率

LAYER1_NODE = 500   # 隐藏层节点个数

def get_weight(shape, regularizer):
    """初始化权重"""

    # 随机生成参数w
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        """如果使用正则化"""

        # 将每个变量的损失加入到总损失losses
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

def get_bias(shape):
    """初始化偏置值"""
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    """该网络的结构有两层，一个隐藏层和一个输出层"""

    # x为1*784的矩阵，w1是一个784*500的矩阵，b1是一个1*500的矩阵
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)  # 最终结果，y1是一个1*500的矩阵

    # y1为1*500的矩阵，w2为500*10的矩阵，b2为1*10的矩阵
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2  # 最终输出，y为一个1*10的矩阵
    return y