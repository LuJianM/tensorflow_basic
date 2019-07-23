"""
前向传播确定网络结构
"""

import tensorflow as tf


INPUT_NODE = 784    # 输入节点 784（一张图片28*28，共有784个像素点）个
OUTPUT_NODE = 10    # 输出10数，表示该图片为0~9之间的概率
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
    """"""
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    """"""
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y =  tf.matmul(y1, w2) + b2
    return y