"""卷积层、池化层样例
输入为：
         1,-1 , 0,
        -1, 2 , 1,
         0, 2 ,-2



"""

import tensorflow as tf
import numpy as np


# 定义一个3*3的矩阵,二维数组
M = np.array([
        [[1],[-1],[0]],
        [[-1],[2],[1]],
        [[0],[2],[-2]]
    ])

print("Matrix shape is: ",M.shape)


# 定义卷积过滤器, 深度为1
# 定义名为weights的卷积核，卷积核的形状为2*2，卷积核的通道数为1，卷积核的个数为1，卷积核为：
# 1，-1
# 0，2
filter_weight = tf.get_variable('weights', [2, 2, 1, 1],
                                initializer=tf.constant_initializer([
                                    [1, -1],
                                    [0, 2]
                                ]))


# 定义偏置，偏置名为biases，形状为1*1，值为1
biases = tf.get_variable('biases', [1], initializer = tf.constant_initializer(1))


M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)

#
x = tf.placeholder('float32', [1, None, None, 1])

# 卷积核
conv = tf.nn.conv2d(x, filter_weight, strides = [1, 2, 2, 1], padding = 'SAME')
bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:

    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias,feed_dict={x:M})
    pooled_M = sess.run(pool,feed_dict={x:M})

    print("convoluted_M: \n", convoluted_M)
    print("pooled_M: \n", pooled_M)








