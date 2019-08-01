import tensorflow as tf

# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([1.0, 2.0], name="b")
#
# result = a + b
#
# print(result)

# 定义一个计算图g1，并且在g1中定义变量v，设置初始值为0
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer(shape=[1]))


# 定义一个计算图g2，并且在g2中定义变量v，设置初始值为1
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer(shape=[1]))

# 在计算图g1中
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
