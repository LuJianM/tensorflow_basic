'''
简单了解手写识别字数据集mnist
'''

import input_data

# 使用官方提供的input_data模块的read_data_sets自动加载数据集，"Mnist_data/"是数据集存放的路径，当前目录的Mnist_data目录下面。并以独热码的方式存取。
# read_data_sets会自动检查本地是否有数据集，如果没有，会自动下载train，
mnist = input_data.read_data_sets("Mnist_data/",one_hot = True)


print("训练集大小为: ", mnist.train.num_examples)

print("验证集大小为：", mnist.validation.num_examples)

print("测试样本大小为：", mnist.test.num_examples)



print("训练集中指定的图片类型为：", mnist.train.images[0].shape)

print("训练集中指定的标签类型为：", mnist.train.labels[0].shape)

print("查看训练集中指定的标签：")
print(mnist.train.labels[0], '\n')

# print("查看训练集中指定的图片：")
# print(mnist.train.images[0])
