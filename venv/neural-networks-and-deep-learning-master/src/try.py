#-*-coding:utf-8-*-
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
# 10个隐藏层
net = network.Network([784, 50, 10])
# 30 次迭代期，⼩批量数⼤⼩为 10，学习速率 η = 3.0，
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
