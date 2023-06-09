# Keras-GRU
基于Keras搭建一个简单的GRU，用IMDB影评数据集对GRU进行训练，完成模型的保存和加载以及测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：存放数据集文件<br />
2. /save_models：保存训练好的模型权重文件<br /><br />

加载模型权重时请确保使用的模型和保存的权重一致<br />
比如，当你保存的是自定义的GRU模型权重，那么同样需要构建自定义的GRU模型来读取这个权重<br />
当使用Keras官方实现的GRU时，就会报错，即使官方实现的GRU和自定义的GRU参数量是一样的<br />
反之亦然<br /><br />

GRU概述<br />
GRU是LSTM网络的一种效果很好的变体，它较LSTM网络的结构更加简单，而且效果也很好，因此也是当前非常流形的一种网络。
GRU既然是LSTM的变体，因此也是可以解决RNN网络中的长依赖问题。
在LSTM中引入了三个门函数：输入门、遗忘门和输出门来控制输入值、记忆值和输出值。
而在GRU模型中只有两个门：分别是更新门和重置门。<br />
1. 更新门用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。
2. 重置门控制前一状态有多少信息被写入到当前的状态上，重置门越小，前一状态的信息被写入的越少。<br />

概括来说，LSTM和CRU都是通过各种门函数来将重要特征保留下来，这样就保证了在long-term传播的时候也不会丢失。
此外GRU相对于LSTM少了一个门函数，因此在参数的数量上也是要少于LSTM的，所以整体上GRU的训练速度要快于LSTM的。
不过对于两个网络的好坏还是得看具体的应用场景。<br /><br />

如同自定义实现简单RNN时所说，实现自己的自定义GRU：<br />
Keras实现自定义循环神经网络需要：<br />
1.实现自定义Cell，比如一个自定义的GRUCell，注意需要定义状态参数维度：self.state_size<br />
2.将实现好的Cell作为参数cell传入Keras.layers.RNN()，让Keras自动推断每个时刻的传递过程<br /><br />

Keras实现的GRU层有两种：<br />
1. Keres.layers.GRU  非常慢，和我自定义的GRU层速度差不多<br />
2. Keres.layers.CuDNNGRU 支持GPU加速，训练非常快，不过激活函数貌似无法更改，工程中推荐用这个<br /><br />

数据集：<br />
IMDB：影评数据集,训练集/测试集包含25000/25000条影评数据<br />
链接：https://pan.baidu.com/s/18nX-2mqJzYU8XKQ5cfhxvw?pwd=52dl 提取码：52dl<br /><br />

通过对训练集切分10%比例用于训练时验证模型<br />
训练好的模型对测试集进行测试评价效果<br />
经测试，实现一个简单的gru在测试集accuracy能达到~80%<br />
