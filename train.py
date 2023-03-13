# coding=gbk

from model import My_GRU
from keras.datasets import imdb
from keras.utils import pad_sequences


#---------------------------------设置参数-------------------------------------
voc_size = 20000        # 词表大小
vec_dim = 64            # 单词向量维度
max_len = 80            # 句子长度
gru_units = 128         # GRU隐层维度
output_dim = 1          # 输出向量维度
batch_size = 128        # 训练批次大小
epochs = 5              # 训练轮数
#-----------------------------------------------------------------------------


#----------------------------------设置路径-----------------------------------
data_path = "D:/科研/python代码/炼丹手册/GRU/datasets/IMDB/imdb.npz"
save_path = "D:/科研/python代码/炼丹手册/GRU/save_models/lstm_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------加载数据集-----------------------------------
(trainX, trainY), _ = imdb.load_data(path=data_path, num_words=voc_size)
print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
#-----------------------------------------------------------------------------


#---------------------------序列预处理，截断或补齐为等长---------------------------
trainX = pad_sequences(trainX, maxlen=max_len)
print('trainX shape:', trainX.shape)
#-----------------------------------------------------------------------------


#-----------------------------------搭建模型----------------------------------
gru = My_GRU(
    voc_size=voc_size,
    vec_dim=vec_dim,
    max_len=max_len,
    units=gru_units,
    output_dim=output_dim,
    )
gru.build((None, 50))
gru.summary()
gru.compile(
    optimizer='adam',                    # adam优化器
    loss='binary_crossentropy',          # 二元交叉熵损失
    metrics='acc'                        # 准确率指标
    )


#----------------------------------训练和保存-----------------------------------
gru.fit(
    trainX, 
    trainY, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=.1           # 取10%样本作验证
    )
gru.save_weights(save_path)
#-----------------------------------------------------------------------------