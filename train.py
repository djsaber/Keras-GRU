# coding=gbk

from model import My_GRU
from keras.datasets import imdb
from keras.utils import pad_sequences


#---------------------------------���ò���-------------------------------------
voc_size = 20000        # �ʱ��С
vec_dim = 64            # ��������ά��
max_len = 80            # ���ӳ���
gru_units = 128         # GRU����ά��
output_dim = 1          # �������ά��
batch_size = 128        # ѵ�����δ�С
epochs = 5              # ѵ������
#-----------------------------------------------------------------------------


#----------------------------------����·��-----------------------------------
data_path = "D:/����/python����/�����ֲ�/GRU/datasets/IMDB/imdb.npz"
save_path = "D:/����/python����/�����ֲ�/GRU/save_models/lstm_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------�������ݼ�-----------------------------------
(trainX, trainY), _ = imdb.load_data(path=data_path, num_words=voc_size)
print('trainX shape:', trainX.shape)
print('trainY shape:', trainY.shape)
#-----------------------------------------------------------------------------


#---------------------------����Ԥ�����ضϻ���Ϊ�ȳ�---------------------------
trainX = pad_sequences(trainX, maxlen=max_len)
print('trainX shape:', trainX.shape)
#-----------------------------------------------------------------------------


#-----------------------------------�ģ��----------------------------------
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
    optimizer='adam',                    # adam�Ż���
    loss='binary_crossentropy',          # ��Ԫ��������ʧ
    metrics='acc'                        # ׼ȷ��ָ��
    )


#----------------------------------ѵ���ͱ���-----------------------------------
gru.fit(
    trainX, 
    trainY, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=.1           # ȡ10%��������֤
    )
gru.save_weights(save_path)
#-----------------------------------------------------------------------------