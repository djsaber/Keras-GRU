# coding=gbk

from keras.layers import Input, Dense, Embedding, RNN, GRU, CuDNNGRU
from keras.layers import Layer
from keras.models import Model
import keras.backend as K
from keras import activations


class My_GRU_Cell(Layer):
    def __init__(
        self,
        units,
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        use_bias = True,
        **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.wr = self.add_weight(name='wr', shape=(input_shape[-1]+self.units, self.units))
        self.wz = self.add_weight(name='wz', shape=(input_shape[-1]+self.units, self.units))
        self.wh = self.add_weight(name='wh', shape=(input_shape[-1]+self.units, self.units))
        if self.use_bias:
            self.br = self.add_weight(name='br', shape=(self.units, ))
            self.bz = self.add_weight(name='bz', shape=(self.units, ))
            self.bh = self.add_weight(name='bh', shape=(self.units, ))

    def call(self, inputs, states):
        h = states[0]
        rt = K.dot(K.concatenate([h, inputs]), self.wr)
        zt = K.dot(K.concatenate([h, inputs]), self.wz)
        if self.use_bias:
            rt = K.bias_add(rt, self.br)
            zt = K.bias_add(zt, self.bz)
        rt = self.recurrent_activation(rt)
        zt = self.recurrent_activation(zt)
        ht_hat = K.dot(K.concatenate([h*rt, inputs]), self.wh)
        if self.use_bias:
            ht_hat = K.bias_add(ht_hat, self.bh)
        ht_hat = self.activation(ht_hat)
        ht = h*(1-zt) + (zt*ht_hat)
        return ht, [ht]


class My_GRU_Layer(Layer):
    def __init__(
        self, 
        units,
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        use_bias = True,
        return_sequences = False,
        **kwargs):
        super().__init__(**kwargs)   
        self.cell = My_GRU_Cell(
            units = units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias
            )
        self.layer = RNN(
            cell=self.cell, 
            return_sequences=return_sequences
            )

    def call(self, inputs):
        return self.layer(inputs)


class My_GRU(Model):
    def __init__(
        self, 
        voc_size = 1000,
        vec_dim=64,
        max_len = 50,
        units=128,
        output_dim=1,
        **kwargs):
        super().__init__(**kwargs)
        self.emb = Embedding(
            input_dim=voc_size,
            output_dim=vec_dim,
            input_length=max_len
            )

        # 自定义实现GRU层
        self.gru_layer = My_GRU_Layer(
            units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias = True,
            return_sequences = False
            )

        # 官方实现1.GRU
        # self.gru_layer = GRU(
        #     units,
        #     activation='tanh',
        #     recurrent_activation='hard_sigmoid',
        #     use_bias = True,
        #     return_sequences = False
        #     )

        # 官方实现2.CuDNNGRU
        # self.gru_layer = CuDNNGRU(
        #     units,
        #     return_sequences = False
        #     )

        self.dense = Dense(output_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.emb(inputs)
        x = self.gru_layer(x)
        x = self.dense(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))