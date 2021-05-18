import tensorflow as tf
from tensorflow import keras
class LRN(keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius=2
        self.bias=1
        self.alpha=1e-4
        self.beta=0.75
    def call(self,x):
        return tf.nn.lrn(x,depth_radius=self.depth_radius,
                         bias=self.bias,alpha=self.alpha,
                         beta=self.beta)
class AlexNet(keras.Model):
    def __init__(self, num_classes=100):
        super(AlexNet,self).__init__()
        self.Conv1 = keras.layers.Conv2D(filters=96,
                              kernel_size=(11,11),
                              strides=4,
                              activation='relu',
                              padding='same',
                              input_shape=(224,224,3))
        self.MaxPool1 = keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.LRN1 = LRN()
        self.Conv2 = keras.layers.Conv2D(filters=256,
                              kernel_size=(5,5),
                              strides=1,
                              activation='relu',
                              padding='same')
        self.Conv3_1 = keras.layers.Conv2D(filters=384,
                              kernel_size=(3,3),
                              strides=1,
                              activation='relu',
                              padding='same')
        self.Conv3_2 = keras.layers.Conv2D(filters=384,
                              kernel_size=(3,3),
                              strides=1,
                              activation='relu',
                              padding='same')
        self.Conv4 = keras.layers.Conv2D(filters=256,
                              kernel_size=(3,3),
                              strides=1,
                              activation='relu',
                              padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense_0 = keras.layers.Dense(4096,activation='relu')
        self.dense_1 = keras.layers.Dense(4096,activation='relu')
        self.dropout = keras.layers.Dropout(0.5)
        self.dense1 = keras.layers.Dense(num_classes,activation="softmax")
    def call(self,inputs,trianing=None):
        x = self.Conv1(inputs)
        x = self.MaxPool1(x)
        x = self.LRN1(x)
        x = self.Conv2(x)
        x = self.MaxPool1(x)
        x = self.LRN1(x)
        x = self.Conv3_1(x)
        x = self.Conv3_2(x)
        x = self.Conv4(x)
        x = self.MaxPool1(x)
        x = self.flatten(x)
        x = self.dense_0(x)
        x = self.dropout(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense1(x)
        return x