import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import sys
def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))
    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))
    return res_block
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())
    def call(self, inputs, training=None):
        identity = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = tf.nn.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = tf.nn.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        output = tf.nn.relu(tf.keras.layers.add([identity, bn3]))
        return output

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3,3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1: #维数不为1的时候需要进行下采样
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x: x
    def call(self, inputs, training=None):
        residual = self.downsample(inputs) #原来的x
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        add = layers.add([bn2, residual]) #shortcut
        out = self.relu(add) #等价于out = tf.nn.relu(add)
        return out

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100): #[2,2,2,2]
        super(ResNet,self).__init__()
        self.stem=Sequential([layers.Conv2D(64,(3,3),strides=(1,1),padding='same'),layers.BatchNormalization(),layers.Activation('relu'),layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # output: [b, 512, h, w]
        self.avgpool = layers.GlobalAveragePooling2D() #功能层
        self.fc = layers.Dense(num_classes) 
    def call(self,inputs,trianing=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    def build_resblock(self, filter_num, blocks, stride=1):
        #may downsample
        res_blocks = keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks
class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=100, activation=tf.keras.activations.softmax)
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)
        return output
def resnet18():
    return ResNet([2,2,2,2])
def resnet34():
    return ResNet([3,4,6,3])
def resnet50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])