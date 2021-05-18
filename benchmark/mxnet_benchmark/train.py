from mxnet import gluon, init, nd
from mxnet.gluon import nn
import sys
import os
import mxnet as mx
from mxnet.gluon import data as gdata
import mxnet.gluon.loss as gloss
import time
from mxnet import autograd
# from model import inceptionv4
def VGGNet(architecture):
    """
    通过引入了函数和循环的方式，可以快速创建任意层数的神经网络
    :return:
    """
    def vgg_block(num_convs, channals):
        """
        定义一个网络的基本结构，由若干卷积层和一个池化层构成
        VGG的一个关键是使用很多有着相对小的kernel（3×3）的卷积层然后接上一个池化层，之后再将这个模块重复多次。因此先定义一个这样的块：
        :param num_convs: 卷积层的层数
        :param channals: 通道数
        :return:
        """
        net = nn.Sequential()
        for _ in range(num_convs):
            net.add(nn.Conv2D(channels=channals, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        return net

    def vgg_stack(architecture):
        """
        定义所有卷积层的网络结构，通过参数将定义的网络结构封装起来
        :param architecture: 指定的网络结构参数
        :return:
        """
        net = nn.Sequential()
        for (num_convs, channals) in architecture:
            net.add(vgg_block(num_convs, channals))
        return net

    # 在卷积层之后，采用了两个全连接层，然后使用输出层输出结果。
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            vgg_stack(architecture),
            nn.Flatten(),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(10)
        )
    return net
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
def train(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    loss = gloss.SoftmaxCrossEntropyLoss()
    x,y = None,None
    for X, Y in train_iter:
        X, Y = X.as_in_context(ctx), Y.as_in_context(ctx)
        x,y=X,Y
        break
    for epoch in range(num_epochs):
        with autograd.record():
            y_hat = net(x)
            l = loss(y_hat, y).sum()
        l.backward()
        trainer.step(batch_size)
#             y = y.astype('float32')
#             n += y.size
# architecture = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
# net = VGGNet(architecture)
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
       nn.BatchNorm(), nn.Activation('relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
net.add(resnet_block(64, 3, first_block=True),
       resnet_block(128, 4),
       resnet_block(256, 6),
       resnet_block(512, 3))
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
lr, num_epochs, batch_size, ctx = 0.05, 500000, 256, try_gpu()
print("batch_size: ",batch_size)
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
train(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)