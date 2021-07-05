import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import sys
import time
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.random.set_seed(2345)
from model.resnet import resnet18,resnet34,resnet50
from model.alexnet import AlexNet
from model.vgg import VGG16,VGG19
from model.inception_v4 import InceptionV4
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 1
    x = tf.image.resize(x,(size[0],size[1]))
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def main():
    # 建立解析对象
    parser = argparse.ArgumentParser() 
    parser.add_argument("model",default=resnet50, type=str)
    parser.add_argument("batchsize",default=16,type=str)
    parser.add_argument("imgsize",default=224,type=str)
    if parser.model=='resnet50':
        model = resnet50()
    elif parser.model=='AlexNet':
        model = AlexNet()
    elif parser.model=='VGG16':
        model = VGG16()
    elif parser.model=='VGG19':
        model = VGG19()
    elif parser.model=='InceptionV4':
        model = InceptionV4()
    else:
        print("model not support")
    size = [parser.imgsize,parser.imgsize]
    model.build(input_shape=(None,size[0],size[1],3))
    
    # build dataset
    batchsz = parser.batchsize
    (x,y), (x_val, y_val) = datasets.cifar100.load_data()
    y = tf.squeeze(y,axis=1)
    y_val = tf.squeeze(y_val,axis=1) #注意维度变换
    print(x.shape,y.shape,x_val.shape,y_val.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x,y))
    train_db = train_db.shuffle(1000).map(preprocess).batch(batchsz)
    test_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    test_db = test_db.map(preprocess).batch(batchsz)
    sample = next(iter(train_db))
    print('batch: ', sample[0].shape, sample[1].shape)
    # build optimizer
    optimizer = optimizers.Adam(lr=1e-4)
    #拼接需要训练的参数 [1,2] + [3,4] = [1,2,3,4]
    #训练过程
    ites = 100
    #while True:
    #    for step,(x,y) in enumerate(train_db):
    since=time.time()
    for i in range(ites):
        with tf.GradientTape() as tape:
            logits = model(sample[0])
            y_onehot = tf.one_hot(sample[1],depth=100)
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(batchsz*100/(time.time()-since))
#     for epoch in range(50):
#         for step, (x,y) in enumerate(train_db):
#             with tf.GradientTape() as tape:
#                 #[b,32,32,3] => [b,1,1,512]
#                 print(x.shape)
#                 sys.exit()
#                 logits = model(x)
#                 y_onehot = tf.one_hot(y, depth=100) #[50k, 10]
#                 y_val_onehot = tf.one_hot(y_val, depth=100)
#                 loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
#                 loss = tf.reduce_mean(loss)
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         total_num = 0
#         total_correct = 0
#         for x, y in test_db:
#             logits = model(x)
#             prob = tf.nn.softmax(logits, axis=1)
#             pred = tf.argmax(prob, axis=1)
#             pred = tf.cast(pred, dtype=tf.int32)
#             correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
#             correct = tf.reduce_sum(correct)
#             total_num += x.shape[0]
#             total_correct += int(correct)
#         acc = total_correct / total_num
        
if __name__ == '__main__':
    main()
