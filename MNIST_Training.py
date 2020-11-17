import numpy as np
# 下面使用MNIST数据集
import tensorflow as tf
mnist = tf.keras.datasets.mnist
#这里先调用上面函数然后下载数据包，下面要填上绝对路径
(x_train, y_train), (x_test, y_test) = mnist.load_data()#(path='C:/Users/wang_xiaohua/Desktop/TF2.0写的书/TF2.0/dataset/mnist.npz')
x_train=x_train[0:6000]
y_train=y_train[0:6000]
x_test=x_test[0:6000]
y_test=y_test[0:6000]
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.expand_dims(x_train,-1)#[1,2]->[[1],[2]]
y_train = np.float32(tf.keras.utils.to_categorical(y_train,num_classes=10))#[4]->[0,0,0,0,1,0,0..]
x_test = tf.expand_dims(x_test,-1)
y_test = np.float32(tf.keras.utils.to_categorical(y_test,num_classes=10))
#这里为了shuffle数据，单独定义了每个batch的大小，batch_size，这与下方的shuffle对应
bacth_size = 512#512 group size
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(bacth_size).shuffle(bacth_size * 10)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(bacth_size)
input_xs = tf.keras.Input([28,28,1])
conv = tf.keras.layers.Conv2D(32,3,padding="SAME",activation=tf.nn.relu)(input_xs)
conv = tf.keras.layers.BatchNormalization()(conv)
conv = tf.keras.layers.Conv2D(64,3,padding="SAME",activation=tf.nn.relu)(conv)
conv = tf.keras.layers.MaxPool2D(strides=[1,1])(conv)
conv = tf.keras.layers.Conv2D(128,3,padding="SAME",activation=tf.nn.relu)(conv)
flat = tf.keras.layers.Flatten()(conv)
dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
logits = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense)
model = tf.keras.Model(inputs=input_xs, outputs=logits)

model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy,metrics = ['accuracy'])
model.fit(train_dataset, epochs=2)#10
model.save("./model.h5")
score = model.evaluate(test_dataset)
print("last score:",score)
