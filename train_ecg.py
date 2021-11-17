import cv2
import numpy as np
import keras
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.models import  load_model
import time
import tensorflow as tf
from datetime import datetime

# Khai báo
X_train=[]
y_train=[]
X_test=[]
y_test=[]

# Biến thay đổi
# 100 - 0.5 - 32 sai lop 1
# 100 - 0.3 - 64:96%test lớp 1 0.7test 0.8train đẹp
# 100 - 0.3 - 32:97% lớp 1 0.6test 0.56train thua
# 100 - 0.4 -32: 96% lớp 1 0.78test 0.84train đẹp hơn
# 100 - 0.4 - 64: 98% lớp 1 thấp
num_epochs = 100
num_drop = 0.5
batch_size = 64

X_train = np.load("X_train_image.npy")
X_test = np.load("X_test_image.npy")
y_train = np.load("y_train_image.npy")
y_test = np.load("y_test_image.npy")

encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

for (i,lab) in enumerate(encoder.classes_):
    print("{}.{}".format(i+1,lab))

base_model = VGG16(input_shape=(128,128,3), weights='imagenet', include_top=False)
    # Dong bang cac layer 4
for layer in base_model.layers:
        layer.trainable = False
    # Them cac layer FC va Dropout
x = Flatten(name='flatten')(base_model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
#x = BatchNormalization()(x) không tăng độ chính xác
x = Dropout(num_drop)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
#x = BatchNormalization()(x)
x = Dropout(num_drop)(x)
x = Dense(5, activation='softmax', name='predictions')(x)
    # Compile
my_model = Model(base_model.input, x)

my_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

print(my_model.summary())

# my_model = keras.utils.plot_model(my_model, "my_first_model.png")

filepath="best_weight/weights-{epoch:02d}-{val_accuracy:.2f}-100-0.5-64.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# callbacks = [EarlyStopping(monitor='val_loss', patience=8),
#              ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
start = datetime.now().time()

history = my_model.fit(
  X_train,y_train,
  validation_data=(X_test,y_test),
  epochs=num_epochs,verbose=2,
  batch_size=batch_size,
  callbacks=callbacks_list
)

my_model.save("models/model_SV_VGG16-100-0.5-64.h5")
np.save("my_history_thu2(100-0.5-64).npy", history.history)
# print("Thoi gian chay", datetime.now().time()-start)

test_loss, test_acc = my_model.evaluate(X_test, y_test,batch_size=batch_size, verbose=2) 
print("Acc:",test_acc * 100)
print("Loss:",test_loss)