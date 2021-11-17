import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import scikitplot as skplt

# load model
model = tf.keras.models.load_model("best_weight/weights-75-0.92.hdf5")
# model = tf.keras.models.load_model("models/model_SV.h5")

X_train=[]
y_train=[]
X_test=[]
y_test=[]

# Đọc dữ liệu
X_train = np.load("X_train_image.npy")
X_test = np.load("X_test_image.npy")
y_train = np.load("y_train_image.npy")
y_test = np.load("y_test_image.npy")

print("predicting....")
pre_test = model.predict(X_test)
pre_train = model.predict(X_train)

encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

y_pre_test= np.argmax(pre_test,axis=1)
y_test_category = y_test.argmax(y_test,axis=1)

# Vẽ chuẩn bị

skplt.metrics.plot_confusion_matrix(y_test_category, y_pre_test, normalize=False)
skplt.metrics.plot_confusion_matrix(y_test_category, y_pre_test, normalize=True)

print("predicting test data....")
plt.show()
print(accuracy_score(y_test_category, y_pre_test))


#====================================

y_pre_train = np.argmax(pre_train,axis=1)
y_train_category = np.argmax(y_train,axis=1)

# Vẽ chuẩn bị

skplt.metrics.plot_confusion_matrix(y_train_category, y_pre_train, normalize=False)
skplt.metrics.plot_confusion_matrix(y_train_category, y_pre_train, normalize=True)

print("predicting train data....")

plt.show()
print(accuracy_score(y_train_category, y_pre_train))


history_model = np.load("my_history_thu2(100-0.4-64).npy",allow_pickle='True').item()

print(np.max(history_model['val_accuracy']))
print(np.max(history_model['accuracy']))

# summarize history for accuracy

plt.plot(history_model['accuracy'])
plt.plot(history_model['val_accuracy'])
plt.plot([0.5])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
plt.show()

# summarize history for loss

plt.plot(history_model['loss'])
plt.plot(history_model['val_loss'])
plt.plot([5.0])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()