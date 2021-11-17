from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('best_weight/weights-93-0.96-100-0.3-64.hdf5')
image = cv2.imread("/content/drive/MyDrive/Colab Notebooks/Dien_Tam_Do/ecg_img/test/F/F/Ffig_135.png")
image = cv2.resize(image, (128,128))
image = tf.keras.preprocessing.image.array_to_img(image)
plt.imshow(image)
plt.show()
ima = np.expand_dims(image, axis=0)
images = np.vstack([ima])
# print("predicting....")

pre_test = model.predict(images)
print(pre_test)
class_cu = ["F","N","Q","S","V"]
print(class_cu[np.argmax(pre_test[0])],100 * np.max(pre_test[0]))

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Ảnh')
plt.xticks([])
plt.yticks([])
plt.imshow(image, cmap=plt.cm.binary)
plt.xlabel("{} {:2.0f}% ".format(class_cu[np.argmax(pre_test[0])],100 * np.max(pre_test[0])))


ax = fig.add_subplot(1, 2, 2)
ax.set_title('Predictor')
thisplot = plt.bar(class_cu, pre_test[0], color="#777777" )
plt.ylim([0, 1])
predicted_label = np.argmax(pre_test[0])
thisplot[predicted_label].set_color('blue')

plt.show()