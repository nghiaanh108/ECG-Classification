import cv2
import numpy as np
import os

# Khai báo 
X_train = []
y_train = []
X_test = []
y_test = []

width, height = 128, 128

data_folder = "ecg_img"
folder_test = "ecg_img/test"
folder_train = "ecg_img/train"

# Chạy vào trong mỗi thư mục lấy ảnh với 128,128
for folder_tra in os.listdir(folder_train):
    curr_path_train = os.path.join(folder_train, folder_tra)
    for file_train in os.listdir(curr_path_train):
        curr_path_train2 = os.path.join(curr_path_train, file_train)
        for file_train2 in os.listdir(curr_path_train2):
            curr_file1= os.path.join(curr_path_train2, file_train2)
            images1 = cv2.imread(curr_file1)
            new_images1 = cv2.resize(images1,(width,height))
            X_train.append(new_images1)
            y_train.append(folder_tra)
            
for folder_tes in os.listdir(folder_test):
    curr_path1 = os.path.join(folder_test, folder_tes)
    for file1 in os.listdir(curr_path1):
        curr_path2 = os.path.join(curr_path1, file1)
        for file2 in os.listdir(curr_path2):
            curr_file2 = os.path.join(curr_path2, file2)
            images2 = cv2.imread(curr_file2)
            new_images2 = cv2.resize(images2,(width,height))
            X_test.append(new_images2)
            y_test.append(folder_tes)   
            
# Tạo mảng
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Lưu vào file .npy để sau load lại khỏi for lại tốn time
np.save("X_train_image", X_train)
np.save("y_train_image", y_train)
np.save("X_test_image", X_test)
np.save("y_test_image", y_test)

# print("X_train-befor:",X_train.shape)    
## load ra thôi (Thử)
# X_train = np.load("X_train_image.npy")
# y_train = np.load("y_train_image.npy")
# X_test = np.load("X_test_image.npy")
# y_test = np.load("y_test_image.npy")

# print(X_train.dtype)
# print("X_train-after:",X_train.shape)    

# encoder = LabelBinarizer()
# y_train =encoder.fit_transform(y_train)
# y_test =encoder.fit_transform(y_test)

# print(y_train.shape)        

# for (i,lab) in enumerate(encoder.classes_):
    # print("{}.{}".format(i+1,lab))