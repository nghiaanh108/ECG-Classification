# try:
#     from PIL import Image
# except ImportError:
#     import Image
# import numpy as np
# import cv2
# img = Image.open("Ffig_3.png")
# ds = 1 + np.random.randint(0,4) * 5
# original_size = img.size
# img = img.resize((img.size[0] / ds, img.size[1] / ds), resample=Image.NEAREST)

# import scipy
# import scipy.signal as sig
# rr = [1.0, 1.0, 0.5, 1.5, 1.0, 1.0] # rr time in seconds
# fs = 8000.0 # sampling rate
# pqrst = sig.wavelets.daub(10) # just to simulate a signal, whatever
# ecg = scipy.concatenate([sig.resample(pqrst, int(r*fs)) for r in rr])
# t = scipy.arange(len(ecg))/fs
# pylab.plot(t, ecg)
# pylab.show()

# from scipy import signal
# x = np.linspace(0, 10, 20, endpoint=False)
# y = np.cos(-x**2/6.0)
# f = signal.resample(y, 100)
# xnew = np.linspace(0, 10, 100, endpoint=False)

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("C:/Users/ASUS/Desktop/New folder/1.jpg")
# crop_img = img[y:y+h, x:x+w]
plt.show(img)
# cv2.waitKey(0)
