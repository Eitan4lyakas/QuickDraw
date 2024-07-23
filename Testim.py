import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread("QuickDrawData\\cat\\1.png")
imgConv = cv2.resize(img, (49, 49))

img_gray = cv2.cvtColor(imgConv, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(imgConv, cv2.COLOR_BGR2RGB)
   
# Creates the environment 
# of the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()