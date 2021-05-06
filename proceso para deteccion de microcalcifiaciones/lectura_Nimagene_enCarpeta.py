import glob
import cv2 as cv
import matplotlib.pyplot as plt

path = glob.glob("../imgs/mamografias 1/*.jpg")

for img in path:
    Nmamo = cv.imread(img)
    plt.imshow(Nmamo)
    plt.show()