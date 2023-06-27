import os

HOME = os.getcwd()
HOME

INPUTS = '../inputs/outputs'
PRODUCTS = '../inputs/products'
INPUTS

sample1 = INPUTS+'/3.jpg'
sample2 = INPUTS+'/img1.jpeg'
sample3 = INPUTS+'/test1.jpg'

sample_paint = PRODUCTS+'Paints/LightBlue.jpg'
sample_stone = PRODUCTS+'/Stone/AdalyaArmani.jpg'
sample_wallpaper = PRODUCTS+'/Wallpapers/3_flower wallpaper anthea by Parato texture-seamless.jpg'

import cv2
import numpy as np
import matplotlib.pyplot as plt

def click_event(event, x, y, flags, params):
    points = []
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image1_copy, (x, y), 4, (0, 0, 255), -1)
        points.append([x, y])
        print(points)
        if len(points) < 4:
            cv2.imshow('image', image1_copy)
            cv2.waitKey(0)
        else:
            cv2.destroyAllWindows()
    return points

def sort_pts(points):
    sorted_pts = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    sorted_pts[0] = points[np.argmin(s)]
    sorted_pts[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    sorted_pts[1] = points[np.argmin(diff)]
    sorted_pts[3] = points[np.argmax(diff)]

    return sorted_pts


image1 = cv2.imread(sample1)
image1_copy = image1.copy()
subject_image = cv2.imread(sample_wallpaper)

cv2.imshow('image', image1_copy)

# cv2.setMouseCallback('image', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


s1_points = [ [331, 6],
             [1466, 520],
             [337, 331],
             [6, 407] 
             ]

print(s1_points)

sorted_pts = sort_pts(s1_points)
print(sorted_pts)