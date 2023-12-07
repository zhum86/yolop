'''
script to test if cv2 gui is working correctly in docker
'''
import cv2

im = cv2.imread('inference/images/0ace96c3-48481887.jpg')
cv2.imshow("", im)
cv2.waitKey()