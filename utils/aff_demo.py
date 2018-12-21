from skimage.transform import estimate_transform,warp
import numpy as np
from skimage.io import imread,imshow
import cv2
import dlib
import face_recognition_models

# ori_pts = np.random.rand(68,2)
# dst_pts = np.random.rand(68,2)
#
# img = imread(r'D:\Job\PyJect\Face\PRNet\TestImages\AFLW2000\0.jpg')
# tform = estimate_transform('similarity', ori_pts, dst_pts)
# cv2.imshow('ss', img)
# cv2.waitKey(0)
# img = img/255.
# cropped_image = warp(img, tform.inverse, output_shape=(256,256))
# cv2.imshow('s', cropped_image)
# cv2.waitKey(0)
# predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
# pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
#
# print(pose_predictor_68_point)

img_size = 256

mean_face_shape_x = np.array([0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689]) * img_size

mean_face_shape_y = np.array([0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182]) * img_size

for point in zip(mean_face_shape_x, mean_face_shape_y):
	print(point)