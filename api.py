import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
import cv2

from predictor import PosPrediction

CENTER = np.array([320, 240])

def find_face(boxes, scores):
    box_cens = np.array([[(box[0]+box[2])/2, (box[1]+box[3])/2] for box in boxes])
    dis = [np.sum(np.abs(CENTER-cen)) for cen in box_cens]
    face_index = dis.index(min(dis))
    return (boxes[face_index],), (scores[face_index],)

def draw_rect_on_image(image, boxes, scores):
    for b,s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    return image

class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, is_faceboxes = True, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load detectors
        if is_dlib:
            import dlib
            self.isDilb = True
            detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)
        if is_faceboxes:
            from faceboxes.face_detector import FaceDetector
            self.isFaceBoxes = True
            MODEL_PATH = './faceboxes/model/adas_model.pb'
            self.face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        # uv file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32)     # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32)   # ntri x 3
        
        self.uv_coords = self.generate_uv_coords()        

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def dlib_detect(self, image):
        return self.face_detector(image, 1)

    def faceboxes_detect(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, scores = self.face_detector(image, score_threshold=0.3)
        # 按照离中心点最近，输入检测框
        return find_face(boxes, scores)

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

    def process(self, input, image_info = None):
        ''' process image with crop operation.
        Args:
            input: (h,w,3) array or str(image path). image value range:1~255. 
            image_info(optional): the bounding box information of faces. if None, will use dlib to detect face. 

        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4: # key points to get bounding box
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :])
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
            else:  # bounding box
                bbox = image_info
                left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)   # 1.6倍的人脸区域
        elif self.isFaceBoxes:
            boxes, scores = self.faceboxes_detect(image)
            if len(scores) == 0:
                print('warning: no detected face')
                return None
            # cv2.imshow('face detection', draw_rect_on_image(image, boxes, scores))
            # cv2.waitKey(1)

            d = boxes[0]
            left = d[1]; right = d[3]; top = d[0]; bottom = d[2]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)
        elif self.isDilb:
            detected_faces = self.dlib_detect(image)
            if len(detected_faces) == 0:
                print('warning: no detected face')
                return None

            d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
            left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
            size = int(old_size*1.58)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2],
                            [center[0]-size/2, center[1]+size/2],
                            [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        # print(src_pts)
        # print('---------------------')
        # print(DST_PTS)
        tform = estimate_transform('similarity', src_pts, DST_PTS)    # 人脸的检测框到256256框的相似变换矩阵
        
        image = image/255.
        # 对输入图像按照相似变换从输入图像中的到RPNet模型需要的输入
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # run our net
        # st = time()
        cropped_pos = self.net_forward(cropped_image)
        # print 'net time:', time() - st

        # restore 
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])

        # 同时返回检测到的pos参数以及对应的人脸区域
        return pos, cropped_image*255
            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt

    def get_landmarks_2d(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 2D landmarks. shape = (68, 2).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :2]
        return kpt

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors








