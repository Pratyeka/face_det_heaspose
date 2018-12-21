import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import argparse
import ast
from api import PRN

from utils.key_point import DlibAlign
from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
from utils.transform import transform_test

ANGULAR = 180 / np.pi
ALIGN_SIZE = 256

def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib, is_faceboxes = args.isFaceBoxes)

    # ---- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, c] = image.shape
        if c>3: image = image[:,:,:3]  # RGBA图中，去除A通道

        # the core: regress position map
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size> 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            pos = prn.process(image) # use dlib to detect face
        elif args.isFaceBoxes:
            pos, cropped_img = prn.process(image)  # use faceboxes to detect face
        else:
            if image.shape[0] == image.shape[1]:
                image = resize(image, (256,256))
                pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
            else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                pos = prn.process(image, box)
        image = image/255.
        if pos is None: continue

        if args.is3d or args.isMat or args.isPose or args.isShow:
            # 3D vertices
            vertices = prn.get_vertices(pos)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:,1] = h - 1 - save_vertices[:,1]

        # 三维人脸旋转对齐方法
        # if args.isImage:
        #     vertices = prn.get_vertices(pos)
        #     scale_init = 180 / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
        #     colors = prn.get_colors(image, vertices)
        #     triangles = prn.triangles
        #     camera_matrix, pose = estimate_pose(vertices)
        #     yaw, pitch, roll = pos * ANGULAR
        #     vertices1 = vertices - np.mean(vertices, 0)[np.newaxis, :]
        #
        #     obj = {'s': scale_init, 'angles': [-pitch, yaw, -roll + 180], 't': [0, 0, 0]}
        #     camera = {'eye':[0, 0, 256], 'proj_type':'perspective', 'at':[0, 0, 0],
        #               'near': 1000, 'far':-100, 'fovy':30, 'up':[0,1,0]}
        #
        #     image1 = transform_test(vertices1, obj, camera, triangles, colors, h=256, w=256) * 255
        #     image1 = image1.astype(np.uint8)
        #     imsave(os.path.join(save_folder, name + '.jpg'), image1)

        if args.is3d:
            # corresponding colors
            colors = prn.get_colors(image, vertices)

            if args.isTexture:
                if args.texture_size != 256:
                    pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
                else:
                    pos_interpolated = pos.copy()
                texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
                if args.isMask:
                    vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                    uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                    uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
                    texture = texture*uv_mask[:,:,np.newaxis]
                write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
            else:
                write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        if args.isDepth:
            depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
            depth = get_depth_image(vertices, prn.triangles, h, w)
            imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)
            sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth':depth})

        if args.isMat:
            sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

        if args.isKpt:
            # get landmarks
            kpt = prn.get_landmarks(pos)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

        if args.is2dKpt and args.is68Align:
            ori_kpt = prn.get_landmarks_2d(pos)
            dlib_aligner = DlibAlign()
            dst_img = dlib_aligner.dlib_68_align(image,ori_kpt, 256, 0.5)
            imsave(os.path.join(save_folder, name + '.jpg'), dst_img)

        if args.isPose:
            # estimate pose
            camera_matrix, pose, rot = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), np.array(pose) * ANGULAR)
            np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix)

        if args.isShow:
            kpt = prn.get_landmarks(pos)
            cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            # cv2.imshow('dense alignment', plot_vertices(image, vertices))
            # cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/adas/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=False, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--isFaceBoxes', default=True, type=ast.literal_eval,
                        help='whether to use faceboxes for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=False, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--is2dKpt', default=True, type=ast.literal_eval,
                        help='whether to save alignmented img(.jpg)')
    parser.add_argument('--is68Align', default=True, type=ast.literal_eval,
                        help='use the 68 2D point to align face')
    parser.add_argument('--isPose', default=True, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=True, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=True, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=False, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    main(parser.parse_args())
