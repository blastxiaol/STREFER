import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points
from scipy.spatial import Delaunay

def load_axis_align_matrix(meta_file):
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix=np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix

def transform_boxes(box):
    '''
    Args: 
        box: 3D box with center [cx, cy, cz, l, w, h, yaw]
    Return: 
        box3d [8, 3]
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    
    cx, cy, cz, l, w, h = box[:6]
    x0 = cx + l/2
    y0 = cy - w/2
    z0 = cz + h/2
    
    x1 = cx + l/2
    y1 = cy + w/2
    z1 = cz + h/2
    
    x2 = cx - l/2
    y2 = cy + w/2
    z2 = cz + h/2
    
    x3 = cx - l/2
    y3 = cy - w/2
    z3 = cz + h/2
    
    x4 = cx + l/2
    y4 = cy - w/2
    z4 = cz - h/2
    
    x5 = cx + l/2
    y5 = cy + w/2
    z5 = cz - h/2
    
    x6 = cx - l/2
    y6 = cy + w/2
    z6 = cz - h/2
    
    x7 = cx - l/2
    y7 = cy - w/2
    z7 = cz - h/2
    
    corner3d = np.array([[x0, y0, z0],
                         [x1, y1, z1],
                         [x2, y2, z2],
                         [x3, y3, z3],
                         [x4, y4, z4],
                         [x5, y5, z5],
                         [x6, y6, z6],
                         [x7, y7, z7]], dtype=np.float32)
    return corner3d

def load_image(img_filename):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    return img

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    l /= 2
    w /= 2
    h /= 2
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def batch_extract_pc_in_box3d(pc, boxes3d, sample_points_num):
    objects_pc = []
    for i in range(len(boxes3d)):
        box3d = my_compute_box_3d(boxes3d[i][0:3], boxes3d[i][3:6], boxes3d[i][6])
        obj_pc, pc_ind = extract_pc_in_box3d(pc.copy(), box3d)
        if obj_pc.shape[0] == 0:
            obj_pc = np.zeros((sample_points_num, 3))
        else:
            obj_pc = random_sampling(obj_pc, sample_points_num)
        objects_pc.append(obj_pc)
    objects_pc = np.stack(objects_pc, axis=0)
    return objects_pc

def box3d_to_2d(box3d, axis_align_matrix, cam_pose, color_intrinsic, format='center'):
    assert format in {'corners', 'center'}
    if format == 'center':
        corners3d = transform_boxes(box3d)
        
    axis_align_matrix_inv = np.linalg.inv(axis_align_matrix)
    corners3d = np.dot(axis_align_matrix_inv, 
                    np.concatenate([corners3d, np.ones([corners3d.shape[0], 1])], axis=1).T).T
    cam_pose_inv = np.linalg.inv(cam_pose)
    corners3d = np.dot(cam_pose_inv, corners3d.T).T
    corners3d[:,0] /= corners3d[:,2]
    corners3d[:,1] /= corners3d[:,2]
    corners3d[:,2] = 1

    corners2d = np.dot(color_intrinsic, corners3d.T).T[:, :2]
    return corners2d.astype(int), corners3d

def batch_compute_box_3d(objects, axis_align_matrix, cam_pose, color_intrinsic, bias=None):
    if bias is not None:
        objects[:, :3] += bias
    corners2d = []
    corners3d = []
    for obj in objects:
        corner_2d, corner_3d = box3d_to_2d(obj, axis_align_matrix, cam_pose, color_intrinsic)
        corners2d.append(corner_2d)
        corners3d.append(corner_3d)
    corners2d = np.stack(corners2d, axis=0)
    corners3d = np.stack(corners3d, axis=0)
    if bias is not None:
        objects[:, :3] -= bias
    return corners2d, corners3d

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image