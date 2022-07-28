import numpy as np
from copy import deepcopy
import math
from shapely.geometry import Polygon

ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
in_matrix = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])

def filter_bbox(bboxes):
    if bboxes.shape[0] == 0:
        return bboxes, []
    points = deepcopy(bboxes[:, :4])
    points_T = np.transpose(points)
    points_T[3, :] = 1.0

    # lidar2camera
    points_T_camera = np.dot(ex_matrix, points_T)

    # camera2pixel
    pixel = np.dot(in_matrix, points_T_camera).T
    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
    pixel_xy = np.around(pixel_xy).astype(int)

    # index_list = []
    # for i in range(pixel_xy.shape[0]):
    #     if pixel_xy[i][0] >= 0 and \
    #         pixel_xy[i][0] <= 1280 \
    #         and pixel_xy[i][1] >= 0 \
    #         and pixel_xy[i][1] <=720 \
    #         and points_T_camera[2, i] > 0:
    #         index_list.append(i)
    index_list = ((pixel_xy[:, 0] >= 0)
                        & (pixel_xy[:, 0] <= 1280)
                        & (pixel_xy[:, 1] >= 0)
                        & (pixel_xy[:, 1] <= 720)
                        & (points_T_camera[2, :] > 0))
    bboxes = bboxes[index_list]
    return bboxes, index_list

def cal_corner_after_rotation(corner, center, r):
    x1, y1 = corner
    x0, y0 = center
    x2 = math.cos(r) * (x1 - x0) - math.sin(r) * (y1 - y0) + x0
    y2 = math.sin(r) * (x1 - x0) + math.cos(r) * (y1 - y0) + y0
    return x2, y2

def cal_inter_area(box1, box2):
    """
    box: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    a=np.array(box1).reshape(4, 2)   #四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 
    
    b=np.array(box2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    
    union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
    if not poly1.intersects(poly2): #如果两四边形不相交
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return poly1.area, poly2.area, inter_area

def eight_points(center, size, rotation=0):
    x, y, z = center
    w, l, h = size
    w = w/2
    l = l/2
    h = h/2

    x1, y1, z1 = x-w, y-l, z+h
    x2, y2, z2 = x+w, y-l, z+h
    x3, y3, z3 = x+w, y-l, z-h
    x4, y4, z4 = x-w, y-l, z-h
    x5, y5, z5 = x-w, y+l, z+h
    x6, y6, z6 = x+w, y+l, z+h
    x7, y7, z7 = x+w, y+l, z-h
    x8, y8, z8 = x-w, y+l, z-h

    if rotation != 0:
        x1, y1 = cal_corner_after_rotation(corner=(x1, y1), center=(x, y), r=rotation)
        x2, y2 = cal_corner_after_rotation(corner=(x2, y2), center=(x, y), r=rotation)
        x3, y3 = cal_corner_after_rotation(corner=(x3, y3), center=(x, y), r=rotation)
        x4, y4 = cal_corner_after_rotation(corner=(x4, y4), center=(x, y), r=rotation)
        x5, y5 = cal_corner_after_rotation(corner=(x5, y5), center=(x, y), r=rotation)
        x6, y6 = cal_corner_after_rotation(corner=(x6, y6), center=(x, y), r=rotation)
        x7, y7 = cal_corner_after_rotation(corner=(x7, y7), center=(x, y), r=rotation)
        x8, y8 = cal_corner_after_rotation(corner=(x8, y8), center=(x, y), r=rotation)

    conern1 = np.array([x1, y1, z1])
    conern2 = np.array([x2, y2, z2])
    conern3 = np.array([x3, y3, z3])
    conern4 = np.array([x4, y4, z4])
    conern5 = np.array([x5, y5, z5])
    conern6 = np.array([x6, y6, z6])
    conern7 = np.array([x7, y7, z7])
    conern8 = np.array([x8, y8, z8])
    
    eight_corners = np.stack([conern1, conern2, conern6, conern5, conern4, conern3, conern7, conern8], axis=0)
    return eight_corners

def cal_iou3d(box1, box2):
    """
    box: [x, y, z, w, h, l, r] center(x, y, z)
    """
    center1 = box1[:3]
    size1 = box1[3:6]
    rotation1 = box1[6]
    eight_corners1 = eight_points(center1, size1, rotation1)
    
    center2 = box2[:3]
    size2 = box2[3:6]
    rotation2 = box2[6]
    eight_corners2 = eight_points(center2, size2, rotation2)
    
    area1, area2, inter_area = cal_inter_area(eight_corners1[:4, :2].reshape(-1), eight_corners2[:4, :2].reshape(-1))
    
    h1 = box1[5]
    z1 = box1[2]
    h2 = box2[5]
    z2 = box2[2]
    volume1 = h1 * area1
    volume2 = h2 * area2
    
    bottom1, top1 = z1 - h1/2, z1 + h1/2
    bottom2, top2 = z2 - h2/2, z2 + h2/2
    
    inter_bottom = max(bottom1, bottom2)
    inter_top = min(top1, top2)
    inter_h = inter_top - inter_bottom if inter_top > inter_bottom else 0
    
    inter_volume = inter_area * inter_h
    union_volume = volume1 + volume2 - inter_volume
    
    iou = inter_volume / union_volume
    
    return iou

def cal_each_image(gt_bboxes, pred_bboxes, confidence):
    # assert len(pred_bboxes) == len(confidence)
    tp = np.zeros(2, dtype=int)
    fp = np.zeros(2, dtype=int)
    gt_total = len(gt_bboxes)

    occupied25 = np.zeros(len(gt_bboxes), dtype=int)
    occupied50 = np.zeros(len(gt_bboxes), dtype=int)
    tp_record25 = np.zeros(len(pred_bboxes), dtype=int)
    fp_record25 = np.zeros(len(pred_bboxes), dtype=int)
    tp_record50 = np.zeros(len(pred_bboxes), dtype=int)
    fp_record50 = np.zeros(len(pred_bboxes), dtype=int)
    for i, pred_box in enumerate(pred_bboxes):
        max_iou = -1
        jmax = -1
        for j, gt_box in enumerate(gt_bboxes):
            iou = cal_iou3d(gt_box, pred_box)
            if iou > max_iou:
                max_iou = iou
                jmax = j
        if max_iou >= 0.5 and occupied50[jmax] == 0:
            tp_record50[i] = 1
            tp_record25[i] = 1
            occupied50[jmax] = 1
            occupied25[jmax] = 1
        elif max_iou >= 0.25 and occupied25[jmax] == 0:
            tp_record25[i] = 1
            occupied25[jmax] = 1
            fp_record50[i] = 1
        else:
            fp_record25[i] = 1
            fp_record50[i] = 1
    tp[0] = tp_record25.sum()
    tp[1] = tp_record50.sum()
    fp[0] = fp_record25.sum()
    fp[1] = fp_record50.sum()
    return tp, fp, gt_total

def loc_pc2img(points):
    x, y, z, w, l, h, r = points
    center = [x, y, z]
    size = [w, l, h]
    points = eight_points(center, size, r)
    points = np.insert(points, 3, values=1, axis=1)
    points_T = np.transpose(points)
    points_T[3, :] = 1.0

    # lidar2camera
    points_T_camera = np.dot(ex_matrix, points_T)
    # camera2pixel
    pixel = np.dot(in_matrix, points_T_camera).T
    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
    pixel_xy = np.around(pixel_xy).astype(int)
    pixel_xy = pixel_xy.T
    return pixel_xy

def point_cloud_position2image_position(points):
    corner_2d = []
    for p in points:
        p = p[:7]
        new_p = loc_pc2img(p)
        corner_2d.append(new_p)
    corner_2d = np.stack(corner_2d, axis=0)
    return corner_2d