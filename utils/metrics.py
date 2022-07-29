import math
import numpy as np
from shapely.geometry import Polygon

def cal_corner_after_rotation(corner, center, r):
        x1, y1 = corner
        x0, y0 = center
        x2 = math.cos(r) * (x1 - x0) - math.sin(r) * (y1 - y0) + x0
        y2 = math.sin(r) * (x1 - x0) + math.cos(r) * (y1 - y0) + y0
        return x2, y2

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

def cal_distance(box1, box2):
    """
    box: [x, y, z, w, h, l, r] center(x, y, z)
    """
    x1, y1, z1, w1, l1, h1, r1 = box1
    x2, y2, z2, w2, l2, h2, r2 = box2
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return dist

def cal_accuracy(pred_bboxes, gt_bboxes):
    total = 0
    tp25 = 0
    tp50 = 0
    miou = 0
    for i in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[i]
        p_bbox = pred_bboxes[i]        
        bbox = p_bbox[:7]
        iou = cal_iou3d(bbox, gt_bbox)

        if iou >= 0.5:
            tp25 += 1
            tp50 += 1
        elif iou >= 0.25:
            tp25 += 1
        total += 1
        miou += iou

    acc25 = round(tp25/total, 4)
    acc50 = round(tp50/total, 4)
    miou = round(miou/total, 4)
    return acc25, acc50, miou

