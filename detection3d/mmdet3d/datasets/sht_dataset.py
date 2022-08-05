from pickle import TRUE
import mmcv
import copy
import numpy as np
from numpy.core.numeric import tensordot
import pandas as pd
import tempfile
import torch
from os import path as osp
from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .pipelines import Compose
import math
from shapely.geometry import Polygon
from copy import deepcopy

ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
in_matrix = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])

def filter_bbox(bboxes):
    points = deepcopy(bboxes[:, :4])
    points_T = np.transpose(points)
    points_T[3, :] = 1.0

    # lidar2camera
    points_T_camera = np.dot(ex_matrix, points_T)

    # camera2pixel
    pixel = np.dot(in_matrix, points_T_camera).T
    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
    pixel_xy = np.around(pixel_xy).astype(int)

    index_list = []
    for i in range(pixel_xy.shape[0]):
        if pixel_xy[i][0] >= 0 and \
            pixel_xy[i][0] <= 1280 \
            and pixel_xy[i][1] >= 0 \
            and pixel_xy[i][1] <=720 \
            and points_T_camera[2, i] > 0:
            index_list.append(i)
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
    
    return pixel_xy.T

def point_cloud_position2image_position(points):
    corner_2d = []
    for p in points:
        p = p[:7]
        new_p = loc_pc2img(p)
        corner_2d.append(new_p)
    corner_2d = np.stack(corner_2d, axis=0)
    return corner_2d

@DATASETS.register_module()
class SHtechDataset(Custom3DDataset):
    def __init__(self,
                data_root,
                ann_file,
                pipeline=None,
                classes=None,
                modality=None,
                box_type_3d='LiDAR',
                filter_empty_gt=True,
                test_mode=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        # self.info = self.data_infos['person']
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(ann_file, file_format='pkl')#[:10]

    # def get_data_info(self,index):
    #     info = self.data_infos[index]
    #     sample_idx = info['image']['image_idx']
    #     img_filename = info['image']['image_path']
    #     pts_filename = info['point_cloud']['point_cloud_path']      #.replace('.bin','.pcd')

    #     # pts_filename = pts_filename.split('/')
    #     # pts_filename.insert(3, 'SHTpersonGrounding')
    #     # pts_filename[7] = 'bin_painting_seg_feature'

    #     # pts_filename[6] = 'bin_painting_seg_feature'
    #     # pts_filename = '/'.join(pts_filename)
    #     # print("the wrong file",pts_filename)

    #     r = info['calib_info']['R'].astype(np.float32)
    #     t = info['calib_info']['T'].astype(np.float32)
    #     lidar2img = np.vstack([np.hstack([r, t]), np.array([0, 0, 0, 1])]).astype(np.float32)
    #     input_dict = dict(
    #         sample_idx=sample_idx,
    #         pts_filename=pts_filename,
    #         img_prefix=None,
    #         img_info=dict(filename=img_filename),
    #         lidar2img=lidar2img)
    #     if not self.test_mode:
    #         annos = self.get_ann_info(index)
    #         input_dict['ann_info'] = annos
    #     # print("the input dict is ",input_dict)
    #     return input_dict
  
    # def get_ann_info(self, index):
    #     annos = self.data_infos[index]['annos']
    #     # we need other objects to avoid collision when sample
    #     loc = np.array(annos['position'])
    #     dims = np.array(annos['dimensions'])
    #     rots = np.array(annos['rotation'])
    #     # gt_names = annos['name']
    #     gt_bboxes = annos['image_bbox']
        
    #     # print("when bugs",loc.shape,dims.shape,rots[..., np.newaxis].shape)
    #     ''' notion: modify x/y because the wried transform after LiDARInstance3DBoxes'''
    #     # gt_bboxes_3d = np.concatenate([loc[:,1].reshape(-1,1), -loc[:,0].reshape(-1,1),loc[:,2].reshape(-1,1),dims, rots[..., np.newaxis]],
    #     #                               axis=1).astype(np.float32)
    #     gt_bboxes_3d = np.concatenate([loc[:,0].reshape(-1,1), loc[:,1].reshape(-1,1),loc[:,2].reshape(-1,1),dims, rots[..., np.newaxis]],
    #                                   axis=1).astype(np.float32)


    #     # gt_bboxes_3d, index_list = filter_bbox(gt_bboxes_3d)
    #     gt_bboxes_3d = np.insert(gt_bboxes_3d, 7, values=0, axis=1)
    #     gt_bboxes_3d = np.insert(gt_bboxes_3d, 8, values=0, axis=1)

    #     # print(loc.shape,dims.shape,rots[..., np.newaxis].shape,gt_bboxes_3d.shape)
    #     # since only one class, the label all 0       
    #     gt_bboxes_3d = LiDARInstance3DBoxes(
    #         gt_bboxes_3d,
    #         box_dim=gt_bboxes_3d.shape[-1],
    #         origin=(0, 0, 0)).convert_to(self.box_mode_3d)   
    #     # assert(np.abs(gt_bboxes_3d.tensor.numpy()[0,0] - loc[0,0]) < 1e-4
    #     # gt_bboxes_3d = Variable(torch.to_tensor(gt_bboxes_3d))
    #     gt_labels = np.array([0 for i in range(len(rots))]).astype(np.int64) # notice modify into 0

    #     # gt_labels = gt_labels[index_list]

    #     gt_labels_3d = copy.deepcopy(gt_labels)

    #     anns_results = dict(
    #         gt_bboxes_3d=gt_bboxes_3d,
    #         gt_labels_3d=gt_labels_3d,
    #         bboxes=gt_bboxes,
    #         labels=gt_labels,
    #         gt_names='person')

    #     return anns_results

    
    def get_data_info(self,index):
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = info['image']['image_path']
        # pts_filename = info['point_cloud']['point_cloud_path'].replace('/bin_v1','/bin_painting_seg_feature')
        # pts_prev_filename = info['annos']['tracking']['prev_name'].replace('/pcd','/bin_painting_seg_feature')
        # pts_next_filename = info['annos']['tracking']['next_name'].replace('/pcd','/bin_painting_seg_feature')
        pts_filename = info['point_cloud']['point_cloud_path']
        pts_prev_filename = info['annos']['tracking']['prev_name'].replace('/pcd','/bin_v1')
        pts_next_filename = info['annos']['tracking']['next_name'].replace('/pcd','/bin_v1')
    
        # print("the wrong file",pts_filename)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            pts_prev_filename = pts_prev_filename,
            pts_next_filename = pts_next_filename)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        # print("the input dict is ",input_dict)
        return input_dict
    
    def get_ann_info(self, index):
        annos = self.data_infos[index]['annos']
        
        # we need other objects to avoid collision when sample
        loc = np.array(annos['position'])
        dims = np.array(annos['dimensions'])
        rots = np.array(annos['rotation'])
        velocity = np.array(annos['tracking']['velocity'])
        # gt_names = annos['name']
        gt_bboxes = annos['image_bbox']
        # anno_head_list = np.array(annos['anno_head'])
        # anno_body_list = np.array(annos['anno_body'])
        # anno_leg_list = np.array(annos['anno_leg'])
        anno_zcenter_list = np.array(annos['anno_zcenter']).reshape(-1,1)
        # print("when bugs",loc.shape,dims.shape,rots[..., np.newaxis].shape)
        ''' notion: modify x/y because the wried transform after LiDARInstance3DBoxes'''
        # gt_bboxes_3d = np.concatenate([loc[:,1].reshape(-1,1), -loc[:,0].reshape(-1,1),loc[:,2].reshape(-1,1),dims, rots[..., np.newaxis]],
        #                               axis=1).astype(np.float32)
        gt_bboxes_3d = np.concatenate([loc[:,0].reshape(-1,1), loc[:,1].reshape(-1,1),loc[:,2].reshape(-1,1),dims, rots[..., np.newaxis],velocity.reshape(-1,2)],
                                      axis=1).astype(np.float32)
                        
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, anno_zcenter_list],axis=1)  # [N, 9 + 1]

        gt_labels = np.array([0 for i in range(len(rots))]).astype(np.int64) # notice modify into 0
        gt_labels_3d = copy.deepcopy(gt_labels)

        if index != 0:
            annos_prev = self.data_infos[index-1]['annos']
            loc = np.array(annos_prev['position'])
            dims = np.array(annos_prev['dimensions'])
            rots = np.array(annos_prev['rotation'])
            velocity = np.array(annos_prev['tracking']['velocity'])
            anno_zcenter_prev = np.array(annos_prev['anno_zcenter']).reshape(-1,1)
            gt_bboxes_3d_prev = np.concatenate([loc[:,0].reshape(-1,1), loc[:,1].reshape(-1,1),loc[:,2].reshape(-1,1),dims,rots[..., np.newaxis],velocity.reshape(-1,2),anno_zcenter_prev],axis=1)
            # N*17 0~9,10~16 prev
            # print(gt_bboxes_3d.shape,gt_bboxes_3d_prev.shape)
            shape_diff = abs(gt_bboxes_3d.shape[0] - gt_bboxes_3d_prev.shape[0])
            if gt_bboxes_3d.shape>gt_bboxes_3d_prev.shape:
                gt_bboxes_3d_prev = np.concatenate([gt_bboxes_3d_prev,np.zeros([shape_diff,gt_bboxes_3d.shape[1]])],0)
                gt_bboxes_3d_prev[gt_bboxes_3d_prev.shape[0]:,:] = -1000
            if gt_bboxes_3d.shape<gt_bboxes_3d_prev.shape:
                gt_bboxes_3d_prev = gt_bboxes_3d_prev[:gt_bboxes_3d.shape[0],:]
            # 目前遇到prev和current 大小不匹配问题。
            # print(gt_bboxes_3d.shape,gt_bboxes_3d_prev.shape)
            assert gt_bboxes_3d.shape == gt_bboxes_3d_prev.shape

            gt_bboxes_3d = np.concatenate([gt_bboxes_3d,gt_bboxes_3d_prev],axis=1)
        else:
            # print(gt_bboxes_3d.shape,gt_bboxes_3d[0:10,:].shape)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d,gt_bboxes_3d],axis=1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0, 0, 0)).convert_to(self.box_mode_3d)   
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names='person')

        return anns_results

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix="detection_result.pkl",
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        from tqdm import tqdm
        import os
        GENERATE = True
        if GENERATE:
            base_path = "../data/pred_bboxes"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            for i in tqdm(range(len(results))):
                pred_bboxes = results[i]['pts_bbox']['boxes_3d'].tensor.numpy()
                info = self.data_infos[i]
                pts_filename = info['point_cloud']['point_cloud_path']
                date, scene_id, _, pts_name = pts_filename.split('/')[-4:]
                scene_path = os.path.join(base_path, date, scene_id)
                if not os.path.exists(scene_path):
                    os.makedirs(scene_path)
                file_path = os.path.join(scene_path, f"{pts_name[:-4]}.npy")

                centers = pred_bboxes[:, :3].copy()
                if centers.shape[0] > 0:
                    points = np.insert(centers, 3, values=1, axis=1)
                    points_T = np.transpose(points)
                    points_T[3, :] = 1.0
                    # lidar2camera
                    points_T_camera = np.dot(ex_matrix, points_T)
                    # camera2pixel
                    pixel = np.dot(in_matrix, points_T_camera).T
                    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
                    pixel_xy = np.around(pixel_xy).astype(int)

                    filter_pred_bboxes = []
                    for k in range(pixel_xy.shape[0]):
                        x, y = pixel_xy[k]
                        if 0 <= x < 1280 and 0 <= y < 720:
                            filter_pred_bboxes.append(pred_bboxes[k])
                    if len(filter_pred_bboxes) > 0:
                        filter_pred_bboxes = np.vstack(filter_pred_bboxes)
                    else:
                        filter_pred_bboxes = np.zeros((0, 9), dtype=np.float32)
                else:
                    filter_pred_bboxes = np.zeros((0, 9), dtype=np.float32)
                np.save(file_path, filter_pred_bboxes)
            exit()

        detection_results = []
        tp = np.zeros(2, dtype=int)
        fp = np.zeros(2, dtype=int)
        gt_total = 0
        for i in tqdm(range(len(results))):
            gt_info = self.data_infos[i]
            pts_filename = gt_info['point_cloud']['point_cloud_path']      #.replace('.bin','.pcd')
            pts_filename = pts_filename.split('/')[-1][:-4]

            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            pred_bboxes = results[i]['pts_bbox']['boxes_3d'].tensor.numpy()
            confidence = results[i]['pts_bbox']['scores_3d'].numpy()

            data_info = dict(pts_filename=pts_filename, gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, confidence=confidence)
            detection_results.append(data_info)
            each_tp, each_fp, each_gt_total = cal_each_image(gt_bboxes, pred_bboxes, confidence)
            tp += each_tp
            fp += each_fp
            gt_total += each_gt_total
        
        precision = tp / (tp + fp)
        recall = tp / gt_total
        ap25, ap50 = precision
        recall25, recall50 = recall
        ret_dict = dict(recall25=recall25, recall50=recall50, ap25=ap25, ap50=ap50)
        
        return ret_dict

    def format_results(self,results,gt_annos,
                    pklfile_prefix="result-one.pkl",
                    empty=False):
        pklfile_path = self.data_root + 'result/' + pklfile_prefix
        det_annos = dict()
        if 'pts_bbox' in results[0].keys():
            det_annos['predict'] = [result['pts_bbox']['boxes_3d'].tensor.numpy() for result in results]
            det_annos['score'] = [result['pts_bbox']['scores_3d'] for result in results]
        else:
            det_annos['predict'] = [result['boxes_3d'].tensor.numpy() for result in results]
            det_annos['score'] = [result['scores_3d'] for result in results]

       
        det_annos['path'] = [info['point_cloud']['point_cloud_path'] for info in self.data_infos]
        number_frame = len(det_annos['path'])
        #--------------------
        #-----check the empty and drop 
        empty = True
        if empty:
            det_annos['no-empty-predict'] = [[] for _ in range(number_frame) ]
            det_annos['no-empty-score'] = [[] for _ in range(number_frame) ]
            
            '''crop the no-empty box'''
            # import open3d as o3d
            import warnings
            warnings.filterwarnings("ignore")
            for i in range(len(det_annos['path'])):
                pcd_path = det_annos['path'][i].replace('pcd','bin_v1')
                points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1,4)
                # pcd =  o3d.io.read_point_cloud(det_annos['path'][i].replace('bin','pcd'))
                # points = np.asarray(pcd.points)
                center = det_annos['predict'][i][:,0:3] #number of pred * 3
                dim = det_annos['predict'][i][:,3:6]
                for j in range(center.shape[0]):
                    ind = np.where(
                        (points[:,0] > center[j,0]-dim[j,0]/2) & (points[:,0] < center[j,0]+dim[j,0]/2) &
                        (points[:,1] > center[j,1]-dim[j,1]/2) & (points[:,1] < center[j,1]+dim[j,1]/2) &
                        (points[:,2] > center[j,2]-dim[j,2]/2) & (points[:,2] < center[j,2]+dim[j,2]/2) )
                    # if i == 10:
                    #     print(np.average(points[ind],0)[2])
                    if ind[0].shape[0] > 3 and np.average(points[ind],0)[2] > -5:
                        # print(det_annos['predict'][i][j].reshape(1,7))
                        pred = det_annos['predict'][i][j]
                        score = det_annos['score'][i][j]
                        det_annos['no-empty-predict'][i].append(pred)
                        det_annos['no-empty-score'][i].append(score)

                # det_annos['no-empty-predict'][i] =  np.asarray(det_annos['no-empty-predict'][i]).reshape(-1,7)
                # det_annos['no-empty-score'][i] =  np.asarray(det_annos['no-empty-score'][i]).reshape(-1,1)
                # print(i,det_annos['no-empty-predict'][i].shape,det_annos['no-empty-score'][i].shape)
        det_annos['annos'] = gt_annos
        det_annos['annos_box'] = [self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy() for i in range(len(gt_annos)) ]
        import pickle
        with open(pklfile_path, 'wb') as f:
            pickle.dump(det_annos,f)
        # mmcv.dump(det_annos, pklfile_path)
        print(f'Saving submission to {pklfile_path}')
        return det_annos

#### using IOU method
    def evaluate2(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix="result-multiloss.pkl",
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):

        generate = False
        if generate:
            import os
            import pickle
            from tqdm import tqdm
            from mmdet.models.builder import build_roi_extractor

            file_list = os.listdir("_detection_info_pc")

            xmin, ymin, xmax, ymax = [0, -20.48, 30.72, 20.48]
            dim = 32
            roi_extractor = build_roi_extractor(dict(type='MySingleRoIExtractor',
                                                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                                out_channels=256,
                                                featmap_strides=[1]))
            img_extractor = build_roi_extractor(dict(type='MySingleRoIExtractor',
                                                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                                out_channels=256,
                                                featmap_strides=[32]))
            img_list = os.listdir("/remote-home/linzhx/Projects/mmdetection/_sht_feature/info")
            img_list = [x[:-4] for x in img_list]
            for i in tqdm(range(len(results))):
                gt_info = self.data_infos[i]
                pts_filename = gt_info['point_cloud']['point_cloud_path']      #.replace('.bin','.pcd')
                pts_filename = pts_filename.split('/')[-1][:-4]

                img_filename = gt_info['image']['image_path'].split('/')[-1][:-4]
                if img_filename not in img_list:
                    print(f"{img_filename} not in img_list")
                    continue

                img_info_path = osp.join("/remote-home/linzhx/Projects/mmdetection/_sht_feature/info", f"{img_filename}.pkl")
                img_info = mmcv.load(img_info_path)
                img_backbone = torch.from_numpy(img_info['feat4'])

                gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
                pred_bboxes = results[i]['pts_bbox']['boxes_3d'].tensor.numpy()

                if pred_bboxes.shape[0] != 0:
                    pred_bboxes, index_list = filter_bbox(pred_bboxes)
                
                confidence = results[i]['pts_bbox']['scores_3d'].numpy()
                backbone_1 = results[i]['backbone_1']
                backbone_2 = results[i]['backbone_2']
                # backbone_3 = results[i]['backbone_3']
                # backbone = torch.from_numpy(backbone_3)
                backbone = torch.from_numpy(backbone_2)

                if f"{pts_filename}.pkl" not in file_list:
                    continue

                test_file = mmcv.load(f"_detection_info_pc/{pts_filename}.pkl")
                backbone_1 = test_file['pc_backbone_1']
                backbone_2 = test_file['pc_backbone_2']
                backbone = torch.from_numpy(backbone_2)

                new_pred_bboxes = []
                for bx in pred_bboxes:
                    corner = eight_points(bx[:3], bx[3:6], bx[6])[:4, :2]
                    x1 = corner[:, 0].min() if corner[:, 0].min() > xmin else xmin
                    y1 = corner[:, 1].min() if corner[:, 1].min() > ymin else ymin
                    x2 = corner[:, 0].max() if corner[:, 0].max() < xmax else xmax
                    y2 = corner[:, 1].max() if corner[:, 1].max() < ymax else ymax
                    new_pred_bboxes.append([x1, y1, x2, y2])
                pb = torch.tensor(new_pred_bboxes, dtype=torch.float32) if len(new_pred_bboxes) > 0 else torch.zeros(size=(0, 4))
                rois = torch.zeros(size=(pb.shape[0], 5))
                rois[:, 1] = dim * (pb[:, 0] - xmin) / (xmax - xmin)
                rois[:, 2] = dim * (pb[:, 1] - ymin) / (ymax - ymin)
                rois[:, 3] = dim * (pb[:, 2] - xmin) / (xmax - xmin)
                rois[:, 4] = dim * (pb[:, 3] - ymin) / (ymax - ymin)

                # pb = torch.from_numpy(pred_bboxes)
                # rois = torch.zeros(size=(pb.shape[0], 5))
                # rois[:, 1:3] = pb[:, :2] - pb[:, 3:5] / 2
                # rois[:, 3:5] = pb[:, :2] + pb[:, 3:5] / 2
                # rois[:, 1] = dim * (rois[:, 1] - xmin) / (xmax - xmin)
                # rois[:, 2] = dim * (rois[:, 2] - ymin) / (ymax - ymin)
                # rois[:, 3] = dim * (rois[:, 3] - xmin) / (xmax - xmin)
                # rois[:, 4] = dim * (rois[:, 4] - ymin) / (ymax - ymin)

                feature = roi_extractor([backbone], rois)
                feature = feature.numpy()

                
                if pred_bboxes.shape[0] != 0:
                    img_rois = []  
                    img_eight_corners = point_cloud_position2image_position(pred_bboxes).swapaxes(1,2)
                    for ec_eight_corners in img_eight_corners:
                        x_left = ec_eight_corners[:, 0].min() if ec_eight_corners[:, 0].min() > 0 else 0
                        x_right = ec_eight_corners[:, 0].max() if ec_eight_corners[:, 0].max() < 1280 else 1280
                        y_left = ec_eight_corners[:, 1].min() if ec_eight_corners[:, 1].min() > 0 else 0
                        y_right = ec_eight_corners[:, 1].max() if ec_eight_corners[:, 1].max() < 720 else 720
                        img_rois.append([0, x_left, y_left, x_right, y_right])
                    img_rois = torch.tensor(img_rois, dtype=torch.float)
                else:
                    img_rois = torch.zeros(size=(0, 2048, 7, 7))
                img_feature = img_extractor([img_backbone], img_rois)
                img_feature = img_feature.numpy()
                # print(img_eight_corners.shape)
                # print(img_eight_corners)

                # data_info = dict(pts_filename=gt_info['point_cloud']['point_cloud_path'], gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, confidence=confidence,
                #                 backbone_1=backbone_1, backbone_2=backbone_2, backbone_3=backbone_3, feature=feature)
                data_info = dict(pts_filename=gt_info['point_cloud']['point_cloud_path'], gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, # confidence=confidence,
                                pc_backbone_1=backbone_1, pc_backbone_2=backbone_2, pc_feature=feature,
                                img_backbone=img_backbone.numpy(), img_feature=img_feature)
                pklfile_path = osp.join('_detection_info_lidar_only', pts_filename + '.pkl')
                with open(pklfile_path, 'wb') as f:
                    pickle.dump(data_info, f)
            exit()


        check_empty = False
        gt_annos = [info['annos'] for info in self.data_infos]
        det_annos = self.format_results(results,gt_annos,pklfile_prefix,empty=check_empty)
        # print(" the saving pkl file is ",pklfile_prefix)
        # print(self.get_ann_info(2)['gt_bboxes_3d'].tensor.numpy())
        # assert(gt_annos[2]['position'][0,0] == self.get_ann_info(2)['gt_bboxes_3d'].tensor.numpy()[0,0])
        assert(len(gt_annos) == len(results))
        # print("the annos keys-----------\n",results[0].keys(),gt_annos[0].keys())
        evaluate_method = 'CENTER'
        if evaluate_method == 'IOU':
            eval_dict = self.my_eval(gt_annos, results)
            mAP_3d = get_mAP(eval_dict['precision'])
            print("map===========",mAP_3d,mAP_3d.shape)
            ret_dict = dict()
            for i in range(2): # for i in min_oberlap shape
                ret_dict['3D'] = mAP_3d[0, 0, i] # class,diff,(3d/bev)
            return ret_dict
        ### using center dis method
        else:
            ret_dict = dict()
            # print(np.array(gt_annos[0]['position']).shape,results[0]['boxes_3d'].tensor.numpy().shape)
            # prec,rec = self.center_eval(gt_annos, results)
            #### see different threshold
            ap_list = []
            gt_match_id = []
            occ_recall_list = np.zeros((1,3))
            for i in list([0.25,0.5, 1.0]):
                prec,recall,ap,match_gt_list,occ_recall = self.center_eval(gt_annos, results,det_annos,dist_th = i,empty=check_empty)
                print("threshold: ",i,"prec , recall and ap: ", prec, "\t " ,recall,"\t " ,ap)
                ap_list.append(ap)
                occ_recall_list = occ_recall_list+occ_recall.reshape(1,3)
                ret_dict[f'prec_{i}'] = prec
                ret_dict[f'recall_{i}'] = recall
                ret_dict[f'ap_{i}'] = ap
            gt_match_id.append(match_gt_list)
            ret_dict['3D'] = sum(ap_list)/3
            print('averge recall:' ,occ_recall_list/3)
            return ret_dict

    def center_eval(self,gt_annos,results,det_annos,dist_th = 1,empty = False):

        from mmdet3d.datasets.pipelines import (LoadAnnotations3D, LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        NormalizePointsColor,
                                        PointSegClassMapping)
        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.
        conf = []  # Accumulator of confidences.
        occlusion_tp = [0 for _ in range(3)] 
        occlusion_gt = [0 for _ in range(3)] 

        total = 0
        frame_num = len(gt_annos)
        recall_list = []
        for i in range(frame_num):
            gt = gt_annos[i]
            if 'pts_bbox' in results[i].keys():
                pre = results[i]['pts_bbox']
            else:
                pre = results[i]
#-------------------------------
#      drop empty bbox
            if empty == True:
                pred_boxes_list = np.array(det_annos['no-empty-predict'][i]) #N*7
                pred_confs = np.array(det_annos['no-empty-score'][i])
                sortind = np.argsort(-np.array(pred_confs[:,0])) # from large to small
            else:
                pred_boxes_list = np.array(pre['boxes_3d'].tensor.numpy()) #N*7
                pred_confs = np.array(pre['scores_3d'])
                sortind = np.argsort(-np.array(pred_confs)) # from large to small
            # ---------------------------------------------
            # Match and accumulate match data.
            # ---------------------------------------------
            taken = set()  # Initially no gt bounding box is matched.
            match_gt_list = list()
            min_dist = np.inf
            total += len(gt['position'])
            # print("--",total)
            # print(sortind.shape,results[i]['pts_bbox']['boxes_3d'].tensor.numpy().shape,pred_boxes_list.shape)
            for ind in sortind:
                pred_box = pred_boxes_list[ind]
                min_dist = np.inf
                match_gt_idx = None
                for gt_idx, gt_box in enumerate(gt['position']):

                    # for the first time. we also count the gt occlusion cases
                    if ind == 0:
                        occlusion_gt[gt['occlusion'][gt_idx]] += 1
                    
                    if pred_box.shape[0] == 1:
                        pred_box = pred_box[0]
                    if not gt_idx in taken:
                        this_distance = np.linalg.norm(np.array(gt_box[:3]) - np.array(pred_box[:3])) # xyz
                        if this_distance < min_dist:
                            min_dist = this_distance
                            match_gt_idx = gt_idx
                if match_gt_idx:
                    match_gt_list.append(match_gt_idx)
                    
                else:
                    match_gt_list.append(-1)

                is_match = min_dist < dist_th
                if is_match:
                    #  Update tp, fp and confs.
                    taken.add(match_gt_idx)
                    tp.append(1)
                    fp.append(0)
                    occlusion_tp[gt['occlusion'][match_gt_idx]] += 1
                    conf.append(pred_confs[ind])

                else:
                    # No match. Mark this as a false positive.
                    tp.append(0)
                    fp.append(1)
                    conf.append(pred_confs[ind])
        # print("current dis is ",min_dist)
        tp_ = np.sum(tp)
        fp_ = np.sum(fp)
        precison = tp_ / float(fp_ + tp_)
        recall = tp_ / float(total)
        conf = np.array(conf)
        # Calculate precision and recall.
        rec_interp = np.linspace(0,1,101)

        sortind = np.argsort(conf)[::-1]
        tp = np.array(tp)
        fp = np.array(fp)

        tp = tp[sortind]
        fp = fp[sortind]

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        prec = tp / (fp + tp)
        rec = tp / (total)
        # rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
        ap = self.voc_ap(precison,recall)
        print('the occlusion recall:',np.array(occlusion_tp)/np.array(occlusion_gt))
        return precison,recall,ap,match_gt_list,np.array(occlusion_tp)/np.array(occlusion_gt)

    def voc_ap(self,rec, prec, use_07_metric=True):
        if use_07_metric:
        # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.01, 0.01):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 101.
        else:
            mrec = np.array([0,rec,1])
            mpre = np.array([0,prec,0])
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    '''input: gt_datas:bbox position,dt_datas:dict'''
    def compute_statistics(self,
                           overlaps,
                           gt_datas,
                           dt_datas,
                           metric,
                           min_overlap,
                           thresh=0):
        gt_size = gt_datas.shape[0]
        det_size = dt_datas['scores_3d'].shape[0]
        dt_scores = dt_datas['scores_3d']
        # dt_alphas = dt_datas['boxes_3d',4]
        # gt_alphas = gt_datas[:, 4]
        dt_bboxes = dt_datas['boxes_3d'][:4]  
        assigned_detection = [False] * det_size
        ignored_threshold = [False] * det_size

        NO_DETECTION = -10000000
        tp, fp, fn, similarity = 0, 0, 0, 0
        thresholds = np.zeros((gt_size, ))
        thresh_idx = 0
        delta = np.zeros((gt_size, ))
        delta_idx = 0
        for i in range(gt_size):
            det_idx = -1
            valid_detection = NO_DETECTION
            max_overlap = 0
            assigned_ignored_det = False
            for j in range(det_size):
                overlap = overlaps[i, j] # modify here
                dt_score = dt_scores[j]
                
                if (overlap > 0.01) and dt_score > valid_detection: # TODO
                    det_idx = j
                    valid_detection = dt_score

            if (valid_detection == NO_DETECTION):
                fn += 1
            elif ((valid_detection != NO_DETECTION)):
                assigned_detection[det_idx] = True
            elif valid_detection != NO_DETECTION:
                tp += 1
                # thresholds.append(dt_scores[det_idx])
                thresholds[thresh_idx] = dt_scores[det_idx]
                thresh_idx += 1
                assigned_detection[det_idx] = True

        return tp, fp, fn, similarity, thresholds[:thresh_idx]

    def my_fused_compute_statistics(self,
                             overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_datas,
                             dt_datas,
                             metric,
                             min_overlap,
                             thresholds):
        gt_num = 0
        dt_num = 0
        for i in range(gt_nums.shape[0]):
            for t, thresh in enumerate(thresholds):
                overlap = overlaps[dt_num:dt_num + dt_nums[i],
                                gt_num:gt_num + gt_nums[i]]

                gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
                dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
                tp, fp, fn, similarity, _ = self.compute_statistics(
                    overlap,
                    gt_data['position'],
                    dt_data,
                    metric,
                    min_overlap=min_overlap,
                    thresh=thresh)
                pr[t, 0] += tp
                pr[t, 1] += fp
                pr[t, 2] += fn
                if similarity != -1:
                    pr[t, 3] += similarity
            gt_num += gt_nums[i]
            dt_num += dt_nums[i]
            

    def my_eval(self,gt_annos,dt_annos,num_parts=200):
        overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5], [0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
        overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                                [0.5, 0.25, 0.25, 0.5, 0.25],
                                [0.5, 0.25, 0.25, 0.5, 0.25]])
        # min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
        min_overlaps = np.array([0.5,0.25])
        num_examples = len(gt_annos)
        #----------------------   TODO:current set the num_parts = num_examples
        num_parts =  num_examples
        #----------------------
        if num_examples < num_parts:
            num_parts = num_examples
        split_parts = get_split_parts(num_examples, num_parts)
        # rets = calculate_iou_partly(dt_annos, gt_annos, 2, num_parts)
        
        ''' calculate_iou_partly -- metric 2 : 3d bbox'''
        example_idx = 0
        parted_overlaps = []
        for num_part in split_parts:  # for i in range(N)
            gt_annos_part = gt_annos[example_idx:example_idx + num_part]
            dt_annos_part = dt_annos[example_idx:example_idx + num_part]
            loc = np.concatenate([a['position'] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            dt_boxes = np.array([a['boxes_3d'].tensor.numpy() for a in dt_annos_part])[0] # TODO deal with the shape into N*7
            # print(dt_annos_part[0]['boxes_3d'].tensor.numpy().shape)
            # print(dt_boxes.shape,gt_boxes.shape)
            
            #------------------------ current deal with it ,the dt shape strange 1*N*7,so I use dt_[0]
            try:
                overlap_part = d3_box_overlap(gt_boxes,
                                            dt_boxes).astype(np.float64)
                parted_overlaps.append(overlap_part)
            except:
                pass
            example_idx += num_part
        
        total_dt_num = np.stack([len(a['boxes_3d']) for a in dt_annos], 0)
        total_gt_num = np.stack([len(a['position']) for a in gt_annos], 0)
        overlaps = []
        example_idx = 0
        for j, num_part in enumerate(split_parts):
            gt_annos_part = gt_annos[example_idx:example_idx + num_part]
            dt_annos_part = dt_annos[example_idx:example_idx + num_part]
            gt_num_idx, dt_num_idx = 0, 0
            for i in range(num_part):
                gt_box_num = total_gt_num[example_idx + i]
                dt_box_num = total_dt_num[example_idx + i]
                overlaps.append(
                    parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                    dt_num_idx:dt_num_idx + dt_box_num])
                gt_num_idx += gt_box_num
                dt_num_idx += dt_box_num
            example_idx += num_part
        
        ''' ending '''
        
        
        N_SAMPLE_PTS = 41
        num_minoverlap = len(min_overlaps)
        num_class = 1 # person
        num_difficulty = 1 # not divide the difficult
        precision = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
        recall = np.zeros(
            [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

        ''' eval_class '''
# current only one diff and class
        current_classes = np.zeros(1)
        difficultys = np.zeros(1)
        
        for m, current_class in enumerate(current_classes):
            for idx_l, difficulty in enumerate(difficultys):
                # rets = _prepare_data(gt_annos, dt_annos, int(current_class), int(difficulty))
                # (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                #     dontcares, total_dc_num, total_num_valid_gt) = rets
                for k, min_overlap in enumerate(min_overlaps): # only consider 0.5,0.25
                    thresholdss = []
                    for i in range(len(gt_annos)):
                        gt_datas = np.array(gt_annos[i]['position'])
                        rets = self.compute_statistics(
                            overlaps[i],
                            gt_datas,
                            dt_annos[i],
                            metric = 2,
                            min_overlap=min_overlap,
                            thresh=0.0)
                        tp, fp, fn, similarity, thresholds = rets
                        thresholdss += thresholds.tolist()
                    thresholdss = np.array(thresholdss)
                    thresholds = get_thresholds(thresholdss, num_examples)
                    thresholds = np.array(thresholds)
                    pr = np.zeros([len(thresholds), 4])
                    idx = 0
                    print('done for first')
                    for j, num_part in enumerate(split_parts):
                        gt_datas_part = gt_annos[example_idx:example_idx + num_part] # dict
                        dt_datas_part = dt_annos[example_idx:example_idx + num_part]
                        

                        self.my_fused_compute_statistics(
                            parted_overlaps[j],
                            pr,
                            total_gt_num[idx:idx + num_part],
                            total_dt_num[idx:idx + num_part],
                            gt_datas_part,
                            dt_datas_part,
                            metric = 2,
                            min_overlap=min_overlap,
                            thresholds=thresholds)
                        idx += num_part
                    print('done')
                    for i in range(len(thresholds)):
                        recall[0, 0, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[0, 0, k, i] = pr[i, 0] / (
                            pr[i, 0] + pr[i, 1])
                    for i in range(len(thresholds)):
                        precision[0, 0, k, i] = np.max(
                            precision[0, 0, k, i:], axis=-1)
                        recall[0, 0, k, i] = np.max(
                            recall[0, 0, k, i:], axis=-1)
        ret_dict = {
            'recall': recall,
            'precision': precision,
        }
        
        return ret_dict
        