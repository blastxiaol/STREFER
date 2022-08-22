import os
from tokenize import group
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from utils import pc_utils, strefer_utils
from utils.metrics import cal_accuracy, cal_iou3d
from pytorch_transformers.tokenization_bert import BertTokenizer
import cv2
from tqdm import tqdm
import math
import cmath
import pickle
from utils import Scene

class STREFER_MF(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.split = split
        self.base_path = args.data_path
        if split == 'train':
            self.data_path = os.path.join('data/train_SHTpersonRefer_ann.json')
        else:
            self.data_path = os.path.join('data/test_SHTpersonRefer_ann.json')
        self.dataset = json.load(open(self.data_path, 'r'))
        self.sample_points_num = args.sample_points_num
        self.x_size, self.y_size = args.img_shape
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)

        self.data_type = torch.float32

        self.use_view = args.use_view
        self.use_vel = args.use_vel
        self.use_xyz = True if not self.use_view and not self.use_vel else False
        self.use_rgb = not args.no_rgb
        self.dim = 23 if self.use_rgb else 4

        self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax = (0, -20.48, -4, 30.72, 20.48, 1)
        self.rmin, self.rmax = (-math.pi, math.pi)
        self.vmin, self.vmax = (0, 3)
        self.lmin, self.wmin, self.hmin = (0, 0, 0)
        self.lmax = self.xmax - self.xmin
        self.wmax = self.ymax - self.ymin
        self.hmax = self.zmax - self.zmin
        self.volume = self.lmax * self.wmax * self.hmax

        self.target_threshold = args.threshold
        self.relative_spatial = args.relative_spatial

        self.frame_num = args.frame_num

        self.spatial_dim = 8
        if args.use_view:
            self.spatial_dim = 7
        if args.use_vel:
            self.spatial_dim = 9
        if args.use_gt:
            self.current2prev = pickle.load(open("data/current2prev_gt.pkl", 'rb'))
        else:
            self.current2prev = pickle.load(open("data/current2prev_mf.pkl", 'rb'))
            # self.current2prev = pickle.load(open("data/current2prev.pkl", 'rb'))
        
        self.no_img = args.no_img

    def __getitem__(self, index):
        data = self.dataset[index]

        group_id = data['group_id']
        scene_id = data['scene_id']

        point_cloud_info = data['point_cloud']
        image_info = data['image']
        language_info = data['language']
        previous_info = data['previous']

        # read image
        image_name = image_info['image_name']
        image_path = os.path.join(self.base_path, group_id, scene_id, 'left', image_name)

        image = strefer_utils.load_image(image_path)
        img_y_size, img_x_size, dim = image.shape
        image = cv2.resize(image, (self.x_size, self.y_size))
        image = image.transpose((2, 0, 1))

        # read points
        point_cloud_name = point_cloud_info['point_cloud_name']
        prev_point_cloud_name = previous_info['point_cloud_name']
        if self.use_rgb:
            point_cloud_path = os.path.join(self.base_path, group_id, scene_id, 'bin_painting_seg_feature', point_cloud_name)
            points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, self.dim)
            prev_point_cloud_path = os.path.join(self.base_path, group_id, scene_id, 'bin_painting_seg_feature', prev_point_cloud_name)
            prev_points = np.fromfile(prev_point_cloud_path, dtype=np.float32).reshape(-1, self.dim)
        else:
            point_cloud_path = os.path.join(self.base_path, group_id, scene_id, 'bin_v1', point_cloud_name)
            points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, self.dim)
            prev_point_cloud_path = os.path.join(self.base_path, group_id, scene_id, 'bin_v1', prev_point_cloud_name)
            prev_points = np.fromfile(prev_point_cloud_path, dtype=np.float32).reshape(-1, self.dim)
        
        # previous box concatation

        objects_pc_list = []
        obj_center_list = []
        boxes3d = np.load(f"data/pred_bboxes/{group_id}/{scene_id}/{point_cloud_name[:-4]}.npy")
        objects_pc = strefer_utils.batch_extract_pc_in_box3d(points, boxes3d, self.sample_points_num, self.dim)
        corners2d = strefer_utils.batch_compute_box_3d(boxes3d)
        obj_center = strefer_utils.norm(boxes3d[:,:3], np.array((self.xmin, self.ymin, self.zmin)), np.array((self.xmax, self.ymax, self.zmax)))
        objects_pc_list.append(objects_pc)
        obj_center_list.append(obj_center)
        
        min_array = []
        max_array = []
        for i, pc in enumerate(objects_pc):
            if not (pc[:, :3] == 0).all():
                pxmin, pymin, pzmin = pc[:, :3].min(axis=0)
                pxmax, pymax, pzmax = pc[:, :3].max(axis=0)
                pmin = np.array([pxmin, pymin, pzmin])
                pmax = np.array([pxmax, pymax, pzmax])
                if (pmin == pmax).any():
                    min_array.append(None)
                    max_array.append(None)
                    objects_pc[i, :, :3] = np.zeros_like(pc[:, :3])
                else:
                    min_array.append(pmin)
                    max_array.append(pmax)
                    objects_pc[i, :, :3] = strefer_utils.norm(pc[:, :3], pmin, pmax)
            else:
                min_array.append(None)
                max_array.append(None)

        prev_objects_pc = np.zeros_like(objects_pc)
        if True:
            curr_frame_num = 1
            curr_scene = self.current2prev[f"{group_id}/{scene_id}/{point_cloud_name[:-4]}"]
            while curr_frame_num < self.frame_num and curr_scene is not None:
                curr_scene = curr_scene.prev_scene 
                if curr_scene is None:
                    break
                curr_boxes = curr_scene.boxes
                curr_object_pc = []
                curr_center = []
                for i, box in enumerate(curr_boxes):
                    if box is None:
                        curr_obj_pc = objects_pc[i]
                        curr_obj_center = boxes3d[i, :3]
                    else:
                        box = np.array([box])[:,:7]
                        curr_obj_pc = strefer_utils.batch_extract_pc_in_box3d(points, box, self.sample_points_num, self.dim)[0]
                        if not (curr_obj_pc[:, :3] == 0).all():
                            pxmin, pymin, pzmin = curr_obj_pc[:, :3].min(axis=0)
                            pxmax, pymax, pzmax = curr_obj_pc[:, :3].max(axis=0)
                            pmin = np.array([pxmin, pymin, pzmin])
                            pmax = np.array([pxmax, pymax, pzmax])
                            if (pmin == pmax).any():
                                curr_obj_pc[:, :3] = np.zeros_like(curr_obj_pc[:, :3])
                            else:
                                curr_obj_pc[:, :3] = strefer_utils.norm(curr_obj_pc[:, :3], pmin, pmax)
                        curr_obj_center = np.array(box)[0, :3]
                    curr_object_pc.append(curr_obj_pc)
                    curr_center.append(curr_obj_center)
                curr_object_pc = np.stack(curr_object_pc, axis=0)
                curr_center = np.stack(curr_center, axis=0)
                curr_center = strefer_utils.norm(curr_center, np.array((self.xmin, self.ymin, self.zmin)), np.array((self.xmax, self.ymax, self.zmax)))
                objects_pc_list.append(curr_object_pc)
                obj_center_list.append(curr_center)

                curr_frame_num += 1
            
            for _ in range(curr_frame_num, self.frame_num):
                objects_pc_list.append(objects_pc)
                obj_center_list.append(obj_center)
            
            objects_pc = np.concatenate(objects_pc_list, axis=1)
            obj_center = np.concatenate(obj_center_list, axis=1)

        if False:
            prev_boxes3d = self.current2prev[f"{group_id}/{scene_id}/{point_cloud_name[:-4]}"]
            prev_obj_center = np.zeros_like(obj_center)
            assert len(prev_boxes3d) == len(objects_pc)
            for i, prev_box in enumerate(prev_boxes3d):
                if prev_box is None:
                    prev_objects_pc[i] = objects_pc[i].copy()
                    prev_obj_center[i] = boxes3d[i,:3]
                else:
                    prev_obj_pc_i = strefer_utils.batch_extract_pc_in_box3d(prev_points, [prev_box], self.sample_points_num, self.dim)[0]
                    if not (prev_obj_pc_i[:, :3] == 0).all():
                        if self.relative_spatial:
                            pmin = min_array[i]
                            pmax = max_array[i]
                        else:
                            pxmin, pymin, pzmin = prev_obj_pc_i[:, :3].min(axis=0)
                            pxmax, pymax, pzmax = prev_obj_pc_i[:, :3].max(axis=0)
                            pmin = np.array([pxmin, pymin, pzmin])
                            pmax = np.array([pxmax, pymax, pzmax])
                        if (pmin == pmax).all():
                            prev_obj_pc_i[:, :3] = np.zeros_like(prev_obj_pc_i[:, :3])
                        else:
                            prev_obj_pc_i[:, :3] = strefer_utils.norm(prev_obj_pc_i[:, :3], pmin, pmax)
                        prev_objects_pc[i] = prev_obj_pc_i
                    prev_obj_center[i] = np.array(prev_boxes3d[i])[:3]

            prev_obj_center = strefer_utils.norm(prev_obj_center, np.array((self.xmin, self.ymin, self.zmin)), np.array((self.xmax, self.ymax, self.zmax)))
            objects_pc = np.concatenate([objects_pc, prev_objects_pc], axis=1)
            obj_center = np.concatenate([obj_center, prev_obj_center], axis=1)
 
        # spatial
        spatial = []
        if self.use_xyz is True:
            max_value = np.array([self.xmax, self.ymax, self.zmax, self.lmax, self.wmax, self.hmax, self.rmax])
            min_value = np.array([self.xmin, self.ymin, self.zmin, self.lmin, self.wmin, self.hmin, self.rmin])
            for i in range(len(boxes3d)):
                x, y, z, l, w, h, r = strefer_utils.norm(boxes3d[i][:7], min_value, max_value)
                volume = (boxes3d[i][3] * boxes3d[i][4] * boxes3d[i][5]) /self.volume
                spatial.append([x, y, z, l, w, h, r, volume])
        elif self.use_vel is True:
            for i in range(len(boxes3d)):
                eachbox_2d = corners2d[i]
                x, _, _, _, _, _, r, xvel, yvel = boxes3d[i][:9]
                x = strefer_utils.norm(x, self.xmin, self.xmax)
                r = strefer_utils.norm(r, self.rmin, self.rmax)
                vel, drt = cmath.polar(complex(xvel, yvel))
                vel = strefer_utils.norm(vel, self.vmin, self.vmax)
                drt = strefer_utils.norm(drt, self.rmin, self.rmax)
                eachbox_2d = corners2d[i]
                eachbox_2d[:, 0][eachbox_2d[:, 0]<0] = 0
                eachbox_2d[:, 0][eachbox_2d[:, 0]>=img_x_size] = img_x_size-1
                eachbox_2d[:, 1][eachbox_2d[:, 1]<0] = 0
                eachbox_2d[:, 1][eachbox_2d[:, 1]>=img_y_size] = img_y_size-1
                xmin, ymin = eachbox_2d.min(axis=0)
                xmax, ymax = eachbox_2d.max(axis=0)
                square = (ymax - ymin) * (xmax - xmin) / (img_x_size * img_y_size)
                xmin /= img_x_size
                xmax /= img_x_size
                ymin /= img_y_size
                ymax /= img_y_size
                spatial.append([xmin, ymin, xmax, ymax, x, r, square, vel, drt])
        elif self.use_view is True:
            for i in range(len(boxes3d)):
                eachbox_2d = corners2d[i]
                x, _, _, _, _, _, r = boxes3d[i][:7]
                x = strefer_utils.norm(x, self.xmin, self.xmax)
                r = strefer_utils.norm(r, self.rmin, self.rmax)
                eachbox_2d = corners2d[i]
                eachbox_2d[:, 0][eachbox_2d[:, 0]<0] = 0
                eachbox_2d[:, 0][eachbox_2d[:, 0]>=img_x_size] = img_x_size-1
                eachbox_2d[:, 1][eachbox_2d[:, 1]<0] = 0
                eachbox_2d[:, 1][eachbox_2d[:, 1]>=img_y_size] = img_y_size-1
                xmin, ymin = eachbox_2d.min(axis=0)
                xmax, ymax = eachbox_2d.max(axis=0)
                square = (ymax - ymin) * (xmax - xmin) / (img_x_size * img_y_size)
                xmin /= img_x_size
                xmax /= img_x_size
                ymin /= img_y_size
                ymax /= img_y_size
                spatial.append([xmin, ymin, xmax, ymax, x, r, square])
        else:
            raise NotImplementedError
                

        spatial = np.array(spatial, dtype=np.float32)
        
        # bboxes2d
        boxes2d = []
        for box in corners2d:
            xmin, ymin = box.min(axis=0)
            xmax, ymax = box.min(axis=0)
        #     xmin, ymin, xmax, ymax = box
            xmin = int(xmin * self.x_size / img_x_size)
            xmax = int(xmax * self.x_size / img_x_size)
            ymin = int(ymin * self.y_size / img_y_size)
            ymax = int(ymax * self.y_size / img_y_size)
            boxes2d.append([xmin, ymin, xmax, ymax])
        boxes2d = np.array(boxes2d)

        # target
        target = np.zeros((len(boxes3d), ), dtype=np.float32)
        target_box = np.array(point_cloud_info['bbox'], dtype=np.float32)
        for i, box in enumerate(boxes3d):
            iou = pc_utils.cal_iou3d(target_box, box)
            target[i] = 1 if iou >= self.target_threshold else 0

        # token
        sentence = language_info['description'].lower()
        sentence = "[CLS] " + sentence + " [SEP]"
        token = self.tokenizer.encode(sentence)
        return image, boxes2d, objects_pc, spatial, token, target, obj_center
    
    def collate_fn(self, raw_batch):
        raw_batch = list(zip(*raw_batch))

        # image
        image_list = raw_batch[0]
        image = []
        for img in image_list:
            img = torch.tensor(img, dtype=self.data_type)
            image.append(img)
        image = torch.stack(image, dim=0)

        # boxes2d
        boxes2d_list = raw_batch[1]
        max_obj_num = 0
        for each_boxes2d in boxes2d_list:
            max_obj_num = max(max_obj_num, each_boxes2d.shape[0])
        
        boxes2d = torch.zeros((len(boxes2d_list), max_obj_num, 4), dtype=self.data_type)
        vis_mask = torch.zeros((len(boxes2d_list), max_obj_num), dtype=torch.long)
        for i, each_boxes2d in enumerate(boxes2d_list):
            each_boxes2d = torch.tensor(each_boxes2d)
            boxes2d[i, :each_boxes2d.shape[0], :] = each_boxes2d
            vis_mask[i, :each_boxes2d.shape[0]] = 1

        # points
        objects_pc = raw_batch[2]
        points = torch.zeros((len(boxes2d_list), max_obj_num, self.sample_points_num * self.frame_num, self.dim), dtype=self.data_type)
        for i, pt in enumerate(objects_pc):
            points[i, :pt.shape[0], :, :] = torch.tensor(pt)

        # spatial
        spatial_list = raw_batch[3]
        spatial = torch.zeros((len(boxes2d_list), max_obj_num, self.spatial_dim), dtype=self.data_type)
        for i, box in enumerate(spatial_list):
            spatial[i, :box.shape[0], :] = torch.tensor(box)

        # token
        token_list = raw_batch[4]
        max_token_num = 0
        for each_token in token_list:
            max_token_num = max(max_token_num, len(each_token))

        token = torch.zeros((len(boxes2d_list), max_token_num), dtype=torch.long)
        for i, each_token in enumerate(token_list):
            token[i, :len(each_token)] = torch.tensor(each_token)

        mask = torch.zeros((len(boxes2d_list), max_token_num), dtype=torch.long)
        mask[token != 0] = 1

        segment_ids = torch.zeros_like(mask)

        # target
        target_list = raw_batch[5]
        target = torch.zeros((len(boxes2d_list), max_obj_num), dtype=self.data_type)
        for i, each_target in enumerate(target_list):
            target[i, :len(each_target)] = torch.tensor(each_target)
        
        # obj_center
        obj_center_list = raw_batch[6]
        obj_center = torch.zeros(len(boxes2d_list), max_obj_num,  3 * self.frame_num, dtype=self.data_type)
        for i, each_obj_center in enumerate(obj_center_list):
            obj_center[i, :len(each_obj_center)] = torch.tensor(each_obj_center)

        if not self.no_img:
            data = dict(image=image, boxes2d=boxes2d, points=points, spatial=spatial,
                        vis_mask=vis_mask, token=token, mask=mask, segment_ids=segment_ids,
                        target=target, obj_center=obj_center)

        else:
            data = dict(image=None, boxes2d=None, points=points, spatial=spatial,
                        vis_mask=vis_mask, token=token, mask=mask, segment_ids=segment_ids,
                        target=target, obj_center=obj_center)

        return data
    
    def __len__(self):
        return len(self.dataset)
    
    def evaluate(self, index_list, result_name=''):
        eval_results = []
        target_boxes = []
        pred_boxes = []
        idx = 0
        for max_index in tqdm(index_list):
            data = self.dataset[idx]
            group_id = data['group_id']
            scene_id = data['scene_id']
            point_cloud_info = data['point_cloud']
            image_info = data['image']
            language_info = data['language']
            point_cloud_name = point_cloud_info['point_cloud_name']
            boxes3d = np.load(f"data/pred_bboxes/{group_id}/{scene_id}/{point_cloud_name[:-4]}.npy")
            pred_box = boxes3d[max_index]
            target = point_cloud_info['bbox']

            if result_name:
                language = data['language']['description']
                point_cloud_path = os.path.join(self.base_path, group_id, scene_id, 'bin_v1', f"{point_cloud_name[:-4]}.bin")
                img_name = os.path.join(self.base_path, group_id, scene_id, 'left', data['image']['image_name'])
                out_data = dict()
                out_data['gt_box'] = target
                out_data['pred_box'] = pred_box
                out_data["point_cloud_path"] = point_cloud_path
                out_data["image_path"] = img_name
                out_data['gt_corner2d'] = strefer_utils.batch_compute_box_3d([np.array(target)])
                out_data['pred_corner2d'] = strefer_utils.batch_compute_box_3d([np.array(pred_box)])
                out_data['language'] = language
                out_data['iou'] = cal_iou3d(pred_box, target)
                eval_results.append(out_data)

            pred_boxes.append(pred_box)
            target_boxes.append(target)
            idx += 1
        
        target_boxes = np.array(target_boxes)
        pred_boxes = np.array(pred_boxes)

        if result_name:
            save_pkl(eval_results, result_name)

        acc25, acc50, miou = cal_accuracy(pred_boxes, target_boxes)
        return acc25, acc50, miou

def save_pkl(file, output_path):
    import pickle
    output = open(output_path, 'wb')
    pickle.dump(file, output)
    output.close()