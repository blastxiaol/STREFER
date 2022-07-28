import os
import json
import pickle
import math
import numpy as np
import math
from tqdm import tqdm

def read_json(fn):
    return json.load(open(fn, 'r', encoding='UTF-8-sig'))

data_root = "/remote-home/share/SHTperson"
anno_path = "/remote-home/linzhx/garbage/shangkeda-3d-post-process_20220420-1633"
anno_path_2d = "/remote-home/share/SHTperson/anno"
ANNO_LIST = os.listdir(anno_path)
ANNO_LIST.remove("SHTpersonRefer.txt")
ANNO_LIST.remove("SHTpersonRefer_ann.json")

painting_feature_path = "bin_painting_seg_feature"
painting_prob_path = "bin_painting_seg_prob"
original_pc_path = "bin_v1"

EX_MATRIX = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
IN_MATRIX = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])


ANNO_DICT = {}
ANNO_DICT_2D = {}
for anno_fn in ANNO_LIST:
    path = os.path.join(anno_path, anno_fn)
    path2d = os.path.join(anno_path_2d, anno_fn)
    file = read_json(path)
    file2d = read_json(path2d)
    ANNO_DICT[anno_fn] = file
    ANNO_DICT_2D[anno_fn] = file2d

PROBLEM_OBJECT_DATA = []
ZEOR_TOKEN = []

import collections
ANN_SET = collections.defaultdict(int)

TRAIN_SET = []
TEST_SET = []

TRAIN_SPLIT = []
TEST_SPLIT = []
with open("train.txt", 'r') as f:
    for line in f.readlines():
        TRAIN_SPLIT.append(line[:-1])
with open("test.txt", 'r') as f:
    for line in f.readlines():
        TEST_SPLIT.append(line[:-1])


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
    
    eight_corners = np.stack([conern1, conern2, conern3, conern4, conern5, conern6, conern7, conern8], axis=0)
    return eight_corners

def loc_pc2img(points):
    x, y, z, w, l, h, r = points
    center = [x, y, z]
    size = [w, l, h]
    points = eight_points(center, size, r)
    points = np.insert(points, 3, values=1, axis=1)
    points_T = np.transpose(points)
    points_T[3, :] = 1.0

    # lidar2camera
    points_T_camera = np.dot(EX_MATRIX, points_T)
    # camera2pixel
    pixel = np.dot(IN_MATRIX, points_T_camera).T
    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
    pixel_xy = np.around(pixel_xy).astype(int)
    
    return pixel_xy.T

def get_frame_info(frame_name, id):
    data_dict = {}
    anno_info = frame_name.split('/')
    point_cloud_name = anno_info[7]
    anno_name = f"{anno_info[4]}-{anno_info[5]}.json"
    group_id = anno_info[4].split()[0]
    scene_id = anno_info[5]
    anno = ANNO_DICT_2D[anno_name]

    frame_id = None
    bbox = None
    image_name = None
    image_bbox = None
    all_bboxes_3d = []
    for frame_id, frame in enumerate(anno['frames']):
        frame_name = frame['frame_name'].split('/')[-1]
        if point_cloud_name == frame_name:
            for item in frame['items']:
                all_bboxes_3d.append([item['position']['x'], item['position']['y'], item['position']['z'], 
                        item['boundingbox']['x'], item['boundingbox']['y'], item['boundingbox']['z'],
                        item['rotation']])
                if id == item['id']:
                    bbox = [item['position']['x'], item['position']['y'], item['position']['z'], 
                        item['boundingbox']['x'], item['boundingbox']['y'], item['boundingbox']['z'],
                        item['rotation']]
            for image in frame['images']:
                image_name = image['image_name'].split('/')[-1]
                for image_item in image['items']:
                    if id == image_item['id']:
                        center_x, center_y = image_item['boundingbox']['x'], image_item['boundingbox']['y']
                        weight, height = image_item['dimension']['x'], image_item['dimension']['y']
                        image_bbox = [int(center_x), int(center_y), int(weight), int(height)]
                        break
            break
    if bbox is None:    # 前一帧 没有这个object
        return None
    assert frame_id is not None
    assert image_name is not None

    try:
        assert image_bbox is not None
    except:
        # print(f"这条Image没有bbox： {group_id}/{scene_id}/left/{image_name}")
        trans_bbox = loc_pc2img(bbox).T
        xmin = trans_bbox[:, 0].min() if trans_bbox[:, 0].min() > 0 else 0 
        ymin = trans_bbox[:, 1].min() if trans_bbox[:, 1].min() > 0 else 0 
        xmax = trans_bbox[:, 0].max() if trans_bbox[:, 0].max() < 1280 else 1280
        ymax = trans_bbox[:, 1].max() if trans_bbox[:, 1].max() < 720 else 720
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        width = xmax - xmin
        height = ymax - ymin
        image_bbox = [center_x, center_y, width, height]

    data_dict['group_id'] = group_id
    data_dict['scene_id'] = scene_id
    data_dict['frame_id'] = str(frame_id)
    data_dict['point_cloud_name'] = point_cloud_name
    data_dict['bbox'] = bbox
    data_dict['image_name'] = image_name
    data_dict['image_bbox'] = image_bbox
    data_dict['all_bboxes_3d'] = all_bboxes_3d
    return data_dict


def trans_type(anno_file):
    """
    Return: List[Dict]
    'group_id'
    'scene_id'
    'frame_id'
    'category'
    'ann_id'
    'description'
    'token'
    'point_cloud_name'
    'original_filename'
    'painting_feature_filename'
    'painting_prob_path'
    'bbox': center_x, center_y, center_z, width, lengh, height, rotation
    'all_bboxes'
    """
    for data in tqdm(anno_file):
        # Log information in annotation.json
        new_data = {}
        sample_id = data.get('sample_id', None)
        group_id = data['group_id'].split()[0]
        scene_id = data['scene_id']
        frame_id = data['frame_id']
        object_id = data['object_id']
        ann_id = data['ann_id']
        description = data['description']
        token = data['token']
        category = data['category']

        ANN_SET[os.path.join(f"{group_id}",f"{scene_id}",f"{frame_id}",f"{object_id}")] += 1

        if len(token) <= 0:
            print(f"Token Length 为 0!!!, idx={sample_id}, group_id={group_id} scene_id={scene_id} frame_id={frame_id}, object_id={object_id}")
            ZEOR_TOKEN.append(data)
            continue

        # find a annotation file
        annotation_file_path = None
        for annotation in ANNO_LIST:
            if annotation.find(group_id) == 0 \
                and not annotation[len(group_id)].isdigit() \
                and f"-{scene_id}.json" in annotation:
                annotation_file_path = annotation
        # find the target point cloud detection annotation
        anno = ANNO_DICT[annotation_file_path]
        anno_2d = ANNO_DICT_2D[annotation_file_path]
        frame_anno = anno['frames'][frame_id]
        frame_anno_2d = anno_2d['frames'][frame_id]
        assert frame_anno['frameId'] == frame_id, f"frame annotation frame id ({frame_anno['frameId']}) != frame id ({frame_id})"
        assert frame_anno_2d['frameId'] == frame_id, f"frame 2d id = {frame_anno_2d['frameId']} != {frame_id}"
       
        point_cloud_name = frame_anno['frame_name'].split('/')[-1]
        image_name = frame_anno['images'][0]['image_name'].split('/')[-1]
        all_items = frame_anno['items']
        
        images2d_3d_items = frame_anno_2d['items']
        image2d_items = frame_anno_2d['images'][0]
        image2d_name = image2d_items['image_name'].split('/')[-1]
        assert image2d_name == image_name, "Two name differnet '{image_name}' and '{image2d_name}'"
        try:
            obj_item = frame_anno['items'][object_id]
        except:
            print(group_id, scene_id, frame_id, object_id, ann_id, description)
            continue
 
        try:
            assert obj_item['object_id'] == object_id, f"frame annotation object id ({obj_item['object_id']}) != object id ({object_id}), idx={sample_id}"
        except:
            print(f"Object ID 不统一： {group_id}-{scene_id} {frame_id}, ({obj_item['object_id']}) != object id ({object_id}), idx={sample_id}")
            PROBLEM_OBJECT_DATA.append(data)
            continue
   
        assert category == obj_item['category'], f"frame object category ({obj_item['category']}) != category ({category})"
        bbox = [obj_item['position']['x'], obj_item['position']['y'], obj_item['position']['z'], 
                        obj_item['boundingbox']['x'], obj_item['boundingbox']['y'], obj_item['boundingbox']['z'],
                        obj_item['rotation']]
        frame_bboxes = [[item['position']['x'], item['position']['y'], item['position']['z'], 
                        item['boundingbox']['x'], item['boundingbox']['y'], item['boundingbox']['z'],
                        item['rotation']] for item in all_items]
        id_3d = obj_item['id']

        prev_frame = frame_anno['previous']
        next_frame = frame_anno['next']
        prev_info = get_frame_info(prev_frame, id_3d) if prev_frame else None
        next_info = get_frame_info(next_frame, id_3d) if next_frame else None

        if prev_info is None:
            # print("No previous frame")
            continue
        # if next_info is None:
        #     print("No next frame")
        #     continue

        # vel_x = next_info['bbox'][0] - bbox[0]
        # vel_y = next_info['bbox'][1] - bbox[1]
        # velocity = [vel_x, vel_y]

        all_image_bboxes = []
        for data in image2d_items['items']:
            center_x, center_y = data['boundingbox']['x'], data['boundingbox']['y']
            weight, height = data['dimension']['x'], data['dimension']['y']
            all_image_bboxes.append([int(center_x), int(center_y), int(weight), int(height)])

        image_bbox = None
        image_obj_id = None
        for i, data in enumerate(image2d_items['items']):
            if data['id'] == id_3d:
                center_x, center_y = data['boundingbox']['x'], data['boundingbox']['y']
                weight, height = data['dimension']['x'], data['dimension']['y']
                image_bbox = [int(center_x), int(center_y), int(weight), int(height)]
                image_obj_id = i
                break

        manual_box = False
        try:
            assert image_bbox is not None, f"image bbox is None"
        except:
            # print(f"这条Image没有bbox: {group_id}/{scene_id}/left/{image_name}")
            trans_bbox = loc_pc2img(bbox).T
            xmin = trans_bbox[:, 0].min() if trans_bbox[:, 0].min() > 0 else 0 
            ymin = trans_bbox[:, 1].min() if trans_bbox[:, 1].min() > 0 else 0 
            xmax = trans_bbox[:, 0].max() if trans_bbox[:, 0].max() < 1280 else 1280
            ymax = trans_bbox[:, 1].max() if trans_bbox[:, 1].max() < 720 else 720
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            width = xmax - xmin
            height = ymax - ymin
            image_bbox = [center_x, center_y, width, height]
            manual_box = True
        if not manual_box:
            assert all_image_bboxes[image_obj_id] == image_bbox, f"Two bboxes are differnet"

        # save into a dictionary
        new_data['sample_index'] = str(sample_id)
        new_data['group_id'] = group_id
        new_data['scene_id'] = scene_id
        new_data['frame_id'] = str(frame_id)
        new_data['object_id'] = str(object_id)
        
        new_data['language'] = {}
        new_data['language']['ann_id'] = ann_id
        new_data['language']['description'] = description
        new_data['language']['token'] = token

        new_data['point_cloud'] = {}
        new_data['point_cloud']['point_cloud_name'] = point_cloud_name
        new_data['point_cloud']['original_filename'] = os.path.join(original_pc_path, point_cloud_name)
        new_data['point_cloud']['painting_feature_filename'] = os.path.join(painting_feature_path, point_cloud_name)
        new_data['point_cloud']['painting_prob_path'] = os.path.join(painting_prob_path, point_cloud_name)
        new_data['point_cloud']['bbox'] = bbox
        new_data['point_cloud']['all_bboxes'] = frame_bboxes
        new_data['point_cloud']['category'] = category
        # new_data['point_cloud']['velocity'] = velocity

        new_data['image'] = {}
        new_data['image']['image_name'] = image_name
        new_data['image']['image_bbox'] = image_bbox
        new_data['image']['image_object_id'] = str(image_obj_id)
        new_data['image']['all_image_bboxes'] = all_image_bboxes

        new_data['previous'] = prev_info
        new_data['next'] = next_info

        # for key in new_data:
        #     if isinstance(new_data[key], dict):
        #         for k in new_data[key]:
        #             print(f"{key}[{k}] == {type(new_data[key][k])}")
        #     else:
        #         print(f"{key} == {type(new_data[key])}")
        # exit()
        scene_class = f"{group_id} {scene_id}"
        if scene_class in TRAIN_SPLIT:
            TRAIN_SET.append(new_data)
        else:
            TEST_SET.append(new_data)
    return 

def save_pkl(file, output_path):
    output = open(output_path, 'wb')
    pickle.dump(file, output)
    output.close()

def check(train, test):
    from check import count_word
    train_calculator, _, _ = count_word(train)
    # print(train_calculator)
    test_calculator, _, _ = count_word(test)

    less_two = set()
    for word in test_calculator:
        times = train_calculator.get(word, 0)
        if times < 2:
            less_two.add((word, times))
    
    if len(less_two) == 0:
        return True
    else:
        return False

import numpy
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':

    all_anno = read_json("SHTpersonRefer_ann.json")

    fn_list = ['train_SHTpersonRefer_ann.pkl', 'test_SHTpersonRefer_ann.pkl']

    trans_type(all_anno)
    print(len(TRAIN_SET), len(TEST_SET))
    if check(TRAIN_SET, TEST_SET):
        with open("train_SHTpersonRefer_ann.json", 'w') as f:       # , encoding='UTF-8'
            f.write(json.dumps(TRAIN_SET, cls=NumpyEncoder))
        with open("test_SHTpersonRefer_ann.json", 'w') as f:        # , encoding='UTF-8'
            f.write(json.dumps(TEST_SET, cls=NumpyEncoder))

    # output_path = fn_list[i]
    # save_pkl(file, output_path)
    # print(f"{fn_list[i]} End")

    # save_pkl(file, "pkl_file/SHTpersonRefer_ann.pkl")
    # print(max(ANN_SET.values()), min(ANN_SET.values()))


    # print(len(PROBLEM_OBJECT_DATA))
    # with open("有问题的数据/问题数据.json", 'w', encoding='UTF-8') as f:
    #     f.write(json.dumps(PROBLEM_OBJECT_DATA))

    # print(PROBLEM_OBJECT_DATA[0])
    # print()
    # print()
    # print(PROBLEM_OBJECT_DATA[20])

    
    

