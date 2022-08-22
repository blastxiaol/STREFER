import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import random
import torch
from datasets import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set config')
    parser.add_argument('--eval_path', type=str)
    
    parser.add_argument('--dataset', default='strefer', type=str, help='dataset')
    parser.add_argument('--data_path', default='/remote-home/share/SHTperson', type=str, help='point cloud path')
    parser.add_argument('--sample_points_num', default=500, type=int, help='number of sampling points')
    parser.add_argument('--img_shape', default=(1280, 720), type=tuple, help='image shape')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--use_view', action='store_true')
    parser.add_argument('--use_vel', action='store_true')
    parser.add_argument('--no_rgb', action='store_true', help="Not use pointpainting feature")
    parser.add_argument('--multi_frame', action='store_true')
    parser.add_argument('--threshold', default=0.25, type=float, help='target threshold')
    parser.add_argument('--rel_dist_threshold', default=1.0, type=float, help='relative distance threshold for multi-objects association')
    parser.add_argument('--matching_threshold', default=0.7, type=float, help='similarity matching threshold for multi-objects association')
    parser.add_argument('--feature_fusion', action='store_true', help="Use feature level fusion")
    parser.add_argument('--frame_num', default=1, type=int, help='frame_num')

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--no_pretrained_resnet', action='store_true', help="If true, not use pretrained resnet")
    parser.add_argument('--resnet_layer_num', default=34, type=int, help="number of rest layers")
    parser.add_argument('--roi_size', default=(7, 7), type=tuple, help="roi align output size")
    parser.add_argument('--spatial_scale', default=1/32, type=float, help="roi align spatial_scale")
    parser.add_argument('--objects_aggregator', default='pool', type=str, help="approach of objects aggregatro", choices=['linear', 'pool'])
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--linear_layer_num', default=3, type=int, help="number of linear layers")
    parser.add_argument('--pc_out_dim', default=512, type=int, help="point cloud output feature dimension")
    parser.add_argument('--vis_out_dim', default=2048, type=int, help="visual information (point cloud and image) output feature dimension")
    parser.add_argument('--vilbert_config_path', default='configs/bert_base_6layer_6conect.json', type=str, help="ViLBert config file path")
    parser.add_argument('--vil_pretrained_file', default='pretrained_model/multi_task_model.bin', type=str, help="ViLBert pretrained file path")
    parser.add_argument('--cat_spatial', action='store_true')
    parser.add_argument('--use_center', action='store_true')
    parser.add_argument('--use_bev', action='store_true')
    parser.add_argument('--no_img', action='store_true')

    parser.add_argument('--use_gt', action='store_true', help="Use GT previous box (Test)")
    parser.add_argument('--relative_spatial', action='store_true', help="Use current box to norm")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_from', default='', type=str, help="ViLBert pretrained file path")
    parser.add_argument('--result_name', default='compare/PF.pkl', type=str)
    
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    if args.debug:
        args.work_dir = "debug"
        args.num_workers = 0
        args.batch_size = 1
    return args


@torch.no_grad()
def test(args, dataset, dataloader, model, criterion=None):
    model.eval()
    loss = 0
    max_index = []
    for data in tqdm(dataloader):
        for key in data:
            try:
                data[key] = data[key].cuda()
            except:
                pass
        logits = model(**data)
        if criterion:
            each_loss = criterion(logits, data['target'])
            loss += (each_loss * logits.shape[0]).item()
        index = torch.flatten(torch.topk(logits, 1).indices).cpu().detach().numpy()
        max_index.append(index)
    max_index = np.hstack(max_index)
    acc25, acc50, m_iou = dataset.evaluate(max_index, args.result_name)
    loss = loss / len(dataset)
    return acc25, acc50, m_iou, loss
    

def main(args):
    set_random_seed(args.seed)

    print("Create dataset")
    dataset = create_dataset(args, 'test')
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_fn)
    
    print("Load Model")
    model = create_model(args).cuda()
    model.load_state_dict(torch.load(args.eval_path))

    print("Run")
    acc25, acc50, m_iou, loss = test(args, dataset, dataloader, model)
    print(f"acc25 = {acc25}")
    print(f"acc50 = {acc50}")
    print(f"miou = {m_iou}")

if __name__ == '__main__':
    args = get_args_parser()
    main(args)