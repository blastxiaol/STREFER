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
from time import time
from utils import Logger
from tqdm import tqdm

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector')
    parser.add_argument('--dataset', default='sunrefer_predet', type=str, help='sunrefer', choices=['sunrefer', 'sunrefer_predet', 'scanrefer'])
    parser.add_argument('--data_path', default='data', type=str, help='point cloud path')
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--sample_points_num', default=3000, type=int, help='number of sampling points')
    parser.add_argument('--img_shape', default=(256, 256), type=tuple, help='image shape')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')

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
    parser.add_argument('--spatial_dim', default=6, type=int, help="spatial dimension size")
    parser.add_argument('--vilbert_config_path', default='configs/bert_base_6layer_6conect.json', type=str, help="ViLBert config file path")
    parser.add_argument('--vil_pretrained_file', default='pretrained_model/multi_task_model.bin', type=str, help="ViLBert pretrained file path")
    
    parser.add_argument('--no_evaluate', action='store_true', help="If true, evaluate when training")
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--val_epoch', default=10, type=int)
    parser.add_argument('--no_verbose', action='store_true', help="If true, not print information")
    parser.add_argument('--work_dir', default='work_dir/vil_bert3d', type=str)

    parser.add_argument('--eval_path', default='work_dir/vil_bert3d_dist/sunrefer/7-20-6-59-50/epoch_55_model.pth', type=str)
    parser.add_argument('--no_vis', action='store_true', help="no visualization")
    parser.add_argument('--vis_path', default='visualize/vilbert3d/sunrefer', type=str)
    args = parser.parse_args()
    return args


@torch.no_grad()
def test(args, dataset, dataloader, model):
    model.eval()
    max_index = []
    for image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids, target in tqdm(dataloader):
        image = image.cuda()
        boxes2d = boxes2d.cuda()
        points = points.cuda()
        spatial = spatial.cuda()
        vis_mask = vis_mask.cuda()
        token = token.cuda()
        mask = mask.cuda()
        segment_ids = segment_ids.cuda()
        target = target.cuda()
        logits = model(image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids)
        index = torch.flatten(torch.topk(logits, 1).indices).cpu().detach().numpy()
        max_index.append(index)
    max_index = np.hstack(max_index)
    acc25, acc50, m_iou = dataset.evaluate(max_index)
    print(f"acc25={acc25}, acc50={acc50}, miou={m_iou}")
    if not args.no_vis:
        dataset.visualize(args, max_index)
    

def main(args):
    set_random_seed(args.seed)

    print("Create dataset")
    dataset = create_dataset(args, 'val')
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_fn)
    
    print("Load Model")
    model = create_model(args).cuda()
    model.load_state_dict(torch.load(args.eval_path))

    print("Run")
    test(args, dataset, dataloader, model)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)