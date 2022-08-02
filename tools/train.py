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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set config')
    parser.add_argument('--dataset', default='strefer_gt', type=str, help='dataset')
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

    parser.add_argument('--no_evaluate', action='store_true', help="If true, evaluate when training")
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)

    parser.add_argument('--use_gt', action='store_true', help="Use GT previous box (Test)")
    parser.add_argument('--relative_spatial', action='store_true', help="Use current box to norm")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--load_from', default='', type=str, help="ViLBert pretrained file path")
    parser.add_argument('--val_epoch', default=1, type=int)
    parser.add_argument('--no_verbose', action='store_true', help="If true, not print information")
    parser.add_argument('--work_dir', default='work_dir/vil_bert3d', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        args.work_dir = "debug"
        # args.num_workers = 0
        # args.batch_size = 1
    return args


def train_epoch(epoch: int, dataloader, model,\
                 criterion, optimizer, scheduler, total_epoch: int, logger=None):
    model.train()
    mean_loss = 0
    for idx, data in enumerate(dataloader):
        start_time = time()
        for key in data:
            try:
                data[key] = data[key].cuda()
            except:
                pass
        scores = model(**data)   
        optimizer.zero_grad()
        loss = criterion(scores, data['target'])
        mean_loss += loss.item() / len(dataloader)
        loss.backward()
        optimizer.step()
        end_time = time()
        if idx % 50 == 0:
            used_time = end_time - start_time
            rest_epoch = total_epoch - epoch
            eta = rest_epoch * len(dataloader) * used_time - idx * used_time  # second
            sec = int(eta % 60)
            minute = int(eta % 3600 // 60)
            hour = int(eta % (24 * 3600) // 3600)
            day = int(eta // (24 * 3600))

            info = f"TRN Epoch[{epoch+1}][{idx}|{len(dataloader)}]\tloss={round(loss.item(), 4)}\t"\
                   f"lr={optimizer.param_groups[0]['lr']}\tbert_lr={optimizer.param_groups[3]['lr']}\t"\
                   f"ETA: {day} days {hour} hours {minute} mins {sec} secs"
            print(info)
            logger(info)
    scheduler.step()
    return mean_loss

@torch.no_grad()
def validate(dataset, dataloader, model, criterion=None):
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
            loss += each_loss.item()
        index = torch.flatten(torch.topk(logits, 1).indices).cpu().detach().numpy()
        max_index.append(index)
    max_index = np.hstack(max_index)
    acc25, acc50, m_iou = dataset.evaluate(max_index)
    loss = loss / len(dataloader)
    return acc25, acc50, m_iou, loss

def train(args, train_dataset, val_dataset, model, criterion, optimizer, scheduler, epoch, logger=None):
    start = time()
    best_score = {
        'acc25': -1, 'acc50': -1, 'm_iou': -1
    }
    best_info = ""
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    val_dataloader_list = [
        DataLoader(val_dataset[i], args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers, collate_fn=train_dataset.collate_fn) for i in range(len(val_dataset))
    ]                               
    val_name = ['Val', 'Overfit']
    for ep in range(epoch):
        mean_loss = train_epoch(ep, train_dataloader, model, criterion, optimizer, scheduler, epoch, logger)
        logger.save_model(model, f"epoch_{ep+1}_model.pth")
        if not args.no_evaluate and ((ep+1) % args.val_epoch == 0):
            for i, val_loader in enumerate(val_dataloader_list):
                acc25, acc50, m_iou, loss = validate(val_dataset[i], val_loader, model, criterion=criterion)
                info = f"{val_name[i]} Epoch[{ep+1}]\tacc25={acc25}\tacc50={acc50}\tm_iou={m_iou}\tloss={loss}"
                print(info)
                logger(info)
                if i == 0 and acc25 > best_score['acc25']:
                    logger.save_model(model, f"best_model.pth")
                    best_score['acc25'] = acc25
                    best_score['acc50'] = acc50
                    best_score['m_iou'] = m_iou
                    best_info = f"Best Epoch[{ep+1}]\tacc25={acc25}\tacc50={acc50}\tm_iou={m_iou}"
                    print(best_info)
                    logger(best_info)
    print(best_info)
    logger(best_info)

    used_time = time() - start
    sec = int(used_time % 60)
    minute = int(used_time % 3600 // 60)
    hour = int(used_time % (24 * 3600) // 3600)
    day = int(used_time // (24 * 3600))
    print(f"-- Totally use {day} days {hour} hour {minute} minute {sec} sec")

def main(args):
    set_random_seed(args.seed)

    print("Create dataset")
    train_dataset = create_dataset(args, 'train')
    val_dataset1 = create_dataset(args, 'test')
    # val_dataset2 = create_dataset(args, 'train')
    # val_dataset = [val_dataset1, val_dataset2]
    val_dataset = [val_dataset1]
                                
    print("Create Model")
    model = create_model(args).cuda()

    print("Create Logger")
    logger = Logger(args.work_dir)

    print("Create optimizer")
    param_list=[
            {'params':model.point_cloud_extractor.parameters(),'lr':args.lr},
            {'params':model.image_extractor.parameters(),'lr':args.lr},
            {'params':model.fusion.parameters(),'lr':args.lr},
            {'params':model.matching.parameters(), 'lr':args.lr_bert},
        ]
    optimizer = torch.optim.AdamW(param_list, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Run")
    train(args, train_dataset, val_dataset, model, criterion, optimizer, scheduler, args.epoch, logger)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)