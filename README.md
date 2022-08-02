# STRefer

# TODO LIST

1. 1. [x] ~~数据集重新划分（把没有前一帧的数据清洗掉）~~
   
2. 2. [ ] 用pointnet++的backbone重新进行单帧输入/多帧输入的实验。
   1. [x] ~~单帧输入三组实验（pos_emb, view-base, vel-base)~~
   2. [x] ~~多帧输入三组实验 (但是效果不如单帧输入)~~
   3. [x] ~~匹配GT-pred-box~~
   4. [x] ~~多帧GT实验：点云融合，norm各自帧位置~~
   5. [ ] 多帧GT实验：点云融合，norm当前帧位置
   6. [ ] 多帧GT实验：特征融合，norm各自帧位置
   7. [ ] 多帧GT实验：特征融合，norm当前帧位置
3. 3. [ ] 增加3DJCG对比实验
4. 4. [ ] 增加velocity误差
5. 5. [ ] 网络描述细化（参数更详细）
6. 6. [ ] 表格命名修订（更细化、明显）
7. 7. [ ] 细化数据集标注规则流程

# Dataset
数据集划分后：总共71个不同场景，56个场景划为训练场景，15个为测试场景。最终4392条训练集数据，1066条测试数据，测试集中出现的词至少在训练集中出现两次。


# Multi-Object Data Association
$$ F_{v}^i(t-1) = \phi(F_{point}^i(t-1), F_{image}^i(t-1)) $$ 
$$ F_{v}^i(t) = \phi(F_{point}^i(t), F_{image}^i(t)) $$

$$\text{where } F_{v}^i \text{ is the feature of i-th object, } \phi(\cdot) \text{ is concatenation. } \\
t \text{ and }t-1 \text{ are current and previous frames.}$$

<br>
Constrained Condition:
<br>

$$ j_{match}^i = argmax_{i}(\left\{
                \begin{array}{ll}
                  SIM(F_{v}^i(t), F_{v}^j(t-1)) ,\ \text{ if } D(P_{i}(t), P_{j}(t-1)) < d_{thr} \\
                  -1 ,\ \text{ else}
                \end{array}
              \right.) $$

$$\text{where } j_{match}^{i} \text{ is the index of matched object in the previous frame for the i-th object in the current frame, } \\
D(\cdot) \text{ is the distance of two points. } P \text{ is the center position of an object. } \\
d_{thr} \text{ is a hyper-paramenter meaning distance threshold which is set as } 1m \text{ here. }$$

# Experiments
## Detection
| P@0.25 | R@0.25 | P@0.5 | R@0.5 |
|:------:|:------:|:-----:|:-----:|
| 88.29 | 90.97| 77.45 | 79.81 |

## Grounding
### Resnet34 & Pointnet++ (single frames input)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   42.40  |  39.77  | 32.73 |   (35/80)
| Ours view-emb     |   44.93  |  42.03  | 34.53 |   (25/80)
| Ours view-vel-emb |   43.34  |  40.43  | 33.04 |   (35/80)

### Resnet34 & Pointnet++ (multi frames input) (Only use current position)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   42.40  |  39.77  | 32.64 | (71/80) 
| Ours view-emb     |   42.03  |  39.59  | 32.34 | (25/80) 
| Ours view-vel-emb |   41.74  |  39.31  | 32.09 | (15/80) 

### Resnet34 & Pointnet++ (multi frames input) (Use both current and previous position)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   42.96  |  40.15  | 32.95 | (25/80)
| Ours view-emb     |   40.71  |  38.74  | 31.44 | (58/80)
| Ours view-vel-emb |   43.15  |  40.53  | 33.32 | (65/80)

### GT Multi frames (normalize to center)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   43.34  |  40.53  | 33.33 | 
| Ours view-emb     |   47.28  |  44.09  | 35.97 | 
| Ours view-vel-emb |   44.56  |  41.46  | 34.25 | 

### GT Multi frames (normalize to current frame's center)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   43.25  |  40.81  |  33.29  | 
| Ours view-emb     |   45.12  |  41.74  |  34.35  | 
| Ours view-vel-emb |   x  |  x  |  x  | 

### GT Multi frames feature fusion(normalize to center)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   x  |  x  |  x  | 
| Ours view-emb     |   x  |  x  |  x  | 
| Ours view-vel-emb |   x  |  x  |  x  | 

### GT Multi frames feature fusion (normalize to current frame's center)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   x  |  x  |  x  | 
| Ours view-emb     |   x  |  x  |  x  | 
| Ours view-vel-emb |   x  |  x  |  x  | 

# Comparison Experiment
| Method | Acc@0.25 | Acc@0.5 | mIOU |
|:------:|:--------:|:-------:|:----:|
| ScanRefer | x | x | x |
| Referit3d | x | x | x |
| InstanceRefer | x | x | x |
| 3DJCG | 15.38 | 13.60 | 9.87 | 
