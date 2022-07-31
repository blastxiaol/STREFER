# STRefer

# TODO LIST

1. 1. [x] ~~数据集重新划分（把没有前一帧的数据清洗掉）~~
   
2. 2. [ ] 用pointnet++的backbone重新进行单帧输入/多帧输入的实验。
   1. [x] ~~单帧输入三组实验（pos_emb, view-base, vel-base)~~
   2. [x] 多帧输入三组实验 (但是效果不如单帧输入)
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
**Constrained Condition:**
$$ j_{match}^i = argmax_{i}(\left\{
                \begin{array}{ll}
                  x ,\ \text{ if } D(P_{i}(t), P_{j}(t-1)) < d_{thr}\\
                  -1 ,\ \text{ else}
                \end{array}
              \right.)
 $$

$$\text{where } j_{match}^{i} \text{ is the index of matched object in the previous frame for the i-th object in the current frame, } \\
D(\cdot) \text{ is the distance of two points. } P \text{ is the center position of an object. } \\
d_{thr} \text{ is a hyper-paramenter meaning distance threshold which is set as } 1m \text{ here. }$$

# Experiments
## Detection
| P@0.25 | R@0.25 | P@0.5 | R@0.5 |
|:------:|:------:|:-----:|:-----:|
| 88.29 | 90.97| 77.45 | 79.81 |

## Grounding
### Resnet34 & Pointnet++ (single frame input)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  |
|:-----------------:|:--------:|:-------:|:-----:|
| Ours pos-emb      |   42.40  |  39.77  | 32.73 |   (35/80)
| Ours view-emb     |   44.93  |  42.03  | 34.53 |   (25/80)
| Ours view-vel-emb |   43.34  |  40.43  | 33.04 |   (35/80)

### Resnet34 & Pointnet++ (multi frame input) (Only use current position)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  | (best Epoch / 目前训练了的Epoch数   总共80个Epoch) |
|:-----------------:|:--------:|:-------:|:-----:|:-----:|
| Ours pos-emb      |   42.21  |  40.43  | 33.00 | (27/55) |
| Ours view-emb     |   42.03  |  39.59  | 32.34 | (25/55) |
| Ours view-vel-emb |   41.74  |  39.31  | 32.09 | (15/51) |

### Resnet34 & Pointnet++ (multi frame input) (Only use current position)
|       Method      | Acc@0.25 | Acc@0.5 | mIOU  | (best Epoch / 目前训练了的Epoch数   总共80个Epoch) |
|:-----------------:|:--------:|:-------:|:-----:|:-----:|
| Ours pos-emb      |   42.96  |  40.15  | 32.95 | (25/27)
| Ours view-emb     |   37.43  |  34.90  | 28.61 | (16/29)
| Ours view-vel-emb |   41.74  |  39.12  | 32.13 | (15/28)