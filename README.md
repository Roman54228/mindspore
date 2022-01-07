# Contents

- [directory](#directory)
- [ResNeSt Instructions] (#resnest Instructions)
- [model-architecture](#model-architecture)
- [dataset](#dataset)
- [feature](#feature)
     - [mixed precision](#mixed precision)
- [environmental requirements] (#environmental requirements)
- [Script description](#Script description)
     - [Script and Sample Code](#Script and Sample Code)
     - [script parameters](#script parameters)
     - [training process](#training process)
         - [usage](#training usage)
         - [examples](#training examples)
     - [evaluation process](#evaluation process)
         - [usage](#evaluate usage)
         - [example](#evaluation example)
         - [results](#evaluation results)
     - [inference process](#inference process)
         - [model export](#model export)
         - [usage](#inference usage)
         - [example](#reasoning example)
         - [result](#inference result)
- [model description](#model description)
     - [performance](#performance)
         - [training performance](#training performance)
         - [inference performance](#inference performance)
- [Random Situation Description](#Random Situation Description)
- [ModelZoo homepage] (#modelzoo homepage)

# ResNeSt Description

ResNeSt is a highly modular image classification network architecture. ResNeSt is designed as a unified, multi-branch architecture that requires only a few hyperparameters to be set. This strategy provides a new dimension, which we call "cardinality" (the size of the transform set), which is an important factor in addition to the depth and width dimensions.

[paper](https://arxiv.org/abs/2004.08955)：  Hang Zhang, Chongruo Wu, Alexander Smola et al. ResNeSt: Split-Attention Networks. 2020.

# Model Architecture

ResNeSt The overall network architecture is as follows:

[Link](https://arxiv.org/abs/2004.08955)

# Dataset

Dataset used：[ImageNet](http://www.image-net.org/)

- Dataset size：1000 classes in total，containing 1.28M colorful images
    - Training set：120G，1.28M images
    - Test set：5G，50k images
- Data format：RGB image.
    - Note: Data will be processed in src/datasets。

# characteristic

## Mixed precision

The training method using [mixed precision](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html) uses support for single-precision and half-precision data to improve the training speed of deep learning neural networks , while maintaining the network accuracy that single-precision training can achieve. Mixed-precision training increases computational speed and reduces memory usage while enabling training of larger models or larger batches on specific hardware.

Taking the FP16 operator as an example, if the input data type is FP32, the MindSpore background will automatically reduce the precision to process the data. You can open the INFO log and search for "reduce precision" to view operators with reduced precision.

# Environmental requirements

- Hardware (Ascend)
    - Use the Ascend processor to build the hardware environment.
- frame
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# script description

## Script and sample code

```path
.
└─ResNeSt50
  ├─README.md
  ├─scripts
    ├─run_train.sh
    ├─run_eval.sh
    ├─run_distribute_train.sh              # Start Ascend distributed training（8p）
    ├─run_distribute_eval.sh               # Start Ascend distributed evaluation (8p)
    └─run_infer_310.sh                     # 启动310推理
  ├─src
    ├─datasets
      ├─autoaug.py                  # 随机数据增强方法
      ├─dataset.py                  # 数据集处理
    ├─models
      ├─resnest.py                  # ResNeSt50网络定义
      ├─resnet.py                   # 主干网络
      ├─splat.py                    # split-attention
      ├─utils.py                    # 工具函数：网络获取、加载权重等
    ├─config.py                       # 参数配置
    ├─crossentropy.py                 # 交叉熵损失函数
    ├─eval_callback.py                # 推理信息打印
    ├─logging.py                      # 日志记录
  ├──eval.py                          # 评估网络
  ├──train.py                         # 训练网络
  ├──export.py                        # 导出Mindir接口
  ├──create_imagenet2012_label.py     # 创建数据集标签用于310推理精度验证
  ├──postprocess.py                   # 后处理
  ├──README.md                        # README文件
```

## 脚本参数

在config.py中可以同时配置训练和评估参数。

```python
"net_name": 'resnest50'                   # 网络选择
"root": '/mass_data/imagenet/imagenet/'   # 数据集路径
"num_classes": 1000,                      # 数据集类数
"base_size": 224,                         # 图像大小
"crop_size": 224,                         # crop大小
"label_smoothing": 0.1,                   # 标签平滑
"batch_size": 64,                         # 输入张量的批次大小，不能超过64
"test_batch_size": 64,                    # 测试批次大小
"last_gamma": True,                       # zero bn last gamma
"final_drop": 1.0,                        # final_drop
"epochs": 270,                            # epochs
"start_epoch": 0,                         # start epochs
"num_workers": 64,                        # num_workers
"lr": 0.025,                              # 基础学习率,多卡训练乘以卡数
"lr_scheduler": 'cosine_annealing',       # 学习率模式
"lr_epochs": '30,60,90,120,150,180,210,240,270',            # LR变化轮次
"lr_gamma": 0.1,                          # 减少LR的exponential lr_scheduler因子
"eta_min": 0,                             # cosine_annealing调度器中的eta_min
"T_max": 270,                             # cosine_annealing调度器中的T-max
"max_epoch": 270,                         # 训练模型的最大轮次数量
"warmup_epochs" : 5,                      # 热身轮次
"weight_decay": 0.0001,                   # 权重衰减
"momentum": 0.9,                          # 动量
"is_dynamic_loss_scale": 0,               # 动态损失放大
"loss_scale": 1024,                       # 损失放大
"disable_bn_wd": True,                    # batchnorm no weight decay
```

## 训练过程

### 训练用法

首先需要在`src/config.py`中设置好超参数以及数据集路径等参数，接着可以通过脚本或者.py文件进行训练

您可以通过python脚本开始训练：

```shell
python train.py --outdir ./output --device_target Ascend
```

或通过shell脚本开始训练：

```shell
Ascend:
    # 分布式训练示例（8卡）
    bash run_distribute_train.sh RANK_TABLE_FILE OUTPUT_DIR
    # 单机训练
    bash run_train.sh OUTPUT_DIR
```

### 训练样例

```shell
# Ascend分布式训练示例（8卡）
bash scripts/run_distribute_train.sh RANK_TABLE_FILE OUTPUT_DIR
# Ascend单机训练示例
bash scripts/run_train.sh OUTPUT_DIR
```

您可以在日志中找到检查点文件和结果。

## 评估过程

### 评估用法

您可以通过python脚本开始评估：

```shell
python eval.py --outdir ./output --resume_path ~/resnest50-270_2502.ckpt
```

或通过shell脚本开始训练：

```shell
# 评估
bash run_eval.sh OUT_DIR PRETRAINED_CKPT_PATH
```

PLATFORM is Ascend, default is Ascend.

### 评估样例

```shell
# 检查点评估
bash scripts/run_eval.sh OUT_DIR PRETRAINED_CKPT_PATH

#或者直接使用脚本运行
python eval.py --outdir ./output --resume_path ~/resnest50-270_2502.ckpt
```

### 评估结果

评估结果保存在脚本路径`/scripts/EVAL_LOG/`下。您可以在日志中找到类似以下的结果。

```log
acc=80.90%(TOP1)
acc=95.51%(TOP5)
```

## 推理过程

在Ascend310执行推理，执行推理之前，需要通过`export.py`文件导出MINDIR模型

### 模型导出

```shell
python export.py --device_id [DEVICE_ID] --ckpt_file [CKPT_PATH] --net_name [NET_NAME] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "ONNX", "MINDIR"].

### 推理用法

通过shell脚本编译文件并在310上执行推理

```shell
# 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

PLATFORM is Ascend310, default is Ascend310.

### 推理样例

```shell
# 直接使用脚本运行
bash run_infer_310.sh /home/stu/lds/mindir/resnest50.mindir /home/MindSpore_dataset/ImageNet2012/val 0
```

### 推理结果

评估结果保存在脚本路径`/scripts/`下。您可以在`acc.log`找到精度结果，在`infer.log`中找到性能结果

```log
acc=0.8088(TOP1)
acc=0.9548(TOP5)
```

# 模型描述

## 性能

### 训练性能

| 参数                       | ResNeSt50                                                  |
| -------------------------- | ---------------------------------------------------------- |
| 资源                       | Ascend 910；CPU：2.60GHz，192核；内存：755GB               |
| 上传日期                   | 2021-11-09                                                 |
| MindSpore版本              | 1.3                                                        |
| 数据集                     | ImageNet                                                   |
| 训练参数                   | src/config.py                                              |
| 优化器                     | Momentum                                                   |
| 损失函数                   | Softmax交叉熵                                              |
| 损失                       | 1.466                                                      |
| 准确率                     | 80.9%(TOP1)                                                |
| 总时长                     | 84h21m39s （8卡）                                          |
| 调优检查点                 | 223 M（.ckpt文件）                                         |

### 推理性能

| 参数                       |                      |
| -------------------------- | -------------------- |
| 资源                       | Ascend 910           |
| 上传日期                   | 2021-11-09           |
| MindSpore版本              | 1.3                  |
| 数据集                     | ImageNet， 5万       |
| batch_size                 | 1                    |
| 输出                       | 分类准确率           |
| 准确率                     | acc=80.9%(TOP1)      |

# 随机情况说明

dataset.py中设置了“ImageNet”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。

