# 飞桨训推一体全流程（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：指Linux GPU/CPU环境下的模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度训练。
- 更多部署方式：包括C++预测、Serving服务化部署、ARM端侧部署等多种部署方式，具体列表见[3.3节](#3.3)
- Slim训练部署：包括PACT在线量化、离线量化。
- 更多训练环境：包括Windows GPU/CPU、Linux NPU、Linux DCU等多种环境。


| 算法论文        | 模型名称               | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 |           更多<br>部署方式           |  Slim<br>训练部署  |   更多<br>训练环境    |
| :---------- | :----------------- | :--: | :--------: | :--------: | :----------------------------: | :------------: | :-------------: |
| MobileNetV3 | mobilenet_v3_small |  分类  |     支持     |    混合精度    | PYTHON 服务化部署<br>Paddle2ONNX 部署 | PACT量化<br>离线量化 | Windows GPU/CPU |


## 3. 测试工具简介

### 3.1 目录介绍

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--train_infer_python.txt   # 基础训练推理测试配置文件
    |--docs                                 # 文档目录
    |   |--test_train_inference_python.md   # 基础训练推理测试说明文档
    |----README.md                          # TIPC说明文档
推理测试数据准备脚本
    |----test_train_inference_python.sh     # TIPC基础训练推理测试解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

### 3.2 测试流程概述

使用本工具，可以测试不同功能的支持情况。测试过程包含：

1. 准备数据与环境
2. 运行测试脚本，观察不同配置是否运行成功。

### 3.3 开始测试

Table7:

​	ETTH1

​	log文件的运行结果：

```
[33m Run successfully with command - ts2vec_pp - python -u tools/train.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --train_model_name latest.pdparams  --epochs=200  !  [0m
[33m Run successfully with command - ts2vec_pp - python -u tools/eval.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --eval!  [0m
```

```
Dataset: ETTh1
Arguments: Namespace(batch_size=8, dataset='ETTh1', epochs=None, eval=True, gpu=0, irregular=0, iters=None, loader='forecast_csv_univar', lr=0.001, max_threads=None, max_train_length=3000, pretrained='./output/etth1_univar.pdparams', repr_dims=320, save_every=None, seed=None)
Loading data... done

Training time: 0:00:01.022761

Evaluation result: {'ours': {24: {'norm': {'MSE': 0.038390016385911364, 'MAE': 0.14776651758714798}}, 48: {'norm': {'MSE': 0.0637512134147867, 'MAE': 0.18987994432178865}}, 168: {'norm': {'MSE': 0.17944338340751317, 'MAE': 0.3419601938126656}}, 336: {'norm': {'MSE': 0.2064567545213771, 'MAE': 0.37233820700516024}}, 720: {'norm': {'MSE': 0.5297436319401028, 'MAE': 0.6759938570370054}}}, 'ts2vec_infer_time': 2.824065685272217, 'lr_train_time': {24: 4.474669456481934, 48: 6.300959348678589, 168: 17.28876042366028, 336: 15.36242938041687, 720: 16.05191206932068}, 'lr_infer_time': {24: 0.09398555755615234, 48: 0.09575629234313965, 168: 0.19899225234985352, 336: 0.0061800479888916016, 720: 0.007113218307495117}}
Finished.
```

​	ETTm1

​	log文件的运行结果：

```
Dataset: ETTm1
Arguments: Namespace(batch_size=8, dataset='ETTm1', epochs=None, eval=True, gpu=0, irregular=0, iters=None, loader='forecast_csv_univar', lr=0.001, max_threads=None, max_train_length=3000, pretrained='./output/ettm1_univar.pdparams', repr_dims=320, save_every=None, seed=None)
Loading data... done

Training time: 0:00:01.024457

Evaluation result: {'ours': {24: {'norm': {'MSE': 0.017456819168878934, 'MAE': 0.09744881495978652}}, 48: {'norm': {'MSE': 0.033934608899689564, 'MAE': 0.13710359628606006}}, 96: {'norm': {'MSE': 0.051660657411031284, 'MAE': 0.17229351659676706}}, 288: {'norm': {'MSE': 0.1029469677511266, 'MAE': 0.24887390469968917}}, 672: {'norm': {'MSE': 0.15287677511920267, 'MAE': 0.30764268231279446}}}, 'ts2vec_infer_time': 6.713715076446533, 'lr_train_time': {24: 9.93132209777832, 48: 16.574370622634888, 96: 25.644906520843506, 288: 49.26114273071289, 672: 64.32194876670837}, 'lr_infer_time': {24: 0.19407391548156738, 48: 0.10126638412475586, 96: 0.20571660995483398, 288: 0.49736833572387695, 672: 0.20410656929016113}}
Finished.
```

Table8:

​	ETTH1:

​	log文件的运行结果：

```
Dataset: ETTh1
Arguments: Namespace(batch_size=8, dataset='ETTh1', epochs=None, eval=True, gpu=0, irregular=0, iters=None, loader='forecast_csv', lr=0.001, max_threads=None, max_train_length=3000, pretrained='./output/etth1.pdparams', repr_dims=320, save_every=None, seed=None)
Loading data... done

Training time: 0:00:00.945650

Evaluation result: {'ours': {24: {'norm': {'MSE': 0.6089094000028505, 'MAE': 0.5479893439424226}}, 48: {'norm': {'MSE': 0.6561628439438086, 'MAE': 0.5791434765169219}}, 168: {'norm': {'MSE': 0.7979862224865039, 'MAE': 0.6612289687468902}}, 336: {'norm': {'MSE': 0.9726944243116041, 'MAE': 0.7475516009993391}}, 720: {'norm': {'MSE': 1.191237690642617, 'MAE': 0.8443857220994363}}}, 'ts2vec_infer_time': 2.6978373527526855, 'lr_train_time': {24: 8.460606336593628, 48: 11.070012331008911, 168: 26.86741542816162, 336: 20.701181173324585, 720: 24.993821620941162}, 'lr_infer_time': {24: 0.1014409065246582, 48: 0.0998222827911377, 168: 0.3959536552429199, 336: 0.19062328338623047, 720: 0.2652101516723633}}
Finished.
```

​	ETTm1:

​	log文件的运行结果：

```
Dataset: ETTm1
Arguments: Namespace(batch_size=8, dataset='ETTm1', epochs=None, eval=True, gpu=0, irregular=0, iters=None, loader='forecast_csv', lr=0.001, max_threads=None, max_train_length=3000, pretrained='./output/ettm1.pdparams', repr_dims=320, save_every=None, seed=None)
Loading data... done

Training time: 0:00:01.003275

Evaluation result: {'ours': {24: {'norm': {'MSE': 0.4966471650339161, 'MAE': 0.45478229388912106}}, 48: {'norm': {'MSE': 0.6555301728378048, 'MAE': 0.5497016127242499}}, 96: {'norm': {'MSE': 0.6904265671938428, 'MAE': 0.581893273987098}}, 288: {'norm': {'MSE': 0.7614055076057212, 'MAE': 0.6315719156618663}}, 672: {'norm': {'MSE': 0.8591821037064751, 'MAE': 0.6922732731379537}}}, 'ts2vec_infer_time': 6.648923397064209, 'lr_train_time': {24: 29.046436309814453, 48: 41.84100937843323, 96: 40.92183828353882, 288: 77.08395218849182, 672: 102.70595598220825}, 'lr_infer_time': {24: 0.2047595977783203, 48: 0.29991793632507324, 96: 0.4040408134460449, 288: 0.7033755779266357, 672: 0.9622237682342529}}
Finished.
```

MAE与MSE误差精度与原论文相比，普遍大一点点，原因可能是paddle函数与原论文torch不太一样，例如AverageModel和init_weight的原因造成的。