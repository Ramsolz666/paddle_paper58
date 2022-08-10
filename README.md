# TS2Vec: Towards Universal Representation of Time Series

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
    - [5.1 基于Inference的推理]()
    - [5.2 基于Serving的服务化部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()


**注意：**

(1) 目录可以使用[gh-md-toc](https://github.com/ekalinin/github-markdown-toc)生成；

(2) 示例repo和文档可以参考：[AlexNet_paddle](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/README.md)。

## 1. 简介



**论文:** [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466)

**参考repo:** https://github.com/yuezhihan/ts2vec

在此非常感谢`yuezhihan`等人贡献的[Ts2Vec](https://github.com/yuezhihan/ts2vec)，提高了本repo复现论文的效率。

**aistudio体验教程:** [地址](url)


## 2. 数据集和复现精度

格式如下：

- 数据集大小：两个有关时间序列的csv文件，分别为ETTh1和ETTm1
- 数据集下载链接：因为文件小，直接在本地文件夹可找到
- 数据格式：csv

## 3. 准备数据与环境

- 硬件：无要求

- 框架：
  - PaddlePaddle >= 2.2.0
  - Paddlets >= 0.1.0

- 安装指令：

  ```
  python setup.py install
  ```

### 3.2 准备数据

数据在datasets文件夹下


### 3.3 准备模型


预训练模型准备好了，在./output文件夹下


## 4. 开始使用


### 4.1 模型训练

训练命令：

```
python -u tools/train.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --train_model_name latest.pdparams  --epochs=200
```

超参数在tools/train.py已经解释的很清楚

训练结果：

ETTh1:

![9099c1d4515fe80482ced3c9c0ebcfb](C:\Users\ADMINI~1\AppData\Local\Temp\WeChat Files\9099c1d4515fe80482ced3c9c0ebcfb.jpg)

### 4.2 模型评估

ETTh1:

```
python -u tools/eval.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --train_model_name latest.pdparams  --epochs=200
```

![097af1bc08f5af9617bdb873d8dd05f](C:\Users\ADMINI~1\AppData\Local\Temp\WeChat Files\097af1bc08f5af9617bdb873d8dd05f.png)

ETTm1:

```
python -u tools/eval.py ETTm1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --train_model_name latest.pdparams  --epochs=200
```



## 6. 自动化测试脚本

参考


## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献