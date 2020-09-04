# YOLOv4

## yolov4相比yolov3的新特性

- [x] Mosaic data pretreatment
- [ ] CutMix
- [ ] DropBlock
- [x] CIOU loss function
- [x] CSP(Cross-Stage-Partial-connection)-Darknet53
- [x] SPPNet
- [x] PANet
- [x] Mish activation
- [x] CosineAnnealing
- [x] Label Smoothing
- [x] Focal loss
- [ ] Self-GAN



## 快速开始

1. 下载代码

```python
$ git clone https://github.com/Runist/YOLOv4.git
```

2. 安装依赖库

```python
$ pip install -r requirements.txt
```

3. 下载权重文件

```python
$ wget https://github.com/Runist/YOLOv4/releases/download/v1.0/pretrain_model.h5   
```

4. 将config/config.py中的文件信息修改至你的路径下

5. 修改predict/predcit.py的图像路径，运行预测代码

```python
$ cd predict
$ python predict.py
```


![show.png](https://i.loli.net/2020/09/04/pRtZ5FYhNc72olu.png)



## 改进

### Mosaic

作者将裁剪过的四张图片合成一张图片，虽然这在我们肉眼下图片内容没有发生什么改变。但从计算机科学的角度出发， 这改变了数据集的数据分布，丰富了背景信息。但直观上看，将四张图合成一张图，那么图片上的先验框就大大缩小了。所以先验框整体都比较大的时候，可以使用Mosaic增强。

### CIOU loss

IoU是比值的概念，对目标物体的scale是不敏感的。在 YOLOv3 中，是用先验框和预测框的wh以及xy计算均方误差。**然而常用的BBox的回归损失优化和IoU优化不是完全等价的，寻常的IoU无法直接优化没有重叠的部分。**

于是有人提出直接使用IOU作为回归优化loss，CIOU是其中非常优秀的一种想法。

CIOU将目标与anchor之间的距离，重叠率、尺度以及惩罚项都考虑进去，**使得目标框回归变得更加稳定，不会像IoU和GIoU一样出现训练过程中发散等问题。而惩罚因子把预测框长宽比拟合目标框的长宽比考虑进去。**

<img src="https://i.loli.net/2020/09/02/wtRgoajvGnq52AZ.png" alt="20200425144646161.png" style="zoom:50%;" />

CIOU公式如下：
$$
CIOU = IOU -  \frac{\rho(b, {bt}^{gt})}{c^2} - \alpha v
$$
其中，$\rho(b, {bt}^{gt})$分别**代表了预测框和真实框的中心点的欧式距离。 c代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离。**

而$\alpha$和$v$的公式如下：
$$
\alpha = \frac{v}{1 - IOU + v}
$$

$$
v = \frac{4}{\pi^2(arctan\frac{w^{gt}}{h^{gt}} - arctan\frac{w}{h})^2}
$$

把1-CIOU就可以得到相应的LOSS了。
$$
LOSS_{CIOU} = 1 - IOU + \frac{\rho(b, {bt}^{gt})}{c^2} + \alpha v
$$
**以上两点是比较值得详细说一下的**



## config - 配置文件

较为常用的配置文件一般是cfg、json格式的文件，因为没有原作者的框架复杂，所以在配置文件采用的是py格式，也方便各个文件的调用。在config.py中也有了较为详细的注释，如果有不懂欢迎在issues中提问。



## 如何训练自己的数据集

首先你的数据集放在什么位置都可以。其次你需要在config.py的annotation_path指定你的图片与真实框信息，如：D:/YOLOv3/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg 156,89,344,279,19

如果想要使用ImageNet的预训练权重

```python
$ https://github.com/Runist/YOLOv4/releases/download/v1.0/pretrain_model.h5
$ https://github.com/Runist/YOLOv4/releases/download/v1.0/pretrain_tiny_model.h5
```

配置好config.py文件后，运行如下代码

```python
$ python train.py
```



## 一些问题和可能会出现的Bug

1. 在pycharm中，import不同文件夹下的包没有问题。但在linux的terminal中调用则会包ImportError。
	- 解决方法是import sys，之后添加sys.path.append("your project path")
2. 在使用pretrain的方式训练模型，必须要指定pretrain_weights_path预训练模型的路径。

## 安装



## Reference

- https://github.com/Ma-Dan/keras-yolo4
- https://github.com/bubbliiiing/yolov4-keras