# 行人属性识别项目（可扩展至其他属性识别）
**From 樊一超**   
这是一个基于pytorch的简单项目，用于在**Market-1501-attribute**和**DukeMTMC-reID-attribute**两个开源数据集上进行**行人属性**[pedestrian attribute recognition]的评估训、测试。    

## 1. 数据集的准备
### 1.1 下载
本人已经将数据集下载好了，存放至dataset及外部的DataSet文件夹。如果需要也可以在线下载：   
[Market-1501-attribute](https://github.com/vana77/Market-1501_Attribute)    
[DukeMTMC-reID-attribute](https://github.com/vana77/DukeMTMC-attribute)    
**数据集说明：**（1）Market-1501-attribute是在清华大学校园内采集，图像来自6个不同的摄像头，其中有一个摄像头为低像素，该数据集包含了训练集12936张图像，测试集19732张图像（训练集一共有751人、测试集一共有750人），图像由检测器自动检测并切割，包含了一些检测误差（接近实际使用情况）。（2）DukeMTMC-reID-attribute是DukeMTMC-attribute的子集，是在杜克大学内采集，图像来自8个不同的摄像头，包含了训练集16522张图像，测试集17661张图像（训练集一共有702人、测试集一共有1110人），是最大的行人重识别数据集，并且提供了行人属性的标注。

### 1.2 解压整理使用
将下载好的数据集进行整理，得到本项目可以使用的格式，同时在目录中创建attribute目录，将标注好的mat文件放进去 如下所示：    
```bash
    dataset/
    ├── DukeMTMC-reID #DukeMTMC-reID-attribute数据集的目录结构
    │   ├── attribute  
    │   │   └── duke_attribute.mat
    │   ├── bounding_box_test
    │   ├── bounding_box_train
    │   └── query
    └── Market-1501 #Market-1501-attribute数据集的目录结构
        ├── attribute  #his file contains "market_attribute.mat".
        │   └── market_attribute.mat
        ├── bounding_box_test  #This file contains 19732 images for testing. 
        ├── bounding_box_train #This file contains 12936 images for training
        ├── gt_bbox
        ├── gt_query
        ├── query #It contains 3368 query images. Search is performed in the "bounding_box_test" file.
        └── readme.txt
```
### 1.3 标签说明
两个开源数据集的标签各不相同，具体如下：   
* Market-1501-attribute标签  
共有27个属性

The 27 attributes are: 

| attribute | representation in file | label |
| :----: | :----: | :----: |
| gender | gender | male(1), female(2) |
| hair length | hair| short hair(1), long hair(2)    |
| sleeve length | up | long sleeve(1), short sleeve(2) |
| length of lower-body clothing | down | long lower body clothing(1), short(2)    |
| type of lower-body clothing| clothes| dress(1), pants(2)    |
| wearing hat| hat | no(1), yes(2) |
| carrying backpack| backpack | no(1), yes(2) |
| carrying bag| bag | no(1), yes(2) |
| carrying handbag| handbag | no(1), yes(2) |
| age| age | young(1), teenager(2), adult(3), old(4) |
| 8 color of upper-body clothing| upblack, upwhite, upred, uppurple, upyellow, upgray, upblue, upgreen | no(1), yes(2) |
| 9 color of lower-body clothing| downblack, downwhite, downpink, downpurple, downyellow, downgray, downblue, downgreen,downbrown | no(1), yes(2) |

Note that the though there are 8 and 9 attributes for upper-body clothing and lower-body clothing, only one color is labeled as yes (2) for an identity.


* DukeMTMC-reID-attribute标签   
共有23个属性  

| attribute | representation in file | label |
| :----: | :----: | :----: |
| gender | gender | male(1), female(2) |
| length of upper-body clothing | top | short upper body clothing(1), long(2)    |
| wearing boots| boots| no(1), yes(2)    |
| wearing hat| hat | no(1), yes(2) |
| carrying backpack| backpack | no(1), yes(2) |
| carrying bag| bag | no(1), yes(2) |
| carrying handbag| handbag | no(1), yes(2) |
| color of shoes| shoes | dark(1), light(2) |
| 8 color of upper-body clothing| upblack, upwhite, upred, uppurple, upgray, upblue, upgreen, upbrown | no(1), yes(2) |
| 7 color of lower-body clothing| downblack, downwhite, downred, downgray, downblue, downgreen, downbrown | no(1), yes(2) |


Note that the though there are 7 and 8 attributes for lower-body clothing and upper-body clothing, only one color is labeled as yes (2) for an identity.



## 2. 预训练模型的准备
从[网盘](https://pan.baidu.com/s/1bByCxZp9bSs8YYZPbuK21A)中获取本工程在duck及market两个数据集上的预训练权重（训练效果较好可以直接使用）    
网盘提取码：jpks    
本人下载好，且保存至如下目录，可以直接使用。
```bash
    checkpoints/
    ├── duke
    │   └── resnet50_nfc
    │       ├── net_last.pth
    │       └── train.jpg
    └── market
        └── resnet50_nfc
            ├── net_last.pth
            └── train.jpg
```
**注意：** 路径在代码中已设置好，不需要修改名称。

## 3. 训练前准备
### 3.1 环境准备
训练镜像：recognition，获取方式docker pull fanacio/recognition:v0 或者本地保存镜像recognition.tar    
具体依赖体现在：    
* Python 3.5
* PyTorch >= 0.4.1
* torchvision >= 0.2.1
* matplotlib, sklearn, prettytable (optional)

### 3.2 预训练权重测试（同训练后测试方法一致）
- part1：测试Market-1501数据集   
执行命令：
```bash
python test.py --data-path dataset/ --dataset market --backbone  resnet50
```
获得结果保存在result/market/resnet50_nfc/acc.mat中,如下所示：
```
+------------+----------+-----------+--------+----------+
| attribute  | accuracy | precision | recall | f1 score |
+------------+----------+-----------+--------+----------+
|   young    |  0.998   |   0.533   | 0.267  |  0.356   |
|  teenager  |  0.892   |   0.927   | 0.951  |  0.939   |
|   adult    |  0.895   |   0.582   | 0.450  |  0.508   |
|    old     |  0.992   |   0.037   | 0.012  |  0.019   |
|  backpack  |  0.883   |   0.828   | 0.672  |  0.742   |
|    bag     |  0.790   |   0.608   | 0.378  |  0.467   |
|  handbag   |  0.893   |   0.254   | 0.065  |  0.104   |
|  clothes   |  0.946   |   0.956   | 0.984  |  0.970   |
|    down    |  0.945   |   0.968   | 0.949  |  0.959   |
|     up     |  0.936   |   0.938   | 0.998  |  0.967   |
|    hair    |  0.877   |   0.871   | 0.773  |  0.819   |
|    hat     |  0.982   |   0.812   | 0.505  |  0.623   |
|   gender   |  0.919   |   0.947   | 0.864  |  0.903   |
|  upblack   |  0.954   |   0.859   | 0.790  |  0.823   |
|  upwhite   |  0.926   |   0.846   | 0.882  |  0.863   |
|   upred    |  0.974   |   0.904   | 0.840  |  0.871   |
|  uppurple  |  0.985   |   0.703   | 0.815  |  0.755   |
|  upyellow  |  0.976   |   0.895   | 0.836  |  0.865   |
|   upgray   |  0.909   |   0.852   | 0.391  |  0.537   |
|   upblue   |  0.946   |   0.868   | 0.420  |  0.566   |
|  upgreen   |  0.966   |   0.790   | 0.713  |  0.750   |
| downblack  |  0.879   |   0.815   | 0.889  |  0.850   |
| downwhite  |  0.956   |   0.608   | 0.550  |  0.578   |
|  downpink  |  0.989   |   0.795   | 0.782  |  0.788   |
| downpurple |  1.000   |     -     |   -    |    -     |
| downyellow |  0.999   |   0.000   | 0.000  |  0.000   |
|  downgray  |  0.878   |   0.756   | 0.443  |  0.559   |
|  downblue  |  0.861   |   0.762   | 0.446  |  0.563   |
| downgreen  |  0.978   |   0.766   | 0.295  |  0.426   |
| downbrown  |  0.958   |   0.754   | 0.590  |  0.662   |
+------------+----------+-----------+--------+----------+
Average accuracy: 0.9361
Average f1 score: 0.6492
```

- part2：测试DukeMTMC-reID-attribute数据集   
执行命令：
```bash
python test.py --data-path dataset/ --dataset duke --backbone  resnet50
```
获得结果保存在result/duke/resnet50_nfc/acc.mat中,如下所示：
```bash
+-----------+----------+-----------+--------+----------+
| attribute | accuracy | precision | recall | f1 score |
+-----------+----------+-----------+--------+----------+
|  backpack |  0.829   |   0.794   | 0.926  |  0.855   |
|    bag    |  0.836   |   0.496   | 0.287  |  0.364   |
|  handbag  |  0.935   |   0.469   | 0.073  |  0.126   |
|   boots   |  0.905   |   0.784   | 0.791  |  0.787   |
|   gender  |  0.858   |   0.806   | 0.828  |  0.817   |
|    hat    |  0.898   |   0.883   | 0.680  |  0.768   |
|   shoes   |  0.916   |   0.756   | 0.414  |  0.535   |
|    top    |  0.893   |   0.590   | 0.381  |  0.463   |
|  upblack  |  0.821   |   0.827   | 0.903  |  0.864   |
|  upwhite  |  0.959   |   0.750   | 0.509  |  0.606   |
|   upred   |  0.973   |   0.745   | 0.649  |  0.694   |
|  uppurple |  0.995   |   0.258   | 0.123  |  0.167   |
|   upgray  |  0.900   |   0.611   | 0.333  |  0.432   |
|   upblue  |  0.943   |   0.766   | 0.519  |  0.619   |
|  upgreen  |  0.975   |   0.463   | 0.403  |  0.431   |
|  upbrown  |  0.980   |   0.481   | 0.328  |  0.390   |
| downblack |  0.787   |   0.740   | 0.807  |  0.772   |
| downwhite |  0.945   |   0.771   | 0.395  |  0.522   |
|  downred  |  0.991   |   0.739   | 0.645  |  0.689   |
|  downgray |  0.927   |   0.471   | 0.238  |  0.317   |
|  downblue |  0.807   |   0.741   | 0.669  |  0.703   |
| downgreen |  0.997   |     -     |   -    |    -     |
| downbrown |  0.979   |   0.871   | 0.594  |  0.706   |
+-----------+----------+-----------+--------+----------+
Average accuracy: 0.9152
Average f1 score: 0.5739
```

- part3：测试单独的图片  
执行命令：
```bash
python inference.py test_sample/test_market.jpg --dataset duke --backbone  resnet50
```
表示将图片test_market.jpg按照duke的23个属性进行识别，其识别结果如下所示：
```bash
    carrying backpack: no
    carrying bag: yes
    carrying handbag: yes
    wearing boots: no
    gender: female
    wearing hat: no
    color of shoes: dark
    length of upper-body clothing: short upper body clothing
    color of upper-body clothing: white
    color of lower-body clothing: brown
```

## 4. 训练模型
以训练DukeMTMC-reID-attribute数据集为例     
执行命令：
```bash
python train.py --data-path dataset/ --dataset duke --backbone  resnet50
```
**注意：** pth权重文件以及loss损失曲线均保存在weights文件夹下，每epoch十次保存一次权重。

## 5. 转onnx
**用途：** 便于可视化模型结构，以及便于部署使用  
执行inference.py文件（注意传参），将 **'--pytorch2onnx'** 传参开关打开即可。

## 6. 其他
**mat文件转csv格式**  
[参考链接](https://blog.csdn.net/hequnwang10/article/details/120311978)    
为了便于查看mat文件中的内容，将其转成csv格式。mat文件不乏有两种，一种是数据集标注的信息，另一种是测试结果（acc.mat）    
在训练中，主要用到的是duke_attribute.mat和market_attribute.mat，其对训练图片按照规定的属性进行了标注。  
直接进入mat2csv目录执行mat2csv.py文件即可。

## 7. 一些关于训练的思考
**对代码的思考**    
代码中写的比较死板，将训练集的名字、目录格式以及属性个数和类别都固定了，如果我们需要训练自己的属性识别模型，则这部分代码需要修改，最好的修改方法是边调试边改。

**对于训练集格式的思考**    
该工程无论是训练还是测试，其真值的标签信息都是以mat的形式保存下来的，这对于python工程而言是不可读的，如若有需求可以改成txt标签或者csv标签。

**关于.mat文件的说明**
关于.mat文件的含义已经存放在./doc/market数据集标签信息.png中，mat表示为一个结构，其中包含了多个mat文件，且mat文件的维度信息完全一致，都为1*n，n表示人数id，mat内容严格按照标签生成。
