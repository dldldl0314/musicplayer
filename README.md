# 卷积神经网络工作原理

> 本文介绍计算机视觉常用的卷积神经网络，并讲解应用领域、卷积、池化（下采样）、全连接、梯度下降和反向传播算法。

## 1.卷积神经网络的应用领域

卷积神经网络长期以来是图像识别领域的核心算法之一，并在学习数据充足时有稳定的表现 。对于一般的大规模图像分类问题，卷积神经网络可用于构建阶层分类器（hierarchical classifier），也可以在精细分类识别（fine-grained recognition）中用于提取图像的判别特征以供其它分类器进行学习 。对于后者，特征提取可以人为地将图像的不同部分分别输入卷积神经网络，也可以由卷积神经网络通过非监督学习自行提取。主要可用于图像识别，图像分类，图像分割等。

**图像分类**
![](https://picx.zhimg.com/v2-6fcd243bab7ae6733aa3fe2632dc5093_720w.jpg?source=d16d100b)

## 2.卷积神经网络的组成

一般来说，最简单的卷积神经网络的组成包含**卷积**、**池化**和**全连接**，接下来将一一讲述每个名词代表的含义。

### 卷积

卷积的本质：将原图中符合卷积核特征的特征提取出来，展示在feature map里面。

![](https://upload-images.jianshu.io/upload_images/29386378-f024266951d70254.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

无论图中的“X”和“O“经过怎样的变换，只要符合特征，都可以进行识别。
![](https://upload-images.jianshu.io/upload_images/29386378-62832ea8bae5536b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
以”X“为例，将这个图以计算机的语言表达出来就是这样的，那么是如何识别他们俩其实都是”X“的呢,就是通过相同的特征图进行对比验证来得出结论，这两个属于同一类图片。**绿色，橙色，紫色的框就是他的卷积核。**
![](https://upload-images.jianshu.io/upload_images/29386378-7d1f72d45592d173.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
如果原图是X，卷积核是X，那么卷积核在原图上卷积运算之后生成的feature map也是X。

如果原图是O，卷积核是O，那么卷积核在原图上卷积运算之后生成的feature map也是O。

如果原图是O，卷积核是X，那么卷积核在原图上卷积运算之后生成的feature map就是乱码。

下图的动画中，绿色表示原图像素值，红色数字表示卷积核中的参数，黄色表示卷积核在原图上滑动。右图表示卷积运算之后生成的feature map。
![](https://pic1.zhimg.com/v2-6428cf505ac1e9e1cf462e1ec8fe9a68_720w.webp?source=d16d100b)
下图展示了RGB三个通道图片的卷积运算过程，共有两组卷积核，每组卷积核都有三个filter分别与原图的RGB三个通道进行卷积。每组卷积核各自生成一个feature map。
![](https://picx.zhimg.com/v2-3e802098ed14b5c14cb9fc0219921bf5_720w.webp?source=d16d100b)
用这三个特征图对原图进行卷积，可以得到如下三个feature map。
![](https://upload-images.jianshu.io/upload_images/29386378-9bd9b9f7db4ec2b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 激活函数（ReLU函数）

激活函数的主要作用是提供网络的非线性建模能力。具体来说就是保留特征，去除一些数据中是的冗余在进行矩阵运算时，负数相乘是比较难计算的，但是使用ReLU激活函数就可以大大缩小计算量。
![](https://pic2.zhimg.com/80/v2-cdaff21d5ae508887c5902353b36bf1d_720w.webp)
**RELU优点：**

x>0 时，梯度恒为1，无梯度弥散问题，收敛快；

增大了网络的稀疏性。当x<0 时，该层的输出为0，训练完成后为0的神经元越多，稀疏性越大，提取出来的特征就约具有代表性，泛化能力越强。即得到同样的效果，真正起作用的神经元越少，网络的泛化性能越好且运算量很小；RELU函数的导数计算更快。

**RELU不足：**

ReLU强制的稀疏处理可能会屏蔽太多，导致模型无法学习到有效特征。由于ReLU在x<0是梯度为0，这样导致负的梯度在这个ReLU被置零，而且这个神经元有可能再也不会被任何数据激活，称为神经元“坏死”,没有更新。

ReLU后的卷积图如图所示
![](https://upload-images.jianshu.io/upload_images/29386378-726920daf04799ba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 池化（下采样）

池化（Pooling）也叫做下采样（subsampling），用一个像素代替原图上邻近的若干像素，在保留feature map特征的同时压缩其大小。
**池化的作用：**
主要用于特征降维，压缩数据和参数的数量，减小过拟合，同时提高模型的容错性。
**池化的分类：**

1. 最大池化（Max Pooling）：选取最大的，我们定义一个空间邻域（比如，2*2的窗口），并从窗口内的修正特征图中取出最大的元素，最大池化被证明效果更好一些。类比近视眼摘掉眼镜，看到的是最明显的那一个。
  
2. 平均池化（Average Pooling）：平均的，我们定义一个空间邻域（比如，2*2的窗口），并从窗口内的修正特征图中算出平均值。
  ![](https://picx.zhimg.com/v2-162b5e2c3bcc2bf1c37d02dbe4ab0503_720w.jpg?source=d16d100b)
  池化的具体方式如图所示
  ![](https://picx.zhimg.com/v2-15e89ec6a866be1f7130655527079786_720w.webp?source=d16d100b)
  最外圈补0：zero padding，可以提取图像边缘的特征。
  关于”X“图的feature map池化之后如下图所示
  
  ![](https://upload-images.jianshu.io/upload_images/29386378-e3e046f30be280d8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
  
  ”X“图经过卷积，激活函数，池化之后是这样的。
  
  ![](https://upload-images.jianshu.io/upload_images/29386378-cd7ae0a4cb6c9cfb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
  
  ![](https://upload-images.jianshu.io/upload_images/29386378-29372ad593c4e8a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
  
  ### 全连接
  
  全连接是指输出的每个神经元都和上一层每一个神经元连接。将池化之后的feature map伸展开，形成神经元。在全连接层中，每个神经元都与上一层的所有神经元相连，每个输入特征都与每个神经元之间都存在一定的连接权重。在训练过程中，神经网络通过反向传播算法来优化每个神经元的权重和偏置，从而使得输出结果能够更好地拟合训练数据。
  ![](https://upload-images.jianshu.io/upload_images/29386378-2774795880fe0ece.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
  
  卷积核的数量、大小、移动步长、补0的圈数是事先人为根据经验指定的，全连接层隐藏层的层数、神经元个数也是人为根据经验指定的（这叫做超参数），但其内部的参数是训练出来的。
  

![](https://picx.zhimg.com/v2-962b574e8da50a30483e767c3088a69c_720w.jpg?source=d16d100b)

### 梯度下降

**损失函数**：计算神经网络的推测结果与图片真实标签的差距，构造损失函数，训练的目标就是将找到损失函数的最小值。

**梯度下降**：将损失函数对各种权重、卷积核参数求导，慢慢优化参数，找到损失函数的最小值。

![](https://upload-images.jianshu.io/upload_images/29386378-7bcf0153af74fb06.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://picx.zhimg.com/v2-6b7ce65bbfbb938c887ed47ffcce0ee0_720w.jpg?source=d16d100b)

**随机梯度下降**：在减小损失函数的过程中，采用步步为营的方法，单个样本单个样本输入进行优化，而不是将全部样本计算之后再统一优化。虽然个别样本会出偏差，但随着样本数量增加，仍旧能够逐渐逼近损失函数最小值。

### 反向传播（Backpropagation）

将神经网络得出的结果与真实的结果进行对比，然后返回给卷积核，调整神经元参数。这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数。
