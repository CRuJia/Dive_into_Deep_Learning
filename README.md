## Dive into Deep Learning 

本项目为2020年2月份动手学深度学习活动，感谢 **Datawhale & 伯禹教育 & 和鲸科技**

2020.2.11

 #### 1、[linear regression](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/linear_regression.ipynb)
 
 2020.2.12
 
 #### 2、[softmax and classification model](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/2.softmax_and_classification_model.ipynb)
 
 #### 3、[multi layer perceptron](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/3.%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA.ipynb)
 
 2020.2.13
 
 #### 4.[文本预处理](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/4.%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86.ipynb)
  - 对于字符串的处理（re正则表达式） 
  - [ ] 复习python正则表达式
  - 建立词典，统计词频
  - 常用的分词工具（Spacy，NLTK）
  - [ ] 看Spacy,NLTK 文档
  
  2020.2.14
  
 #### 5.[语言模型（基于统计的语言模型）](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/5.%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%95%B0%E6%8D%AE%E9%9B%86.ipynb)
 
 - 马尔科夫假设（一个词的出现，只与前面n个词有关）
 - n元语法（n-grams）：基于n-1阶马尔可夫链的语言模型 
 缺陷：1.参数空间过大 2.数据稀疏问题
 - 时序数据的采样：随机采样（在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。）、相邻采样（在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。）
 - [ ] 对于采样再看一下
 
 #### 6.[RNN 基础](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/6.%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.ipynb)
 
 - 每个字符都是字典大小的向量，每个样本都是时间步数个序列，每个batch都是batch_size个样本
 - 第一个(batch_size,num_vocab)的矩阵：取出了第一批样本中的每个序列的第一个字符，并将每个字符展开成字典大小的向量
 ，就形成了第一个时间步所表示的矩阵
 - 第二个（batch_size,num_vocab）的矩阵：取出一个批量样本中每个序列的第二个字符，并将每个字符展开成词典大小的向量，就形成了第二个时间步所表示的矩阵
- 最后就形成了时间步个(批量大小，词典大小)的矩阵，这也就是每个batch最后的形式

 
 
 - 梯度裁剪：应对梯度爆照，设置阈值 $\theta$,裁剪后的梯度的L2范式不超过$\theta$
 - 困惑度：困惑度是对交叉熵损失函数做指数运算后得到的值,任何一个有效模型的困惑度必须小于类别个数
 
 2020.2.15
 
 #### 7.[过拟合、欠拟合及解决方案](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/7.%E8%BF%87%E6%8B%9F%E5%90%88%E6%AC%A0%E6%8B%9F%E5%90%88%E5%8F%8A%E5%85%B6%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88.ipynb)
 
 - 权重衰减对过拟合现象有效：添加L2正则化项
 - Dropout
 
 2020.02.16
 
 #### 8.[梯度消失、梯度爆炸](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/8.%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E3%80%81%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8.ipynb)
 
 - 随机初始化参数模型
 
 - [ ] Xavier随机初始化
 它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

- 斜变量偏移、标签偏移、概念偏移


模型训练步骤：
1、获取数据集
2、数据预处理(标准化)
3、模型设计
4、模型验证和模型调整（调参）
5、模型预测及提交

- [ ] kaggle房价预测

 #### 9.[RNN进阶](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/9.ModernRNN.ipynb) 

- GRU   
重置门有助于捕捉时间序列里短期的依赖关系;  
更新门有助于捕捉时间序列里长期的依赖关系。

- LSTM  
遗忘门:控制上一时间步的记忆细胞 
输入门:控制当前时间步的输入  
输出门:控制从记忆细胞到隐藏状态  
记忆细胞：⼀种特殊的隐藏状态的信息的流动  

 #### 10.[CNN基础](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/10.%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80.ipynb)

- 卷积：1x1卷积可以在不改变宽度高度的情况下改变通道数。
- 池化：池化成主要用于缓解卷积层对位置的过度敏感性

#### 11.[Lenet](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/11.LeNet.ipynb)

net.train():启用BatchNormal和Dropout  
net.eval():不启用BatchNormal和Dropout

#### 12.[CNN进阶](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/12.ModernCNN.ipynb)
- AlexNet
- VGG：多个VGG block组成
- NiN：其中的1*1的卷积层相当于全连接层的作用 
- GooleNet ： Inception 块 中的1*1的卷积层起到减少通道的作用

#### 13.[机器翻译](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/13.%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91.ipynb)
输出序列长度可能和源序列长度不同。

数据预处理、分词、建立词典

数据预处理中分词（Tokenization）的作用是把字符形式的句子转换为单词组成的列表。  
Encoder：输入端到隐藏状态
Decoder：隐藏状态到输出端

Beam Search

Sequence to Squence 模型在训练时，decoder每个单元输出的单词不作为下个单元的输入次，只有在预测时decoder才把每个单元输出的单词作为下个单元的输入单词。  

#### 14.[注意力机制与Seq2seq模型](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/14.%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E5%92%8CSeq2seq%E6%A8%A1%E5%9E%8B.ipynb)

在seq2seq模型中，解码器只能隐式地从编码器的最终状态中选择相应的信息。然而，attention机制可以从这种选择过程中显式地建模。  
Attention 是一种通用的带权池化方法，输入由两部分构成：询问（query）和键值对（key-value pairs）。 attention layer 得到的输出和value的维度一致。

- [ ] Dot-product Attention

- [ ] Multilayer Perceptron Attention

#### 15.[Transformer](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/15.Transformer.ipynb)
可以并行计算
- [ ] << Attention is all you need >> papers  
- Multi-head Attention Layer
- FFN
与多头注意力层相似，FFN层同样只会对最后一维的大小进行改变；除此之外，对于两个完全相同的输入，FFN层的输出也将相等。 

#### 16.[批量归一化和残差网络](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/16.%E6%89%B9%E9%87%8F%E5%BD%92%E4%B8%80%E5%8C%96%E5%92%8C%E6%AE%8B%E5%B7%AE%E7%BD%91%E7%BB%9C.ipynb)

- BatchNorm  
训练：以batch为单位，对每个batch计算均值和方差。  
预测：用移动平均估算整个训练数据集的样本均值和方差。
- ResNet：使用+连接

- DensNet：使用concat连接，通道数相加
稠密块(dense block):定义了输入和输出是如何连接的
过渡层(transition layer):控制通道数，使之不过大

#### 17.[凸优化](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/17.%E5%87%B8%E4%BC%98%E5%8C%96.ipynb)
- 优化方法目标：训练集损失函数值
- 深度学习目标：测试集损失函数值（泛化性）

凸集合

凸函数的性质：1、无局部极小值 2、

#### 18.[梯度下降](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/18.%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.ipynb)
沿梯度的反方向移动自变量可以减小函数值。

梯度下降 时间复杂度O(n)
多维梯度下降  
- 牛顿法
- 随机梯度下降 时间复杂度：O(1)
- 动态学习率
- batch
- mini-batch

# Task07

Task07 和Task08 为NLP方向
#### 19.[优化算法进阶]()

- Momentum


#### 20.[word2vec]()

#### 21.[词嵌入进阶]()

# Task08

#### 22.[文本分类]()
#### 23.[数据增强](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/23.%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA.ipynb)
反转和剪裁  
变化颜色、色调  
叠加多个图像增广的方法  
#### 24.[模型微调](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/24.%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83.ipynb)
torchvision的[`models`](https://pytorch.org/docs/stable/torchvision/models.html)包提供了常用的预训练模型。如果希望获取更多的预训练模型，可以使用使用[`pretrained-models.pytorch`](https://github.com/Cadene/pretrained-models.pytorch)仓库。
在使用模型的时候，指定pretrained=True自动下载并加载预训练的模型参数。
一般情况下，将pretrained_net的fc层随机初始化，使用较大的学习率来调整fc层的参数，使用较小的学习率调整其他层的参数。
# Task09
Task09 和Task10为CV方向

#### 25.[目标检测基础](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/25.%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%9F%BA%E7%A1%80.ipynb)
交并比计算

输出预测边界框：在模型预测阶段，我们先为图像生成多个锚框，并为这些锚框一一预测类别和偏移量。随后，我们根据锚框及其预测偏移量得到预测边界框。当锚框数量较多时，同一个目标上可能会输出较多相似的预测边界框。为了使结果更加简洁，我们可以移除相似的预测边界框。常用的方法叫作非极大值抑制（non-maximum suppression，NMS）。

#### 小结
* 以每个像素为中心，生成多个大小和宽高比不同的锚框。
* 交并比是两个边界框相交面积与相并面积之比。
* 在训练集中，为每个锚框标注两类标签：一是锚框所含目标的类别；二是真实边界框相对锚框的偏移量。
* 预测时，可以使用非极大值抑制来移除相似的预测边界框，从而令结果简洁。

多尺度目标检测

#### 26.[图像风格迁移](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/26.%E5%9B%BE%E5%83%8F%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB.ipynb)
预处理和后处理图像  
使用VGG-19抽取图像特征  
内容损失：MSE  
样式损失：MSE  
总损失函数：L1 loss

实验中，我们选择第四卷积块的最后一个卷积层作为内容层，以及每个卷积块的第一个卷积层作为样式层。  



##### 小结

* 样式迁移常用的损失函数由3部分组成：内容损失使合成图像与内容图像在内容特征上接近，样式损失令合成图像与样式图像在样式特征上接近，而总变差损失则有助于减少合成图像中的噪点。
* 可以通过预训练的卷积神经网络来抽取图像的特征，并通过最小化损失函数来不断更新合成图像。
* 用格拉姆矩阵表达样式层输出的样式。

这里不改变网络模型参数，只对合成图像的内容进行训练更新。  
训练完成后，合成图像就是最终的结果，不需要再经过网络卷积块的计算

#### 27.[图像分类案例1](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/27.%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E6%A1%88%E4%BE%8B1.ipynb)
创建Image
# Task10

#### 28.[图像分类案例2](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/28.%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E6%A1%88%E4%BE%8B2.ipynb)
调参过程：1训练集测试集带入到训练函数。
2.将完整的训练集带入训练
3。对测试集进行分类
#### 29.[GAN](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/29.%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C.ipynb)
判别学习：P(y|x)
生成学习：P(x)

生成器Generator  
判别器Discriminator：binary classifier  
为了避免梯度消失问题，我们对生成器的损失函数进行了一些更改

#### 30.[DCGAN](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/30.DCGAN.ipynb)
nn.ConvTranspose2d
转置卷积层可以将图像变大

- [ ] 训练的网络有问题

## Reference
[学员手册](https://shimo.im/docs/GtHjWK93yyT6KVR8/read)