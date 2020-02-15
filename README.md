## Dive into Deep Learning 

本项目为2020年2月份动手学深度学习活动，感谢 **Datawhale & 伯禹教育 & 和鲸科技**

2020.2.11

 1、[linear regression](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/linear_regression.ipynb)
 
 2020.2.12
 
 2、[softmax and classification model](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/2.softmax_and_classification_model.ipynb)
 
 3、[multi layer perceptron](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/3.%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA.ipynb)
 
 2020.2.13
 
 4.[文本预处理](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/4.%E6%96%87%E6%9C%AC%E9%A2%84%E5%A4%84%E7%90%86.ipynb)
  - 对于字符串的处理（re正则表达式） 
  - [ ] 复习python正则表达式
  - 建立词典，统计词频
  - 常用的分词工具（Spacy，NLTK）
  - [ ] 看Spacy,NLTK 文档
  
  2020.2.15
  
 5.[语言模型（基于统计的语言模型）](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/5.%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%95%B0%E6%8D%AE%E9%9B%86.ipynb)
 
 - 马尔科夫假设（一个词的出现，只与前面n个词有关）
 - n元语法（n-grams）：基于n-1阶马尔可夫链的语言模型 
 缺陷：1.参数空间过大 2.数据稀疏问题
 - 时序数据的采样：随机采样（在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。）、相邻采样（在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。）
 - [ ] 对于采样再看一下
 
 6.[RNN 基础](https://github.com/CRuJia/Dive_into_Deep_Learning/blob/master/6.%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.ipynb)
 
 - 每个字符都是字典大小的向量，每个样本都是时间步数个序列，每个batch都是batch_size个样本
 - 第一个(batch_size,num_vocab)的矩阵：取出了第一批样本中的每个序列的第一个字符，并将每个字符展开成字典大小的向量
 ，就形成了第一个时间步所表示的矩阵
 - 第二个（batch_size,num_vocab）的矩阵：取出一个批量样本中每个序列的第二个字符，并将每个字符展开成词典大小的向量，就形成了第二个时间步所表示的矩阵
- 最后就形成了时间步个(批量大小，词典大小)的矩阵，这也就是每个batch最后的形式

 
 
 - 梯度裁剪：应对梯度爆照，设置阈值 $\theta$,裁剪后的梯度的L2范式不超过$\theta$
 - 困惑度：困惑度是对交叉熵损失函数做指数运算后得到的值,任何一个有效模型的困惑度必须小于类别个数
 
 
 
 