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

- 卷积、池化
