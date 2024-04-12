## 单层 transformer 基本结构



https://zhuanlan.zhihu.com/p/338817680

![img](transformer/v2-8b442ffd03ea0f103e9acc37a1db910a_720w.png)

![img](transformer/v2-f6380627207ff4d1e72addfafeaff0bb_720w.png)

只使用左侧的encoder

![img](transformer/v2-6444601b4c41d99e70569b0ea388c3bd_720w.png)

![img](transformer/v2-9699a37b96c2b62d22b312b5e1863acd_720w.png)

多头就是使用多组权重矩阵，做上面的操作

![img](transformer/v2-35d78d9aa9150ae4babd0ea6aa68d113_720w.png)

## 我的任务

将时间-温度序列数据转化为一维特征向量

- 输入(seq_len,batch_size,dim)，输出(1,dim)

  ```
  [
  	[
  		[12,30]
  	],
  	[
  		[13,31]
  	],
  	[
  		[14,20]
  	]
  ]
  ```

  

## 预训练模型

现有的都是nlp的词嵌入模型，不是时间序列模型，但也可以使用词嵌入的预训练模型来做时间序列的编码。
