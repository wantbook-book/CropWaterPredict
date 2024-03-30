## torch

1. unsqueeze

   Returns a new tensor with a dimension of size one inserted at the specified position.

   ```python
   >>> x = torch.tensor([1, 2, 3, 4])
   >>> torch.unsqueeze(x, 0)
   tensor([[ 1,  2,  3,  4]])
   >>> torch.unsqueeze(x, 1)
   tensor([[ 1],
           [ 2],
           [ 3],
           [ 4]])
   ```

   

2. 将数据集分为验证集和训练集

   ```python
   from torch.utils.data.dataset import random_split
   
   # 假设 dataset 已经被定义
   dataset_size = len(dataset)  # 获取数据集的总大小
   train_size = int(dataset_size * 0.8)  # 定义训练集大小，比如说80%
   val_size = dataset_size - train_size  # 验证集大小
   
   # 随机分割数据集
   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   
   # 然后，你可以为训练集和验证集创建 DataLoader
   train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
   val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)
   
   ```

3. 





## tim

1. ### `timm.data.resolve_model_data_config(model)`

   - **功能**：这个函数的目的是从给定的模型中解析出数据配置。这包括模型期望的输入图像的大小、是否需要归一化、归一化使用的均值和标准差等。这些配置对于正确处理输入图像至关重要，确保模型能够以其训练时相同的方式处理输入数据。
   - **参数**：
     - `model`：传入的预训练模型或自定义模型对象。
   - **返回值**：返回一个包含数据配置的字典，例如输入图像的大小、归一化参数等。

2. ### `timm.data.create_transform(**data_config, is_training=False)`

   - **功能**：根据提供的数据配置（例如，解析得到的模型数据配置）和训练状态（训练或评估），创建一个PIL图像到张量的变换序列。这个序列可以包括调整大小、裁剪、归一化等步骤。这些步骤是预处理图像以供模型使用的关键部分。
   - **参数**：
     - `**data_config`：一个包含数据处理所需配置的字典，这通常由`resolve_model_data_config`函数提供。
     - `is_training`：一个布尔值，指示创建的变换是否用于训练。这会影响所选择的图像变换类型，例如，在训练时可能包含数据增强步骤，在评估或推理时则不包含。
   - **返回值**：返回一个预处理步骤的组合，这可以直接应用于图像数据，准备它们以供模型进行预测或训练。
   
3. 

## datetime

1. 时间字符串转化为时间结构体

   ```python
   from datetime import datetime
   
   time_str = "2023-11-08 10:07:33"
   time_format = "%Y-%m-%d %H:%M:%S"
   
   dt_obj = datetime.strptime(time_str, time_format)
   
   ```

   

2. 获取年月日时分秒信息

   ```python
   year = dt_obj.year
   month = dt_obj.month
   day = dt_obj.day
   hour = dt_obj.hour
   minute = dt_obj.minute
   second = dt_obj.second
   
   print(year, month, day, hour, minute, second)
   
   ```

3. 转化为时间戳

   ```python
   from datetime import datetime
   
   timestamp = dt_obj.timestamp()
   print(timestamp)
   # 假设有一个Unix时间戳
   # 以秒为单位
   timestamp = 1609459200  # 2021-01-01 00:00:00 UTC
   
   # 使用fromtimestamp()将时间戳转换为datetime对象
   dt_obj = datetime.fromtimestamp(timestamp)
   
   print(dt_obj)
   
   ```

4. 时间作差

   ```python
   # 将时间字符串转换为datetime对象
   dt_obj1 = datetime.strptime(time_str1, time_format)
   dt_obj2 = datetime.strptime(time_str2, time_format)
   
   # 计算两个时间相差的天数
   delta = dt_obj2 - dt_obj1
   days_diff = delta.days
   ```

## pandas

1. 读取csv文件

   ```python
   filename = 'E:\onedrive\桌面\毕业设计\算法\data\series_data\温湿度时间序列数据.csv'
   df = pd.read_csv(filename, encoding='GBK')  # 返回一个DataFrame对象
   ```

   可以设置文件的字符编码

2. 读取列

   ```python
   df['time']
   ```

3. 读取行

   - ```python
     df.loc[0]
     ```

   - 读取行中的指定列信息

     ```python
     df.loc[0]['time']
     ```

     

4. 遍历行

   ```python
   for index, row in df.iterrows():
       print(index)
       print(row)
       break
   ```

5. 查看列名

   ```python
   df.columns
   ```

   

6. 空值可能为NaN也可能为NaT

   在Pandas中，判断一个值是否为`NaT`（Not a Time），可以使用`pandas.isna()`函数或者`pandas.isnull()`函数。这两个函数对于`NaT`和`NaN`都会返回`True`，因此它们可以用来检测缺失的日期时间数据以及数值数据。

7. 指定类型

   - 转化为datetime日期类型

     ```python
     self.labels_csv['date'] = pd.to_datetime(self.labels_csv['date'], format='%Y/%m/%d')
     ```

     

   - 直接指定

     ```python
     self.labels_csv = pd.read_csv(labels_file_path, dtype={'day':int})
     ```

     

   - 后期转化

     ```python
     for pot in pots_column_names:
                 self.labels_csv[pot] = self.labels_csv[pot].astype(float)
     ```

     

   - 

8. 



## json

1. 写json文件

   ```python
   with open('data.json', 'w', encoding='utf-8') as f:
       json.dump(data, f, ensure_ascii=False, indent=4)
   
   ```

   - `'w'`表示写入模式。
   - `encoding='utf-8'`确保正确处理Unicode字符。
   - `ensure_ascii=False`允许非ASCII字符直接作为Unicode字符写入文件，而不是被转义。
   - `indent=4`提供了一个漂亮的格式化输出，使得JSON文件更易于阅读。

2. 读json文件

   ```python
   with open('data.json', 'r', encoding='utf-8') as f:
       data = json.load(f)
   data['key']
   ```

   



## pathlib

1. 获取绝对路径

   ```python
   from pathlib import Path
   
   # 创建Path对象
   path = Path('relative/path/to/file')
   
   # 获取绝对路径
   absolute_path = path.absolute()
   print(absolute_path)
   
   ```

   ```python
   from pathlib import Path
   
   # 创建Path对象
   path = Path('relative/path/to/file')
   
   # 获取绝对路径，并解析符号链接
   resolved_path = path.resolve()
   print(resolved_path)
   
   ```

   - resolve可以解析符号链接，并检查路径是否存在，如果不存在会抛出异常

2. 遍历目录，判断路径是文件还是目录

   ```python
   from pathlib import Path
   
   # 创建Path对象指向想要遍历的目录
   directory_path = Path('/your/directory/path')
   
   # 遍历目录
   for path in directory_path.iterdir():
       # 判断是文件还是目录
       if path.is_file():
           print(f"文件: {path}")
       elif path.is_dir():
           print(f"目录: {path}")
   ```

   

3. 获取文件扩展名

   ```python
   if file_path.suffix == '.txt':
       print("这是一个文本文件。")
   else:
       print("这不是一个文本文件。")
   ```

4. 创建目录

   ```python
   from pathlib import Path
   
   # 创建单个目录
   Path("my_directory").mkdir(exist_ok=True)
   
   # 创建多级目录
   Path("my_directory/nested_directory").mkdir(parents=True, exist_ok=True)
   ```

   

5. 

   

   

## python 函数

1. 字符串中字母大小写转换

   ```python
   original_str.lower()
   ```

   

2. 

   

## torch

1. 损失函数nn.CrossEntropyLoss

   用于分类任务，要求label是long

2. 回归任务用

   - **均方误差损失（Mean Squared Error Loss, MSE）**：计算预测值和真实值之间差的平方的平均值。在PyTorch中，这可以通过`nn.MSELoss()`实现。
   - **平均绝对误差损失（Mean Absolute Error Loss, MAE）**：计算预测值和真实值之间绝对差的平均值。在PyTorch中，这可以通过`nn.L1Loss()`实现。



## matplotlib

1. 散点

   ```python
   ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
   ```

   

2. 添加形状

   add_patch可以向画布中添加不同的形状

   ```python
   ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
   ```

3. 显示图片

   ```python
   plt.figure(figsize=(10,10))
   plt.imshow(image)
   plt.axis('on')
   plt.show()
   ```

   在这个例子中，`figsize=(10,10)` 表示创建一个宽度和高度都为10英寸的正方形图形。这个尺寸不包括任何标签或标题的空间——它仅仅指的是绘图区域的大小。

   `matplotlib` 默认的DPI（dots per inch，每英寸点数）值是100，所以如果不手动设置DPI，上面的图形在像素尺寸上将会是1000x1000像素。如果你想要更高或更低的图像分辨率，可以通过`dpi`参数来调整，例如：

   ```python
   plt.figure(figsize=(10,10), dpi=200)
   ```

4. 获取当前绘图区域

   gca, get current axes

   ```python
   plt.plot([1, 2, 3], [4, 5, 6])
   ax = plt.gca()  # 获取当前的Axes对象
   ax.set_title("Example Plot")  # 设置图表标题
   plt.show()
   ```

5. 子图

   ```python
   import matplotlib.pyplot as plt
   
   # 创建一个2x2的子图布局，fig是整个图形，axs是子图轴的数组
   fig, axs = plt.subplots(2, 2)
   
   # 通过索引访问特定的子图来绘制内容
   axs[0, 0].plot([0, 1], [0, 1]) # 在第一个子图上绘制
   axs[0, 0].set_title('First Plot') # 设置第一个子图的标题
   
   axs[0, 1].plot([0, 1], [1, 0]) # 在第二个子图上绘制
   axs[0, 1].set_title('Second Plot') # 设置第二个子图的标题
   
   axs[1, 0].plot([1, 0], [0, 1]) # 在第三个子图上绘制
   axs[1, 0].set_title('Third Plot') # 设置第三个子图的标题
   axs[1, 1].plot([1, 0], [1, 0]) # 在第四个子图上绘制
   axs[1, 1].set_title('Fourth Plot') # 设置第四个子图的标题
   
   # 自动调整子图参数，以给定的填充区域
   plt.tight_layout()
   
   plt.show()
   ```

   

6. 绘制torch.tensor类型的图片

   ```python
   import matplotlib.pyplot as plt
   import torch
   
   # 假设img_tensor是你的图片tensor，形状为(C, H, W)
   # 为了示例，这里创建一个随机的图片tensor
   img_tensor = torch.rand(3, 128, 128)  # 创建一个随机的RGB图片
   
   # 将tensor转换为numpy数组，并调整通道的顺序
   img_numpy = img_tensor.numpy().transpose(1, 2, 0)
   
   # 显示图片
   plt.imshow(img_numpy)
   plt.axis('off')  # 不显示坐标轴
   plt.show()
   ```

7. 



## cv

1. 读取图像

   ```python
   image = cv2.imread('../images/Canon/01 - Nov - 23/JPG/DSC_0538 (Pot 3).JPG')
   ```

   image类型是numpy，但是bgr，plt.show需要反转

2. 转换图像

   ```python
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   ```

   这也可以转灰度

3. 融合图片

   ```python
   image_processed = cv2.addWeighted(image, 0.4, mask_image, 0.6, 0)
   ```

   - `src1`: 第一个输入图像。
   - `alpha`: `src1`图像的权重。这个权重值与图像像素值相乘。
   - `src2`: 第二个输入图像。`src1`和`src2`必须有相同的尺寸和类型。
   - `beta`: `src2`图像的权重。同样，这个权重值与`src2`图像像素值相乘。
   - `gamma`: 标量值，加到最终的加权和上。
   - `dst`: 输出图像，它具有与输入图像相同的尺寸和类型。
   - `dtype`: 输出数组的可选深度，有默认值-1。当两个输入数组具有相同的深度时，这个参数被设置为-1，意味着`dst`将和输入数组有相同的深度。

4. 



## numpy

1. 复制维度

   ```python
   mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
   ```

2. 将(3, 200,200)的mask取或变成(200,200)

   ```python
   mask = np.any(masks, axis=0)
   ```

3. (200,200,3)指定二维区域设为(0,0,0)

   ```python
   arr[mask] = [0,0,0] 
   ```

   

4. 


## dataset

1. 要实现的函数

   - `__len__(self)`
   - `__getitem__(self, idx)`

2. dataloader

   ```python
   # 实例化Dataset
   dataset = CropInfoDataset(data, labels)
   
   # 实例化DataLoader
   dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
   
   # 使用DataLoader
   for i, (inputs, labels) in enumerate(dataloader):
       # 在这里处理你的数据
       pass
   ```

   

3. 





## 内置函数

1. 判断bool列表全为true

   ```python
   all(bool_list)
   ```

   

2. 判断bool列表全为false

   ```python
   not any(bool_list)
   ```

   

3. 
