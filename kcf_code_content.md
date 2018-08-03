程序主入口在`runtracker.cpp`

# KCFTracker类

## 构造函数

根据一系列输入的选项（hog； fixed_window； multiscale lab）对类的一些参数进行初始化
- `lambda=0.0001` 、 `padding=2.5` 、`output_sigma_factor=0.125`
- 特征提取相关参数： `interp_factor` ； `sigma`；`cell_size`
- 多尺度参数：`template_size=96` ；`scale_step=1.05` ；`scale_weight=0.95`

## 初始化函数KCFTracker::init(const cv::Rect &roi, cv::Mat image)

- 初始化 `_roi` 变量；
- 调用 `getFeatures()` 函数对模板块进行特征提取，特征存入_tmpl变量中。具体操作：首先对原始roi区域按 `padding` 倍进行box扩展，之后padding_box在保证长宽比不变的前提下将其缩放到 `template_size` ，再对长宽进行微调（使其为偶数或是16倍数）。缩放后的padding_box图片记作 `z` ，提取z的hog特征（灰度、lab特征）；
- createGaussianPeak()：求取一个高斯系数矩阵，尺寸与所提特征的维度一致，返回值为高斯系数求取`fft`后的结果，返回值保存在 `_prob` 中；
- 计算alphaf矩阵（size_patch[0]xsize_patch[1]）;
- train()：训练模型，为下一检测做准备
   - 求取K，见论文中式（31）;
   - 计算`_alphaf`;
   - 计算 `_tmple`

## 更新函数update()

根据每个新输入的视频帧对模型进行更迭
- 执行detect()：求取K。计算响应，根据式(22)，响应最大值所对应的位置为本帧的检测结果；
- 进行多尺度检测；
- 在本帧检测结果的基础之上，训练计算新的模型参数：`_alhpaf` 和 `_tmple`；
