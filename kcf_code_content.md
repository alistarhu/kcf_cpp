程序主入口在`runtracker.cpp` ，`kcftracker.cpp`对KCFTracker类的一些函数进行了定义，是整个程序的核心算法主要集中在此处,`fhog.cpp`定义了一些提取HOG特征所需的一些函数。

下面按照程序执行的流程对相应的函数进行解读：

## KCFTracker类构造函数：

函数原型`KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)`

这里主要根据输入的选项（hog； fixed_window； multiscale；lab）对类的一些参数进行初始化
- `lambda=0.0001` 、 `padding=2.5` 、`output_sigma_factor=0.125`
- 特征提取相关参数： `interp_factor=0.012` 、 `sigma=0.6` 、 `cell_size=4` 、 `_hogfeature=true`
- 多尺度参数：`template_size=96` 、 `scale_step=1.05` 、 `scale_weight=0.95` 、 `fixed_window=true`

## 初始化函数

`void KCFTracker::init(const cv::Rect &roi, cv::Mat image)`

1、根据所给定的模板box信息对 `Rect _roi`进行初始化

2、执行`_tmpl = getFeatures(image, 1)`

`cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)`

- 对原始 `_roi`按照`padding`系数进行区域扩展，扩大之后宽和高记作padding_w和padding_h
- 按照关系式：`padding_w、h/_scale=template_size`，其中`template_size`为我们所设定的模板大小，这样可以计算得到缩放系数`_scale`
- 在`_templ_sz`中存入由`padding_w、h`缩放到模板大小的宽和高（ **`_templ_sz`在初始化之后就不会变，可以理解为保证跟踪区域模板长宽比不变情况下缩放到template_size的长宽大小** ）
- 因为需要在模板大小下进行特征特征提取，为了方便HOG特征提取，需要对输入尺寸`_templ_sz`进行调整，使其满足是`cell_size`的整数倍
- 由于模板的`_templ_sz`做了调整，因此padding_roi的大小也需要按照公式`_templ_sz*_scale`作相应调整，结果存在`extracted_roi`中
- 根据`extracted_roi`在图像中截取其相应区域，并缩放到模板大小`_templ_sz`，缩放后的区域记作z
- 对图片z进行HOG特征提取：
    - 将特征图的尺寸x、y、channel依次存入`size_patch[3]`中
    - 提取出的特征维度：channel*(x*y) （整个特征看成两个维度！！！）
- 调用`createHaningMats()`函数产生汉宁系数，维度与特征维度一致
- 将HOG特征与汉宁系数进行对应元素点乘后输出

3、调用`_prob = createGaussianPeak(size_patch[0], size_patch[1])`

产生一个中心二维高斯系数矩阵，将系数的fft后的结果返回。这个高斯矩阵可以看成是类标签，中心点对应跟踪的中心，响应最大，越靠近外围响应相应的衰弱

4、变量初始化`_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0))`

5、执行`train(_tmpl, 1.0)`

根据当前的输入进行模型学习，得到参数 **`_alphaf`** 和 **模板特征`_tmpl`**，以便下一次的检测

- `cv::Mat k = gaussianCorrelation(x, x)`按照论文中的公式（31）实现对K的求取
- `cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda))`根据论文中公式（17）计算得到alphaf
- 将**`_alphaf`** 和 **模板特征`_tmpl`** 进行更新，按照一定系数与上一次的值进行加权求和

## 更新函数

`cv::Rect KCFTracker::update(cv::Mat image)`

该函数主要完成对新输入帧的跟踪以及相应的模型更新

1、对当前帧进行检测`cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value)`

检测阶段利用上一帧已经学得的`_alphaf` 和模板特征`_tmpl`，对新输入帧（上一帧检测结果的中心点为基准，根据roi区域大小进行特征提取）利用公式（22）进行相应图求解

- `cv::Mat k = gaussianCorrelation(x, z)`论文公式（31）求互相关
- `cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)))`得到检测响应图式（22）
- 将相应图的最大值位置返回

2、多尺度检测

在特征提取中`extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width`，其中：`_tmpl_sz`在第一帧初始化之后就不变了，`_scale`沿用上一帧的比例关系，因此在检测过程中可以将特征提取中参数`scale_adjust=1、1/scale_step、scale_step`就可以得到三种不同尺度的roi区域（roi区域中心点还是上一帧检测结果的中心点）
- 当`scale_adjust=1`时，得到最大响应值记作`peak_value`
- 当`scale_adjust=1/scale_step`时，如果`scale_wight*new_peak_value>peak_value`，则对roi相关参数进行更新：`_scale /= scale_step`、`_roi.width /= scale_step`、`_roi.height /= scale_step`
- - 当`scale_adjust=scale_step`时，如果`scale_wight*new_peak_value>peak_value`，则对roi相关参数进行更新：`_scale *= scale_step`、`_roi.width *= scale_step`、`_roi.height *= scale_step`

3、根据原始roi图与特征相应位置的对应关系，更新本帧的跟踪中心点的新的位置

4、根据本帧的检测结果，执行`train()`，对`_alphaf` 和模板特征`_tmpl`等参数进行更新，方便下一帧图像的检测。
