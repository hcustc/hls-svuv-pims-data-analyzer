# BL03U Mass Spectrum Tool 核心算法备忘

本文档记录软件中适合写入论文方法学或讨论部分的核心算法，并尽量对应到当前代码实现。

## 代码依据

- `src/bl03u_masstool/core/peak_detection.py`: 传统局部极大值寻峰、prominence 寻峰、高斯拟合与峰边界估计。
- `src/bl03u_masstool/core/cwt_peak_detection.py`: CWT 连续小波寻峰候选方案。
- `src/bl03u_masstool/core/integration.py`: 峰面积积分、基线扣除、高斯面积计算。
- `src/bl03u_masstool/core/pie_analysis.py`: PIE 曲线生成、PICS 数据库读取、物种组合最优拟合。
- `src/bl03u_masstool/core/temperature_scan.py`: 温度扫描曲线生成、光强归一化、Kr 膨胀系数校正。
- `src/bl03u_masstool/core/calibration.py`: TOF 到 m/z 的二次定标及定标系数拟合。
- `src/bl03u_masstool/core/isotope.py`: 分子式解析、精确质量/名义质量计算和同位素分布卷积。
- `config/peak_detection.yaml`: 自动寻峰默认参数。

## 1. TOF 到 m/z 的二次定标

当前定标模型为：

```text
m/z = c + b * TOF + a * TOF^2
```

定标系数可由标准点通过最小二乘拟合：

```text
X = [TOF^2, TOF, 1]
beta = argmin ||X beta - m/z||_2
```

反向换算 `m/z -> TOF` 通过二次方程正根完成；当二次项近似为 0 时退化为线性模型。该定标为自动寻峰、卡峰积分、PIE 拟合提供统一物理坐标。

## 2. 自动寻峰算法

当前主工作台自动寻峰支持三种算法，由 `config/peak_detection.yaml` 中的 `algorithm` 选择：

```text
prominence  # 默认推荐
legacy      # 传统局部极大值
cwt         # 连续小波变换
```

### 2.1 Prominence 寻峰

实现位于 `src/bl03u_masstool/core/peak_detection.py::detect_peaks_prominence`。流程：

1. 从 `detection_min_idx` 之后开始分析，避免低 TOF 噪声。
2. 使用 Savitzky-Golay 对强度序列平滑。
3. 使用 rolling percentile 估计局部基线。
4. 对平滑谱图扣除基线并截断为非负信号。
5. 调用 `scipy.signal.find_peaks`，用强度、prominence、峰宽和峰间距筛选候选峰。
6. 使用 `scipy.signal.peak_widths` 估计初始边界。
7. 局部高斯拟合成功时，用高斯中心和 FWHM 细化峰位与边界。

核心筛选参数：

```text
min_intensity
prominence_ratio
duplicate_window
min_peak_width
max_peak_width
baseline_window
baseline_percentile
```

论文讨论点：该方法比单纯局部极大值更适合存在基线漂移、弱峰和噪声尖刺的质谱数据，因为 prominence 描述的是峰相对局部背景的突起程度，而不是绝对强度。

### 2.2 传统局部极大值寻峰

实现位于 `src/bl03u_masstool/core/peak_detection.py::detect_peaks_in_range`。候选峰满足：

```text
y[i] > min_intensity
y[i] >= y[i-1]
y[i] >= y[i+1]
```

候选峰按强度排序后，使用 `nearby_peak_window`、`duplicate_window` 和弱拖尾规则去重：

```text
current_peak * weak_tail_ratio <= previous_peak
```

峰边界从中心向两侧扩展至低于 `threshold_end`，然后尝试高斯拟合修正中心和窗口。

论文讨论点：这是针对 BL03U 强峰和拖尾特征的经验寻峰方案，保留为可复现实验历史结果的 legacy 模式。

### 2.3 CWT 小波寻峰

实现位于 `src/bl03u_masstool/core/cwt_peak_detection.py::detect_peaks_cwt`。流程：

1. Savitzky-Golay 平滑。
2. rolling percentile 基线校正。
3. 使用 PyWavelets 计算连续小波变换，默认 wavelet 为 `mexh`。
4. 对多尺度 CWT 响应取最大值。
5. 使用 `find_peaks` 对 CWT 响应做 prominence、宽度和距离筛选。
6. 使用 `peak_widths` 与高斯拟合估计边界。

关键参数包括：

```text
wavelet_widths           小波尺度范围，决定对不同峰宽的响应
wavelet                  小波基，默认 mexh
prominence_ratio         CWT 响应峰的相对 prominence 阈值
min_peak_distance        候选峰最小间隔
min_peak_width/max_peak_width  CWT 响应峰宽筛选范围
baseline_window_factor   基线估计窗口相对平滑窗口的放大倍数
```

论文讨论点：CWT 可作为多尺度峰检测方法，用于弱峰、峰宽变化明显或肩峰较多的数据。代价是参数更多、计算更慢。

## 3. 峰边界与高斯拟合

自动峰检测得到候选中心后，软件尝试拟合：

```text
G(x) = amplitude * exp(-(x - mean)^2 / (2 * std_dev^2)) + baseline
FWHM = 2 * sqrt(2 * ln(2)) * std_dev
```

拟合成功时，用：

```text
left  = mean - gaussian_boundary_scale * FWHM
right = mean + gaussian_boundary_scale * FWHM
```

再叠加 `boundary_padding` 得到最终卡峰窗口。拟合失败时保留阈值或 `peak_widths` 给出的窗口。

## 4. 峰面积积分

实现位于 `src/bl03u_masstool/core/integration.py`。

基线扣除梯形积分：

```text
area = trapz(y[left:right] - min(y[left:right]))
```

高斯积分：

```text
area = amplitude * FWHM * sqrt(pi / (4 * ln(2)))
```

当 `prefer_gaussian=True` 时，软件会在当前谱图中重新拟合峰，而不是复用参考谱中的高斯参数。拟合失败时回退到梯形积分。

## 5. PIE 物种组合拟合

PIE 分析将实验曲线与本地 PICS 数据库中的候选物种进行组合拟合。对候选物种的 PICS 曲线插值到实验 photon energy 网格后，构建设计矩阵：

```text
y_exp ~= X * coef
```

其中每一列代表一个候选物种的理论或数据库 PICS 曲线。软件支持锁定物种、系数约束和组合评分，用于讨论同一 m/z 下多物种贡献的最优解释。

## 6. 分子式、质量和同位素分布

实现位于 `src/bl03u_masstool/core/isotope.py`。

- 分子式解析：将 `C6F11O2H` 等字符串解析为元素-计数字典。
- 单同位素精确质量：按各元素最丰同位素质量求和。
- 名义质量：按主要同位素质量数求和。
- 同位素分布：对每个元素的同位素分布做多项式卷积，并按丰度阈值截断。
- 分子式候选：在给定元素范围内枚举组合，保留质量误差窗口内的候选式。

论文讨论点：该模块可用于把 m/z 峰与可能分子式、同位素包络和数据库物种信息关联起来。

## 7. 温度曲线自动分组

实现位于 `src/bl03u_masstool/core/temperature_scan.py::classify_temperature_curve`。软件按 m/z 聚合温度扫描积分结果后，基于曲线形状分为：

```text
formation      生成(升高)
consumption    消耗(减少)
intermediate   中间体(先升后降低)
unclassified   暂未区分
```

分类依据为温度升序后的积分强度序列。算法用原始曲线定位全局峰值，用 3 点滚动中位数估计低温端和高温端强度，然后比较低温端、高温端和全局峰值：

- 若高温端相对低温端明显升高，并接近全局峰值，归为生成。
- 若低温端相对高温端明显更高，并接近全局峰值，归为消耗。
- 若全局峰值位于中间温度，并且低温端和高温端都明显低于峰值，归为中间体。
- 若有效温度点不足、信号为零、动态范围过小或趋势不满足上述规则，归为暂未区分。

默认相对变化阈值为 25%，端点接近峰值的判据为 65%。分类结果会写入温度扫描结果表：

```text
curve_class
curve_class_label
curve_class_reason
```

论文讨论点：该分组可用于从温度依赖行为上区分产物型、反应物消耗型和中间体型物种，为后续反应路径讨论提供自动化初筛。
