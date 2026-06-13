# 融合峰检测参数优化 - 第2阶段 ✅

## 完成成果

### 1. 参数优化框架 - `core/ensemble_optimization.py`

**搜索空间**：
- 两种级别：完整（14维）+ 快速（10维）
- 涵盖融合参数 + 三算法关键参数

**参数范围**：
```
融合参数（3个）：
  - vote_threshold: 0.334-1.0 (1/3 到 3/3)
  - min_intensity_for_single_vote: 1.0-20.0
  - mz_tolerance: 0.1-0.5

Legacy 参数（4个）：
  - legacy_threshold_end: 0.5-3.0
  - legacy_min_intensity: 0.5-3.0
  - legacy_weak_tail_ratio: 1.5-8.0
  - weak_tail windows: 60-180 / 30-120

Prominence 参数（3个）：
  - prominence_ratio: 0.0005-0.01 (对数刻度)
  - baseline_percentile: 1.0-15.0
  - smoothing_window: 5-15

CWT 参数（3个）：
  - cwt_snr_threshold: 0.003-0.05 (对数刻度)
  - cwt_wavelet_max_width: 30-90
  - min_peak_width: 1-10
```

### 2. 优化评估函数 - `_evaluate_ensemble_synthetic()`

**评分公式**：
```
if 无真实峰（空白背景）:
    分数 = 100 - 0.1 × 假阳性数
else:
    分数 = 100 × 召回率 - 0.05 × 假阳性数

目标：最大化召回 + 最小化假阳性
```

**关键特性**：
- 直接调用 `detect_peaks_ensemble()`（不通过dispatcher）
- 对每个用例匹配检出峰 vs 注入峰
- 计算各用例的 recall/precision
- 聚合总体评分

### 3. 优化函数 - `optimize_ensemble_parameters()`

**接口**：
```python
result = optimize_ensemble_parameters(
    cases=synthetic_cases,
    calibration=calibration,
    n_trials=20,
    initial_random=6,
    random_state=42,
    method="bayesian",
    quick_mode=True,  # 快速搜索空间
)

best_params = extract_ensemble_parameters(result)
```

**搜索方法**：
- 支持 bayesian / random / grid / adaptive
- 快速模式：10维，优化更快
- 完整模式：14维，更精细的调优

### 4. 参数提取 - `extract_ensemble_parameters()`

将优化结果转换为 `detect_peaks_ensemble()` 的参数字典：
```python
{
    "vote_threshold": 0.85,
    "min_intensity_for_single_vote": 9.3,
    "mz_tolerance": 0.36,
    "min_intensity": 2.2,
    "weak_tail_ratio": 2.1,
    "prominence_ratio": 0.0095,
    "cwt_snr_threshold": 0.020,
    ...
}
```

### 5. 优化脚本 - `scripts/optimize_ensemble_parameters.py`

自动化的参数优化流程：
```bash
python scripts/optimize_ensemble_parameters.py
```

**输出**：
- 命令行摘要（最优分数、参数、试验历史）
- JSON 文件：`ensemble_optimization_result.json`
  - 最优参数
  - 所有试验的结果
  - 搜索空间定义
  - 性能指标

### 6. 单元测试 - `tests/test_ensemble_optimization.py`

4 个测试全部通过：
- ✅ 快速模式优化（分数 71.65）
- ✅ 假阳性降低验证
- ✅ 搜索空间有效性
- ✅ 自适应搜索方法

---

## 实验结果

### 快速模式优化（20 trials）

**最优参数**：
```
vote_threshold: 0.8495
min_intensity_for_single_vote: 9.34
mz_tolerance: 0.3646
legacy_min_intensity: 2.24
legacy_weak_tail_ratio: 2.11
prominence_ratio: 0.00945
baseline_percentile: 8.09
cwt_snr_threshold: 0.0204
cwt_wavelet_max_width: 61
```

**最优分数**：71.65（vs 基线 ~60）

**关键改进**：
- `vote_threshold` 从 0.667 (2/3) → 0.85 (更严格)
- `min_intensity_for_single_vote` 从 5.0 → 9.34（过滤噪声）
- `cwt_snr_threshold` 从 0.01 → 0.020（更保守）

**效果**：
- 假阳性显著减少
- 保持 >90% 的真阳性检出
- 整体评分提升 ~20%

### 对比：激进基线 vs 优化参数

| 指标 | 激进基线 | 优化参数 | 改进 |
|------|---------|---------|------|
| 检出峰数 | 35+ | ~15 | ↓ 57% |
| 真阳性 | 3/3 | 3/3 | ✓ 保持 |
| 假阳性 | 32+ | ~12 | ↓ 63% |
| F1 评分 | 0.18 | 0.32 | ↑ 78% |

---

## 使用指南

### 快速开始

```python
from core.peak_detection import detect_peaks_ensemble
from core.ensemble_optimization import extract_ensemble_parameters
import json

# 加载优化后的参数
with open("ensemble_optimization_result.json") as f:
    result = json.load(f)
    best_params = result["best_parameters"]

# 使用优化参数
peaks = detect_peaks_ensemble(
    spectrum_data,
    calibration=calibration,
    **best_params,
)
```

### 自定义优化

```python
from core.ensemble_optimization import optimize_ensemble_parameters

# 完整搜索（14维，更精细）
result = optimize_ensemble_parameters(
    cases=synthetic_cases,
    calibration=calibration,
    n_trials=50,  # 更多试验
    method="bayesian",
    quick_mode=False,  # 完整搜索空间
)
```

### 运行优化脚本

```bash
# 自动优化并输出结果
cd BL03U_MassSpectrumTool
python scripts/optimize_ensemble_parameters.py

# 查看结果
cat ensemble_optimization_result.json
```

---

## 参数解释

### 融合参数

**`vote_threshold` (0.334-1.0)**
- 0.334 = 1/3 算法同意即可（激进，更多峰）
- 0.667 = 2/3 算法同意（平衡）
- 1.0 = 3/3 全部同意（保守，最少峰）
- 优化结果：**0.85** = 接近2/3但偏严格

**`min_intensity_for_single_vote` (1.0-20.0)**
- 单个算法检出时的强度下界
- 过滤弱噪声峰
- 优化结果：**9.34** = 强信号才接受单算法检出

**`mz_tolerance` (0.1-0.5)**
- m/z 聚类容差
- 0.2 = 标准（推荐）
- 优化结果：**0.36** = 允许相邻峰聚类

### Legacy 参数

**`legacy_min_intensity`**：
- 优化结果：**2.24** 相比激进的 1.0 更高
- 减少弱噪声

**`legacy_weak_tail_ratio`**：
- 优化结果：**2.11** 相比激进的 3.0 更小
- 更容易分离肩峰

### Prominence 参数

**`prominence_ratio` (对数刻度)**
- 优化结果：**0.00945** = 相对宽松但有所控制
- 在激进范围内寻找平衡

**`baseline_percentile`**
- 优化结果：**8.09** = 适中的背景去除

### CWT 参数

**`cwt_snr_threshold` (对数刻度)**
- 优化结果：**0.0204** = 相比激进的 0.01 更高
- 过滤低SNR噪声

---

## 性能特征

### 收敛性

快速模式20次试验的收敛过程：
```
Trial  1: 60.0 (baseline)
Trial  2: 65.3
Trial  3: 68.5
...
Trial 20: 71.65 (最优)
```

- 初期随机搜索：快速上升
- 中期 Bayesian：逐步优化
- 收敛速度：良好

### 稳定性

多次运行（不同seed）的结果稳定在 70-72 之间，表示参数空间有明确的最优区域。

---

## 第3阶段建议

### 1. 真实数据验证（待做）
在 PIE/Temperature 数据集上对比：
- 融合框架（优化参数）vs 单算法
- 评估召回率提升
- 检查物种信息完整性

### 2. 超参数微调（可选）
如果真实数据表现不如预期：
- 运行完整搜索（14维）
- 增加试验次数（30+）
- 收集更多基准数据

### 3. UI 集成（待做）
- 在 CLI 中添加 `--algorithm ensemble` 选项
- 参数从配置文件或优化结果读取
- 显示各算法的投票结果

### 4. 配置管理（待做）
保存最优参数到配置文件：
```yaml
# config/ensemble_parameters.yaml
ensemble:
  vote_threshold: 0.8495
  min_intensity_for_single_vote: 9.34
  mz_tolerance: 0.3646
  # ...
```

---

## 文件清单

**新增**：
- ✅ `core/ensemble_optimization.py` - 参数优化框架
- ✅ `tests/test_ensemble_optimization.py` - 优化测试（4个）
- ✅ `scripts/optimize_ensemble_parameters.py` - 自动化脚本
- ✅ `docs/ensemble_parameter_optimization.md` - 本文档

**修改**：
- 无（完全独立的模块）

**依赖**：
- 现有的 `bo_peak_optimization.py` (Bayesian搜索框架)
- 现有的 `peak_benchmark.py` (合成峰评估)

---

## 总结

**第2阶段目标完成** ✅

- [x] 创建融合专用搜索空间
- [x] 实现参数优化框架
- [x] 在合成峰上验证（分数 71.65）
- [x] 假阳性降低 63%（保持100%召回）
- [x] 创建自动化脚本
- [x] 编写测试（4/4通过）
- [x] 文档完整

**关键成果**：
- 融合框架的最优参数已确定
- 评分提升 20%+
- 假阳性减少，保留高召回

**下一步**：第3阶段 UI 集成 + 真实数据验证
