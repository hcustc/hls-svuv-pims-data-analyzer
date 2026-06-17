# 融合峰检测框架实现完成 ✅

## 第1阶段总结

**目标**：通过三算法融合（Legacy、Prominence、CWT）最大化峰召回率

### 实现成果

#### 1. **融合检测函数** - `detect_peaks_ensemble()`
位置：`src/bl03u_masstool/core/peak_detection.py`

**核心特性**：
- 并行运行三个算法（可选启用）
- 按 m/z ±0.2 聚类峰
- 投票规则（可配置阈值）：
  - 2/3 算法同意 → 高置信峰 ✓
  - 1/3 算法 + 强度>阈值 → 低置信峰 ✓

**关键参数**：
```python
detect_peaks_ensemble(
    y_data,
    calibration=calibration,
    # 融合控制
    use_legacy=True,          # 启用Legacy算法
    use_prominence=True,       # 启用Prominence算法
    use_cwt=True,             # 启用CWT算法
    mz_tolerance=0.2,         # m/z聚类容差
    vote_threshold=0.667,     # 投票阈值（2/3）
    min_intensity_for_single_vote=5.0,  # 单算法检出的强度下界
    # 各算法的recall-focused参数
    threshold_end=1.0,        # 更宽松的边界判断
    min_intensity=1.0,        # 更低的强度阈值
    # ... 其他参数见文档
)
```

#### 2. **修复硬编码参数**
**Issue**：`peak_detection.py:273` 的 `15000` 阈值是硬编码

**解决**：
- 添加参数 `weak_tail_cutoff_idx: int = 15000`
- 使算法适应不同仪器/数据集

#### 3. **辅助函数**
- `_cluster_peaks_by_mz(peaks, tolerance)` - m/z聚类
- `_merge_cluster_peaks(cluster)` - 聚类峰合并（加权均值 + 最强信号）

#### 4. **测试覆盖** ✅
位置：`tests/test_ensemble_peak_detection.py`

6个测试全部通过：
- ✅ 检出合成弱峰
- ✅ 分离重叠峰
- ✅ 3个动态范围峰（100% 召回率）
- ✅ 单算法降级模式
- ✅ 投票阈值灵敏度
- ✅ 投票规则过滤噪声

### 关键发现

**融合的优势**：
```
融合结果（投票2/3）：
  - 多动态范围：100% 召回率（3/3真实峰）
  - 重叠峰：能分离
  - 弱峰：能检出

关键数字：
  - m/z容差±0.2：避免峰重复，同时聚类相邻的峰
  - vote_threshold=0.667：2/3一致性 → 高置信
  - 单算法强度阈值：防止噪声通过投票
```

### 性能基准（合成峰）

| 测试用例 | 注入峰数 | 检出数 | 真阳性 | 假阳性 | 召回率 |
|---------|---------|--------|--------|---------|--------|
| weak_single | 1 | ≥1 | 1 | 0+ | 100% |
| overlap_equal | 2 | ≥2 | 2 | 0+ | 100% |
| multi_dynamic_range | 3 | 35 | 3 | 32 | 100% |

**注**：假阳性可通过提高 `min_intensity_for_single_vote` 进一步减少

---

## 使用示例

### 基本用法
```python
from bl03u_masstool.core.peak_detection import detect_peaks_ensemble
from bl03u_masstool.core.calibration import Calibration

calibration = Calibration()
peaks = detect_peaks_ensemble(
    spectrum_data,
    calibration=calibration,
)
```

### 激进模式（最大召回）
```python
peaks = detect_peaks_ensemble(
    spectrum_data,
    calibration=calibration,
    vote_threshold=0.334,  # 1/3即可 → 更多峰
    min_intensity_for_single_vote=1.0,
    min_intensity=0.5,     # 更低的绝对阈值
)
```

### 保守模式（最少假阳性）
```python
peaks = detect_peaks_ensemble(
    spectrum_data,
    calibration=calibration,
    vote_threshold=1.0,    # 需要3/3同意（仅3算法都检出）
    min_intensity_for_single_vote=20.0,
    min_intensity=5.0,
)
```

### 仅用单个算法
```python
# 等同于 detect_peaks_in_range() 但用融合框架
peaks = detect_peaks_ensemble(
    spectrum_data,
    use_legacy=True,
    use_prominence=False,
    use_cwt=False,
)
```

---

## 集成建议

### 第2阶段：参数优化（后续任务）

现在的基础参数是"激进的"（最大召回）：
```python
# 当前defaults（recall-focused）
min_intensity=1.0,
threshold_end=1.0,
prominence_ratio=0.002,
cwt_snr_threshold=0.01,
```

**建议**：
1. 用 `bo_peak_optimization.py` 的 Bayesian 搜索优化三个算法的参数
2. 针对融合方案创建新的搜索空间（投票参数 + 各算法关键参数）
3. 在真实数据集（PIE/Temperature）上验证

### 第3阶段：UI 集成（未来）

```python
# cli/api 中的使用
result = detect_peaks_ensemble(
    spectrum_data,
    calibration=config.calibration,
    use_legacy=config.use_legacy,
    use_prominence=config.use_prominence,
    use_cwt=config.use_cwt,
    vote_threshold=config.vote_threshold,  # 用户配置
    # ... 其他参数从配置读取
)
```

---

## 技术细节

### 聚类逻辑
```
m/z排序 → 遍历相邻峰
  if peak.mz - cluster[0].mz <= 0.2:
    加入同一聚类
  else:
    开始新聚类
```

### 投票规则
```
enabled_algorithms = 3  # 或 1/2/3

for cluster in clusters:
    vote_count = len(cluster)  # 有多少算法检出
    vote_threshold_count = ceil(enabled_algorithms * 0.667)

    if vote_count >= vote_threshold_count:
        保留 ✓
    elif vote_count == 1 and intensity > 5.0:
        保留 ✓ （单算法但强信号）
    else:
        过滤
```

### 峰合并策略
```
weighted_mz = Σ(peak.mz * peak.intensity) / Σ(peak.intensity)
max_intensity = max(peak.intensity for peak in cluster)
使用最强峰的边界和高斯参数
```

---

## 下一步

### 立即可做 ✅
- [x] 融合框架实现
- [x] 三个算法集成
- [x] 投票规则应用
- [x] 单元测试（6个）

### 建议接下来 ⏳
1. **参数优化**（估计 1-2周）
   - 用 Bayesian 搜索优化投票参数
   - 为三个算法分别调整 recall-focused 基准

2. **真实数据验证**（估计 3-5天）
   - 在 PIE/Temperature 测试集上对比
   - 与现有检测结果对比（召回率提升多少？）

3. **UI 集成**（估计 3-5天）
   - 添加融合方法到 CLI 选项
   - 添加参数配置面板

---

## Q&A

**Q: 为什么 multi_dynamic_range 有 32 个假阳性？**

A: 因为我们用了激进的参数（min_intensity=1.0）来最大化弱峰检出。在第2阶段可通过优化参数减少。

**Q: 能否用不同的 m/z 容差？**

A: 可以！`mz_tolerance` 参数可调：
- 0.1 → 更严格的聚类（可能分割肩峰）
- 0.2 → 平衡（推荐）
- 0.5 → 宽松（可能合并不同峰）

**Q: 是否必须启用全部三个算法？**

A: 不必。可以任意组合（1/2/3个）。投票规则会自动调整。

**Q: 能否扩展到其他算法？**

A: 可以！框架可扩展：
1. 在融合函数中添加新算法分支
2. 更新 `enabled_algorithms` 计数
3. 投票规则自动适配
