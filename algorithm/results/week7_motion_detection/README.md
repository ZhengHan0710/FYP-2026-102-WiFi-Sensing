# Week 7: CSI 运动检测初步实验

## 目标

使用小组采集的 ESP32-S3 CSI 数据，先验证简单时间变化特征能否识别“人员运动”。本轮不做无人、人员运动和人员静止三分类，而是先将任务简化为：

- `moving`：人员运动；
- `nonmoving`：空场景或人员静止。

## 数据与处理

- 场景：`empty`、`person_moving`、`person_stationary`；
- 每个场景包含 3 次独立采集；
- 采样率约为 20 帧/秒；
- 每帧包含 52 个复数 CSI 位置，本实验将实部和虚部转换为幅度；
- 每 64 帧构成一个时间窗口，约为 3.2 秒；
- 窗口步长为 16 帧，约为 0.8 秒；
- 序号不连续时切断数据，不让一个时间窗口跨越丢帧位置。

## 特征

1. `temporal_std`：一个窗口内 CSI 幅度随时间的波动程度。
2. `relative_temporal_diff`：相邻帧幅度变化除以该窗口幅度均值，用于减少整体信号强弱的影响。

特征分布表明：人员运动时两项特征整体明显高于空场景和人员静止场景；但空场景与人员静止仍明显重叠。

## 数据划分

- 第 1、2 次采集：用于确定阈值或训练模型；
- 第 3 次采集：仅用于最终测试。

不随机打乱窗口后再切分，避免同一次采集中的高度重叠窗口同时出现在训练集和测试集。

## 初步结果

| 方法 | 测试准确率 | 平衡准确率 | 运动召回率 | 非运动召回率 | 误报 FP | 漏检 FN |
|---|---:|---:|---:|---:|---:|---:|
| 单特征阈值 | 76.65% | 79.14% | 95.77% | 62.50% | 36 | 3 |
| 双特征逻辑回归 | 89.22% | 89.71% | 92.96% | 86.46% | 13 | 5 |

双特征逻辑回归将误报从 36 个降到 13 个，但漏检从 3 个增加到 5 个。这说明同时利用两种时间变化特征，比单一阈值更稳定。

## 局限

- 目前每个场景只有 3 次采集，测试量较小；
- 目前结果是同一实验环境内的初步验证；
- 当前特征适合检测明显运动，尚不能充分区分空场景与人员静止。

## 文件

- `window_features.csv`：每个时间窗口的特征；
- `scenario_summary.json`：三个场景的特征中位数汇总；
- `scenario_feature_comparison.png`：特征分布对比图；
- `motion_threshold_report.json`：单特征阈值实验结果；
- `motion_logistic_report.json`：双特征逻辑回归实验结果。

## 复现命令

```bash
python algorithm/analyze_scenarios.py \
  --data-root data \
  --output /tmp/fyp_scenario_analysis \
  --window 64 \
  --step 16

python algorithm/plot_scenario_features.py \
  /tmp/fyp_scenario_analysis/window_features.csv \
  --output /tmp/fyp_scenario_analysis/scenario_feature_comparison.png

python algorithm/motion_threshold_baseline.py \
  /tmp/fyp_scenario_analysis/window_features.csv \
  --output /tmp/fyp_motion_baseline

python algorithm/motion_logistic_baseline.py \
  /tmp/fyp_scenario_analysis/window_features.csv \
  --output /tmp/fyp_motion_logistic
```
