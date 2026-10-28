# Algorithm and Data Processing

该目录保存 CSI 数据检查、时间窗口构建、特征提取和初步分类实验代码。

## 目录结构

```text
algorithm/
├── prepare_project_csi.py          # 检查小组 ESP32-S3 CSV 并构造时间窗口
├── analyze_scenarios.py            # 提取三场景的幅度时间特征
├── plot_scenario_features.py       # 绘制三场景特征分布对比图
├── motion_threshold_baseline.py    # 单特征阈值运动检测基线
├── motion_logistic_baseline.py     # 双特征逻辑回归运动检测基线
├── motion_cnn_baseline.py          # 完整时间×CSI幅度图的轻量CNN
├── train_lstm.py                    # 公开 HAR 数据集的 BiLSTM/CNN 对照实验
├── benchmarks/
│   └── run_ut_har_cnn_compatible.py # SenseFi UT-HAR CNN-5兼容运行脚本
├── WEEK5_WEEK6_SUMMARY.md           # 第五、六周公开数据集实验总结
├── WEEK9_SUMMARY.md                 # 第九周公开基准与项目CNN总结
└── results/
    ├── week6_comparison/           # 类别不均衡处理对比
    ├── week7_motion_detection/     # 真实 ESP32-S3 三场景初步实验
    └── week9_cnn_baseline/         # SenseFi及项目CNN结果
```

## Week 7 复现流程

以项目根目录为当前目录：

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

实验设计、指标和结果见
[`results/week7_motion_detection/README.md`](results/week7_motion_detection/README.md)。

## 公开 HAR 数据实验

`train_lstm.py` 使用的公开 HAR 数据未上传到本仓库。运行时需指定本地数据集目录：

```bash
python algorithm/train_lstm.py same_room \
  --data-root /path/to/wifi_csi_har_dataset \
  --output-root /tmp/fyp_har_outputs
```

第五、六周的已完成结果见
[`WEEK5_WEEK6_SUMMARY.md`](WEEK5_WEEK6_SUMMARY.md)。

## Week 9 CNN 实验

项目实采数据的轻量 CNN：

```bash
python algorithm/motion_cnn_baseline.py \
  --data-root data \
  --output /tmp/fyp_motion_cnn \
  --input-mode centered \
  --test-recording 3 \
  --epochs 30 \
  --seed 42
```

SenseFi / UT-HAR 公开基准与项目 CNN 的设置、结果和结论边界见
[`results/week9_cnn_baseline/README.md`](results/week9_cnn_baseline/README.md)。

## 依赖

```bash
python -m pip install -r algorithm/requirements.txt
```
