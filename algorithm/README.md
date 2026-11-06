# Algorithm and Data Processing

该目录保存 CSI 数据检查、时间窗口构建、特征提取和初步分类实验代码。

新组员先看 [CNN 本地训练与协作入口](GETTING_STARTED_CNN.md)，其中包含安装、训练、固定模型评估、学习曲线和无人机数据适配说明。代码位于 `algorithm-week9-cnn-baseline` 分支。

## 目录结构

```text
algorithm/
├── prepare_project_csi.py          # 检查小组 ESP32-S3 CSV 并构造时间窗口
├── analyze_scenarios.py            # 提取三场景的幅度时间特征
├── plot_scenario_features.py       # 绘制三场景特征分布对比图
├── motion_threshold_baseline.py    # 单特征阈值运动检测基线
├── motion_logistic_baseline.py     # 双特征逻辑回归运动检测基线
├── motion_cnn_baseline.py          # 完整时间×CSI幅度图的轻量CNN
├── evaluate_saved_motion_cnn.py    # 加载固定CNN，只评估，不重新训练
├── diagnose_cnn_learning_curves.py # 每轮训练/留出loss与跨批次诊断曲线
├── GETTING_STARTED_CNN.md          # 新组员运行入口与数据适配边界
├── CROSS_BATCH_PROTOCOL.md         # 批次划分与采集评价规则
├── tests/                         # 合成数据功能回归测试
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

## 独立新批次测试

新批次按 `Stop`、`Person`、`person_stop` 保存，支持保留采集时间子目录：

```bash
python algorithm/motion_cnn_baseline.py \
  --data-root data \
  --test-data-root data_batches/date_B \
  --output /tmp/fyp_cross_batch_date_B \
  --input-mode centered --epochs 30 --seed 42
```

该模式使用旧批次全部录制训练、新批次全部录制测试，标准化只使用训练数据，并拒绝两边
完全相同的 CSV。采集步骤、当前日期/会话限制和评估规则见
[`CROSS_BATCH_PROTOCOL.md`](CROSS_BATCH_PROTOCOL.md)。11月4日早批已有固定模型复测与批次内诊断。新批次结果依赖所用模型和划分，不能套用旧批次100%的结论。

## Week 10 模型保存和重新加载

运行 `motion_cnn_baseline.py` 后，输出目录会同时保存 `motion_cnn.pt`：模型权重、仅由训练数据
拟合的标准化均值/标准差、输入模式、窗口与步长、类别映射、训练配置和训练CSV的SHA-256。
这个文件用于固定模型推理。训练脚本会从原始测试窗口重新预处理并加载权重，检查预测标签
完全一致、运动概率在数值容差内一致，结果写入JSON的 `checkpoint.reload_verification`。

第10周的保存验证保持原第3折设置：每类第1、2段训练、第3段测试。它仍是旧数据内的测试。
以仓库根目录为当前目录，在已安装依赖的Python环境中运行：

```bash
python algorithm/motion_cnn_baseline.py \
  --data-root data --output algorithm/results/week10_checkpoint/fold3 \
  --input-mode centered --test-recording 3 --epochs 30 --seed 42 --device cpu

python algorithm/evaluate_saved_motion_cnn.py \
  --checkpoint algorithm/results/week10_checkpoint/fold3/motion_cnn.pt \
  --data-root data --test-recording 3 \
  --output algorithm/results/week10_checkpoint/fold3_reloaded
```

第二条命令是独立进程，只加载模型和训练标准化参数，不训练、不重新拟合标准化。
它按保存的窗口设置解析CSI，输出 `frozen_evaluation_report.json` 和 `test_predictions.csv`。
省略 `--test-recording` 时评估根目录内的全部录制，并拒绝与训练CSV字节完全相同的文件。

新批次真实数据到位后，可以用保存的同一个模型测试整个新目录：

```bash
python algorithm/evaluate_saved_motion_cnn.py \
  --checkpoint algorithm/results/week10_checkpoint/fold3/motion_cnn.pt \
  --data-root data_batches/date_B \
  --output /tmp/fyp_frozen_date_B
```

上例的固定模型使用6段旧CSV训练，不是9段全训练；和逻辑回归比较时也要使用相同训练来源。
文件哈希防止直接重用训练数据，但不能证明两个批次在日期、设备位置上独立；仍需保存采集记录。
已有的人体跨批次诊断不代表无人机验证。权重文件由根目录的 `*.pt` 规则忽略；从代码复现时先运行训练入口生成权重，已有本地权重应单独备份。

## 逐轮学习曲线与协作复现

`diagnose_cnn_learning_curves.py` 分别在两批数据内部训练，并记录旧模型在新批次上的逐轮表现。
每个批次按完整录制划分，不随机拆分重叠窗口。新批次路径、历史参考报告路径都可配置；
缺少旧权重或旧报告不会阻止运行。完整命令、数据布局与输出说明见
[CNN 本地训练与协作入口](GETTING_STARTED_CNN.md)。

图中的留出集和跨批次数据是诊断用途，不是训练选轮后仍未使用的最终测试集。
目前没有实现自动早停或混合批次训练，不能将这些计划写成已完成成果。

## 依赖

```bash
python -m pip install -r algorithm/requirements.txt
```
