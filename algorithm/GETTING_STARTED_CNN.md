# CNN 本地训练与协作入口

当前分支：`algorithm-week9-cnn-baseline`。从 GitHub 下载时请选择这个分支。

这套代码使用 Python、PyTorch、NumPy 和 Matplotlib，在本地训练。普通 CPU 即可运行现有小批次；不依赖网页训练平台、GPU、云端账户或原作者电脑里的 Python 路径。

## 1. 下载代码、安装依赖

建议使用 Python 3.11 或 3.12。在已安装 Git 的终端中执行：

```bash
git clone --branch algorithm-week9-cnn-baseline https://github.com/ZhengHan0710/FYP-2026-102-WiFi-Sensing.git
cd FYP-2026-102-WiFi-Sensing
```

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r algorithm/requirements.txt
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r algorithm/requirements.txt
```

以下示例使用 `python`，运行时将它替换成上述虚拟环境的 Python 路径。CPU 实验不需要安装 CUDA。安装失败时保留完整报错，先核实 Python 版本与 PyTorch 对当前系统的支持。

## 2. 先复现项目的 CNN 基线

在仓库根目录执行：

```bash
python algorithm/motion_cnn_baseline.py \
  --data-root data \
  --output outputs/cnn_old_fold3 \
  --input-mode centered --test-recording 3 \
  --epochs 30 --seed 42 --device cpu
```

Windows 可将以上命令写成一行。这里 `data/` 中是原三场景录制：

| 文件夹 | 场景 | 当前二分类标签 |
|---|---|---|
| `Stop` | 无人 | 非运动 0 |
| `Person` | 人员运动 | 运动 1 |
| `person_stop` | 人员静止 | 非运动 0 |

每类第1、2段训练，第3段测试。完整文件分别划分后再切64帧窗口，步长16，窗口不跨文件或序号断点。预处理为每个窗口去时间均值，再使用训练录制拟合的标准化参数。网络是3层卷积、2层全连接；参数设置不代表已经经过最优搜索。

输出包括：

- `motion_cnn.pt`：网络权重、训练标准化参数、窗口设置、标签和训练来源。
- `motion_cnn_report.json`：训练设置、每轮训练loss、最终指标、加载一致性检查。
- `test_predictions.csv`：每个测试窗口的输出。

该入口本身只记录逐轮训练loss。需要逐轮留出loss时使用第4步的诊断脚本。不要把同批次成绩当成跨批次或无人机检测成绩。

## 3. 固定模型评估

先完成第2步生成权重，然后可以只加载模型，不重新训练：

```bash
python algorithm/evaluate_saved_motion_cnn.py \
  --checkpoint outputs/cnn_old_fold3/motion_cnn.pt \
  --data-root data --test-recording 3 \
  --output outputs/cnn_old_reloaded --device cpu
```

独立批次准备成同样的三场景目录后，将 `--data-root` 改为新目录并省略 `--test-recording`。程序会拒绝训练CSV的字节副本，但这不能替代实际的批次、标签和采集条件核查。

如果只评估一种场景，可以显式使用 `--scenarios person_moving`、`empty` 或 `person_stationary`。缺少运动或非运动时，缺失类别的召回率和平衡准确率不可估计，不要把单类结果当成完整检测能力。

## 4. 逐轮训练、留出及跨批次曲线

新批次需要另外准备。此分支不重复上传 `main` 中的原始新批次，也不上传本地重复暂存的 `data_batches/`。准备后的布局为：

```text
data_batches/batch_B/
├── Stop/         # 无人CSV，每类文件名末尾保留 _1.csv、_2.csv、_3.csv
├── Person/       # 人员运动CSV
└── person_stop/  # 人员静止CSV
```

可保留每次采集的子目录。现有 11月4日早批来源在 `main` 的 `data/Stop2.0`、`data/Person2.0`、`data/Person_stop2.0`，分别复制到上述三个对应目录即可；原始CSV保持不变。晚批 `Person(2m/3m/5m)` 的实际动作仍需核实，不要仅凭文件夹名把它们标为运动。

先只检查数据，不训练：

```bash
python algorithm/diagnose_cnn_learning_curves.py \
  --old-data-root data --new-data-root data_batches/batch_B \
  --output outputs/learning_curves --check-only
```

检查通过后去掉 `--check-only`，即进行两次固定配置训练：

```bash
python algorithm/diagnose_cnn_learning_curves.py \
  --old-data-root data --new-data-root data_batches/batch_B \
  --output outputs/learning_curves --epochs 30 --seed 42
```

此脚本使用CPU。输出目录必须是新目录，防止覆盖已有结果。不需要先下载任何旧模型权重或本地历史报告：存在时会额外比较，不存在时跳过。输出包括：

- `within_batch_learning_curves.png/.pdf`：两批各自的训练与留出loss、准确率。
- `cross_batch_learning_curves.png/.pdf`：旧模型各轮在新批次上的loss与运动召回。
- `epoch_metrics.csv`、`old/learning_history.json`、`new/learning_history.json`：逐轮数值。
- `old/motion_cnn.pt`、`new/motion_cnn.pt`：两次实验最后一轮的模型。
- `summary.json`：配置、原文件未变化检查、可选历史权重一致性检查。

图上训练loss和留出loss都在评估模式下计算，关闭Dropout，使用同一训练类别权重；训练优化时的loss另行保存。`original_training_loss.png` 只在本地已有历史报告时额外生成。

这是诊断回放，不会早停或按最优轮数选模型。看到某轮更好，不应直接把该轮结果当成新的独立测试成绩。每轮观察过的批次属于开发验证数据，最终仍需保留新的未使用批次。

## 5. 新组员做无人机数据实验时怎样复用

可以复用 `SmallMotionCNN` 的特征提取结构和训练/评估思路。现有读取器专用于本项目ESP32 CSV，标签专用于人体运动二分类。

无人机数据必须先确认：是CSI、原始射频、通信流量还是图像；无人机是被感知目标还是通信端点；是否有真实的有/无无人机或其他类别标签。按新数据编写读取器、定义窗口和分割规则，必要时修改最后的输出类别数，并重新训练。不要把数据强行变成64×52后当作同一任务，也不要直接用人体训练权重宣称完成无人机检测。

公开集文件、预训练权重和本地新批次不随代码上传。此前UT-HAR结果来自SenseFi CNN-5；它不等同于当前项目小型CNN在公开集上的成绩。

## 6. 功能检查

```bash
python -m unittest discover -s algorithm/tests -p 'test_motion_cnn*.py' -v
python -m unittest discover -s algorithm/tests -p 'test_diagnose_cnn_learning_curves.py' -v
```

这些是合成数据功能回归检查，验证划分、模型保存/加载和诊断脚本可运行，不代表真实识别准确率。
