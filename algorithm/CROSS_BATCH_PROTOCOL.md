# 下一轮采集与跨批次评估

## 已有结果的范围

现有 9 个 CSV 的 CNN 三折结果为 100%，但三折都是在每种场景同一次采集会话的三段数据之间轮换。
它检验了原采集条件下录制段之间的稳定性，该100%本身不代表独立会话、日期或房间的泛化。

`data/Stop/session.json` 记录空场景采于 2026-10-11；`Person` 和 `person_stop` 采于
2026-10-18。因而空场景与另外两类还存在日期差异。运动和静止同日采集，所以这不等于
模型只凭日期就能完成全部分类，但目前无法排除会话、设备布置和背景变化的影响。

## 下一轮需要采集什么

每个新日期、每种设备布置都采齐三种场景：empty、person_moving、person_stationary。
每种场景至少 3 段，每段 60 秒；每段之间留出恢复和改变动作的时间。

先在原房间、固定设备位置下做独立日期复测；再增加人员位置和路线，最后单独增加设备位置
或房间变化。一次只改变一类条件，便于定位性能变化的来源。

为减少类别与采集顺序绑定，建议按轮交替采集三种场景，例如第一轮 empty → moving →
stationary，第二轮 moving → stationary → empty，第三轮 stationary → empty → moving。
每次先让环境稳定再开始录制；不要把进入场地、离开场地和移动设备的过渡过程算作静止样本。

保持固件、热点、信道、带宽、LTF、子载波顺序和缩放配置一致。每次保留 CSV、`session.json`
和 `serial.log`，另记日期、房间、设备位置/距离/高度、人员位置、路线和是否有其他移动物体。
约 20 fps 是原数据的实测值；新批次仍需检查时间戳，不能仅凭配置假设采样率相同。

## 保存方式与采集命令

在仓库根目录下保存新批次，例如：

```text
data_batches/date_B/
├── Stop/<采集时间>/..._104_1.csv
├── Person/<采集时间>/..._104_1.csv
└── person_stop/<采集时间>/..._104_1.csv
```

CNN 脚本递归读取这三个文件夹内的 CSV，因此可以保留 capture.py 自动生成的时间子目录。
文件名末尾继续保留 `_<段号>.csv`。算法按文件夹识别场景。

在 Windows 采集电脑上，在仓库根目录执行，`COM4` 替换为实际端口：

```powershell
python esp32s3_csi/capture.py --port COM4 --mode lltf52 --seconds 60 --segments 1 --label empty --out data_batches/date_B/Stop
python esp32s3_csi/capture.py --port COM4 --mode lltf52 --seconds 60 --segments 1 --label person_moving --out data_batches/date_B/Person
python esp32s3_csi/capture.py --port COM4 --mode lltf52 --seconds 60 --segments 1 --label person_stationary --out data_batches/date_B/person_stop
```

这三条命令对应一轮，重复至少三轮并轮换场景顺序。每次命令执行前准备好对应场景。
`capture.py` 自带预热，并在时间子目录保留完整日志和元数据。

注意：如果每轮单独启动 `--segments 1`，不同会话的文件末尾段号可能都从1开始。默认训练入口按文件末尾的段号留出第3段，归档时应保留原始文件，并在数据副本中按实际轮次明确编号1、2、3，保存副本与原始文件的对应清单。不要将全部名为 `_1.csv` 的录制直接送入默认第3段留出实验。

## 直接用已保存的 CNN 测试新批次

原实验使用第10周保存的 `results/week10_checkpoint/fold3/motion_cnn.pt`。权重不随Git上传；新组员请先按 [运行入口](GETTING_STARTED_CNN.md) 训练生成自己的权重，并在下方命令中替换路径。
该模型按第9周第3折设置重新训练后保存：6份旧CSV训练、3份旧CSV测试，分别有359和167个窗口。
模型文件包含权重、训练集标准化参数、64帧窗口、16帧步长及centered输入设置。
新批次到齐后直接加载并测试，不重新训练，也不重新拟合标准化参数。

在仓库根目录执行：

```bash
python algorithm/evaluate_saved_motion_cnn.py \
  --checkpoint algorithm/results/week10_checkpoint/fold3/motion_cnn.pt \
  --data-root data_batches/date_B \
  --output /tmp/fyp_frozen_cnn_date_B \
  --device cpu
```

不指定 `--test-recording`，评估新目录中全部有效窗口。去时间均值仍在每个窗口内计算，
标准化沿用保存的训练参数。脚本通过SHA-256拒绝与模型训练来源字节完全相同的CSV，
即使这些CSV改名或放进子目录。它不能检测所有经过裁剪或重写的重复数据，因此仍需保留原始来源，
避免把同一段录制拆到两个批次。

截至2026-11-06，11月4日早批已完成固定模型复测和批次内重训诊断。旧模型对新批次运动漏检明显，批次内重训后有所恢复；这些已用于诊断的数据不能再当作未使用的最终测试集。晚批2m、3m、5m已确认是手机与ESP32的收发间距，实际动作标签仍需采集者确认。

## 可选实验：用旧9份CSV重新训练后测试新批次

如果另行开展旧数据全量训练实验，可使用下列入口。它会用旧目录全部9份CSV训练新模型，
与当前已保存的6份训练模型属于不同实验，必须分别记录模型和结果。
训练设置在第一次新批次评估前确定，避免反复查看测试结果后调参。

```bash
python algorithm/motion_cnn_baseline.py \
  --data-root data \
  --test-data-root data_batches/date_B \
  --output /tmp/fyp_cross_batch_date_B \
  --input-mode centered \
  --epochs 30 \
  --seed 42
```

此模式使用旧目录的所有录制训练，新目录只用于最终评估；`--test-recording` 不参与划分。
去时间均值分别在每个窗口内计算，标准化参数只由训练数据拟合。脚本拒绝相同训练/测试目录，
并通过 SHA-256 拒绝两边字节完全相同的 CSV（即使改名或放进子目录）。这不能检测所有经过
裁剪或重写的重复数据，所以仍应保留原始来源，避免把同一段录制拆到两个批次。

原来的 `--test-recording 1/2/3` 入口仍用于旧数据内部留一测试。不要把多个日期混进同一个根目录
后按文件末尾的段号划分，并将其称作跨日期评估。

## 结果应如何报告

- 整体准确率、宏平均召回/平衡准确率、运动召回和非运动召回。
- 混淆矩阵中的 TP、TN、FP、FN。
- empty、moving、stationary 各自被预测为运动的比例。
- 训练和测试的日期、房间、文件清单、帧数、窗口数、采样间隔、丢帧和断点情况。

本轮以CNN的新批次结果为主。旧双特征逻辑回归可作为可选历史参照，不是新批次验证的必做项。
如需比较两种方法，必须使用相同训练和测试来源；旧测试集的89.22%不能直接与新测试集CNN比较。

外部批次尚未采集时只能验证运行入口，不能产生或宣称跨日期识别率。第一次新测试若导致
调参，该批次应转为验证集，再保留一个未使用的日期/会话作为最终测试集。
