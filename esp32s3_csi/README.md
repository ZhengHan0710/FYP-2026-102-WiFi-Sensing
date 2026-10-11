# ESP32-S3 手机热点 CSI 采集

本项目采集真实 CSI，按用户提供的 csi_data.zip 的 28 列 CSV 格式保存。没有生成或提供虚构的实测数据。固件面向 ESP32-S3、ESP-IDF 5.4.x、VS Code ESP-IDF 扩展；其他版本需核对 API。

## 参考数据集

压缩包内有 7 个 CSV，表头一致：

```text
type,seq,timestamp,taget_seq,taget,mac,rssi,rate,sig_mode,mcs,cwb,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel_primary,channel_secondary,local_timestamp,ant,sig_len,rx_state,len,first_word_invalid,data
```

保留 `taget`、`taget_seq` 拼写，避免破坏读取程序。文件夹中的缩写含义不能仅从文件推断；不视为无人机类别。本项目将 taget 设置为 empty，将 taget_seq 设为当前采集段号 1/2/3，这是本项目明确的标注约定，不声称与原数据集标签含义相同。

## 配置与烧录

1. VS Code 打开本项目目录，使用 ESP-IDF 扩展。
2. 手机开启 2.4 GHz 热点，建议 WPA2，关闭热点自动关闭功能。默认 SSID 为 CSI_TEST，密码 12345678。在 main/main.c 修改成实际名称和密码。
3. 手机和开发板固定摆放，起步可相距约 2 米。不要在三段采集之间改变位置。采集期间人员离开链路附近。
4. USB 连接开发板。如果有两个 USB 接口，起步建议使用标注 UART 的接口。本项目默认 UART 控制台；使用原生 USB Serial/JTAG 需要在 menuconfig 修改控制台选项及选择对应串口。
5. 命令面板执行 ESP-IDF: Open ESP-IDF Terminal，然后依次执行下面命令。COM5 替换成你的端口。

```powershell
idf.py --version
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p COM5 flash monitor
```

在 menuconfig 搜索 CSI，确认启用 Wi-Fi CSI；默认串口波特率为 115200。sdkconfig.defaults 只对尚未确定的配置项生效，已有 sdkconfig 时务必检查。

看到 READY 以及连续 CSI_DATA 行后按 Ctrl+] 退出 Monitor。不要同时用两种程序打开同一个串口。CSI_STATS 行记录累计接收数、固件丢弃数、Ping 成功数及超时数。手机热点可能不回复 Ping；若 ping_ok 一直为零，应先排查热点，而非认为代码一定会得到固定速率。

## 保存三段数据

在项目目录安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

建议第一次保存完整原始数据：

```powershell
python capture.py --port COM5 --mode raw --seconds 60 --gap 10 --label empty
```

要生成与参考样例相同的 104 个数的数组：

```powershell
python capture.py --port COM5 --mode lltf52 --seconds 60 --gap 10 --label empty
```

两种模式默认预热 15 秒，采三段，每段 60 秒，间隔 10 秒。输出位于 data/本次时间/，包含三个 CSV、完整串口 serial.log 和 session.json。CSV 不额外添加字段或空白行。lltf52 文件名采用“时间_104_段号.csv”，raw 文件名明确标注 raw。

## raw 与 lltf52 的区别

- raw：28 列表头与参考一致，原始 CSI 数组和 first_word_invalid 不改，len 是驱动提供的字节数。仅启用 LLTF，常见长度为 128。
- lltf52：要求 LLTF 原始长度 128、接收带宽 20 MHz、没有副信道、sig_mode 为 0 或 1、first_word_invalid 为 0。选取 [-26..-1, +1..+26] 共 52 个复数，输出虚部、实部交错的 104 个整数，len 变为 104。不插值、不补零、不缩放，不支持的帧直接跳过并记录数目。
- first_word_invalid=1 的帧在 raw 中完整保留；后续分析原始数据时应屏蔽前两个复数。在 lltf52 模式中整帧拒绝，以免所选 +1 子载波含无效值。若没有帧可保存，请使用 raw 先排查。
- 52 个 LLTF 子载波包含导频，不能称为 52 个纯数据子载波。本项目并未改用 HT-LTF，也没有从幅值反推原始复数。
- 参考样例中 len=104、cwb=1、channel_secondary=2，并不是本项目 HT20 的配置。CSV 没有提供该数组的选择顺序、LTF 类型或是否经过预处理，因此“字段一致、长度一致”不等于与参考具有相同物理含义。若要混合训练，必须先找到原采集/预处理代码，统一 LTF、带宽、子载波顺序、缩放、采样率和实验标签。

timestamp 为电脑接收串口行的北京时间，格式 yyyy-MM-dd HH:mm:ss.SSS；local_timestamp 为设备接收时间戳（微秒），会发生 32 位回绕。串口时间含传输延迟，不是严格的射频到达时间。seq 保留设备序号；跳号有助发现固件/串口丢失。mac、RSSI、制式、信道等均来自真实接收帧，不复制参考值。

采集脚本遇到断线或序号回退会停止，将 session.json 标为 incomplete，不自动重连拼接基线。预热和间隔持续读取串口；一般无法保证数据严格等间隔。默认 Ping 请求 20 次/秒，不代表实际 CSI 一定 20 帧/秒。

## 实验验证与限制

检查三个 CSV 持续有数据、信道一致、无断线/重启；CSI_STATS 中 dropped 不持续增加。若 dropped 增加，可以将 main.c 中 PING_INTERVAL_MS 改为 100 后重新烧录。手机热点不能保证稳定的发包率；参考数据集约 100 帧/秒不意味着当前手机方案能复现同样速率。

这些数据是无遮挡基线，无法单独训练无人机分类器。后续需要真实无人机经过以及人员、风扇等干扰对照数据。CSV 中的字段和采集项目不证明具有无人机检测能力。

Python 格式转换会进行离线验证。固件需要在用户 ESP-IDF 环境编译及 ESP32-S3 上实际烧录验证；本项目未声称已实机通过。

官方依据：

- https://docs.espressif.com/projects/esp-idf/en/v5.4/esp32s3/api-guides/wifi.html#wi-fi-channel-state-information
- https://github.com/espressif/esp-csi/tree/master/examples/get-started/csi_recv_router
