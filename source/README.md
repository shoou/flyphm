# UAV / UGV PHM 最小可执行基线

该项目包含数据生成、时间窗口特征、按设备分组拆分训练/测试集、故障分类、无监督异常检测、演示性健康评分，以及简单退化外推。仅用于算法研发与工程接口验证，**不可用于飞行安全控制、运营许可、实际寿命判断或自动返航/刹车**。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python phm_demo.py --mode all
```

生成 `synthetic.csv`、`artifacts/models.joblib`、`artifacts/metrics.json` 和 `predictions.csv`。模型所报指标只针对合成样本，不代表真实泛化能力。

## 接入真实 CSV

必须有：`device_id,device_type,timestamp_s,motor_current_a,motor_temp_c,vibration_rms,tracking_error,battery_voltage_v,battery_temp_c,nav_residual,comm_loss_pct,fault_label`。

- `device_id`：设备唯一标识；`device_type`：UAV 或 UGV；`timestamp_s`：一个设备内单调递增的秒时间戳；`fault_label`：healthy/motor/battery/actuator/navigation/communication。
- 其余列的单位在文件名中：电流 A、电机及电池温度 °C、电压 V、通信丢包百分数 0~100。振动 RMS、跟踪误差及导航残差在本示例中是**归一化的示意量**，真实接入必须明确物理单位并重新校准阈值。
- 若没有故障标签，先建立仅健康数据的异常检测模型，不可直接采用此监督训练入口，标签不可填造。
- `--mode train --input your_data.csv` 训练；`--mode predict --input your_data.csv` 推理（要求列结构一致）。

## 对应真实数据源

- RflyMAD: https://github.com/rfly-openha/documents/blob/main/4_resources/dataset.md
- NASA IMS/FEMTO/Battery: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA ProgPy: https://github.com/nasa/progpy
- ROS 2 Diagnostics: https://github.com/ros/diagnostics

公开原始数据的列名/频率不相同，需独立编写适配器，不能直接把原始文件交给此脚本。不同故障标注及采样时序必须确认。

## 必须进一步工程化的项目

1. 将同一任务/飞行的时间窗口保持在同一数据拆分，避免任务泄漏；训练与验证分设备、型号及环境报告。
2. 实现各设备、工况和负载的基准线；分系统健康指标校准到可解释的退化状态。
3. 在显式已知退化目标、失效阈值、寿命标签后才能部署 RUL。增加置信区间、漂移监测和校准。
4. 设备丢数、时钟异常、单位错误、越界数据时输出 UNKNOWN；禁止健康假象。
5. 关键故障由本地经过验证的保护控制器处理；PHM 作为监测与决策辅助系统。

## 交互原型

真实数据下载、训练、界面和验证见项目根目录 `README.md`。本目录仍是合成基线。已修复：RUL 秒/小时换算、缺失数据输出 UNKNOWN、无标签推理窗口以及同设备多任务分窗。当前真实界面不读取本目录的合成成绩。
