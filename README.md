# 无人系统 PHM 可交互原型

基于公开真实数据展示无人机螺旋桨故障诊断、电池 SOH/RUL、独立测试结果与告警复核。现有 `source/` 合成示例保留，**不会作为真实实验的自动替代数据**。

## 启动

推荐 Python 3.11–3.13。首次安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements-lock.txt` 记录本次实际验证环境的精确依赖版本；复现实验时可使用 `pip install -r requirements-lock.txt`。

已有本地实验结果时，直接运行：

```bash
./run.sh
```

浏览器打开 http://127.0.0.1:8501 。Windows 可执行 `.venv\Scripts\python -m streamlit run src/app.py`。

首次取得真实数据并生成实验：

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python -m phm.training --prepare
.venv/bin/python scripts/summarize_results.py
```

最后一条命令同步 `validation_status.json`，并生成可离线打开的 `runs/validation_report.html`（图表资源内嵌）。

下载依赖公开服务器。支持重试，已完成文件会复用；失败会明确报错，不会生成假成绩。若机器设置了失效代理，应先修正代理设置。数据与模型仅保存在本地，原始数据不提交版本库。

可单独执行 `--dataset uav` / `--dataset battery` 下载，以及 `--task uav --prepare` / `--task battery --prepare` 训练。界面“模型训练与验证”可从已下载数据重新训练并保存新的实验。

## 页面与演示顺序

1. **设备与案例总览**：真实任务数量、测试指标、数据覆盖、质量报告。导航/通信/伺服/UGV 显示未覆盖，整机健康状态为无法评估。
2. **故障诊断回放**：默认独立测试任务；播放、暂停、倍速、时间窗口滑块；遥测、异常分数、分类结果、证据及导出。当前窗口可加入待复核记录。
3. **电池健康与寿命**：选择电池和观测截止循环，显示观测容量/SOH、该电池留一模型的 RUL 与探索性区间；未来真实容量默认隐藏，可打开作事后验证。
4. **模型训练与验证**：训练/验证/测试对照、混淆矩阵、逐类召回、全局特征重要性、误报事件、逐电池与分阶段误差、JSON/CSV 下载。
5. 回到总览，更新事件状态与复核意见，验证本地持久化。

演示不需要现场重新下载或训练。任务 ID 表示飞行任务，不是独立飞机编号。

## 数据与实验定义

### 无人机：UAV Realistic Fault Dataset

来源：https://github.com/tiiuae/UAV-Realistic-Fault-Dataset

- 首次下载固定当前提交哈希；默认每个类别选择按时间均匀分布的 8 个任务，共 40 个任务。
- 仅使用 SensorCombined 的三轴加速度（m/s²）及角速度（rad/s）；时间戳微秒转秒，按任务归零。音频不下载。
- 任务级标签 0–4 表示损坏螺旋桨数量，不用于具体电机定位。没有飞行中故障起点标签，因此不报告检测延迟。
- 原始信号排序、时间去重、非有限值与加速度裁剪样本剔除，并输出质量报告。2 秒窗口、1 秒步长；不足 100 样本或内部间隔超过 0.2 秒的窗口不推理。
- 每类任务按时间排序，默认 4 个训练 / 2 个验证 / 2 个测试。完整任务隔离；相邻窗口不跨集合。仍不代表跨机体泛化。
- 随机森林五分类；加速度波动阈值作为二分类基线；Isolation Forest 只在健康训练窗口拟合。阈值来自健康验证窗口的 95% 分位数。
- 连续异常窗口合并为误报事件，健康有效窗口数 × 1 秒为曝光时长；短时曝光下每小时率可能很高且不稳定。
- 动力 HI 仅为相对健康训练样本的振动偏离指标，未做工程标定；分类概率不是校准过的置信度。
- 原仓库未发现明确的数据许可声明。保留来源用于本地研究演示；再次分发原始数据或商用前需确认授权。

### 电池：NASA Battery

来源：https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

下载器从 NASA 官方 ZIP 中按范围提取 `BatteryAgingARC-FY08Q4.zip`，校验 ZIP CRC，读取 B0005/B0006/B0007/B0018 及 README。记录下载来源、压缩包 SHA256 和原始 MAT 哈希。

- 原始 README 明确额定容量 2 Ah，失效判据为容量下降 30% 至 1.4 Ah。本原型取首次观测容量 ≤1.4 Ah 的循环为 EOL。
- SOH = 实测容量 / 2 Ah，这是观测健康量，不是模型预测成绩。
- 容量读取后以放电记录从 1 开始计数；RUL 单位为放电循环，不是小时。
- 未观察到阈值穿越的电池作为右删失样本，保留曲线、不给真实 RUL 标签。只对 EOL 前且有至少 10 个历史循环的样本评分。
- 留一电池验证，输入仅含当前循环、SOH、最近至多 20 循环的容量均值与斜率；未来容量只用于生成评估真值。
- 对比线性阈值外推和随机森林。线性斜率非负时不输出寿命，对比表限定在共同有效样本上。
- 区间半径来自训练电池内部再次留一的绝对误差 90% 分位数，测试电池不参与。独立电池数量很少，属于探索性区间，不声称具有校准覆盖保证。
- 报告 MAE、RMSE、逐电池误差、等权电池 MAE、早中晚阶段误差、实测区间覆盖率和宽度。

引用：B. Saha and K. Goebel (2007), “Battery Data Set”, NASA Ames Research Center.

## 代码与产物

```text
src/app.py                 Streamlit 界面
phm/data.py                真实数据适配、质量与来源
phm/training.py            两条训练验证流水线
phm/events.py              SQLite 事件复核记录
scripts/download_data.py   可重试的数据下载与来源清单
source/phm_demo.py         原合成算法示例（已修正缺失值、单位、无标签窗口）
validation/                原 RflyMAD 通用适配及独立验证命令
runs/<task>_<id>/           配置、来源、划分、模型、预测、指标与状态
data/raw/                  本地原始数据（不提交）
data/processed/            可重建特征及曲线（不提交）
tests/                     数据泄漏、质量、单位与界面回归测试
```

每次训练新建独立实验目录；异常记录为 failed，不会在界面冒充完成实验。模型 joblib 仅加载本程序生成的可信文件。`runs/events.sqlite` 保存事件处理意见；人工意见不会自动变成训练真值。

RflyMAD 的原映射模板仍需依据实际字段和任务目录配置；当前交互原型默认使用已取得的 TII 数据，未声称完成 RflyMAD 实验。ROS 2/CAN/MAVLink 在线接入、伺服专项模型、UGV 实车验证、经过工程校准的整机风险与完整 MLOps 留待下一阶段。

## 检查

```bash
.venv/bin/python -m pytest
```

自动化测试检查时间单位、未知健康状态、任务隔离、因果前缀特征、删失处理、告警合并及持久化；有真实实验产物时会进一步检查界面加载和关键交互。测试中的小型人造夹具仅用于软件回归，不计入展示指标。
