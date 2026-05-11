# nem_forecast

澳大利亚 NEM 区域负荷与 5 分钟 RRP 预测的 **Python 工程脚手架**：配置驱动、CSV 占位加载、预处理与滚动验证骨架、SARIMA / XGBoost / LSTM / TFT 占位、尖峰两阶段与分位数预警占位。

## 约定

- **数据**：将按区域的 CSV 放入 `data/raw/`（路径见 `configs/default.yaml`）。默认文件名 `nem_{REGION}_5min.csv`，列约定见 `src/nem_forecast/data/schemas.py`。本仓库 **不包含** AEMO 自动下载；格式多变时请自行调整 `loaders.py`。
- **安装**：在 `nem_forecast` 目录执行 `pip install -e .`
- **预处理**：`python scripts/preprocess_once.py --config configs/default.yaml`
- **滚动 CV**：`python scripts/train_eval_rollcv.py --config configs/default.yaml --model xgb --target rrp --horizon 24h`
- **TFT**：`models/tft_pt.py` 为薄封装占位；生产环境可替换为 [pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting) 等实现。

无有效 CSV 时脚本会提示并退出（exit code 1）。若仅有部分区域文件，`preprocess_once.py` 与 `train_eval_rollcv.py` 会跳过缺失区域。

依赖中已将 `numpy` 上限设为 `<2`，以降低与部分 SciPy / XGBoost 二进制 wheel 的兼容性风险；若你使用 NumPy 2.x 环境，请自行对齐相关二进制版本。
