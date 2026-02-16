# AI量化选股与调仓系统 V1.0.2

本项目提供一个可本地运行、可迁移到聚宽的量化选股框架，核心是“三模型融合 + 动态阈值控制”。

## 0. 30 秒快速检查（终端）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py inspect-models --root .
```

预期：终端输出 `factor_count`、`prediction_shape`、`prediction_min/max/mean`。
说明：该命令用于“模型可用性体检”，不生成结果文件。

## 1. 项目内容

- 本地能力：模型检查、特征打分、TopN 选股导出
- 聚宽能力：读取预训练模型并按月调仓
- 模型文件：`model_reg_final.pkl`、`model_cls_final.pkl`、`model_dir_final.pkl`
- 因子清单：`selected_factors.csv`
- 研报文件：`reports/量化研报_动态阈值三模型_2019-2025_v4.pdf`

## 2. 目录结构

```text
quant/
  main.py
  inspect_models.py
  joinquant_strategy_template.py
  model_reg_final.pkl
  model_cls_final.pkl
  model_dir_final.pkl
  selected_factors.csv
  src/quant_soft/
    cli.py
    config.py
    feature_io.py
    model_service.py
    pipeline.py
    scoring.py
  data/train.csv
  docs/
    用户手册.md
    软件设计说明.md
    测试报告.md
    GITHUB发布前检查清单.md
  模型说明.md
  CHANGELOG.md
  DISCLAIMER.md
  LICENSE
  requirements.txt
```

## 3. 环境安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 运行命令

模型检查：

```bash
python main.py inspect-models --root .
```

本地打分与选股：

```bash
python main.py rank --root . --features your_features.csv --top-n 10 --out rank_result.csv
```

兼容脚本：

```bash
python inspect_models.py
```

## 5. 输入输出约束

输入特征文件要求：

- 必须包含 `code` 列
- 必须包含 `selected_factors.csv` 中全部因子列
- 因子列名和顺序应与训练时定义一致

输出文件 `rank_result.csv` 主要字段：

- `pred_reg`：回归模型输出
- `pred_cls`：分类模型输出
- `pred_dir`：方向模型输出
- `ai_score`：融合评分
- `consistency`：一致性指标

## 6. 复现口径（建议在研报中保持一致）

- 基准：`000985.XSHG`
- 交易设置：实盘价、避未来函数、固定滑点 0、含交易成本
- 调仓频率：月频（每月首个交易日）
- 股票池过滤：ST、停牌、涨跌停开盘、新规板块代码过滤
- 模型更新：回测期内模型固定，不在策略中训练

## 7. 聚宽部署

1. 打开 `joinquant_strategy_template.py`
2. 粘贴到聚宽策略编辑器
3. 上传三个模型文件和因子文件
4. 运行回测或模拟盘

## 8. 免责声明

本项目仅用于技术研究与工程演示，不构成任何投资建议。历史回测结果不代表未来收益。

更多说明见 `DISCLAIMER.md`。开源许可见 `LICENSE`。
