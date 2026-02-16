# Changelog

## V1.0.3 - 2026-02-16
- 完善 GitHub 发布材料：新增 `LICENSE`、`DISCLAIMER.md`、`.gitignore`、`run_check.bat`。
- 重写 `README.md`，增加“30 秒快速检查”与结果解释。
- 新增 `docs/GITHUB发布前检查清单.md`，用于发布前自查。
- 更新 `模型说明.md` 为可公开审阅版本。

## V1.0.2 - 2026-02-15
- 回退聚宽模板近期策略改动，恢复为当前稳定版本（以 `joinquant_strategy_template.py` 实际代码为准）。
- 同步清理版本记录，移除与当前代码不一致的描述。

## V1.0.1 - 2026-02-15
- 完善软件架构：新增 `src/quant_soft/pipeline.py` 作为应用服务层，统一模型检查与选股流程。
- 重构 CLI：`src/quant_soft/cli.py` 改为薄入口，仅负责参数解析与输出。
- 优化启动入口：`main.py` 显式挂载 `src` 路径后再启动包内 CLI。
- 兼容旧脚本：`inspect_models.py` 改为调用统一服务层，避免与 CLI 逻辑分叉。
- 包导出更新：`src/quant_soft/__init__.py` 增加 `pipeline`。

## V1.0.0 - 2026-02-15
- 建立可独立运行的量化程序骨架（CLI + 模型推理 + 评分选股）。
- 新增本地命令：`inspect-models`、`rank`。
- 新增聚宽模板：`joinquant_strategy_template.py`。
- 新增文档模板：用户手册、软件设计说明、测试报告。
- 新增项目内训练数据文件：`data/train.csv`，并将文档路径统一为项目相对路径。
