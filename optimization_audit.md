# FedML 优化审计

## 结论

项目还能优化，但当前最值得做的是先恢复可运行性和实验边界，再做性能优化。现有改动已经把 FedRep、DMCFE、隐私基线、结果记录器和对比脚本同时接入 FedAvg 核心路径，功能扩展速度快于验证和隔离速度。

## P0：先处理

1. 修复 `python/examples/cross_cloud/mqtt_s3_fedavg_mnist_lr_example/step_by_step/config/fedml_config.yaml:44-45`：当前值是 `config/mqtt_` 和 `config/s3_`，对应配置文件不存在。应恢复为 `config/mqtt_config.yaml` 和 `config/s3_config.yaml`，然后做一次 YAML 加载与路径存在性检查。
2. 检查并轮换 tracked YAML 中的 MLOps/W&B key。当前文件包含非空凭据字段；本次没有验证它们是否有效。有效凭据不应继续写入仓库。
3. 在提交前隔离生成物：当前有 11 个 tracked modified files、75 个 untracked files，包含 `.vs`、实验 `results`、`.7z` 和临时测试文件。`.gitignore` 目前没有覆盖这些类型。

## P1：本轮建议

1. 将 DMCFE/FedRep 适配器从 `python/fedml/simulation/sp/fedavg/fedavg_api.py` 拆出。该文件目前 1,156 行，并在第 13-50 行无条件导入多组 DMCFE smoke 模块和 `privacy_baselines`。普通 FedAvg 不应因实验算法缺包而无法导入；建议采用独立 aggregation strategy 或延迟导入。
2. 统一依赖来源。根目录 `requirements.txt` 删除了 `wandb`，但 `fedavg_api.py:7` 无条件 `import wandb`；新 DMCFE 代码使用 `sympy.randprime`，而 `requirements.txt` 和 `python/setup.py` 都没有声明 `sympy`。应按运行模式拆分基础依赖、SP/FedRep/DMCFE 额外依赖，并在干净环境执行安装验证。
3. 明确实验口径。当前模型的共享表示有 100,480 个参数，而 `dmcfe_validation_dim` 是 7,850，仅覆盖 7.81%。`fedavg_api.py:765-774` 会把未检查的后缀保留为明文 FedAvg 结果。因此 README 和论文数据不能称其为完整模型 DMCFE 聚合；要么完整覆盖，要么把该运行命名为 partial-coordinate smoke test。
4. 为 baseline 建立最小 CI：普通 `FedAvg` 导入/启动、FedRep 参数划分、DMCFE 等价性、dropout recovery、结果 CSV schema 各保留一个测试；测试环境至少要安装 `torch`、`multiprocess` 和对应算法 extra。

## P2：之后再做

- 将结果目录改为显式的 run-specific output path，避免默认相对当前工作目录写入 `results/*.csv`。
- 清理 `fedavg_api.py` 中重复的日志、配置读取和张量复制，减少每轮不必要的 CPU/GPU 往返。
- 用 `git diff --check` 清掉新增代码中的 trailing whitespace，再做格式化和类型检查。

## 已执行检查

- `git diff --check`：发现新增 diff 中存在 trailing whitespace。
- `python -m compileall`（使用 `D:\anaconda\python.exe`）：FedAvg/FedRep 相关源码通过。
- 目标 pytest：未通过收集阶段，原因是环境缺少 `torch`，不是测试断言失败。
- `import fedml`：未执行成功，原因是环境缺少 `multiprocess`。
- YAML 读取：成功确认 cross-cloud 配置值为截断路径，且目标文件不存在。

## 本次未改动

没有修改实现代码、配置值、依赖文件或删除任何用户文件。本次新增的审计辅助文件是 `task_plan.md`、`notes.md` 和本文件。
