# Neuro-CXL-PIM

基于强化学习的 CPU/PIM 卸载决策研究原型。仓库把虚拟模型 profiling、Ramulator2 内存仿真、能耗/延迟建模和 PPO 策略训练串在一起，用来探索在 CXL/PIM 风格系统中哪些层更适合放到近内存侧执行。

## 当前状态

当前代码可以对虚拟 BERT-Base 或 ResNet-18 工作负载生成层级特征，并在 Gymnasium 环境中训练一个二分类策略：`CPU` 或 `PIM`。仓库中包含一个已保存的 `pim_offload_policy.zip`、若干 trace/config 示例和测试文件。

这仍是架构探索原型。`main_dse.py` 中的硬件参数、功耗、EDP 奖励和工作负载 profiling 都是建模假设，不是真实硅片测量。

## 主要模块

- `main_dse.py`：Gymnasium 环境、Ramulator2 调用、PPO 训练与评估入口。
- `workload_analysis.py`：虚拟 BERT-Base 和 ResNet-18 层级 workload profiler。
- `cxl_pim_config.yaml`：CXL/PIM 风格配置参考。
- `ramulator2/`：内存系统仿真器相关代码。
- `traces/`：自动生成或保留的 trace/config 文件。
- `tests/`：硬件参数和 workload profiler 的基础测试。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

运行主实验前需要构建 Ramulator2，或通过环境变量指定可执行文件：

```bash
export RAMULATOR_PATH=/path/to/ramulator2
python main_dse.py
```

如果不设置 `RAMULATOR_PATH`，代码会默认查找 `ramulator2/build/ramulator2`。

## 运行与验证

训练并评估 PPO 卸载策略：

```bash
python main_dse.py
```

运行测试：

```bash
pytest
```

## 注意事项

- `main_dse.py` 会在 `traces/` 下生成每层仿真 trace 和 YAML 配置。
- Ramulator2 失败或超时时，代码会回退到惩罚性的默认 cycle 数；正式结果需要检查日志。
- PPO 当前固定在 CPU 上训练，避免策略训练本身和系统仿真混用 GPU 资源。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
