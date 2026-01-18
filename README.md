# Neuro-CXL-PIM

<p align="center">
  <b>基于强化学习的智能 CPU/PIM 卸载决策系统</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/RL-PPO-green" alt="RL">
  <img src="https://img.shields.io/badge/Memory-Ramulator2-orange" alt="Memory">
</p>

## 📖 概述

Neuro-CXL-PIM 是一个基于强化学习(PPO)的智能计算卸载系统，专为 **CXL 连接的 Processing-In-Memory (PIM)** 架构设计。系统能够为神经网络工作负载（如 BERT、ResNet）自动学习最优的层级别 CPU/PIM 卸载策略，以最小化 **Energy-Delay Product (EDP)**。

### 核心特性

- 🧠 **智能决策**: 使用 PPO 算法学习 CPU 与 PIM 间的最优计算划分
- ⚡ **周期精确仿真**: 集成 Ramulator2 进行真实内存系统建模
- 📊 **多模型支持**: 支持 BERT-Base、ResNet-18 等神经网络
- 🔧 **可扩展架构**: 易于添加新的模型和硬件配置

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Neuro-CXL-PIM 系统                        │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐  │
│  │ WorkloadProfiler│ → │  PPO Agent    │ → │  Decision   │  │
│  │  (BERT/ResNet)  │    │ (Gymnasium)   │    │ CPU or PIM  │  │
│  └───────────────┘    └───────┬───────┘    └─────────────┘  │
│                               │                              │
│                               ▼                              │
│                    ┌───────────────────┐                     │
│                    │    Ramulator2     │                     │
│                    │  (Memory Cycles)  │                     │
│                    └───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CMake 3.14+ (用于编译 Ramulator2)
- g++-12 或 clang++-15

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/Neuro-CXL-PIM.git
cd Neuro-CXL-PIM

# 安装 Python 依赖
pip install -r requirements.txt

# 编译 Ramulator2
cd ramulator2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..
```

### 运行

```bash
# 训练 PPO 模型 (默认 10000 timesteps)
python main_dse.py

# 训练后模型保存为 pim_offload_policy.zip
```

## 📁 项目结构

```
Neuro-CXL-PIM/
├── main_dse.py           # 主程序: RL环境 + PPO训练
├── workload_analysis.py  # 模型Profiler (BERT/ResNet)
├── cxl_pim_config.yaml   # CXL内存配置示例
├── requirements.txt      # Python依赖
├── ramulator2/           # Ramulator2 内存仿真器
│   ├── build/            # 编译输出
│   └── src/              # 源代码
├── traces/               # 仿真trace文件(自动生成)
└── tests/                # 单元测试
```

## ⚙️ 配置

### 硬件参数 (`main_dse.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CPU_FREQ_GHZ` | 3.0 | CPU 频率 |
| `CPU_FP32_OPS_PER_CYCLE` | 32 | CPU 每周期 FP32 操作数 (AVX-512) |
| `PIM_FREQ_GHZ` | 1.0 | PIM 单元频率 |
| `PIM_FP32_OPS_PER_CYCLE` | 8 | PIM 每周期 FP32 操作数 |

### 训练参数

```python
train_and_evaluate(total_timesteps=10000)  # 调整训练步数
```

## 📊 输出示例

```
Neuro-CXL-PIM: RL-based CPU/PIM Offloading Decision
============================================================
Model: BERT-Base (97 layers)
Training timesteps: 10000
============================================================

Layer  0 [Memory  ]: PIM
Layer  1 [Compute ]: CPU
Layer  2 [Attention]: PIM
...

Decision Summary:
  CPU layers: 48 (49.5%)
  PIM layers: 49 (50.5%)
----------------------------------------
Total Latency: 1.23e+08 cycles
Total Energy:  4.56e-02 J
```

## 🔬 技术细节

### 奖励函数

系统使用 **Energy-Delay Product (EDP)** 作为优化目标:

```
reward = -EDP_normalized + heuristic_bonus
```

其中 `heuristic_bonus` 鼓励:
- Memory-bound 层在 PIM 执行
- Compute-bound 层在 CPU 执行

### 层类型分类

| TypeID | 类型 | 特征 | 倾向 |
|--------|------|------|------|
| 0 | Compute | Linear/Conv | CPU |
| 1 | Memory | LayerNorm/Activation | PIM |
| 2 | Attention | Self-Attention | PIM |

## 📚 引用

如果本项目对您的研究有帮助，请引用:

```bibtex
@misc{neuro-cxl-pim,
  title={Neuro-CXL-PIM: RL-based Intelligent CPU/PIM Offloading for Neural Networks},
  author={Your Name},
  year={2026},
  url={https://github.com/your-username/Neuro-CXL-PIM}
}
```

## 📄 许可证

MIT License

## 🙏 致谢

- [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) - 周期精确的内存仿真器
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO 实现
