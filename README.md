# Neuro-CXL-PIM 🚀🧠

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/RL-PPO-green" alt="RL">
  <img src="https://img.shields.io/badge/Memory-Ramulator2-orange" alt="Memory">
</p>

> Neuro-CXL-PIM — an RL-based design-space exploration framework for CPU/PIM offloading with cycle-accurate memory modeling. Cute, practical, and research-ready! 🐣💡

Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Build Ramulator2](#build-ramulator2)
  - [Run Examples](#run-examples)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works (High-level)](#how-it-works-high-level)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Citing This Work](#citing-this-work)
- [License & Acknowledgements](#license--acknowledgements)
- [Contact](#contact)

---

## Overview

Neuro-CXL-PIM is a design-space exploration (DSE) framework that uses reinforcement learning (PPO) to learn intelligent offloading decisions between CPU and a CXL-attached PIM device for neural network workloads. The framework integrates a cycle-accurate memory simulator (Ramulator 2.0) to evaluate memory-system performance impact, enabling research-grade evaluation of CPU/PIM partitioning strategies. 💖

This repository contains:
- RL environment and training code (PPO)
- Workload profiling tools (e.g., BERT, ResNet)
- Ramulator2 integration for cycle-accurate memory simulation
- Config files and tracing tools for reproducible experiments

---

## Key Features

- 🧠 RL-driven offload decision-making using PPO
- ⚡ Cycle-accurate memory modeling via Ramulator 2.0
- 📊 Support for multiple neural models (BERT-Base, ResNet-18, ...)
- 🔧 YAML configuration for easy parameter sweeps and reproducibility
- ♻️ Modular structure to add new models / memory configs / policies
- 🧪 Trace generation and evaluation tooling

---

## Quick Start

### Prerequisites

- Python 3.8+ (venv recommended)
- pip
- CMake 3.14+
- A C++20-capable compiler (tested with g++-12, clang++-15)
- make, git, and typical build utilities

Recommended (Ubuntu-like):

```bash
# Install essential tools (example)
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3-pip cmake build-essential git
# If you need a specific compiler:
sudo apt install -y g++-12
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/lkcfqy/Neuro-CXL-PIM.git
cd Neuro-CXL-PIM
```

2. Create a Python virtual environment and install required packages:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: `requirements.txt` should list RL libraries (e.g., stable-baselines3), PyTorch (if used for workloads), YAML, and any other Python dependencies.

### Build Ramulator2

Ramulator2 is embedded under `ramulator2/`. Build it to obtain the `ramulator2` executable and/or `libramulator.so`.

```bash
cd ramulator2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
# After build, you should have `ramulator2` and possibly `libramulator.so` in the top-level of ramulator2
cd ../..
```

If the Ramulator build pulls third-party libs via CMake/FetchContent, ensure your system can access the network.

### Run Examples

- Train the default PPO model (quick run — uses the main DSE script):

```bash
# From repo root
source .venv/bin/activate
python main_dse.py
```

This runs the default experiment (the script typically trains for a default number of timesteps and saves a policy file such as `pim_offload_policy.zip`).

- Train with custom settings (example arguments — check `main_dse.py` for exact CLI flags):

```bash
python main_dse.py --config cxl_pim_config.yaml --timesteps 10000 --save-dir ./models
```

- Evaluate a trained policy (example):

```bash
python evaluate_policy.py --model ./models/pim_offload_policy.zip --config cxl_pim_config.yaml
```

(If the repo contains different filenames or argument names, adapt commands to the script's CLI. See the code for exact flag names.)

---

## Configuration

The framework uses YAML configuration files (example: `cxl_pim_config.yaml`) to describe:
- Memory system parameters (CXL/PIM sizing, DRAM timing)
- RL environment settings (observation/action spaces, reward function)
- Model/workload selection (BERT, ResNet, parameters)
- RL hyperparameters (PPO learning rate, batch size, timesteps)

Example (high-level) YAML structure:

```yaml
Environment:
  workload: bert
  dataset: ...
  memory_system:
    type: CXL
    pim_capacity: 8GB
    host_memory: 64GB
  ramulator_config: ramulator_config.yaml

RL:
  algorithm: PPO
  timesteps: 100000
  learning_rate: 3e-4
  gamma: 0.99
  n_steps: 2048
```

Tip: Keep the `ramulator_config` file separate and version it so experiments are reproducible.

---

## Project Structure

```
Neuro-CXL-PIM/
├── main_dse.py           # Main training / DSE entry (RL environment + PPO orchestration)
├── workload_analysis.py  # Workload profiler and feature extractor (BERT/ResNet helpers)
├── cxl_pim_config.yaml   # Example CXL/PIM + RL configuration
├── requirements.txt      # Python dependencies
├── ramulator2/           # Ramulator2 DRAM simulator (subtree)
│   ├── build/            # CMake build output
│   └── src/              # Ramulator source
├── traces/               # Generated traces for simulations
├── tests/                # Unit and integration tests
├── README.md             # <-- This file
└── ...                   # Other scripts / utilities
```

---

## How It Works (High-level)

1. The workload profiler extracts memory and compute characteristics from neural network models (e.g., BERT, ResNet).
2. The RL environment simulates execution where the agent chooses offloading actions (what to run on CPU vs PIM) given observations of model characteristics and memory state.
3. Ramulator2 runs cycle-accurate memory simulations for each decision to compute accurate memory performance metrics — these feed into reward computation.
4. PPO trains a policy to maximize a reward that encodes throughput/latency/energy trade-offs.
5. Traces and logs are saved for offline analysis and visualization. 📈

---

## Extending the Framework

- Add a new workload:
  - Add a workload parser/profiler in `workload_analysis.py` (or create a new module).
  - Register the workload in your config YAML and the RL environment.

- Add a new memory configuration:
  - Create/modify a `ramulator` config YAML and update `cxl_pim_config.yaml`.
  - Rebuild Ramulator2 if needed.

- Add a new RL algorithm:
  - Swap PPO with another algorithm in the training script (e.g., from `stable-baselines3`) and tune hyperparameters.

---

## Troubleshooting & Tips

- Ramulator build fails: ensure C++20-compatible compiler and required build tools are installed. Check `ramulator2/CMakeLists.txt` for dependency details.
- Long training runs: start with a small number of timesteps or a reduced environment to validate the training loop quickly.
- Reproducibility: fix random seeds in both Python and the simulator for deterministic runs when needed.
- GPU vs CPU: If workloads use PyTorch, ensure CUDA is configured if you want GPU-accelerated profiling.

---

## Citing This Work

If Neuro-CXL-PIM helps your research, please cite it:

```bibtex
@misc{neuro-cxl-pim,
  title={Neuro-CXL-PIM: RL-based Intelligent CPU/PIM Offloading for Neural Networks},
  author={Your Name},
  year={2026},
  url={https://github.com/lkcfqy/Neuro-CXL-PIM}
}
```

---

## License & Acknowledgements

- License: MIT
- Acknowledgements:
  - [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) — cycle-accurate memory simulator ❤️
  - [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation
  - Other open-source projects and community contributions

---

## Contact

Maintainer: lkcfqy  
If you'd like me to add more examples, CI config, or automatically create a PR to update this README in your repo, say the word and I'll do it for you! 🐱‍🏍

Have fun exploring CPU/PIM design space — and stay cute while running experiments! ✨🐥
