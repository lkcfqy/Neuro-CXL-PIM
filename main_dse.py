import os
import subprocess
import re
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from workload_analysis import WorkloadProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 路径配置 ====================
RAMULATOR_PATH = os.environ.get(
    'RAMULATOR_PATH', 
    os.path.join(os.path.dirname(__file__), 'ramulator2', 'build', 'ramulator2')
)

# ==================== 硬件配置参数 ====================
class HardwareConfig:
    """基于真实硬件参数的配置类"""
    # CPU 参数 (假设 Intel Xeon, AVX-512)
    CPU_FREQ_GHZ = 3.0
    CPU_FP32_OPS_PER_CYCLE = 32  # AVX-512: 2 FMA units * 16 FP32 = 32 ops/cycle
    CPU_POWER_IDLE_W = 10.0
    CPU_POWER_PEAK_W = 150.0
    
    # PIM 参数 (假设 HBM-PIM 类似架构)
    PIM_FREQ_GHZ = 1.0
    PIM_FP32_OPS_PER_CYCLE = 8  # 每 bank 1-2 MAC units
    PIM_POWER_W = 15.0
    
    # CXL 参数
    CXL_LINK_POWER_PER_ACCESS_NJ = 5.0  # 每次访问的能耗 (nJ)
    
    # 归一化基线
    BASELINE_EDP = 1e12  # 用于奖励归一化


class RewardLogger(BaseCallback):
    """记录训练过程中的奖励用于可视化"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True


class PIMDesignSpace(gym.Env):
    def __init__(self, model_name='bert'):
        super(PIMDesignSpace, self).__init__()
        
        # 使用真实模型 Profiling
        print(f"Profiling Model ({model_name.upper()})...")
        profiler = WorkloadProfiler(model_name)
        self.layers = profiler.profile()
        # layers row: [FLOPs, DataSize_Bytes, TypeID]
        # TypeID: 0=Compute, 1=Mem(Simple), 2=Attention(Complex)
        
        self.current_step = 0
        self.hw = HardwareConfig()
        
        # 记录每个 episode 的决策和指标
        self.episode_decisions = []
        self.episode_metrics = {'latency': 0, 'energy': 0}
        
        # Action: 0 = CPU, 1 = PIM
        self.action_space = spaces.Discrete(2)
        
        # Observation: [FLOPs_norm, DataSize_norm, TypeID, ComputeIntensity]
        # 添加计算密度作为额外特征
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # 计算特征归一化参数
        self._compute_normalization_params()

        os.makedirs("traces", exist_ok=True)

    def _compute_normalization_params(self):
        """预计算特征归一化参数"""
        self.max_flops = np.max(self.layers[:, 0]) + 1
        self.max_data_size = np.max(self.layers[:, 1]) + 1
        
    def _get_normalized_obs(self, layer_idx):
        """获取归一化的观察值"""
        flops, data_size, type_id = self.layers[layer_idx]
        
        # 计算密度 = FLOPs / DataSize (高密度适合 CPU，低密度适合 PIM)
        compute_intensity = flops / (data_size + 1)
        max_intensity = self.max_flops / 1024  # 大约估计
        
        return np.array([
            flops / self.max_flops,
            data_size / self.max_data_size,
            type_id / 2.0,  # TypeID 归一化到 [0, 1]
            min(compute_intensity / max_intensity, 1.0)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_decisions = []
        self.episode_metrics = {'latency': 0, 'energy': 0}
        return self._get_normalized_obs(0), {}

    def step(self, action):
        layer_features = self.layers[self.current_step]
        flops, data_size, type_id = layer_features
        
        # 1. 运行仿真获取 Memory Cycles
        mem_cycles = self._run_simulation(action, layer_features)
        
        # 2. 计算延迟 (基于真实硬件参数)
        compute_cycles = self._compute_latency(action, flops, type_id)
        total_cycles = mem_cycles + compute_cycles
        
        # 3. 能耗模型 (更精细)
        energy = self._compute_energy(action, compute_cycles, mem_cycles)
        
        # 4. 累积指标
        self.episode_metrics['latency'] += total_cycles
        self.episode_metrics['energy'] += energy
        self.episode_decisions.append('PIM' if action == 1 else 'CPU')
            
        # 5. 奖励计算 (改进的奖励函数)
        reward = self._compute_reward(total_cycles, energy, type_id, action)
        
        # 6. 状态更新
        self.current_step += 1
        done = self.current_step >= len(self.layers)
        
        if done:
            next_obs = np.zeros(4, dtype=np.float32)
        else:
            next_obs = self._get_normalized_obs(self.current_step)
        
        return next_obs, reward, done, False, {
            'latency': total_cycles,
            'energy': energy,
            'decision': 'PIM' if action == 1 else 'CPU'
        }
    
    def _compute_latency(self, action, flops, type_id):
        """基于硬件参数计算延迟"""
        if action == 0:  # CPU
            # CPU 计算能力强，但受内存带宽限制
            ops_per_second = self.hw.CPU_FP32_OPS_PER_CYCLE * self.hw.CPU_FREQ_GHZ * 1e9
            compute_time_s = flops / ops_per_second
            compute_cycles = compute_time_s * self.hw.CPU_FREQ_GHZ * 1e9
        else:  # PIM
            # PIM 计算能力弱，但近数据计算
            ops_per_second = self.hw.PIM_FP32_OPS_PER_CYCLE * self.hw.PIM_FREQ_GHZ * 1e9
            compute_time_s = flops / ops_per_second
            compute_cycles = compute_time_s * self.hw.PIM_FREQ_GHZ * 1e9
            
            # Attention (Type 2) 对 PIM 更复杂，需要额外开销
            if type_id == 2:
                compute_cycles *= 1.5
                
        return compute_cycles
    
    def _compute_energy(self, action, compute_cycles, mem_cycles):
        """计算能耗"""
        if action == 0:  # CPU
            # CPU 功耗 = 基础功耗 + 动态功耗
            utilization = min(1.0, compute_cycles / (compute_cycles + mem_cycles + 1))
            cpu_power = self.hw.CPU_POWER_IDLE_W + utilization * (self.hw.CPU_POWER_PEAK_W - self.hw.CPU_POWER_IDLE_W)
            
            total_time_s = (compute_cycles + mem_cycles) / (self.hw.CPU_FREQ_GHZ * 1e9)
            compute_energy = cpu_power * total_time_s
            
            # CXL 链路能耗
            link_energy = mem_cycles * self.hw.CXL_LINK_POWER_PER_ACCESS_NJ * 1e-9
            
            return compute_energy + link_energy
        else:  # PIM
            total_time_s = (compute_cycles + mem_cycles) / (self.hw.PIM_FREQ_GHZ * 1e9)
            return self.hw.PIM_POWER_W * total_time_s
    
    def _compute_reward(self, latency, energy, type_id, action):
        """改进的奖励函数"""
        # EDP (Energy-Delay Product) 归一化
        edp = energy * latency
        normalized_edp = edp / self.hw.BASELINE_EDP
        
        # 基础奖励 (线性负奖励，避免 log 不稳定)
        base_reward = -normalized_edp
        
        # 启发式奖励: 鼓励合理的决策
        heuristic_bonus = 0.0
        
        # Memory-bound (Type 1) 层适合 PIM
        if type_id == 1 and action == 1:
            heuristic_bonus += 0.1
        
        # Compute-bound (Type 0) 层适合 CPU
        if type_id == 0 and action == 0:
            heuristic_bonus += 0.05
            
        return float(base_reward + heuristic_bonus)

    def _run_simulation(self, action, features):
        step_id = self.current_step
        trace_file = f"traces/step_{step_id}.trace"
        config_file = f"traces/step_{step_id}.yaml"
        
        self._generate_trace_file(action, features, trace_file)
        self._generate_config_file(trace_file, config_file)
        
        cmd = [
            RAMULATOR_PATH,
            "-f", config_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            output = result.stdout
            match = re.search(r"memory_system_cycles:\s+(\d+)", output)
            if match:
                return int(match.group(1))
            else:
                logger.warning(f"Step {step_id}: Could not parse memory_system_cycles from output")
                return 10000
        except subprocess.CalledProcessError as e:
            logger.warning(f"Step {step_id}: Ramulator2 failed with return code {e.returncode}")
            return 20000
        except subprocess.TimeoutExpired:
            logger.warning(f"Step {step_id}: Ramulator2 simulation timed out (30s)")
            return 20000
        except FileNotFoundError:
            logger.error(f"Ramulator2 not found at: {RAMULATOR_PATH}")
            raise RuntimeError(f"Ramulator2 executable not found. Please build it or set RAMULATOR_PATH environment variable.")

    def _generate_trace_file(self, action, features, filepath):
        _, data_size, _ = features
        # Generate trace based on data size
        num_reqs = max(1, int(data_size / 64))
        limit = 100  # 增加 trace 限制以获得更准确的仿真结果
        
        with open(filepath, "w") as f:
            if action == 0:  # CPU - Load/Store 模式
                for i in range(min(num_reqs, limit)): 
                    addr = hex(0x100000 + i * 64)
                    f.write(f"LD {addr}\n")
            else:  # PIM - Use write-heavy pattern to simulate PIM behavior
                for i in range(min(num_reqs, limit)):
                    addr = hex(0x100000 + i * 64)
                    f.write(f"ST {addr}\n")  # ST represents write-heavy PIM workload 

    def _generate_config_file(self, trace_path, config_path):
        # Note: Using DDR4 as fallback. HBM3/PIM_HBM3 requires additional 
        # ramulator2 compatibility fixes for pseudochannel/rank hierarchy
        yaml_content = f"""
Frontend:
  impl: LoadStoreTrace
  path: {trace_path}
  clock_ratio: 1
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 1
  DRAM:
    impl: DDR4
    org:
      preset: DDR4_8Gb_x8
      channel: 1 
    timing:
      preset: DDR4_2400R
  Controller:
    impl: Generic
    Scheduler:
      impl: FRFCFS
    RefreshManager:
      impl: AllBank
    RowPolicy:
      impl: ClosedRowPolicy
      cap: 4
  AddrMapper:
    impl: RoBaRaCoCh
"""
        with open(config_path, "w") as f:
            f.write(yaml_content)


def train_and_evaluate(total_timesteps=10000, verbose=True):
    """训练并评估模型"""
    env = PIMDesignSpace('bert')
    
    if verbose:
        print("=" * 60)
        print("Neuro-CXL-PIM: RL-based CPU/PIM Offloading Decision")
        print("=" * 60)
        print(f"Model: BERT-Base ({len(env.layers)} layers)")
        print(f"Training timesteps: {total_timesteps}")
        print("=" * 60)
    
    # 初始化 RL Agent
    print("\nInitializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1 if verbose else 0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device='cpu'  # PPO 在 CPU 上通常更高效
    )
    
    # 训练
    print(f"\nStarting Training ({total_timesteps} timesteps)...")
    reward_logger = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=reward_logger)
    
    # 保存模型
    model.save("pim_offload_policy")
    print("\nModel saved to 'pim_offload_policy.zip'")
    
    # 评估
    print("\n" + "=" * 60)
    print("Evaluation on Full Model")
    print("=" * 60)
    
    obs, _ = env.reset()
    total_layers = len(env.layers)
    decisions = {'CPU': 0, 'PIM': 0}
    layer_decisions = []
    
    for i in range(total_layers):
        action, _ = model.predict(obs, deterministic=True)
        choice = "PIM" if action == 1 else "CPU"
        decisions[choice] += 1
        layer_decisions.append(choice)
        
        layer_type = ['Compute', 'Memory', 'Attention'][int(env.layers[i][2])]
        
        if verbose and i < 20:  # 只打印前 20 层
            print(f"Layer {i:2d} [{layer_type:8s}]: {choice}")
        
        obs, _, done, _, info = env.step(action)
        if done:
            break
    
    if verbose and total_layers > 20:
        print(f"... ({total_layers - 20} more layers)")
    
    # 统计结果
    print("\n" + "-" * 40)
    print("Decision Summary:")
    print(f"  CPU layers: {decisions['CPU']} ({100*decisions['CPU']/total_layers:.1f}%)")
    print(f"  PIM layers: {decisions['PIM']} ({100*decisions['PIM']/total_layers:.1f}%)")
    print("-" * 40)
    print(f"Total Latency: {env.episode_metrics['latency']:.2e} cycles")
    print(f"Total Energy:  {env.episode_metrics['energy']:.2e} J")
    print("=" * 60)
    
    return model, reward_logger.episode_rewards, layer_decisions


if __name__ == "__main__":
    model, rewards, decisions = train_and_evaluate(total_timesteps=10000)
