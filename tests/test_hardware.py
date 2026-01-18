"""
Unit tests for HardwareConfig and related functions
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_dse import HardwareConfig, PIMDesignSpace


class TestHardwareConfig:
    """Tests for HardwareConfig class"""
    
    def test_cpu_freq_reasonable(self):
        """CPU frequency should be reasonable (1-5 GHz)"""
        assert 1.0 <= HardwareConfig.CPU_FREQ_GHZ <= 5.0
    
    def test_pim_freq_reasonable(self):
        """PIM frequency should be reasonable (0.5-2 GHz)"""
        assert 0.5 <= HardwareConfig.PIM_FREQ_GHZ <= 2.0
    
    def test_cpu_faster_than_pim(self):
        """CPU should have higher compute throughput than PIM"""
        cpu_throughput = HardwareConfig.CPU_FREQ_GHZ * HardwareConfig.CPU_FP32_OPS_PER_CYCLE
        pim_throughput = HardwareConfig.PIM_FREQ_GHZ * HardwareConfig.PIM_FP32_OPS_PER_CYCLE
        
        assert cpu_throughput > pim_throughput
    
    def test_power_values_positive(self):
        """Power values should be positive"""
        assert HardwareConfig.CPU_POWER_IDLE_W > 0
        assert HardwareConfig.CPU_POWER_PEAK_W > 0
        assert HardwareConfig.PIM_POWER_W > 0
    
    def test_idle_less_than_peak(self):
        """CPU idle power should be less than peak"""
        assert HardwareConfig.CPU_POWER_IDLE_W < HardwareConfig.CPU_POWER_PEAK_W
    
    def test_pim_more_efficient(self):
        """PIM should be more power efficient than CPU at peak"""
        assert HardwareConfig.PIM_POWER_W < HardwareConfig.CPU_POWER_PEAK_W


class TestPIMDesignSpace:
    """Tests for PIMDesignSpace Gym environment"""
    
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return PIMDesignSpace('bert')
    
    def test_action_space(self, env):
        """Action space should be Discrete(2) - CPU or PIM"""
        assert env.action_space.n == 2
    
    def test_observation_space_shape(self, env):
        """Observation space should have 4 features"""
        assert env.observation_space.shape == (4,)
    
    def test_reset_returns_valid_obs(self, env):
        """Reset should return valid observation"""
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert np.all(obs >= 0) and np.all(obs <= 1)
    
    def test_step_returns_correct_format(self, env):
        """Step should return (obs, reward, done, truncated, info)"""
        env.reset()
        obs, reward, done, truncated, info = env.step(0)  # Action 0 = CPU
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode_terminates(self, env):
        """Episode should terminate after all layers processed"""
        obs, _ = env.reset()
        num_layers = len(env.layers)
        done = False
        steps = 0
        
        while not done and steps < num_layers + 10:
            obs, _, done, _, _ = env.step(0)
            steps += 1
        
        assert done
        assert steps == num_layers


class TestLatencyCalculation:
    """Tests for latency and energy calculations"""
    
    @pytest.fixture
    def env(self):
        return PIMDesignSpace('bert')
    
    def test_cpu_compute_latency(self, env):
        """CPU compute latency calculation"""
        flops = 1e9  # 1 GFLOPs
        latency = env._compute_latency(0, flops, type_id=0)
        
        assert latency > 0
        assert isinstance(latency, float)
    
    def test_pim_compute_latency(self, env):
        """PIM compute latency calculation"""
        flops = 1e9
        latency = env._compute_latency(1, flops, type_id=0)
        
        assert latency > 0
    
    def test_attention_penalty_on_pim(self, env):
        """Attention layers should have penalty on PIM"""
        flops = 1e9
        normal_latency = env._compute_latency(1, flops, type_id=0)
        attention_latency = env._compute_latency(1, flops, type_id=2)
        
        assert attention_latency > normal_latency
    
    def test_energy_positive(self, env):
        """Energy should always be positive"""
        compute_cycles = 1e6
        mem_cycles = 1e5
        
        cpu_energy = env._compute_energy(0, compute_cycles, mem_cycles)
        pim_energy = env._compute_energy(1, compute_cycles, mem_cycles)
        
        assert cpu_energy > 0
        assert pim_energy > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
