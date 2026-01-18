"""
Unit tests for WorkloadProfiler
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload_analysis import WorkloadProfiler


class TestWorkloadProfiler:
    """Tests for WorkloadProfiler class"""
    
    def test_bert_profile_returns_numpy_array(self):
        """Test that BERT profiling returns numpy array"""
        profiler = WorkloadProfiler('bert')
        layers = profiler.profile()
        
        assert isinstance(layers, np.ndarray)
        assert layers.dtype == np.float32
    
    def test_bert_profile_shape(self):
        """Test BERT profile output shape: (N, 3) where 3 = [FLOPs, DataSize, TypeID]"""
        profiler = WorkloadProfiler('bert')
        layers = profiler.profile()
        
        assert layers.ndim == 2
        assert layers.shape[1] == 3  # [FLOPs, DataSize, TypeID]
        assert layers.shape[0] > 0   # At least one layer
    
    def test_bert_layer_count(self):
        """BERT-Base should have ~97 layers (embeddings + 12 encoder blocks * 8 sublayers)"""
        profiler = WorkloadProfiler('bert')
        layers = profiler.profile()
        
        # 1 embedding + 12 * (QKV + Attention + OutProj + Norm + FF1 + Act + FF2 + Norm)
        # = 1 + 12 * 8 = 97 layers
        assert layers.shape[0] == 97
    
    def test_bert_type_ids_valid(self):
        """TypeIDs should be 0, 1, or 2"""
        profiler = WorkloadProfiler('bert')
        layers = profiler.profile()
        
        type_ids = layers[:, 2]
        assert np.all((type_ids >= 0) & (type_ids <= 2))
    
    def test_bert_flops_positive(self):
        """FLOPs should be positive"""
        profiler = WorkloadProfiler('bert')
        layers = profiler.profile()
        
        flops = layers[:, 0]
        assert np.all(flops > 0)
    
    def test_bert_data_size_positive(self):
        """Data sizes should be positive"""
        profiler = WorkloadProfiler('bert')
        layers = profiler.profile()
        
        data_sizes = layers[:, 1]
        assert np.all(data_sizes > 0)
    
    def test_resnet_profile(self):
        """Test ResNet-18 profiling"""
        profiler = WorkloadProfiler('resnet')
        layers = profiler.profile()
        
        assert isinstance(layers, np.ndarray)
        assert layers.ndim == 2
        assert layers.shape[1] == 3
        assert layers.shape[0] == 7  # Conv1, MaxPool, Layer1-4, FC
    
    def test_unknown_model_defaults_to_bert(self):
        """Unknown model should default to BERT"""
        profiler = WorkloadProfiler('unknown_model')
        layers = profiler.profile()
        
        # Should return same as BERT
        bert_profiler = WorkloadProfiler('bert')
        bert_layers = bert_profiler.profile()
        
        assert layers.shape == bert_layers.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
