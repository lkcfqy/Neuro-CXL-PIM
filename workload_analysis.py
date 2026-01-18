import torch
import torch.nn as nn
import numpy as np

class WorkloadProfiler:
    def __init__(self, model_name='bert'):
        self.model_name = model_name
        self.layers_info = []

    def profile(self):
        """
        Returns: List of [FLOPs, DataSize_Bytes, TypeID]
        TypeID: 0=ComputeBound(Conv/Linear), 1=MemoryBound(Act/Norm), 2=Attention(PIM Friendly)
        """
        if 'bert' in self.model_name.lower():
            return self._profile_bert_base()
        elif 'resnet' in self.model_name.lower():
            return self._profile_resnet18()
        else:
            return self._profile_bert_base()

    def _profile_bert_base(self):
        """
        Simulates BERT-Base Architecture (12 Layers, Hidden=768, Heads=12)
        Total Params: ~110M
        """
        print("Profiling Virtual BERT-Base Model...")
        layers = []
        
        # BERT Config
        batch_size = 1
        seq_len = 128
        hidden_size = 768
        intermediate_size = 3072 # 4 * hidden
        num_attention_heads = 12
        
        # 1. Embeddings (Lookups - Pure Memory Bound)
        vocab_size = 30522
        # Lookup: No FLOPs really, just data movement
        embed_size = vocab_size * hidden_size * 4 # 4 bytes
        layers.append([seq_len * hidden_size, embed_size, 1]) 

        # 12 Encoder Layers
        for i in range(12):
            # --- Attention Block ---
            
            # Q, K, V Projections (Linear)
            # 3 * (Hidden x Hidden)
            # FLOPs: 3 * 2 * B * S * H * H
            # Wts: 3 * H * H * 4
            qkv_flops = 3 * 2 * batch_size * seq_len * hidden_size * hidden_size
            qkv_wts = 3 * hidden_size * hidden_size * 4
            layers.append([qkv_flops, qkv_wts, 0]) # Compute Heavy
            
            # Scaled Dot Product Attention (MatMul + Softmax) -> Very Memory Bound for large seq
            # Q*K: B * Hds * S * S
            # Softmax
            # A*V: B * Hds * S * H_head
            # Approx FLOPs: 2 * B * S * S * H
            # Data: Reading Q, K, V, writing A.
            att_flops = 2 * batch_size * seq_len * seq_len * hidden_size 
            att_data = (batch_size * seq_len * hidden_size * 4) * 3 # Read Q,K,V
            layers.append([att_flops, att_data, 2]) # Attention (Type 2)
            
            # Output Projection (Linear)
            out_flops = 2 * batch_size * seq_len * hidden_size * hidden_size
            out_wts = hidden_size * hidden_size * 4
            layers.append([out_flops, out_wts, 0])
            
            # Layer Norm + Residual (Memory Bound)
            norm_flops = batch_size * seq_len * hidden_size * 5 # simple op
            norm_data = batch_size * seq_len * hidden_size * 4 * 2 # Read + Write
            layers.append([norm_flops, norm_data, 1])

            # --- Feed Forward Block (FFN) ---
            
            # Linear 1 (H -> 4H)
            ff1_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size
            ff1_wts = hidden_size * intermediate_size * 4
            layers.append([ff1_flops, ff1_wts, 0])
            
            # Activation (GELU) - Memory Bound
            layers.append([batch_size * seq_len * intermediate_size, batch_size * seq_len * intermediate_size * 4, 1])

            # Linear 2 (4H -> H)
            ff2_flops = 2 * batch_size * seq_len * intermediate_size * hidden_size
            ff2_wts = intermediate_size * hidden_size * 4
            layers.append([ff2_flops, ff2_wts, 0])
            
            # Layer Norm + Residual
            layers.append([norm_flops, norm_data, 1])

        return np.array(layers, dtype=np.float32)

    def _profile_resnet18(self):
        # 简化的 ResNet Profile (Fallback)
        # [FLOPs, DataSize, Type]
        print("Profiling Virtual ResNet-18...")
        return np.array([
            [1.18e8, 9408*4, 0],   # Conv1
            [0, 802816*4, 1],      # MaxPool
            [4.6e8, 147456*4, 0],  # Layer1
            [4.6e8, 589824*4, 0],  # Layer2
            [4.6e8, 2359296*4, 0], # Layer3
            [4.6e8, 9437184*4, 0], # Layer4
            [1e5, 512*1000*4, 0],  # FC
        ], dtype=np.float32)

if __name__ == "__main__":
    profiler = WorkloadProfiler('bert')
    layers = profiler.profile()
    print(f"BERT Layers: {len(layers)}")
    print(layers[:5])