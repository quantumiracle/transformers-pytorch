class ImprovedMemoryModule(nn.Module):
    def __init__(self, hidden_size, num_heads, memory_size, memory_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # For memory queries, keys, values projections
        self.query_proj = nn.Linear(memory_dim, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # For structured memory representation
        self.struct_proj = nn.Linear(hidden_size, memory_dim)
        
        # Learnable parameters
        self.decay = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.9)  # Initial decay of 0.9
        self.update_rate = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.1)  # Learning rate for delta rule
        
        # Memory gate - controls information flow
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size + memory_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def update_memory(self, memory, keys, values):
        """
        memory: [B, H, L, D] - Previous memory state
        keys: [B, H, N, D] - New input keys
        values: [B, H, N, D] - New input values
        """
        batch_size, n_heads = memory.shape[0], memory.shape[1]
        
        # Step 1: Generate memory queries for retrieval
        q_mem = self.query_proj(memory)  # [B, H, L, D]
        
        # Step 2: Predict values from current memory
        # First retrieve what memory predicts for new keys
        attn_weights = torch.matmul(q_mem, keys.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        attn_weights = F.softmax(attn_weights, dim=-1)
        predicted_values = torch.matmul(attn_weights, values)  # [B, H, L, D]
        
        # Step 3: Compute prediction error (delta)
        # Attention from keys to memory
        key_to_mem_attn = torch.matmul(keys, memory.transpose(-2, -1)) / math.sqrt(self.memory_dim)
        key_to_mem_attn = F.softmax(key_to_mem_attn, dim=-1)
        # Map values to memory space
        target_memory = torch.matmul(key_to_mem_attn.transpose(-2, -1), values)  # [B, H, L, D]
        
        # Error between target and current memory (delta rule)
        delta = target_memory - predicted_values
        
        # Step 4: Update memory with decay and delta correction
        # Compute importance weighting for current memory
        mem_importance = torch.sigmoid(memory.mean(dim=-1, keepdim=True))
        
        # Apply adaptive decay - more important memories decay slower
        effective_decay = self.decay * (1.0 + 0.1 * mem_importance)
        effective_decay = torch.clamp(effective_decay, 0.5, 0.99)
        
        # Apply decay and update
        new_memory = effective_decay * memory + self.update_rate * delta
        
        # Step 5: Gate the update to prevent catastrophic forgetting
        gate_input = torch.cat([
            memory.mean(dim=1, keepdim=True).expand(-1, n_heads, -1, -1),
            delta.mean(dim=1, keepdim=True).expand(-1, n_heads, -1, -1)
        ], dim=-1)
        gate = self.memory_gate(gate_input)
        
        # Final memory update with gating
        updated_memory = memory * (1 - gate) + new_memory * gate
        
        # Normalize memory to prevent value explosion
        updated_memory = F.layer_norm(
            updated_memory, 
            normalized_shape=[updated_memory.size(-1)]
        )
        
        return updated_memory
        
    def retrieve_from_memory(self, queries, memory):
        """
        queries: [B, H, N, D] - Queries from current tokens
        memory: [B, H, L, D] - Memory state
        """
        # Project memory to key and value spaces
        mem_keys = self.key_proj(memory)
        mem_values = self.value_proj(memory)
        
        # Apply rotary embeddings for positional information
        queries, mem_keys = apply_rotary_emb(queries, mem_keys)
        
        # Attend to memory
        attn_scores = torch.matmul(queries, mem_keys.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Retrieve memory content
        retrieved = torch.matmul(attn_probs, mem_values)
        
        return retrieved