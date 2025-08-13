import mlx.core as mx
import mlx.nn as nn
import json
import os
from collections import namedtuple

ModelOutput = namedtuple("ModelOutput", ["logits", "hidden_states", "past_key_values"])

class GPT2Config:
    def __init__(self, config_path=None, **kwargs):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_head = 12
        self.n_layer = 12
        self.layer_norm_epsilon = 1e-5
        self.attn_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.activation_function = "gelu_new"
        self.bos_token_id = 50256
        self.eos_token_id = 50256
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self._load_from_dict(config_dict)
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _load_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.bias = mx.tril(mx.ones((config.n_positions, config.n_positions)))
        self.bias = self.bias.reshape(1, 1, config.n_positions, config.n_positions)
        
    def __call__(self, x, past_key_value=None):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        
        present_key_value = (k, v)
        
        kv_seq_len = k.shape[2]
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        
        mask = self.bias[:, :, kv_seq_len-T:kv_seq_len, :kv_seq_len]
        mask = (1.0 - mask) * -1e10
        scores = scores + mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = mx.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        
        return out, present_key_value

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * self.n_embd, self.n_embd, bias=True)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.activation = self._get_activation(getattr(config, 'activation_function', 'gelu'))
        
    def _get_activation(self, activation_function):
        if activation_function == "gelu_new":
            return nn.gelu_approx
        elif activation_function == "relu":
            return nn.relu
        else:
            return nn.gelu
        
    def __call__(self, x):  
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.resid_dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        
    def __call__(self, x, past_key_value=None):
        attn_out, present_key_value = self.attn(self.ln_1(x), past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_positions = config.n_positions
        
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.embd_dropout = nn.Dropout(config.embd_pdrop)
        
        self.h = [Block(config) for _ in range(self.n_layer)]
        
        self.ln_f = nn.LayerNorm(self.n_embd, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte
        
    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, past_key_values=None, output_hidden_states=False):
        if input_ids is not None:
            B, T = input_ids.shape
            token_embeds = self.wte(input_ids)
        elif inputs_embeds is not None:
            B, T, _ = inputs_embeds.shape
            token_embeds = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        if position_ids is not None:
            position_embeds = self.wpe(position_ids)
        else:
            positions = mx.arange(T)
            position_embeds = self.wpe(positions)
        
        x = token_embeds + position_embeds
        x = self.embd_dropout(x)
        
        hidden_states = []
        if output_hidden_states:
            hidden_states.append(x)
        
        present_key_values = []
        
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            x, present_kv = block(x, past_key_value=layer_past)
            present_key_values.append(present_kv)
            
            if output_hidden_states:
                hidden_states.append(x)
        
        x = self.ln_f(x)
        if output_hidden_states:
            hidden_states.append(x)
        
        logits = mx.matmul(x, self.wte.weight.T)
        
        return ModelOutput(
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
            past_key_values=present_key_values
        )
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.0, eos_token_id=None):
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        
        _, input_length = input_ids.shape
        generated = input_ids.tolist()
        past_key_values = None
        
        for step in range(max_new_tokens):
            if step == 0:
                current_input = input_ids
                position_ids = mx.arange(input_length).reshape(1, -1)
            else:
                current_input = mx.array([[generated[0][-1]]])
                position_ids = mx.array([[input_length + step - 1]])
            
            outputs = self(
                input_ids=current_input,
                position_ids=position_ids,
                past_key_values=past_key_values
            )
            
            past_key_values = outputs.past_key_values
            
            next_token_logits = outputs.logits[0, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = mx.softmax(next_token_logits)
                next_token = mx.random.categorical(mx.log(probs)).item()
            else:
                next_token = mx.argmax(next_token_logits).item()
            
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            generated[0].append(next_token)
        
        return mx.array(generated)