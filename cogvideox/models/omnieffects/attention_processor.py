from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, LinearActivation, SwiGLU


class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width=512,
        cond_height=512,
        number=0,
        n_loras=1,
        lora_type='cond'
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number
        self.n_loras = n_loras
        self.lora_type = lora_type

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        #### img condition
        batch_size = hidden_states.shape[0]
        cond_size = (self.cond_width // 16) * (self.cond_height // 16)
        block_size =  hidden_states.shape[1] - cond_size * self.n_loras
        shape = (batch_size, hidden_states.shape[1], 3072)
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype) 
        if self.lora_type == 'cond':
            mask[:, :block_size+self.number*cond_size, :] = 0
            mask[:, block_size+(self.number+1)*cond_size:, :] = 0
            hidden_states = mask * hidden_states
        else:
            mask[:, block_size:, :] = 0
            hidden_states = mask * hidden_states
        ####
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class NoisyTopkRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, bias=False):
        super().__init__()
        #layer for router logits
        self.topkroute_linear = nn.Linear(hidden_dim, num_experts, bias=bias)
        self.noise_linear = nn.Linear(hidden_dim, num_experts, bias=bias)

    def forward(self, hidden_states):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(hidden_states)

        # Noise logits
        noise_logits = self.noise_linear(hidden_states)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        return noisy_logits


class LoraExpert(nn.Module):
    def __init__(
        self, 
        in_features,
        out_features,
        rank,
        network_alpha,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.down = nn.Linear(
            in_features, 
            rank, 
            bias=False, 
            device=device, 
            dtype=dtype
        )
        self.up = nn.Linear(
            rank, 
            out_features, 
            bias=False,
            device=device, 
            dtype=dtype
        )
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class MoELoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features: int,
        rank: int = 128,
        network_alpha: int = 128,
        num_experts: int = 4,
        top_k: int = 1,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.in_features = in_features

        self.gate = NoisyTopkRouter(in_features, self.num_experts, bias=False)
        self.lora_experts = nn.ModuleList(
            [
                LoraExpert(
                    in_features, 
                    out_features,
                    rank,
                    network_alpha
                ) for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.lora_experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class CogVideoXAttnLoraProcessor2_0(nn.Module):
    def __init__(
        self, 
        dim: int, 
        ranks=[], 
        lora_weights=[], 
        network_alphas=[], 
        num_experts=4, 
        top_k=1,
        device=None, 
        dtype=None, 
        cond_width=512, 
        cond_height=512, 
        n_loras=1,
        cond_lora=True,
        base_lora=True,
        text_visual_attention=False,
        m2v_mask=False,
        full_attention=False
    ):
        super().__init__()
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.cond_lora = cond_lora
        self.base_lora = base_lora
        self.text_visual_attention = text_visual_attention
        self.m2v_mask = m2v_mask
        self.full_attention = full_attention

        if cond_lora and base_lora:
            base_rank = ranks[0]
            ranks = ranks[1:]
            base_network_alpha = network_alphas[0]
            network_alphas = network_alphas[1:]
        
        if self.cond_lora:
            self.cond_q_loras = nn.ModuleList([
                LoRALinearLayer(
                    dim, dim, 
                    ranks[i], network_alphas[i], 
                    device=device, dtype=dtype, 
                    cond_width=cond_width, 
                    cond_height=cond_height, 
                    number=i, n_loras=n_loras,
                    lora_type='cond'
                )
                for i in range(n_loras)
            ])
            self.cond_k_loras = nn.ModuleList([
                LoRALinearLayer(
                    dim, dim, 
                    ranks[i], network_alphas[i], 
                    device=device, dtype=dtype, 
                    cond_width=cond_width, 
                    cond_height=cond_height, 
                    number=i, n_loras=n_loras,
                    lora_type='cond'
                )
                for i in range(n_loras)
            ])
            self.cond_v_loras = nn.ModuleList([
                LoRALinearLayer(
                    dim, dim, 
                    ranks[i], network_alphas[i], 
                    device=device, dtype=dtype, 
                    cond_width=cond_width, 
                    cond_height=cond_height, 
                    number=i, n_loras=n_loras,
                    lora_type='cond'
                )
                for i in range(n_loras)
            ])
            self.cond_proj_loras = nn.ModuleList([
                LoRALinearLayer(
                    dim, dim, 
                    ranks[i], network_alphas[i], 
                    device=device, dtype=dtype, 
                    cond_width=cond_width, 
                    cond_height=cond_height, 
                    number=i, n_loras=n_loras,
                    lora_type='cond'
                )
                for i in range(n_loras)
            ])
        if self.base_lora:
            self.q_loras = LoRALinearLayer(
                dim, dim,
                base_rank, base_network_alpha,
                device=device, dtype=dtype,
                cond_width=cond_width, 
                cond_height=cond_height, 
                number=0, n_loras=n_loras,
                lora_type='base'
            )
            self.k_loras = LoRALinearLayer(
                dim, dim,
                base_rank, base_network_alpha,
                device=device, dtype=dtype,
                cond_width=cond_width, 
                cond_height=cond_height, 
                number=0, n_loras=n_loras,
                lora_type='base'
            )
            self.v_loras = LoRALinearLayer(
                dim, dim,
                base_rank, base_network_alpha,
                device=device, dtype=dtype,
                cond_width=cond_width, 
                cond_height=cond_height, 
                number=0, n_loras=n_loras,
                lora_type='base'
            )
            self.proj_loras = LoRALinearLayer(
                dim, dim,
                base_rank, base_network_alpha,
                device=device, dtype=dtype,
                cond_width=cond_width, 
                cond_height=cond_height, 
                number=0, n_loras=n_loras,
                lora_type='base'
            )
        self.lora_weights = lora_weights
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        if self.cond_lora:
            per_text_seq_length = text_seq_length // self.n_loras
        visual_seq_length = hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if self.base_lora:
            query = query + self.lora_weights[0] * self.q_loras(hidden_states)
            key = key + self.lora_weights[0] * self.k_loras(hidden_states)
            value = value + self.lora_weights[0] * self.v_loras(hidden_states)

        if self.cond_lora:
            for i in range(self.n_loras):
                query = query + self.lora_weights[i] * self.cond_q_loras[i](hidden_states)
                key = key + self.lora_weights[i] * self.cond_k_loras[i](hidden_states)
                value = value + self.lora_weights[i] * self.cond_v_loras[i](hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        cond_size = (self.cond_width // 16) * (self.cond_height // 16)
        block_size =  hidden_states.shape[1] - cond_size * self.n_loras
        scaled_cond_size = cond_size
        scaled_seq_len = query.shape[2]
        scaled_block_size = scaled_seq_len - cond_size * self.n_loras

        num_cond_blocks = self.n_loras
        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
        if self.full_attention:
            mask *= 0
        else:
            mask[text_seq_length:scaled_block_size, :] = 0  # visual block row
            if self.m2v_mask:
                mask[text_seq_length:scaled_block_size, scaled_block_size:] = 1

            for i in range(num_cond_blocks):
                start_text = i * per_text_seq_length
                end_text   = (i + 1) * per_text_seq_length
                start_cond = i * scaled_cond_size + scaled_block_size
                end_cond   = (i + 1) * scaled_cond_size + scaled_block_size

                mask[start_text:end_text, start_text:end_text] = 0
                mask[start_text:end_text, start_cond:end_cond] = 0
                mask[start_cond:end_cond, start_text:end_text] = 0
                mask[start_cond:end_cond, start_cond:end_cond] = 0

                if self.text_visual_attention:
                    mask[start_text:end_text, text_seq_length:scaled_block_size] = 0

        mask = mask * -1e20
        mask = mask.to(query.dtype)
        attention_mask = mask

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        if self.base_lora:
            hidden_states = hidden_states + self.lora_weights[0] * self.proj_loras(hidden_states)

        if self.cond_lora:
            for i in range(self.n_loras):
                hidden_states = hidden_states + self.lora_weights[i] * self.cond_proj_loras[i](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        # scaled_block_size - text_seq_length ？
        cond_hidden_states = hidden_states[:, scaled_block_size-text_seq_length:, :]
        hidden_states = hidden_states[:, :scaled_block_size-text_seq_length, :]

        return hidden_states, encoder_hidden_states, cond_hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.dim = dim
        self.dim_out = dim_out

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def add_moe_lora(
        self,
        rank,
        network_alpha,
        num_experts,
        top_k,
        device,
        dtype
    ):
        self.lora_moe_block = MoELoRALinearLayer(
            in_features=self.dim,
            out_features=self.dim_out,
            rank=rank,
            network_alpha=network_alpha,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        residual = hidden_states
        
        for module in self.net:
            hidden_states = module(hidden_states)

        if hasattr(self, "lora_moe_block"):
            hidden_states_lora, router_logits = self.lora_moe_block(residual)
            hidden_states = hidden_states + hidden_states_lora
            return hidden_states, router_logits
        else:
            return hidden_states, None