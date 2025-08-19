import os
import sys
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None


from diffusers.utils import export_to_video, load_image

from cogvideox.models.omnieffects.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from cogvideox.models.omnieffects.attention_processor import CogVideoXAttnLoraProcessor2_0
from cogvideox.pipelines.omnieffects.pipeline_omnieffects import CogVideoXOmniEffectsPipeline

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    return datas

def generate_video(args):
    device = f"cuda:{args.device_id}"
    os.makedirs(f"results/{args.output}", exist_ok=True)

    ## load pretrained
    pretrained_model_name_or_path = args.model_path # "/path/to/CogVideoX-5b-I2V"
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer"
    )

    ## load lora
    lora_name = "adapter"
    lora_path = args.lora_path # "/path/to/lora_path"
    moe_lora_path = os.path.join(args.lora_path, "moe_lora.pth")
    cond_lora_path = os.path.join(args.lora_path, "cond_lora.pth")

    ## set lora parameters
    n_loras = args.num_cond
    transformer.n_loras = n_loras
    num_experts = args.num_experts
    top_k = args.top_k
    rank = 128
    lora_weight = 1.0
    network_alpha = 128
    dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    

    ## load moe lora parameters
    transformer.add_moe_lora(
        rank=rank,
        network_alpha=network_alpha,
        num_experts=num_experts,
        top_k=top_k,
        device=device, 
        dtype=torch.bfloat16,
    )
    moe_lora_state_dict = torch.load(moe_lora_path)
    transformer.load_state_dict(moe_lora_state_dict, strict=False)

    ## load condition branch lora parameters
    attn_processor_type = CogVideoXAttnLoraProcessor2_0
    dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    ranks = [rank] * n_loras
    lora_weights = [lora_weight] * n_loras
    network_alphas = [network_alpha] * n_loras

    if args.base_lora:
        ranks = [rank] + ranks
        lora_weights = [lora_weight] + lora_weights
        network_alphas = [network_alpha] + network_alphas

    cond_lora_state_dict = torch.load(cond_lora_path)
    attn_processors = {}

    for key, value in transformer.attn_processors.items():
        block_idx = int(key.split(".")[1])

        if block_idx in list(range(transformer.config.num_layers)):
            attn_processor = attn_processor_type(
                dim=dim, 
                ranks=ranks,
                lora_weights=lora_weights,
                network_alphas=network_alphas,
                device=device, 
                dtype=torch.bfloat16,
                cond_width=720,
                cond_height=480,
                n_loras=n_loras,
                cond_lora=True,
                base_lora=args.base_lora,
                text_visual_attention=args.text_visual_attention,
                m2v_mask=args.m2v_mask,
                full_attention=args.full_attention

            )
            for param in attn_processor.parameters():
                param.requires_grad_(False)
            attn_processors[key] = attn_processor

            for idx in range(n_loras):
                attn_processors[key].cond_q_loras[idx].down.weight.data = cond_lora_state_dict.get(f'{key}.cond_q_loras.0.down.weight', None)
                attn_processors[key].cond_q_loras[idx].up.weight.data   = cond_lora_state_dict.get(f'{key}.cond_q_loras.0.up.weight', None)
                attn_processors[key].cond_k_loras[idx].down.weight.data = cond_lora_state_dict.get(f'{key}.cond_k_loras.0.down.weight', None)
                attn_processors[key].cond_k_loras[idx].up.weight.data   = cond_lora_state_dict.get(f'{key}.cond_k_loras.0.up.weight', None)
                attn_processors[key].cond_v_loras[idx].down.weight.data = cond_lora_state_dict.get(f'{key}.cond_v_loras.0.down.weight', None)
                attn_processors[key].cond_v_loras[idx].up.weight.data   = cond_lora_state_dict.get(f'{key}.cond_v_loras.0.up.weight', None)
                attn_processors[key].cond_proj_loras[idx].down.weight.data = cond_lora_state_dict.get(f'{key}.cond_proj_loras.0.down.weight', None)
                attn_processors[key].cond_proj_loras[idx].up.weight.data = cond_lora_state_dict.get(f'{key}.cond_proj_loras.0.up.weight', None)
                attn_processors[key].to(device)
        else:
            attn_processors[key] = value

    transformer.set_attn_processor(attn_processors)
    transformer.add_cond_linear(
        480 // 8, 
        720 // 8,
        cond_linear=args.cond_linear
    )
    
    transformer.load_state_dict(cond_lora_state_dict, strict=False)
    transformer.to(dtype=torch.bfloat16)

    pipe = CogVideoXOmniEffectsPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to(device)

    if args.mha_lora:
        pipe.load_lora_weights(
            lora_path, 
            weight_name="pytorch_lora_weights.safetensors", 
            adapter_name=lora_name
        )
        pipe.set_adapters([lora_name], [1.0])
        pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    datas = load_json(args.data_path)

    if args.num_to_evaluate > 0:
        num_to_evaluate = min(args.num_to_evaluate, len(datas))
    else:
        num_to_evaluate = len(datas)

    for i in tqdm(range(num_to_evaluate)):
        data = datas[i]
        image = load_image(data['file_path'])
        prompt = data['text']
        cond_image = []
        video_name = "_".join(prompt)

        if n_loras == 1:
            cond_image.append(load_image(data['mask_path']))
        elif n_loras > 1:
            cond_image = [load_image(data[f'mask_path{i+1}']) for i in range(n_loras)]
        else:
            raise ValueError("n_loras must be greater than 0")
            
        video = pipe(
            image=image,
            cond_image=cond_image,
            prompt=prompt,
            height=480,
            width=720,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(42),
        ).frames[0]

        export_to_video(video, f"results/{args.output}/{video_name}_{os.path.basename(data['file_path']).split('.')[0]}.mp4", fps=8)


def main():
    parser = argparse.ArgumentParser(description="Generate video using CogVideoX and LoRA weights")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b-I2V", help="Base Model path or HF ID")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--lora_name", type=str, default="lora_adapter", help="Name of the LoRA adapter")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--mha_lora", action="store_true")
    parser.add_argument("--text_visual_attention",action="store_true")
    parser.add_argument("--base_lora",action="store_true")
    parser.add_argument("--cond_linear",action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_cond", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--m2v_mask", action="store_true")
    parser.add_argument("--num_to_evaluate", type=int, default=-1)
    parser.add_argument("--full_attention", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=6.0)


    args = parser.parse_args()

    generate_video(args)


if __name__ == '__main__':
    main()
