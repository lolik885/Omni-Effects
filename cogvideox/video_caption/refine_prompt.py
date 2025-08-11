import os
import argparse

from vlm import VLM
from vlm_config import LLM_SERVER_URL, LLM_API_KEYS

from caption_rewrite import extract_output


def parse_args():
    parser = argparse.ArgumentParser(description="Beautiful prompt.")
    parser.add_argument("--vlm_model_type", type=str, required=True, help="The OpenAI model or the path to your local LLM.")
    parser.add_argument(
        "--template",
        type=str,
        default="cogvideox/video_caption/prompt/beautiful_prompt.txt",
        help="A string or a txt file contains the template for beautiful prompt."
    )
    parser.add_argument(
        "--max_retry_nums",
        type=int,
        default=5,
        help="Maximum number of retries to obtain an output that meets the JSON format."
    )
    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    vlm = VLM(
        url=LLM_SERVER_URL,
        keys=LLM_API_KEYS,
        model=args.vlm_model_type
    )

    if args.template.endswith(".txt") and os.path.exists(args.template):
        with open(args.template, "r") as f:
            args.template = "".join(f.readlines())

    image_path = args.image_path
    prompt_text = 'Please generate prompt words to be used for AI-generated videos. Output with the following json format: {"initial description": "your detailed description here"}'
    for _ in range(args.max_retry_nums):
        initial_prompt = vlm.forward(image_path, prompt_text)
        if initial_prompt is None:
            continue
        initial_prompt = extract_output(initial_prompt, prefix='"initial description": ')
        print(initial_prompt)
        refine_prompt = vlm.forward([], args.template + "\n" + initial_prompt)
        print(refine_prompt)
        output = extract_output(refine_prompt, prefix='"detailed description": ')
        if output is not None:
            break
    print(f"Beautiful prompt: {output}")


if __name__ == '__main__':
    main()
