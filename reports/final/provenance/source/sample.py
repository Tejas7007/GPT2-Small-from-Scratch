"""Record reproducible generations for a fixed prompt suite and checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("configs/prompts.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        parser.error("Prompts must be a nonempty JSON list of strings.")
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer(checkpoint["data_meta"]["tokenizer"])
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for index, prompt in enumerate(prompts):
            # A fixed seed per suite index permits checkpoint comparisons with
            # identical sampling settings. Preserve prompt order across runs.
            seed = args.seed + index
            torch.manual_seed(seed)
            ids = tokenizer.encode(prompt) or [tokenizer.eot_id]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            result = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
            record = {
                "prompt_index": index,
                "prompt": prompt,
                "text": tokenizer.decode(result[0].tolist()),
                "seed": seed,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "checkpoint": args.checkpoint.name,
                "step": checkpoint["step"],
                "training_tokens": checkpoint["tokens_seen"],
            }
            handle.write(json.dumps(record) + "\n")
    print(f"Recorded {len(prompts)} generations in {args.out}")


if __name__ == "__main__":
    main()
