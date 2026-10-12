"""Generate text from a saved checkpoint; uses its tokenizer and architecture."""

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="The student")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer(checkpoint["data_meta"]["tokenizer"])
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    ids = tokenizer.encode(args.prompt) or [tokenizer.eot_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
