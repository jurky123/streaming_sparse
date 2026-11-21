from pathlib import Path
import torch
from local_experiments.broad_common import PROMPTS
from utils.wan_wrapper import WanTextEncoder

with torch.inference_mode():
    model = WanTextEncoder().to(device="cuda", dtype=torch.bfloat16)
    for split in PROMPTS.values():
        for name, prompt in split:
            if name in {"forest_run", "kitchen_cook", "snow_ski", "bird_flight", "city_walk"}:
                out = Path(__file__).parent / "embeddings" / (name + ".pt")
                torch.save(model([prompt])["prompt_embeds"].cpu(), out)
