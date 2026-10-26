"""Entry point for the verified frozen Tile20 runtime; no retraining required.

Usage (under torch.no_grad and the existing sdpa_backend context):
    net, metadata = load()
    net.select('Tile20_after')  # reset conditioning/history ownership per video
    chunks, diagnostics = trajectory(net, prompt_embeds, prompt_id, seed, True)

The first rollout builds shape-specific CUDA Graphs. Timings in REPORT.txt
exclude that warmup. The supported model is the existing BF16 T2V B=1 model.
"""
from .rollout import trajectory

def load():
 from ..opt20.network import load as parent_load
 from .step_graph import configure_graph
 from .rope import install
 net,metadata=parent_load();configure_graph(net,kernel=True);install()
 return net,metadata
