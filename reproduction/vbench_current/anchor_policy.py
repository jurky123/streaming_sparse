def install(runtime, policy):
    assert policy in ('chunk_history10','frame_fixed15')
    assert runtime.graph_batches and runtime.legacy_attention
    runtime.set_first_frame_fixed15(True)
    runtime.batch_graph.anchor_policy=policy
    runtime.resident_route=False
