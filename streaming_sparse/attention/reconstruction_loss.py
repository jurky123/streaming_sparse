def normalized_attention_mse(sparse, dense, epsilon=1e-8):
    return (sparse.float() - dense.float()).square().sum() / (dense.float().square().sum() + epsilon)
