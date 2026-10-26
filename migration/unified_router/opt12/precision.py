"""Deployment-matched FP16 values with FP32 training gradients."""
import torch


class ForwardValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, differentiable, deployed):
        return deployed.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def align(differentiable, deployed):
    return ForwardValue.apply(differentiable, deployed.detach())


class HalfBmmFP32Backward(torch.autograd.Function):
    """Keep the original FP16 BMM forward; avoid FP16 gradient overflow.

    Inputs enter this Function before casts, so autograd never casts the
    upstream gradient to half. The Jacobian uses the same rounded operands
    but both gradient matrix products and their outputs remain FP32.
    """
    @staticmethod
    def forward(ctx, left, right):
        a, b = left.half(), right.half()
        ctx.save_for_backward(a.float(), b.float())
        return torch.bmm(a, b).float()

    @staticmethod
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        g = gradient.float()
        left_grad = torch.bmm(g, b.transpose(1, 2)) if ctx.needs_input_grad[0] else None
        right_grad = torch.bmm(a.transpose(1, 2), g) if ctx.needs_input_grad[1] else None
        return left_grad, right_grad


def fp16_round(value):
    """Round the forward value to FP16 without casting its gradient to FP16."""
    return align(value, value.detach().half().float())


@torch.no_grad()
def clip_grad_norm(parameters, maximum=1.):
    """Strict finite check; FP64 norm avoids overflow when FP32 grads are finite."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.)
    total = torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
    if not torch.isfinite(total):
        raise RuntimeError('Non-finite gradient in FP32 surrogate backward')
    scale = (maximum / (total + 1e-6)).clamp(max=1.)
    for gradient in gradients:
        gradient.mul_(scale.to(gradient.dtype))
    return total
